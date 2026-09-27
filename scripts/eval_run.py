"""评测运行器（Phase D §5.6：**报告制，不做硬门禁**）。

WHY 与 `tests/test_eval_trace_golden.py` 分工：那份 golden set 是**门禁**（断言 → 红灯），
本脚本是**度量**（跑一遍、算指标、落报告）—— 两者共用同一份语料（`CASES` 单一来源，
避免「门禁测的那批输入」与「报告展示的那批输入」悄悄分叉，那会让答辩数据与质量结论脱钩）。

三种模式：
    ① 离线（默认）：语料经离线 fake 模型跑真图，产出**轨迹指标**（意图/终态分布、降级率、
       澄清率、工具调用数、LLM 调用数=成本代理、引用覆盖率、护栏不变量违规数）。
       零网络、零 API Key、可重复 —— 答辩时随时可跑、可与历史报告对比趋势。
    ② `--live`：连**真 LLM**（+ 已接入的 MCP 工具）端到端跑同一批输入，额外记录延迟与真实
       成本；`--judge`（默认开）走现有 `LLMService` 结构化输出做 5 级 rubric 评分 ——
       不引入新 provider（plan §5.6）。
    ③ `--self-check`：只校验度量数学（命中率@k / MRR / 聚合）本身，零依赖。

**不做门禁**：本脚本永不因指标阈值返回非零退出码（除用法错误）——「评测分数决定构建成败」
会让团队为了绿而调语料。质量红线在 golden set 与单测里，本脚本只负责「把趋势说清楚」。

RAG 指标（命中率@k / MRR）需要**带标注的检索样本**（query → 相关 source_id）。RAG 属独立
MCP 服务、其检索结果到 citations 的接线尚未落地，故当前报告里如实记为
`{"status": "skipped", "reason": …}`；计算路径已实现（`hit_rate_at_k` / `reciprocal_rank`），
标注到位后只需喂数据、无需改代码。

用法（仓库根目录，依赖 `.env`）：
    uv run python -m scripts.eval_run                     # 离线轨迹指标
    uv run python -m scripts.eval_run --live              # 真 LLM 端到端 + judge
    uv run python -m scripts.eval_run --live --limit 5 --no-judge
    uv run python -m scripts.eval_run --self-check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field, SecretStr

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.llm import LLMService
from agent.memory import MemoryStore, wait_pending_saves
from agent.response.models import AgentResponse
from settings import get_settings
from shared.logging import (
    EVENT_ANSWER_GENERATED,
    EVENT_INTENT_CLASSIFIED,
    EVENT_TOOL_CALLED,
    LoggingConfig,
    RecordingListener,
    configure_logging,
)
from tests.conftest import RecordingFakeChatModel
from tests.test_eval_trace_golden import CASES, TraceCase

logger = logging.getLogger(__name__)

# 报告版本：结构变更加一（历史报告可比性靠它判定，而不是靠字段缺失猜）。
REPORT_VERSION: Final = 1

# 默认 RAG 命中率口径的 k（与记忆/RAG 召回 top_k 同量级，plan §5.6）。
DEFAULT_K: Final = 5

# 评测专用身份（**与真实用户隔离**：记忆按 user_id 隔离，评测数据绝不进入真实用户召回）。
EVAL_USER_ID: Final = "eval-runner"

# judge 的 5 级 rubric（1..5；等级含义写进提示词，避免模型自创刻度）。
JUDGE_PROMPT: Final = """你是对话质量评审。请对「助手回答」按 5 级 rubric 打分：
5 = 准确、完整且直接有用；
4 = 基本正确，略有遗漏但不误导；
3 = 可用但不完整或偏泛（用户仍需追问）；
2 = 部分无关或含糊，基本没帮上忙；
1 = 错误、答非所问或空话。
只依据用户输入与助手回答判断，不臆测内部实现，不要因为回答礼貌就加分。
"""


# —— 度量数学（纯函数：便于 --self-check 与单独推理）——


def hit_rate_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """命中率@k：前 k 个检索结果里是否含任一相关项（1.0 / 0.0；无标注项记 0）。"""
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """首个相关项的倒数排名（1/rank；一个都没有记 0）—— MRR 的单查询分量。"""
    for rank, item in enumerate(retrieved, start=1):
        if item in set(relevant):
            return 1.0 / rank
    return 0.0


def aggregate_rag(
    labeled: Sequence[tuple[Sequence[str], Collection[str]]], *, k: int = DEFAULT_K
) -> dict[str, Any]:
    """聚合 RAG 指标；**无标注样本时如实报 skipped**（而不是报 0 分）。

    WHY 关键：把「没测」记成 0 分，会让报告在 RAG 接线当天看起来「指标暴跌」，
    而实际是从「未测」变成了「测了」—— 这是最容易被误读成回归的一类假信号。
    """
    if not labeled:
        return {
            "status": "skipped",
            "k": k,
            "labeled_queries": 0,
            "hit_rate_at_k": None,
            "mrr": None,
            "reason": "语料中没有带标注的检索样本（query → 相关 source_id）；"
            "RAG 检索结果到 citations 的接线尚未落地，无法计算命中率/MRR",
        }
    hits = [hit_rate_at_k(r, rel, k) for r, rel in labeled]
    rr = [reciprocal_rank(r, rel) for r, rel in labeled]
    return {
        "status": "ok",
        "k": k,
        "labeled_queries": len(labeled),
        "hit_rate_at_k": round(sum(hits) / len(hits), 4),
        "mrr": round(sum(rr) / len(rr), 4),
        "reason": None,
    }


# —— 观测与聚合 ——


@dataclass(frozen=True)
class Observation:
    """一条语料的观测结果（报告的最小单元；失败也留痕，不静默丢掉）。"""

    case: str
    tags: tuple[str, ...]
    text: str
    intent: str | None
    tools: tuple[str, ...]
    tool_statuses: tuple[str, ...]
    tool_error_codes: tuple[str, ...]
    finished_reason: str | None
    degraded: bool | None
    statuses: tuple[str, ...]
    clarification: bool
    citations: tuple[str, ...]
    expected_citations: tuple[str, ...]
    retrieved: tuple[str, ...]
    llm_calls: int
    reply: str  # judge 需要原文（只留长度不够），报告对回答做长度截断见 _write_report
    reply_chars: int
    latency_ms: float | None
    error: str | None = None


def _rate(numerator: int, denominator: int) -> float:
    """比例（分母为 0 记 0：报告里 null 与 0 的混用比直接给 0 更难读）。"""
    return round(numerator / denominator, 4) if denominator else 0.0


def aggregate(
    observations: Sequence[Observation], *, mode: str, k: int = DEFAULT_K
) -> dict[str, Any]:
    """把逐条观测折叠成指标（口径与 `/metrics` 一致：全部来自事件与 AgentResponse）。"""
    total = len(observations)
    intents = Counter(o.intent for o in observations if o.intent)
    reasons = Counter(o.finished_reason for o in observations if o.finished_reason)
    tool_calls = sum(len(o.tools) for o in observations)
    llm_calls = sum(o.llm_calls for o in observations)
    with_tools = sum(1 for o in observations if o.tools)
    degraded = sum(1 for o in observations if o.degraded)
    clarified = sum(1 for o in observations if o.clarification)
    errors = sum(1 for o in observations if o.error is not None)

    # 引用：只统计「有期望引用」的轮次（把无引用的闲聊轮算进去会稀释覆盖率，读起来像退步）。
    cites_expected = [o for o in observations if o.expected_citations]
    covered = 0
    for o in cites_expected:
        if set(o.citations) & set(o.expected_citations):
            covered += 1
    # 护栏不变量：回答引用的来源必须真在本次检索集内（违反即 0 容忍，但仍只报告不断言）。
    violations = [
        {"case": o.case, "unknown_sources": sorted(set(o.citations) - set(o.retrieved))}
        for o in observations
        if set(o.citations) - set(o.retrieved)
    ]
    latencies = [o.latency_ms for o in observations if o.latency_ms is not None]

    return {
        "corpus_size": total,
        "errors": errors,
        "intent_distribution": dict(intents),
        "finished_reason_distribution": dict(reasons),
        "degradation_rate": _rate(degraded, total),
        "clarification_rate": _rate(clarified, total),
        "turns_with_tools_rate": _rate(with_tools, total),
        "tool_calls_total": tool_calls,
        "tool_calls_per_turn_avg": round(tool_calls / total, 3) if total else 0.0,
        "tool_error_rate": _rate(
            sum(1 for o in observations for s in o.tool_statuses if s == "error"), tool_calls
        ),
        "llm_calls_total": llm_calls,
        "llm_calls_per_turn_avg": round(llm_calls / total, 3) if total else 0.0,
        "avg_reply_chars": (
            round(sum(o.reply_chars for o in observations) / total, 1) if total else 0.0
        ),
        "latency": (
            {
                "status": "ok",
                "samples": len(latencies),
                "avg_ms": round(sum(latencies) / len(latencies), 1),
                "max_ms": round(max(latencies), 1),
            }
            if latencies
            else {
                "status": "skipped",
                "reason": (
                    "离线 fake 模型（微秒级内存调用），延迟无参考意义；仅 --live 报延迟"
                    if mode == "offline"
                    else "本轮无成功样本"
                ),
            }
        ),
        "citation": {
            "turns_with_expected_citations": len(cites_expected),
            "coverage_rate": _rate(covered, len(cites_expected)),
            "turns_with_citations": sum(1 for o in observations if o.citations),
            "invariant_checks": total,
            "invariant_violations": len(violations),
            "violations": violations,
        },
        # RAG 指标：标注缺失 → skipped（见 aggregate_rag 的 WHY）。
        "rag": aggregate_rag([], k=k),
    }


# —— ① 离线：语料经 fake 模型跑真图 ——


async def _offline_observation(case: TraceCase) -> Observation:
    """跑一条语料（离线 fake 模型 + 真图），收集观测（**不做断言** —— 那是门禁的活）。"""
    store = InMemoryStore()
    for item in case.seeds:
        await MemoryStore(store).save(item)

    service = LLMService(
        config=LLMConfig(api_key=_dummy_key()),
        chat_model=RecordingFakeChatModel(messages=iter(case.messages)),
    )
    graph = build_agent_graph(service, case.config, tools=list(case.tools), store=store)
    listener = RecordingListener().install()
    response: AgentResponse | None = None
    error: str | None = None
    try:
        _input: dict[str, str] = {"input": case.text, "session_id": f"eval-{case.name}"}
        if case.user_id:
            _input["user_id"] = case.user_id
        async for chunk in graph.astream(
            _input,
            config={"configurable": {"thread_id": f"eval-{case.name}"}},
            stream_mode="updates",
        ):
            for updates in chunk.values():
                if updates and "response" in updates:
                    response = updates["response"]
    except Exception as exc:  # 单条失败不终止整轮评测（报告里如实记录）
        error = f"{type(exc).__name__}: {exc}"
    finally:
        listener.uninstall()
        await wait_pending_saves()

    return _observation_from(
        case,
        response=response,
        listener=listener,
        llm_calls=len(getattr(service.chat_model, "prompts", [])),
        # 离线**不报延迟**：fake 模型是微秒级内存调用，报出来只会让「平均 0.4ms」这类
        # 数字被误当作性能结论（延迟只在 --live 有意义）。
        latency_ms=None,
        error=error,
    )


def _dummy_key() -> SecretStr:
    """离线构造 LLMService 用的占位 key（SecretStr；离线路径绝不触网、绝不进日志）。"""
    return SecretStr("sk-offline-eval")


def _observation_from(
    case: TraceCase,
    *,
    response: AgentResponse | None,
    listener: RecordingListener,
    llm_calls: int,
    latency_ms: float | None,
    error: str | None,
) -> Observation:
    """从 AgentResponse + 事件轨迹抽取观测字段（两条口径同源，不另立指标口径）。"""
    intents = listener.find(EVENT_INTENT_CLASSIFIED)
    answers = listener.find(EVENT_ANSWER_GENERATED)
    tools = listener.find(EVENT_TOOL_CALLED)
    tool_trace = list(response.tool_trace) if response is not None else []
    # 工具序列优先取 AgentResponse.tool_trace（含成败与耗时），事件侧仅作交叉校验来源。
    tool_names = (
        [record.tool_name for record in tool_trace]
        if tool_trace
        else [event.tool_name or "" for event in tools]
    )
    retrieved = {item.id for item in case.seeds}
    for record in tool_trace:
        for citation in getattr(getattr(record, "result", None), "citations", None) or []:
            retrieved.add(citation.source_id)

    return Observation(
        case=case.name,
        tags=case.tags,
        text=case.text,
        intent=intents[0].status if intents else None,
        tools=tuple(tool_names),
        tool_statuses=tuple(t.status for t in tools),
        tool_error_codes=tuple(t.code or "" for t in tools if t.code),
        finished_reason=response.finished_reason if response is not None else None,
        degraded=(bool(answers[-1].fields.get("degraded")) if answers else None),
        statuses=tuple(e.status for e in (response.status_trace if response is not None else [])),
        clarification=bool(response is not None and response.clarification is not None),
        citations=tuple(c.source_id for c in (response.citations if response is not None else [])),
        expected_citations=tuple(sorted(case.expect_citations)),
        retrieved=tuple(sorted(retrieved)),
        llm_calls=llm_calls,
        reply=(response.reply if response is not None else ""),
        reply_chars=len(response.reply) if response is not None else 0,
        latency_ms=latency_ms,
        error=error,
    )


# —— ② --live：真 LLM 端到端 ——


class JudgeVerdict(BaseModel):
    """judge 的结构化输出（5 级 rubric；走现有 LLMService，不引入新 provider）。"""

    score: int = Field(ge=1, le=5, description="1-5 分（5 最好）")
    rationale: str = Field(default="", description="一句话理由")


async def _live_observation(case: TraceCase, runtime: Any) -> Observation:
    """真 LLM 跑一条语料（会话/用户与真实用户隔离，见 EVAL_USER_ID）。"""
    listener = RecordingListener().install()
    started = time.perf_counter()
    response: AgentResponse | None = None
    error: str | None = None
    try:
        response = await runtime.chat(
            user_id=EVAL_USER_ID, session_id=f"eval-{case.name}", text=case.text
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        listener.uninstall()
        await wait_pending_saves()
    elapsed_ms = (time.perf_counter() - started) * 1000

    return _observation_from(
        case,
        response=response,
        listener=listener,
        # live 模式下 LLM 调用次数由事件侧无发行，报告里记 0 并在 notes 说明
        # （真实成本以 token 事件为准，见 C4）。
        llm_calls=0,
        latency_ms=None if error else elapsed_ms,
        error=error,
    )


async def _judge(observations: Sequence[Observation], llm: LLMService) -> dict[str, Any]:
    """LLM-as-judge：5 级 rubric 打分（报告项，不参与门禁）。

    judge 走**现有** `LLMService` + 结构化输出（plan §5.6：不引入新 provider）；
    调用失败只少一项报告，绝不影响轨迹指标。
    """
    verdicts: list[dict[str, Any]] = []
    for obs in observations:
        if obs.error or not obs.reply.strip():
            continue
        prompt = [
            HumanMessage(
                content=(
                    f"{JUDGE_PROMPT}\n用户输入：{obs.text}\n助手回答：{obs.reply}\n"
                    f"本轮终态：{obs.finished_reason}（degraded={obs.degraded}）"
                )
            )
        ]
        try:
            verdict = await llm.ainvoke_structured(JudgeVerdict, prompt)
        except Exception as exc:
            logger.warning("judge 失败（%s）：%s", obs.case, exc)
            continue
        if verdict is not None:
            verdicts.append(
                {"case": obs.case, "score": verdict.score, "rationale": verdict.rationale}
            )
    if not verdicts:
        return {"status": "skipped", "reason": "未产生有效评分（无回答或 judge 调用失败）"}
    scores = [v["score"] for v in verdicts]
    return {
        "status": "ok",
        "rubric_scale": "1-5",
        "judged_samples": len(scores),
        "mean_score": round(sum(scores) / len(scores), 2),
        "distribution": dict(sorted(Counter(scores).items())),
        "verdicts": verdicts,
    }


# —— 报告 ——


def _git_sha() -> str:
    """当前提交短 sha（便于「这份报告对应哪次改动」）。

    WHY 直接读 `.git` 而不起子进程：脚本不该因为 PATH 上没有 `git`、
    或沙箱禁子进程而拿不到这一行元数据（读取失败也照常产出报告）。
    """
    try:
        head = (Path(__file__).resolve().parents[1] / ".git" / "HEAD").read_text(encoding="utf-8")
    except OSError:
        return os.environ.get("GIT_SHA", "unknown")
    ref = head.strip()
    if ref.startswith("ref:"):
        try:
            ref = (
                (Path(__file__).resolve().parents[1] / ".git" / ref.split(" ", 1)[1].strip())
                .read_text(encoding="utf-8")
                .strip()
            )
        except OSError:
            return os.environ.get("GIT_SHA", "unknown")
    return ref[:7] or "unknown"


def _write_report(report: dict[str, Any], out_dir: Path, tag: str | None) -> Path:
    """落 `eval_reports/<date>[-tag].json`（趋势对比靠日期文件名，无需数据库）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    date = report["generated_at"][:10]
    path = out_dir / f"{date}{f'-{tag}' if tag else ''}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _log_summary(metrics: dict[str, Any], mode: str) -> None:
    """人读摘要（报告是给机器与趋势的，日志是给人当场看的）。"""
    logger.info("── 评测摘要（%s）──", mode)
    logger.info("语料 %d 条｜错误 %d 条", metrics["corpus_size"], metrics["errors"])
    logger.info("意图分布：%s", metrics["intent_distribution"] or "无")
    logger.info("终态分布：%s", metrics["finished_reason_distribution"] or "无")
    logger.info(
        "降级率 %.2f%%｜澄清率 %.2f%%｜含工具轮次 %.2f%%",
        metrics["degradation_rate"] * 100,
        metrics["clarification_rate"] * 100,
        metrics["turns_with_tools_rate"] * 100,
    )
    logger.info(
        "工具调用 %d 次（均 %.2f/轮）｜LLM 调用 %d 次（均 %.2f/轮）｜平均回答 %s 字",
        metrics["tool_calls_total"],
        metrics["tool_calls_per_turn_avg"],
        metrics["llm_calls_total"],
        metrics["llm_calls_per_turn_avg"],
        metrics["avg_reply_chars"],
    )
    citation = metrics["citation"]
    logger.info(
        "引用：期望引用轮次 %d｜覆盖率 %.2f%%｜护栏违规 %d（0 为达标）",
        citation["turns_with_expected_citations"],
        citation["coverage_rate"] * 100,
        citation["invariant_violations"],
    )
    logger.info(
        "RAG 指标：%s（%s）",
        metrics["rag"]["status"],
        metrics["rag"]["reason"] or f"命中率@k={metrics['rag']['hit_rate_at_k']}",
    )
    logger.info("延迟：%s", metrics["latency"]["status"])


def _self_check() -> None:
    """度量数学自检（零依赖：不建图、不连模型）。"""
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, 2) == 0.0  # k 截断生效
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_rate_at_k(["a"], set(), 5) == 0.0  # 无标注不虚报
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    skipped = aggregate_rag([])
    assert skipped["status"] == "skipped" and skipped["hit_rate_at_k"] is None
    ok = aggregate_rag([(["a", "b"], {"b"}), (["x"], {"x"})], k=2)
    assert ok["status"] == "ok" and ok["hit_rate_at_k"] == 1.0 and ok["mrr"] == 0.75
    agg = aggregate([], mode="offline")
    assert agg["corpus_size"] == 0 and agg["citation"]["invariant_violations"] == 0
    logger.info("度量数学自检通过（hit_rate@k / MRR / 聚合的空集与截断口径）")


async def _run_offline(cases: Sequence[TraceCase]) -> list[Observation]:
    logger.info("离线模式：%d 条语料（fake 模型 + 真图，零网络）", len(cases))
    observations: list[Observation] = []
    for case in cases:
        obs = await _offline_observation(case)
        if obs.error:
            logger.warning("用例失败：%s → %s", case.name, obs.error)
        observations.append(obs)
    return observations


async def _run_live(
    cases: Sequence[TraceCase], *, judge: bool, settings: Any
) -> tuple[list[Observation], dict]:
    """真 LLM 端到端（+ 可选 judge）。返回 (观测, judge 结果)。

    WHY judge 另建 LLMService 而不是复用图内实例：`AgentRuntime` 不对外暴露 `llm`
    （装配细节不外泄，这是刻意的边界）—— judge 用**同一份 `LLM_*` 配置**构造，
    故 provider 与模型一致，只是多一个独立客户端（用完即弃，不影响运行时生命周期）。
    """
    from agent.runtime import AgentRuntime

    logger.info("live 模式：%d 条语料（真 LLM + 已接入 MCP 工具）", len(cases))
    cfg = AgentFrameworkConfig.get_default()
    judge_llm = (
        LLMService(
            config=LLMConfig.from_runtime(
                provider=settings.llm_provider,
                api_key=settings.llm_api_key.get_secret_value(),
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                behavior=cfg.llm_behavior,
            )
        )
        if judge
        else None
    )
    runtime = await AgentRuntime.create()
    observations: list[Observation] = []
    judged: dict[str, Any] = {"status": "skipped", "reason": "未开启 judge"}
    try:
        for case in cases:
            obs = await _live_observation(case, runtime)
            if obs.error:
                logger.warning("用例失败：%s → %s", case.name, obs.error)
            observations.append(obs)
        if judge_llm is not None:
            judged = await _judge(observations, judge_llm)
    finally:
        await runtime.aclose()
    return observations, judged


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数：模式、范围、输出位置。"""
    parser = argparse.ArgumentParser(description="评测运行器（报告制，不做硬门禁）")
    parser.add_argument("--live", action="store_true", help="连真 LLM 端到端（需 LLM_API_KEY）")
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="live 模式下做 LLM-as-judge 5 级评分（默认开；--no-judge 只跑轨迹）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条语料（0 = 全部）")
    parser.add_argument("--out", default="eval_reports", help="报告目录（默认 %(default)s）")
    parser.add_argument("--tag", default=None, help="报告文件名后缀（如 baseline / rag-on）")
    parser.add_argument("--self-check", action="store_true", help="只校验度量数学并退出")
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    # 日志单点配置：进程身份记 `eval_run`（事件里一眼看出这条来自评测而非线上请求）。
    configure_logging(LoggingConfig.from_settings(settings, service="eval_run"))
    if args.self_check:
        _self_check()
        return 0

    cases: Sequence[TraceCase] = CASES[: args.limit] if args.limit > 0 else CASES
    notes: list[str] = []
    mode = "live" if args.live else "offline"
    if args.live and not settings.llm_api_key.get_secret_value():
        logger.warning("未配置 LLM_API_KEY，--live 不可用 → 退回离线模式（轨迹指标照常产出）")
        notes.append("--live 请求因缺少 LLM_API_KEY 退回离线模式")
        mode = "offline"

    judged: dict[str, Any] = {"status": "skipped", "reason": "离线模式不做 judge"}
    if mode == "live":
        observations, judged = await _run_live(cases, judge=args.judge, settings=settings)
        notes.append("live 模式的 LLM 调用次数未逐轮统计（由 C4 token 事件承载真实成本）")
    else:
        observations = await _run_offline(cases)

    metrics = aggregate(observations, mode=mode)
    metrics["judge"] = judged
    report = {
        "report_version": REPORT_VERSION,
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "env": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_enabled": settings.embedding_enabled,
            "observability_enabled": settings.langfuse_enabled,  # 只记开关，绝不记密钥
        },
        "corpus": {
            "source": "tests/test_eval_trace_golden.py:CASES",
            "size": len(CASES),
            "selected": len(cases),
        },
        "metrics": metrics,
        "observations": [asdict(o) for o in observations],
        "notes": notes,
    }
    path = _write_report(report, Path(args.out), args.tag)
    _log_summary(metrics, mode)
    logger.info("报告已写入：%s（报告制，不构成门禁）", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
