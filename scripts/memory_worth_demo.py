"""记忆价值判定可观测演示（v6.0 T1：V6-M1 / V6-M2 验收）。

WHY 独立于 demo_chat：demo_chat 走完整图（LLM 意图 + 抽取 + 带外保存），
「哪些被判定为不值得记、为什么」会被意图识别与对话流程的噪声淹没。
本脚本把「抽取 → 值得性判定 → 落库」三段分开观测：

  ① 判据表（离线，零依赖）：固定 22 例 golden set（10 保留 / 10 拒收 / 2 显式「记住」）
     逐条打印 category / worth_score / 期望 vs 实际决策 —— **V6-M1** 的可复现证据
     （常识/寒暄/助手产出被拒收，身份/偏好/计划被保留）。
  ② 真实写入场景（离线，固定抽取结果）：三类场景（混合 / 阈值 / 显式记住）走
     `save_conversation_memory` 完整落库路径，打印逐条丢弃日志与最终落库内容，
     并演示每轮条数上限的截断 —— **V6-M2**（显式记住零误伤）的可复现证据。
  ③ `--live`（需 LLM_API_KEY）：用**真实** `MemoryExtractor` + 精化后的提示词跑真实对话，
     先打印模型结构化输出（含 category/worth_score/worth_reason），再过判定落库 ——
     验证提示词对「常识 vs 关于用户的事实」的判别质量。

用法（仓库根目录，依赖 `.env`）：
    uv run python -m scripts.memory_worth_demo
    LLM_API_KEY=sk-xxx uv run python -m scripts.memory_worth_demo --live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.core.config import AgentFrameworkConfig, LLMConfig, MemoryWorthConfig
from agent.llm import LLMService
from agent.memory import (
    KEEP_EXPLICIT,
    LONG_TERM_NAMESPACE,
    MemoryExtraction,
    MemoryExtractor,
    MemoryStore,
    build_memory_backends,
    save_conversation_memory,
    should_keep,
)
from agent.share.eventloop import ensure_selector_event_loop
from settings import configure_logging, get_settings
from shared.embeddings import EmbeddingConfig
from tests.test_memory_worth_golden import ALL_CASES, WorthCase, judged_as_expected

logger = logging.getLogger(__name__)

# 演示用户：独立命名空间，与其它 demo 隔离（避免历史数据干扰）。
DEMO_USER_ID = "worth-demo-user"


class _FixedExtractor:
    """固定返回给定抽取结果的 stub 抽取器（离线精确观测判定与落库，排除模型噪声）。"""

    def __init__(self, extractions: Sequence[MemoryExtraction]) -> None:
        self._extractions = list(extractions)

    async def extract(self, messages: Sequence[BaseMessage]) -> list[MemoryExtraction]:
        return list(self._extractions)


def _case_extraction(case: WorthCase) -> MemoryExtraction:
    """把 golden 用例转成抽取结果（进入落库路径的真实形态）。"""
    return MemoryExtraction(
        kind=case.kind,
        content=case.content,
        importance=0.6,
        category=case.category,
        worth_score=case.worth_score,
        worth_reason=case.note,
    )


# —— ① 判据表（离线）——


def _run_judging_table(cfg: MemoryWorthConfig) -> bool:
    """逐条展示判据决策；返回是否全部符合期望（零误丢 + 拒收生效）。"""
    logger.info("── ① 判据表（golden set %d 例，阈值 %.2f）──", len(ALL_CASES), cfg.min_worth_score)
    failures: list[str] = []
    for case in ALL_CASES:
        ok = judged_as_expected(case, cfg=cfg)
        actual = should_keep(_case_extraction(case), cfg=cfg)
        marker = "✓" if ok else "✗"
        logger.info(
            "  %s [%s] 期望=%s 实际=%s | category=%-14s score=%.2f | %s",
            marker,
            case.text,
            "落库" if case.keep else "丢弃",
            "落库" if actual else "丢弃",
            case.category,
            case.worth_score,
            case.content,
        )
        if not ok:
            failures.append(f"{case.content}（期望 {'落库' if case.keep else '丢弃'}）")
    logger.info(
        "  判据表结论：%d/%d 符合期望%s",
        len(ALL_CASES) - len(failures),
        len(ALL_CASES),
        "" if not failures else f" | ✗ 不符：{failures}",
    )
    return not failures


# —— ② 真实写入场景（离线固定抽取结果）——


@dataclass(frozen=True)
class WriteScenario:
    """一个落库场景：一段对话 + 该对话的固定抽取结果 + 期望落库内容。"""

    name: str
    messages: tuple[str, ...]
    extractions: tuple[MemoryExtraction, ...]
    expect_landed: tuple[str, ...]
    note: str = ""


def _scenarios() -> tuple[WriteScenario, ...]:
    """三类场景：常识混合 / 阈值判定 / 显式记住（含每轮上限截断）。"""
    return (
        WriteScenario(
            name="V6-M1 常识混合（常识被拒收、个人事实保留）",
            messages=("我们公司年假怎么算？另外我叫小明，是医生",),
            extractions=(
                MemoryExtraction(
                    kind="fact",
                    content="用户叫小明，职业是医生",
                    importance=0.8,
                    category="identity",
                    worth_score=0.85,
                    worth_reason="用户身份事实",
                ),
                MemoryExtraction(
                    kind="episode",
                    content="员工年假按工龄计算，满一年 5 天",
                    importance=0.5,
                    category="commonsense",
                    worth_score=0.2,
                    worth_reason="公共制度知识，由知识库负责",
                ),
                MemoryExtraction(
                    kind="preference",
                    content="用户对电子产品有消费意愿",
                    importance=0.6,
                    category="derivable",
                    worth_score=0.25,
                    worth_reason="由用户购物事实推导，无新增信息",
                ),
            ),
            expect_landed=("用户叫小明，职业是医生",),
            note="常识与推导品不落库，只留用户画像",
        ),
        WriteScenario(
            name="V6-M1 阈值情形（拒收类别 + 高分 → 保守保留）",
            messages=("（模型自相矛盾：标了常识却给高分）",),
            extractions=(
                MemoryExtraction(
                    kind="fact",
                    content="用户提到过三次握手这个概念",
                    importance=0.6,
                    category="commonsense",
                    worth_score=0.9,
                    worth_reason="模型标注与分数矛盾",
                ),
            ),
            expect_landed=("用户提到过三次握手这个概念",),
            note="类别与分数矛盾时保守保留（宁漏不误丢），并记 warning 暴露提示词漂移",
        ),
        WriteScenario(
            name="V6-M2 显式记住（旁路判定层，竞争每轮上限）",
            messages=("记住 TCP 三次握手是什么；另外我下周三去上海出差",),
            extractions=(
                MemoryExtraction(
                    kind="fact",
                    content="用户要求记住 TCP 三次握手的流程",
                    importance=0.8,
                    category=KEEP_EXPLICIT,
                    worth_score=0.95,
                    worth_reason="用户显式要求记住",
                ),
                MemoryExtraction(
                    kind="episode",
                    content="用户下周三要去上海出差",
                    importance=0.7,
                    category="plan",
                    worth_score=0.75,
                    worth_reason="有行动指向的计划",
                ),
                MemoryExtraction(
                    kind="episode",
                    content="用户上周买了个投影仪，花了 3200",
                    importance=0.6,
                    category="identity",
                    worth_score=0.8,
                    worth_reason="个人消费事实",
                ),
            ),
            expect_landed=(
                "用户要求记住 TCP 三次握手的流程",
                "用户上周买了个投影仪，花了 3200",
            ),
            note="每轮上限 2：显式记住(0.95) 与投影仪(0.80) 胜出，出差计划(0.75) 被截断",
        ),
    )


async def _reset_namespace(memory_store: MemoryStore, user_id: str) -> None:
    """清空演示用户的长期记忆命名空间（每个场景独立观测，避免累积干扰条目数）。"""
    namespace = (user_id, LONG_TERM_NAMESPACE)
    hits = await memory_store._store.asearch(namespace, limit=200)
    for hit in hits:
        await memory_store._store.adelete(namespace, hit.key)


async def _run_write_scenarios(
    memory_store: MemoryStore, user_id: str, *, worth: MemoryWorthConfig
) -> bool:
    """跑三类落库场景：打印丢弃/落库明细，比对落库内容是否符合期望。"""
    logger.info("── ② 真实写入场景（每轮上限 %d）──", worth.memories_max_per_turn)
    all_ok = True
    for scenario in _scenarios():
        await _reset_namespace(memory_store, user_id)  # 场景间隔离
        logger.info("── %s ──", scenario.name)
        logger.info("  对话：%s", " / ".join(scenario.messages))
        logger.info("  说明：%s", scenario.note)
        await save_conversation_memory(
            [HumanMessage(content=m) for m in scenario.messages],
            session_id="worth-demo",
            user_id=user_id,
            extractor=_FixedExtractor(scenario.extractions),
            store=memory_store,
            worth=worth,
        )
        landed = await _landed_contents(memory_store, user_id)
        expected = sorted(scenario.expect_landed)
        ok = landed == expected
        all_ok = all_ok and ok
        logger.info("  %s 本轮落库：%s", "✓" if ok else "✗", landed or "（无）")
        if not ok:
            logger.info("  ✗ 期望：%s", expected)
    return all_ok


async def _landed_contents(memory_store: MemoryStore, user_id: str) -> list[str]:
    """已落库内容（排序后返回；每场景前已清空命名空间，故即本轮结果）。

    WHY 走 `recall` 而不是直读 store：演示要展示的是「用户最终能召回什么」；
    条数远小于 top_k，不存在被截断的问题。
    """
    result = await memory_store.recall(user_id=user_id, top_k=100)
    return sorted(m.content for m in result.items)


# —— ③ --live：真实抽取器 + 精化提示词 ——

LIVE_CONVERSATIONS: tuple[tuple[str, ...], ...] = (
    ("我们公司年假怎么算？我下周三要去上海出差",),
    ("我叫小明，是医生。Python 是解释型语言吧？",),
    ("记住：我们公司的报销标准是每天 200",),
)


async def _run_live(
    extractor: MemoryExtractor, memory_store: MemoryStore, user_id: str, worth: MemoryWorthConfig
) -> bool:
    """真实 LLM 抽取：先看模型输出的值得性字段，再过判定落库（验证提示词质量）。

    返回 False 表示**观察到提示词质量问题**（模型把助手产出也抽成了记忆），
    与离线用例的硬断言不同 —— 这是观测项，不是门禁（门禁在 golden set）。
    """
    logger.info("── ③ 真实抽取器（--live）：提示词判别质量 ──")
    await _reset_namespace(memory_store, user_id)  # 与 ② 隔离：累计统计只反映 ③ 的产出
    extracted_assistant_content = False
    for i, conversation in enumerate(LIVE_CONVERSATIONS, start=1):
        logger.info("── 对话 %d：%s", i, " / ".join(conversation))
        messages = [HumanMessage(content=m) for m in conversation] + [
            # 助手一轮回答，用于验证「助手产出不得被抽取」这条既有约束。
            AIMessage(content="（助手对制度与常识的解答，属生成内容，不应成为记忆）")
        ]
        extractions = await extractor.extract(messages)
        if not extractions:
            logger.info("  模型判定：无需记忆（memories=[]）")
        for e in extractions:
            decision = "落库" if should_keep(e, cfg=worth) else "丢弃"
            logger.info(
                "  模型输出 → kind=%-10s category=%-14s score=%.2f | %s | 判定=%s（%s）",
                e.kind,
                e.category,
                e.worth_score,
                e.content,
                decision,
                e.worth_reason or "无",
            )
            if "助手" in e.content:
                extracted_assistant_content = True
        await save_conversation_memory(
            messages,
            session_id=f"live-{i}",
            user_id=user_id,
            extractor=_FixedExtractor(extractions),
            store=memory_store,
            worth=worth,
        )
    landed = await _landed_contents(memory_store, user_id)
    logger.info("  该用户累计落库 %d 条：", len(landed))
    for content in landed:
        logger.info("    · %s", content)
    if extracted_assistant_content:
        logger.warning("观察到提示词漂移：模型把助手自身产出也抽成了记忆（应拒收 self_generated）")
    return not extracted_assistant_content


# —— CLI ——


def _parse_args() -> argparse.Namespace:
    """解析命令行参数：目标用户命名空间、阈值覆盖、是否跑真实抽取器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        default=DEMO_USER_ID,
        help="演示用户命名空间（默认 %(default)s）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MemoryWorthConfig().min_worth_score,
        help="值得性阈值覆盖（默认 %(default)s，越低调越少误丢、噪声越多）",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="用真实 MemoryExtractor + 精化提示词跑真实对话（需 LLM_API_KEY）",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    settings = get_settings()
    configure_logging(settings)
    cfg = AgentFrameworkConfig.get_default()
    # 演示用的值得性配置：阈值可覆盖，每轮上限收紧到 2 以便观察截断行为。
    worth = MemoryWorthConfig(min_worth_score=args.threshold, memories_max_per_turn=2)
    # 语义底座（与其它 demo 同惯例注入 EMBEDDING_*；未配置时自动降级确定性召回）。
    cfg.memory.embedding = EmbeddingConfig.from_runtime(
        enabled=settings.embedding_enabled,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
    )
    extractor: MemoryExtractor | None = None
    if args.live:
        api_key = settings.llm_api_key.get_secret_value()
        if not api_key:
            logger.warning("未配置 LLM_API_KEY，--live 跳过（离线部分照常演示）")
        else:
            llm_config = LLMConfig.from_runtime(
                provider=settings.llm_provider,
                api_key=api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
            )
            extractor = MemoryExtractor(LLMService(config=llm_config))

    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    backends = await build_memory_backends(cfg, database_url=database_url)
    try:
        memory_store = MemoryStore(backends.store, recall_config=cfg.memory.recall)
        logger.info("=== 记忆价值判定演示（v6.0 T1）===")
        logger.info("后端: %s", "AsyncPostgresStore" if database_url else "InMemoryStore")
        logger.info(
            "判定配置: enabled=%s | min_worth_score=%.2f | memories_max_per_turn=%d",
            worth.enabled,
            worth.min_worth_score,
            worth.memories_max_per_turn,
        )
        table_ok = _run_judging_table(worth)
        write_ok = await _run_write_scenarios(memory_store, args.user, worth=worth)
        live_ok = True
        if extractor is not None:
            live_ok = await _run_live(extractor, memory_store, args.user, worth)
        logger.info("── 汇总 ──")
        logger.info("  ① 判据表（V6-M1 判据）: %s", "✓ 全部符合期望" if table_ok else "✗ 存在不符")
        logger.info("  ② 落库场景（V6-M1/M2）: %s", "✓ 全部符合期望" if write_ok else "✗ 存在不符")
        if extractor is not None:
            logger.info("  ③ 真实抽取器（--live）: %s", "✓ 已观测" if live_ok else "✗ 异常")
        else:
            logger.info("  ③ 真实抽取器: 未启用（加 --live 且配置 LLM_API_KEY 可验证提示词质量）")
        logger.info(
            "  关闭开关即回 v5.1 行为：MemoryWorthConfig(enabled=False) → 全部落库（见单测断言）"
        )
    finally:
        await backends.aclose()  # InMemory 路径为安全 no-op


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
