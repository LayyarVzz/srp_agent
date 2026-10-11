"""带外保存去重可观测演示：验证「重复事实第二次保存被 upsert 合并」。

WHY 独立于 demo_chat：demo_chat 走完整图（LLM 意图 + 抽取 + 带外保存），去重的
命中与否被抽取器/模型噪声淹没；本脚本只驱动 `MemoryStore.upsert`，用三组受控输入
精确观测两层去重（D4）：
  - L1 content-hash 精确层：完全重复 / 仅标点空白差异 → merged（id 复用、条数不涨）；
  - L2 语义近似层（需 embeddings）：同事实不同表述 → merged；
  - 未命中 → inserted（不同事实条数 +1）。

语义层（L2）触发必需（与 RAG 对齐，见 shared/embeddings.py）：`EMBEDDING_ENABLED=true`
+ `EMBEDDING_MODEL/DIMS/BASE_URL/API_KEY`；未配置时自动降级为仅 L1 hash 精确去重
（D3 零回归），脚本会显著标注当前模式。

`--judge` 开关额外注入真实 `MemoryRelationJudge`（D4-2，需 `LLM_API_KEY`）：
对组合/否定词等事实性场景直接展示三分类判定并端到端走带去重保存
（经 `save_conversation_memory` + 真实判定器），观察「合并 / 新写」的最终决策。
缺省保持纯确定性演示（无需生成式 LLM）。

用法（仓库根目录，依赖 `.env`）：
    uv run python -m scripts.memory_dedup_demo
    LLM_API_KEY=sk-xxx uv run python -m scripts.memory_dedup_demo --judge
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from langchain_core.messages import BaseMessage, HumanMessage

from agent.core.config import AgentFrameworkConfig, DedupConfig, LLMConfig
from agent.llm import LLMService
from agent.memory import (
    KIND_FACT,
    KIND_PREFERENCE,
    MemoryExtraction,
    MemoryItem,
    MemoryRelationJudge,
    MemoryStore,
    build_memory_backends,
    normalize_content_hash,
    save_conversation_memory,
    store_has_embeddings,
)
from agent.share.eventloop import ensure_selector_event_loop
from settings import configure_logging, get_settings
from shared.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

# 演示用户：独立命名空间，与 demo_chat / semantic_recall_demo 隔离，避免历史数据干扰。
DEMO_USER_ID = "dedup-demo-user"

# 语义近似层默认阈值（与 DedupConfig.semantic_threshold 对齐，脚本内直读配置）。
DEDUP_THRESHOLD = 0.92


@dataclass(frozen=True)
class UpsertScenario:
    """一条 upsert 探针：期望动作（merged=同事实重复 / inserted=新事实）。"""

    desc: str
    content: str
    kind: str
    expect: str  # "merged" | "inserted"
    note: str = ""


# 场景设计：①L1 完全重复 ②L1 标点变体 ③L2 语义改写（需语义模式）④L3 全新事实。
SCENARIOS: tuple[UpsertScenario, ...] = (
    UpsertScenario(
        desc="完全重复（L1 精确层）",
        content="用户早餐只喝美式咖啡，不加糖不加奶",
        kind=KIND_PREFERENCE,
        expect="merged",
        note="与种子逐字一致 → content-hash 精确命中",
    ),
    UpsertScenario(
        desc="标点/空白变体（L1 精确层）",
        content="用户早餐只喝美式咖啡，不加糖不加奶。",
        kind=KIND_PREFERENCE,
        expect="merged",
        note="仅句号差异 → 归一化后 hash 命中",
    ),
    UpsertScenario(
        desc="语义改写（L2 近似层，需语义模式）",
        content="用户每天早饭只喝黑咖啡，不加糖也不加奶",
        kind=KIND_PREFERENCE,
        expect="merged",
        note="换词改写 → 语义模式 ON 时 cosine≥阈值合并；OFF 时降级为 inserted",
    ),
    UpsertScenario(
        desc="不同事实（L3 新写）",
        content="用户是程序员，主攻 Python",
        kind=KIND_FACT,
        expect="inserted",
        note="与既有记忆不同 → 新写一条",
    ),
)

# 种子记忆：一条 preference 作为重复目标（固定 id，aput 幂等）。
SEED_ITEMS: tuple[MemoryItem, ...] = (
    MemoryItem(
        id="dedup-base-coffee",
        kind=KIND_PREFERENCE,
        content="用户早餐只喝美式咖啡，不加糖不加奶",
        session_id="seed",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=0.7,
        content_hash=normalize_content_hash("用户早餐只喝美式咖啡，不加糖不加奶"),
    ),
)


class _FixedExtractor:
    """固定返回给定抽取结果的 stub 抽取器（--judge 端到端驱动，避免真实抽取器噪声）。"""

    def __init__(self, extraction: MemoryExtraction) -> None:
        self._extraction = extraction

    async def extract(self, messages: Sequence[BaseMessage]) -> list[MemoryExtraction]:
        return [self._extraction]


async def _seed(memory_store: MemoryStore, user_id: str) -> None:
    """幂等种入种子记忆（固定 id 走 save，重复运行不产生脏数据）。"""
    for item in SEED_ITEMS:
        await memory_store.save(item.model_copy(update={"user_id": user_id}))
    logger.info("已种入种子记忆：%s", SEED_ITEMS[0].content)


async def _count_items(memory_store: MemoryStore, user_id: str) -> int:
    """当前用户命名空间的条目数（演示规模用 top-50 召回近似，足够准确）。"""
    result = await memory_store.recall(user_id=user_id, top_k=50)
    return len(result.items)


async def _run_scenarios(memory_store: MemoryStore, user_id: str) -> None:
    """顺序执行场景：每步 upsert 后打印动作 / 合并 id / 条目数，判定是否达预期。"""
    logger.info("── 去重场景 ──")
    for i, sc in enumerate(SCENARIOS, start=1):
        before = await _count_items(memory_store, user_id)
        item = MemoryItem(
            id=f"scenario-{i}",
            kind=sc.kind,
            content=sc.content,
            session_id="scenario",
            user_id=user_id,
            timestamp=datetime.now(UTC),
            provenance="dedup-demo",
            importance=0.6,
            content_hash=normalize_content_hash(sc.content),
        )
        outcome = await memory_store.upsert(item, semantic_threshold=DEDUP_THRESHOLD)
        after = await _count_items(memory_store, user_id)
        ok = outcome.action == sc.expect
        marker = "✓" if ok else "✗"
        logger.info("── %s %s「%s」──", marker, sc.desc, sc.content)
        logger.info(
            "  action=%s | id=%s | 条目数 %d → %d%s",
            outcome.action,
            outcome.item.id,
            before,
            after,
            "（不涨 ✓ 合并生效）" if after == before else "",
        )
        logger.info("  期望 %s | %s", sc.expect, sc.note)
        if not ok:
            logger.info("  ✗ 未达预期：实际 %s ≠ 期望 %s", outcome.action, sc.expect)


# D4-2 判定演示（--judge）：既有偏好种子 + 事实性探针，展示三分类判定与最终决策。
JUDGE_SEED_ITEMS: tuple[MemoryItem, ...] = (
    MemoryItem(
        id="judge-base-milk",
        kind=KIND_PREFERENCE,
        content="用户喜欢喝牛奶",
        session_id="judge",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="dedup-demo",
        importance=0.7,
        content_hash=normalize_content_hash("用户喜欢喝牛奶"),
    ),
    MemoryItem(
        id="judge-base-coffee",
        kind=KIND_PREFERENCE,
        content="用户喜欢喝咖啡",
        session_id="judge",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="dedup-demo",
        importance=0.7,
        content_hash=normalize_content_hash("用户喜欢喝咖啡"),
    ),
)


@dataclass(frozen=True)
class JudgeProbe:
    """一条判定探针：content 为待保存记忆，note 描述期望的三分类与决策。"""

    desc: str
    content: str
    note: str


JUDGE_PROBES: tuple[JudgeProbe, ...] = (
    JudgeProbe(
        desc="完全重复（L1 精确层）",
        content="用户喜欢喝牛奶",
        note="与种子逐字一致 → content-hash 精确命中，不经判定器直接合并",
    ),
    JudgeProbe(
        desc="组合新事实（牛奶和咖啡）",
        content="用户喜欢喝牛奶和咖啡",
        note="同时命中牛奶+咖啡 → 期望两条 overlap → 保守新写，不腐蚀任一既有事实",
    ),
    JudgeProbe(
        desc="否定词翻转（不喜欢咖啡）",
        content="用户不喜欢喝咖啡",
        note="观点相反 → 期望 not_duplicate → 新写（余弦再高也不误并）",
    ),
)


async def _run_judge_demo(
    memory_store: MemoryStore, judge: MemoryRelationJudge, user_id: str
) -> None:
    """D4-2 判定演示：真实 LLM 三分类 + 端到端带去重保存（需 LLM_API_KEY + embeddings）。

    先展示判定器对每条约保存记忆 vs 既有偏好的三分类 verdicts，
    再经 `save_conversation_memory`（stub 抽取器 + 真实判定器）走完整 `_save_deduped`
    决策，用条目数变化观测最终动作（merged=不涨 / inserted=+1）。
    """
    logger.info("── D4-2 事实感知合并判定（--judge，真实 MemoryRelationJudge）──")
    for item in JUDGE_SEED_ITEMS:
        await memory_store.save(item.model_copy(update={"user_id": user_id}))
    logger.info("已种入判定演示种子：牛奶 / 咖啡")
    for probe in JUDGE_PROBES:
        before = await _count_items(memory_store, user_id)
        # ① 直接展示三分类（候选 = 两条既有偏好；L1 场景仅作对照，实际不经判定器）。
        if probe.content != JUDGE_SEED_ITEMS[0].content:
            item = MemoryItem(
                id=f"probe-{probe.desc}",
                kind=KIND_PREFERENCE,
                content=probe.content,
                session_id="judge",
                user_id=user_id,
                timestamp=datetime.now(UTC),
                provenance="dedup-demo",
                importance=0.6,
                content_hash=normalize_content_hash(probe.content),
            )
            verdicts = await judge.judge(
                item, [s for s in JUDGE_SEED_ITEMS if s.content != probe.content]
            )
            logger.info("── %s「%s」──", probe.desc, probe.content)
            for v in verdicts:
                logger.info(
                    "  判定器 verdict: index=%s relation=%s reason=%s",
                    v.index,
                    v.relation,
                    v.reason,
                )
        else:
            logger.info("── %s「%s」──（L1 精确层，不经判定器）", probe.desc, probe.content)
        # ② 端到端带去重保存（stub 抽取器固定产出该探针 → 真实 _save_deduped）。
        await save_conversation_memory(
            [HumanMessage(content=probe.content)],
            session_id="judge",
            user_id=user_id,
            extractor=_FixedExtractor(
                MemoryExtraction(kind=KIND_PREFERENCE, content=probe.content, importance=0.6)
            ),
            store=memory_store,
            dedup=DedupConfig(),
            judge=judge,
        )
        after = await _count_items(memory_store, user_id)
        action = "merged" if after == before else "inserted"
        logger.info(
            "  保存决策: action=%s | 条目数 %d → %d | %s", action, before, after, probe.note
        )


def _parse_args() -> argparse.Namespace:
    """解析命令行参数：目标用户命名空间 + 是否注入真实判定器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        default=DEMO_USER_ID,
        help="演示用户命名空间（默认 %(default)s）",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="注入真实 MemoryRelationJudge（D4-2），展示三分类判定；需 LLM_API_KEY 与 embeddings",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    demo_user = args.user
    settings = get_settings()
    configure_logging(settings)
    cfg = AgentFrameworkConfig.get_default()
    # 语义底座：把 settings 的 EMBEDDING_* 注入 cfg.memory.embedding（P3-1 装配约定）。
    cfg.memory.embedding = EmbeddingConfig.from_runtime(
        enabled=settings.embedding_enabled,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
    )
    # --judge：构造真实 MemoryRelationJudge（与抽取器共用 LLMService，D4-2 接线同 demo_chat）。
    judge: MemoryRelationJudge | None = None
    if args.judge:
        api_key = settings.llm_api_key.get_secret_value()
        if not api_key:
            logger.warning("未配置 LLM_API_KEY，--judge 跳过（可先跑 pytest 离线验证判定路径）")
        else:
            llm_config = LLMConfig.from_runtime(
                provider=settings.llm_provider,
                api_key=api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
            )
            judge = MemoryRelationJudge(LLMService(config=llm_config))
    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    backends = await build_memory_backends(cfg, database_url=database_url)
    try:
        memory_store = MemoryStore(backends.store, recall_config=cfg.memory.recall)
        semantic = store_has_embeddings(backends.store)
        logger.info("=== 记忆保存去重演示 ===")
        logger.info("后端: %s", "AsyncPostgresStore" if database_url else "InMemoryStore")
        mode_label = (
            "ON（store 已配 embeddings，L1+L2 两层去重）"
            if semantic
            else "OFF（仅 L1 content-hash 精确去重，D3 降级）"
        )
        logger.info("语义层: %s | semantic_threshold=%.2f", mode_label, DEDUP_THRESHOLD)
        logger.info(
            "判定器: %s", "MemoryRelationJudge（D4-2，--judge）" if judge else "无（阈值路径）"
        )
        if not semantic:
            logger.warning(
                "未配置 EMBEDDING_* 或 embedding 不可用——语义改写场景将按新事实落盘；"
                "配好 EMBEDDING_ENABLED=true 后重跑即为 L1+L2 完整演示"
            )
        await _seed(memory_store, demo_user)
        logger.info("种入后条目数: %d", await _count_items(memory_store, demo_user))
        await _run_scenarios(memory_store, demo_user)
        if judge is not None:
            if not semantic:
                logger.warning("无 embeddings → 无语义候选，判定器不会触发；跳过 --judge 演示")
            else:
                await _run_judge_demo(memory_store, judge, demo_user)
        logger.info("── 汇总 ──")
        logger.info(
            "语义层: %s | 语义改写场景 %s",
            "ON" if semantic else "OFF",
            "由 L2 余弦 ≥ 阈值判定合并" if semantic else "降级为 L1，新事实落盘",
        )
        if judge is not None:
            logger.info(
                "判定器: 已注入 | 事实性场景（组合/否定词）由 MemoryRelationJudge 三分类决策"
            )
    finally:
        await backends.aclose()  # InMemory 路径为安全 no-op


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
