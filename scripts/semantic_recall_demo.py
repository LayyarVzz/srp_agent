"""语义召回可观测演示：预植入长期记忆，验证「不同表述命中同一事实」。

WHY 独立于 demo_chat：demo_chat 走完整图（LLM 意图 + 工具 + 带外保存），语义召回
的命中与否被模型/意图噪声淹没；本脚本只驱动 `MemoryStore.recall`，用「语义等价、
词面不同」的 query 精确观测三个信号：
  - score 回填：float=语义模式、None=确定性降级——语义路径的判定信号；
  - 期望记忆是否进入 top_k（命中）——语义命中而非词面子串匹配的证明；
  - 两套权重（content / preference）对同一 query 的排序差异——职责辨析。

触发语义模式必需（与 RAG 对齐，见 shared/embeddings.py）：`EMBEDDING_ENABLED=true`
+ `EMBEDDING_MODEL/DIMS/BASE_URL/API_KEY`；未配置时自动降级确定性 importance 排序，
脚本会显著标注当前模式。不需要 `LLM_API_KEY`（本脚本不调用生成式 LLM）。

用法（仓库根目录，依赖 `.env`）：
    uv run python -m scripts.semantic_recall_demo
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from agent.core.config import AgentFrameworkConfig
from agent.memory import (
    KIND_EPISODE,
    KIND_FACT,
    KIND_PREFERENCE,
    MemoryItem,
    MemoryStore,
    build_memory_backends,
    store_has_embeddings,
)
from agent.share.eventloop import ensure_selector_event_loop
from settings import configure_logging, get_settings
from shared.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

# 演示用户：独立命名空间，与 demo_chat 的 demo-user 隔离，避免历史数据干扰。
DEMO_USER_ID = "semantic-demo-user"


@dataclass(frozen=True)
class RecallProbe:
    """一条召回探针：query 如何问、期望命中哪条记忆。"""

    desc: str
    query: str
    expect: str


# 探针设计：控制组（词面重叠）对比挑战组（词面零重叠）→ 证明召回靠语义而非子串。
PROBES: tuple[RecallProbe, ...] = (
    RecallProbe(
        desc="直接问花生过敏（词面重叠，控制组）",
        query="我对花生过敏吗",
        expect="seed-allergy",
    ),
    RecallProbe(
        desc="吃宫保鸡丁（词面零重叠，验证语义泛化）",
        query="我中午吃了宫保鸡丁，会不会有问题",
        expect="seed-allergy",
    ),
    RecallProbe(desc="口语化问作息习惯", query="我该几点上床？", expect="seed-sleep"),
    RecallProbe(
        desc="黑咖啡=美式咖啡（偏好，词面部分重叠）",
        query="我早上喝的那种黑咖啡",
        expect="seed-coffee",
    ),
    RecallProbe(
        desc="口语化 paraphrase「简洁回答」偏好",
        query="回复的时候能简短一点吗",
        expect="seed-answer",
    ),
)

# 种子记忆：一条高 importance 安全事实（花生过敏）+ 两条 preference + 一条 fact + 一条干扰 episode。
SEED_MEMORIES: tuple[MemoryItem, ...] = (
    MemoryItem(
        id="seed-allergy",
        kind=KIND_FACT,
        content="用户对花生严重过敏，误食会导致过敏性休克",
        session_id="seed",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=1.0,
    ),
    MemoryItem(
        id="seed-coffee",
        kind=KIND_PREFERENCE,
        content="用户早餐只喝美式咖啡，不加糖不加奶",
        session_id="seed",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=0.7,
    ),
    MemoryItem(
        id="seed-sleep",
        kind=KIND_FACT,
        content="用户是中学生，通常晚上九点睡觉",
        session_id="seed",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=0.8,
    ),
    MemoryItem(
        id="seed-answer",
        kind=KIND_PREFERENCE,
        content="用户偏好简洁回答，重要信息需二次确认",
        session_id="seed",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=0.9,
    ),
    MemoryItem(
        id="seed-deadline",
        kind=KIND_EPISODE,
        content="用户上周提交了毕业论文初稿",
        session_id="seed",
        user_id=DEMO_USER_ID,
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=0.4,
    ),
)


async def _seed(memory_store: MemoryStore, user_id: str) -> None:
    """幂等种入演示记忆：固定 id 走 aput upsert，重复运行不产生脏数据。

    `model_copy` 重绑 user_id：同一批 SEED_MEMORIES 可种到任意用户命名空间
    （默认语义演示用户；`--seed-user demo-user` 预植给 demo_chat）。
    """
    for item in SEED_MEMORIES:
        await memory_store.save(item.model_copy(update={"user_id": user_id}))
    logger.info(
        "已种入 %d 条演示记忆（花生过敏/咖啡偏好/作息习惯/回答风格/论文干扰项）",
        len(SEED_MEMORIES),
    )


async def _run_probe(
    memory_store: MemoryStore, probe: RecallProbe, *, user_id: str, top_k: int = 3
) -> bool:
    """执行一条探针并打印排序：score 回填、kind/importance、期望命中标记。"""
    result = await memory_store.recall(user_id=user_id, top_k=top_k, query=probe.query)
    hit = any(m.id == probe.expect for m in result.items)
    logger.info("── 探针「%s」──", probe.desc)
    logger.info("  query: %s | 期望命中: %s", probe.query, probe.expect)
    for i, (m, src) in enumerate(zip(result.items, result.sources, strict=True), start=1):
        # score 非 None = 语义模式（余弦回填）；None = 确定性降级（importance 排序）。
        score = f"{src.score:.3f}" if src.score is not None else "None(确定性)"
        marker = "✓" if m.id == probe.expect else " "
        logger.info(
            "  #%d %s %-13s kind=%-10s imp=%.2f score=%s",
            i,
            marker,
            m.id,
            m.kind,
            m.importance,
            score,
        )
        logger.info("      └ %s", m.content)
    if not hit:
        logger.info(
            "  ✗ 期望 %s 未进入 top-%d（语义泛化不足或词面相关性被高 importance 干扰）",
            probe.expect,
            top_k,
        )
    return hit


async def _compare_weights(
    memory_store: MemoryStore, probe: RecallProbe, *, user_id: str, top_k: int = 3
) -> None:
    """同一 query 用两套职责权重各召回一次，对比排序差异（importance 主导 vs query 主导）。"""
    logger.info("── 权重对比「%s」（content vs preference）──", probe.desc)
    variants: tuple[tuple[str, tuple[float, float, float] | None], ...] = (
        ("content(0.6,0.25,0.15) query主导   ", None),
        ("preference(0.2,0.6,0.2) importance主导", (0.2, 0.6, 0.2)),
    )
    for label, weights in variants:
        result = await memory_store.recall(
            user_id=user_id, top_k=top_k, query=probe.query, hybrid_weights=weights
        )
        order = " > ".join(f"{m.id}(imp={m.importance:.1f})" for m in result.items)
        logger.info("  [%s] %s", label, order)
    logger.info(
        "  说明: 排序一致也属正常——语义分与 importance 同向时两套权重收敛；出现差异即职责辨析生效"
    )


def _parse_args() -> argparse.Namespace:
    """解析命令行参数：目标用户命名空间 + 是否仅种入。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-user",
        default=DEMO_USER_ID,
        help="种入记忆并召回的用户命名空间（默认 %(default)s；预植给 demo_chat 用 demo-user）",
    )
    parser.add_argument(
        "--plant-only",
        action="store_true",
        help="仅种入种子记忆后退出（供 demo_chat 预植），不跑召回探针",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    seed_user = args.seed_user
    settings = get_settings()
    configure_logging(settings)
    cfg = AgentFrameworkConfig.get_default()
    # 语义召回底座：把 settings 的 EMBEDDING_* 注入 cfg.memory.embedding（P3-1 装配约定）。
    cfg.memory.embedding = EmbeddingConfig.from_runtime(
        enabled=settings.embedding_enabled,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
    )
    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    backends = await build_memory_backends(cfg, database_url=database_url)
    try:
        memory_store = MemoryStore(backends.store, recall_config=cfg.memory.recall)
        semantic = store_has_embeddings(backends.store)
        logger.info("=== 语义召回可观测演示 ===")
        logger.info("后端: %s", "AsyncPostgresStore" if database_url else "InMemoryStore")
        mode_label = (
            "ON（store 已配 embeddings，语义召回生效）"
            if semantic
            else "OFF（降级确定性 importance 排序）"
        )
        logger.info("语义模式: %s", mode_label)
        if not semantic:
            logger.warning(
                "未配置 EMBEDDING_* 或 embedding 不可用——以下探针验证的是 importance 排序，"
                "不是语义命中；配好 EMBEDDING_ENABLED=true 后重跑即为语义演示"
            )
        await _seed(memory_store, seed_user)
        if args.plant_only:
            logger.info("仅种入完成（user=%s），退出", seed_user)
            return
        hits = 0
        for probe in PROBES:
            if await _run_probe(memory_store, probe, user_id=seed_user):
                hits += 1
        await _compare_weights(memory_store, PROBES[4], user_id=seed_user)
        logger.info("── 汇总 ──")
        logger.info(
            "语义模式: %s | 命中 %d/%d"
            "（「宫保鸡丁」等零词面重叠探针未命中，通常说明 embedding 语义泛化不足）",
            "ON" if semantic else "OFF",
            hits,
            len(PROBES),
        )
    finally:
        await backends.aclose()  # InMemory 路径为安全 no-op


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
