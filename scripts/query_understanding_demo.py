"""查询理解可观测演示（v6.0 T2：V6-M3 / V6-M4 验收）。

WHY 独立脚本：查询理解的效果体现在「同一问题的检索结果变了」——把它放在 demo_chat 里
会被意图识别、规划、工具链的噪声淹没。本脚本分四段观测：

  ① 改写产出（离线固定结果）：四类改写的结构化展示（主改写 / 子查询 / 同义 / HyDE），
     含「复合问题拆子查询」「短查询生成假设文档」等典型形态 —— **V6-M3** 的证据。
  ② 门控（离线，零依赖）：寒暄/极短输入**不调 LLM**，与真问题对照 —— **V6-M3** 的证据。
  ③ 记忆召回的改写前后对比：同一批记忆，用原始输入 vs 改写查询各召回一次，
     对比命中集合（**V6-M4** 的证据；不要求改写一定更好，但必须可见差异）。
  ④ `--live`（需 LLM_API_KEY）：用真实 LLM 跑改写，观察模型对真实问题的产出质量。

用法（仓库根目录，依赖 `.env`）：
    uv run python -m scripts.query_understanding_demo
    LLM_API_KEY=sk-xxx uv run python -m scripts.query_understanding_demo --live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass

from agent.core.config import AgentFrameworkConfig, LLMConfig, QueryUnderstandingConfig
from agent.llm import LLMService
from agent.memory import KIND_FACT, MemoryItem, MemoryStore, build_memory_backends
from agent.query import QueryRewriter, QueryUnderstanding, should_understand
from agent.share.eventloop import ensure_selector_event_loop
from settings import configure_logging, get_settings
from shared.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

DEMO_USER_ID = "query-demo-user"


# —— ① 改写形态（离线固定结果，展示四类字段的实际用途）——


@dataclass(frozen=True)
class RewriteCase:
    """一条改写样例：用户原话 + 期望的改写形态（+ 说明为什么这么拆）。"""

    text: str
    understanding: QueryUnderstanding
    note: str


REWRITE_CASES: tuple[RewriteCase, ...] = (
    RewriteCase(
        text="那它的上限呢",
        understanding=QueryUnderstanding(
            main_query="用户所在公司的年假天数上限",
            synonyms=["带薪年假上限"],
            reason="补全指代：结合会话摘要把「它」补成「年假」",
        ),
        note="主改写补全指代（脱离上下文读不懂的短句 → 可检索查询）",
    ),
    RewriteCase(
        text="我们公司年假怎么算，另外我上次说的出差补贴还记得吗",
        understanding=QueryUnderstanding(
            main_query="用户所在公司的年假计算方式与出差补贴标准",
            sub_queries=["年假的计算方式", "出差补贴标准"],
            synonyms=["带薪年假"],
            reason="含两个可独立检索的信息需求 → 拆两个子问题",
        ),
        note="子查询：仅当有多个可独立检索的需求时才拆（避免每轮白拆）",
    ),
    RewriteCase(
        text="报销标准",
        understanding=QueryUnderstanding(
            main_query="用户所在公司的差旅报销标准",
            synonyms=["费用报销标准", "报销额度"],
            hypothetical_answer=(
                "公司差旅报销按职级设定标准：交通费凭票实报实销，住宿费按城市分档，"
                "餐补按出勤天数计算。"
            ),
            reason="短查询 + 术语化 → 生成假设文档用于嵌入检索（HyDE）",
        ),
        note="HyDE：假答案只作嵌入输入（不进引用、不进生成 prompt），且仅短查询启用",
    ),
)


def _run_rewrite_forms(cfg: QueryUnderstandingConfig, *, max_variants: int) -> None:
    """离线展示四类改写的产出形态与各自消费方。"""
    logger.info("── ① 改写产出形态（离线样例，字段来源与消费方）──")
    for case in REWRITE_CASES:
        logger.info("── 用户原话：%s", case.text)
        logger.info("   %s", case.note)
        logger.info("   main_query（记忆召回 + RAG 实参）: %s", case.understanding.main_query)
        if case.understanding.sub_queries:
            logger.info("   sub_queries（召回不足时补检）: %s", case.understanding.sub_queries)
        if case.understanding.synonyms:
            logger.info("   synonyms（召回不足时补检）: %s", case.understanding.synonyms)
        if case.understanding.hypothetical_answer:
            logger.info(
                "   hypothetical_answer（仅嵌入输入，长度 %d）: %s…",
                len(case.understanding.hypothetical_answer),
                case.understanding.hypothetical_answer[:24],
            )
        logger.info(
            "   变体池（max_variants=%d）: %s",
            max_variants,
            case.understanding.retrieval_queries(max_variants=max_variants),
        )


# —— ② 门控（零 LLM 成本）——


GATE_SAMPLES: tuple[str, ...] = (
    "你好",
    "好的",
    "谢谢！",
    "?",
    "我们公司年假怎么算",
    "那它的上限呢",
    "帮我看看",
)


def _run_gate(cfg: QueryUnderstandingConfig) -> None:
    """展示哪些输入被门控拦下（不调 LLM）、哪些放行。"""
    logger.info("── ② 确定性预门控（命中则一次 LLM 都不调）──")
    for text in GATE_SAMPLES:
        passed = should_understand(text, settings=cfg)
        logger.info(
            "   %s %-24s → %s",
            "放行" if passed else "跳过",
            text,
            "调用 LLM 做改写" if passed else "零 LLM 成本（按 retrieval_needed=False 走）",
        )
    logger.info("   注：门控不按长度猜闲聊 —— 「帮我看看」是请求，必须放行（见 gate.py 取舍说明）")


# —— ③ 记忆召回：改写前后对比 ——


def _seed_items(user_id: str) -> list[MemoryItem]:
    """构造演示用记忆（固定 id → 幂等重种，重复运行不产生脏数据）。"""
    from datetime import UTC, datetime

    contents: tuple[tuple[str, str], ...] = (
        ("seed-annual", "用户所在公司的年假按工龄计算，满一年 5 天"),
        ("seed-travel", "用户所在公司的差旅补贴为每天 200 元"),
        ("seed-city", "用户在杭州工作，住在滨江"),
        ("seed-coffee", "用户早餐只喝美式咖啡，不加糖不加奶"),
    )
    return [
        MemoryItem(
            id=item_id,
            kind=KIND_FACT,
            content=content,
            session_id="seed",
            user_id=user_id,
            timestamp=datetime.now(UTC),
            provenance="seed",
            importance=0.6,
        )
        for item_id, content in contents
    ]


async def _run_recall_comparison(
    memory_store: MemoryStore, user_id: str, *, max_variants: int
) -> None:
    """同一批记忆：原始输入 vs 改写查询各召回一次，对比命中集合。"""
    logger.info("── ③ 记忆召回：原始输入 vs 改写查询（V6-M4）──")
    for item in _seed_items(user_id):
        await memory_store.save(item)

    raw_input = "那它的上限呢"
    rewritten = QueryUnderstanding(
        main_query="用户所在公司的年假天数上限",
        synonyms=["带薪年假天数"],
        reason="补全指代",
    )
    variants = rewritten.retrieval_queries(max_variants=max_variants)[1:]

    for label, query, extra in (
        ("原始输入", raw_input, []),
        ("改写查询（含变体）", rewritten.main_query, variants),
    ):
        result = await memory_store.recall(
            user_id=user_id, kinds=[KIND_FACT], top_k=5, query=query, variant_queries=extra
        )
        logger.info("   %s：query=%s | 变体=%d", label, query, len(extra))
        if not result.items:
            logger.info("      （未召回任何记忆）")
        for item, citation in zip(result.items, result.sources, strict=True):
            score = f"{citation.score:.3f}" if citation.score is not None else "n/a"
            logger.info("      · [%s] %s（score=%s）", item.id, item.content, score)
    logger.info(
        "   注：改写是否更好取决于语义底座；本段用于**观测差异**，"
        "golden set 回归（test_memory_semantic）负责锁住「不劣化」"
    )


# —— ④ --live：真实 LLM 改写 ——

LIVE_INPUTS: tuple[str, ...] = (
    "那它的上限呢",
    "我们公司年假怎么算，另外我上次说的出差补贴还记得吗",
    "报销标准",
    "你好",
)


async def _run_live(rewriter: QueryRewriter, cfg: QueryUnderstandingConfig) -> None:
    """真实 LLM 改写：观察模型在真实问题上的产出（含 HyDE 门控是否生效）。"""
    logger.info("── ④ 真实改写器（--live）──")
    for text in LIVE_INPUTS:
        if not should_understand(text, settings=cfg):
            logger.info("── %s → 门控跳过（不调 LLM）", text)
            continue
        result = await rewriter.rewrite(text)
        if result is None:
            logger.warning("── %s → 改写失败（回退原始输入，零回归）", text)
            continue
        logger.info("── %s", text)
        logger.info("   main: %s", result.main_query)
        logger.info("   sub: %s | 同义: %s", result.sub_queries, result.synonyms)
        logger.info(
            "   hyde: %s | retrieval_needed=%s | 理由: %s",
            "有" if result.hypothetical_answer else "无",
            result.retrieval_needed,
            result.reason or "无",
        )


# —— CLI ——


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=DEMO_USER_ID, help="演示用户命名空间（默认 %(default)s）")
    parser.add_argument(
        "--live",
        action="store_true",
        help="用真实 LLM 跑改写（需 LLM_API_KEY）；缺省只跑离线部分",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    settings = get_settings()
    configure_logging(settings)
    cfg = AgentFrameworkConfig.get_default()
    qcfg = cfg.query_understanding
    # 语义底座：注入 EMBEDDING_*（与记忆/RAG 对齐；未配置时召回自动降级确定性路径）。
    cfg.memory.embedding = EmbeddingConfig.from_runtime(
        enabled=settings.embedding_enabled,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
    )
    api_key = settings.llm_api_key.get_secret_value()
    rewriter: QueryRewriter | None = None
    if args.live and api_key:
        rewriter = QueryRewriter(
            LLMService(
                config=LLMConfig.from_runtime(
                    provider=settings.llm_provider,
                    api_key=api_key,
                    base_url=settings.llm_base_url,
                    model=settings.llm_model,
                )
            ),
            enable_hypothetical=qcfg.hypothetical_enabled,
            hypothetical_max_query_chars=qcfg.hyde_max_query_chars,
        )
    elif args.live:
        logger.warning("未配置 LLM_API_KEY，--live 跳过（离线部分照常演示）")

    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    backends = await build_memory_backends(cfg, database_url=database_url)
    try:
        memory_store = MemoryStore(backends.store, recall_config=cfg.memory.recall)
        logger.info("=== 查询理解演示（v6.0 T2）===")
        logger.info(
            "配置：enabled=%s | 子查询上限=%d | 同义上限=%d | HyDE=%s（≤%d 字）",
            qcfg.enabled,
            qcfg.sub_query_max,
            qcfg.synonym_max,
            qcfg.hypothetical_enabled,
            qcfg.hyde_max_query_chars,
        )
        _run_rewrite_forms(qcfg, max_variants=cfg.retrieval.max_variants)
        _run_gate(qcfg)
        await _run_recall_comparison(
            memory_store, args.user, max_variants=cfg.retrieval.max_variants
        )
        if rewriter is not None:
            await _run_live(rewriter, qcfg)
        else:
            logger.info("── ④ 真实改写器：未启用（加 --live 且配置 LLM_API_KEY 可验证提示词质量）")
        logger.info("── 汇总 ──")
        logger.info("  ① 四类改写形态已展示（V6-M3）")
        logger.info("  ② 确定性门控：寒暄/极短输入零 LLM 成本（V6-M3）")
        logger.info("  ③ 记忆召回改写前后对比已展示（V6-M4）")
        logger.info(
            "  关闭开关即回 v5.1：QueryUnderstandingConfig(enabled=False) → 全部回退原始输入"
        )
    finally:
        await backends.aclose()  # InMemory 路径为安全 no-op


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
