"""V3-M2 语义召回 + 混合重排（双模式）测试（离线，无 numpy / 无 DB）。

覆盖：语义模式命中排序、kind 过滤顺序、Citation.score 回填、空 query / 无 embeddings
降级、recency tie-break、两套职责权重（content / preference）次序、
`_recency_signal`/`_clamp01`/`_aware_utc` 单测、asearch limit 契约、
图级 load_context / recall_memory 传 query 与各自权重接线。

WHY 用 `_MapEmbedder` 而非 `tests.fakes.FakeEmbeddings`：FakeEmbeddings 的 hash 向量
对「不同文本」的 cosine 不可解析预测（实测两段中文短语可达 0.75），无法可靠构造
「语义压倒 importance」的断言；`_MapEmbedder` 按文本映射预置向量，未注册文本返回与
注册向量正交的默认向量（cosine 恰 0），score 精确可控、断言确定。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from agent.intent.models import Intent, IntentResult
from agent.memory import KIND_FACT, KIND_PREFERENCE, MemoryItem, MemoryStore
from agent.memory.adapter import _aware_utc, _clamp01, _recency_signal
from tests.conftest import (
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
    understand_message,
)

_DIMS = 8
# 注册向量沿 e0；默认向量沿 e1（与 e0 正交 → 未注册文本 cosine 恰 0）。
_E0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# 第二条正交注册向量（沿 e2）：用于构造「两路各自独占命中一条」的多查询场景。
_E1 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _MapEmbedder(Embeddings):
    """文本→预置向量映射的受控 embedder：精确控制余弦（不依赖 FakeEmbeddings 的不可解析 hash）。

    注册文本返回注册向量；未注册文本返回默认向量（沿 e1，与注册的 e0 正交 → score 恰 0）。
    仅供测试：真实环境一律经根级 `shared/embeddings.EmbeddingsFactory` 构造。
    """

    def __init__(self) -> None:
        self._map: dict[str, list[float]] = {}
        self._default = [0.0] * _DIMS
        self._default[1] = 1.0

    def register(self, text: str, vector: list[float]) -> None:
        if len(vector) != _DIMS:
            raise ValueError("vector dims mismatch")
        self._map[text] = vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._map.get(t, self._default) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._map.get(text, self._default)


def _semantic_store(embedder: Embeddings) -> InMemoryStore:
    """构造带语义索引的 InMemoryStore（镜像 build_memory_backends 的 index 配置，同步、无 DB）。"""
    return InMemoryStore(index={"dims": _DIMS, "embed": embedder, "fields": ["content"]})


class _RecordingStore(InMemoryStore):
    """透传代理：记录最近一次 asearch 的 query/limit（锁定 fetch_factor 契约防回归）。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.last_query: str | None = None
        self.last_limit: int | None = None

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, object] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[object]:
        self.last_query = query
        self.last_limit = limit
        return await super().asearch(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )


def _make_item(
    *,
    id: str,
    content: str,
    kind: str = KIND_FACT,
    importance: float = 0.5,
    timestamp: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=id,
        kind=kind,
        content=content,
        session_id="s1",
        user_id="u1",
        timestamp=timestamp or datetime.now(UTC),
        provenance="test",
        importance=importance,
    )


# —— 单元：混合重排信号 ——


def test_aware_utc_normalizes_naive() -> None:
    """naive timestamp 视为 UTC（混合比较不抛 TypeError）。"""
    naive = datetime(2026, 1, 1)
    assert _aware_utc(naive).tzinfo is UTC
    aware = datetime(2026, 1, 1, tzinfo=UTC)
    assert _aware_utc(aware) == aware


def test_clamp01() -> None:
    """负 cosine / 超 1 统一归到 [0,1]（混合信号非负区间）。"""
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(1.5) == 1.0
    assert _clamp01(0.3) == 0.3


def test_recency_signal() -> None:
    """近因信号：now→1.0；half_life→0.5；远过去→≈0；未来 clamp 1.0；naive 不抛。"""
    now = datetime.now(UTC)
    half = 30.0
    assert _recency_signal(now, half_life_days=half) == pytest.approx(1.0)
    assert _recency_signal(now - timedelta(days=30), half_life_days=half) == pytest.approx(0.5)
    far = _recency_signal(now - timedelta(days=300), half_life_days=half)
    assert far == pytest.approx(1.0 / 11.0)
    assert _recency_signal(now + timedelta(days=10), half_life_days=half) == pytest.approx(1.0)
    assert 0.0 <= _recency_signal(datetime(2026, 1, 1), half_life_days=half) <= 1.0


def test_hybrid_rerank_weights_order() -> None:
    """两套职责权重次序：content（score 主导）a 胜；preference（importance 主导）b 胜。"""
    store = MemoryStore(InMemoryStore())
    now = datetime.now(UTC)
    a = _make_item(id="a", content="a", importance=0.0, timestamp=now)
    b = _make_item(id="b", content="b", importance=1.0, timestamp=now)
    # content (0.6,0.25,0.15)：score 1.0 vs 0.0 → a 胜（0.6 > 0.25）。
    ranked = store._hybrid_rerank([(a, 1.0), (b, 0.0)], weights=(0.6, 0.25, 0.15))
    assert [m.id for m, _ in ranked] == ["a", "b"]
    # preference (0.2,0.6,0.2)：importance 主导 → b 胜（0.6 > 0.2）。
    ranked_pref = store._hybrid_rerank([(a, 1.0), (b, 0.0)], weights=(0.2, 0.6, 0.2))
    assert [m.id for m, _ in ranked_pref] == ["b", "a"]


def test_hybrid_rerank_score_missing_and_negative() -> None:
    """raw=None / 负数按 0 计；输出保留原始 raw（供 Citation.score 回填）。"""
    store = MemoryStore(InMemoryStore())
    now = datetime.now(UTC)
    zero = _make_item(id="zero", content="z", importance=0.0, timestamp=now)
    low = _make_item(id="low", content="l", importance=0.1, timestamp=now)
    ranked = store._hybrid_rerank([(zero, -0.5), (low, None)], weights=(0.6, 0.25, 0.15))
    # zero：score 0 + imp 0；low：score 0 + imp 0.1 → low 胜。
    assert [m.id for m, _ in ranked] == ["low", "zero"]
    ranked_pairs = store._hybrid_rerank([(zero, -0.5), (low, 0.7)], weights=(0.6, 0.25, 0.15))
    # 原始 raw 原样保留（-0.5 与 0.7）。
    assert dict((m.id, raw) for m, raw in ranked_pairs) == {"zero": -0.5, "low": 0.7}


# —— 集成：双模式 recall ——

_SEMANTIC_QUERY = "用户喜欢美式咖啡"


def _semantic_ms() -> tuple[MemoryStore, _MapEmbedder]:
    embedder = _MapEmbedder()
    embedder.register(_SEMANTIC_QUERY, _E0)
    return MemoryStore(_semantic_store(embedder)), embedder


async def test_semantic_prefers_semantic_match_over_importance() -> None:
    """语义压倒 importance：精确命中（score 1.0，低 importance）排在无关高 importance 前。"""
    ms, _ = _semantic_ms()
    await ms.save(_make_item(id="match", content=_SEMANTIC_QUERY, importance=0.1))
    await ms.save(_make_item(id="irrelevant", content="用户是程序员", importance=0.9))
    result = await ms.recall(user_id="u1", query=_SEMANTIC_QUERY)
    assert [m.id for m in result.items] == ["match", "irrelevant"]
    # Citation.score 回填原始余弦：精确命中 1.0，无关项 0.0。
    assert result.sources[0].source_id == "match"
    assert result.sources[0].score == pytest.approx(1.0)
    assert result.sources[1].score == pytest.approx(0.0)


async def test_recall_hybrid_weights_override() -> None:
    """recall 的 hybrid_weights 参数覆盖默认 content_weights：
    (0.2,0.6,0.2) 下 importance 主导 → 无关高 importance 排在精确命中前。"""
    ms, _ = _semantic_ms()
    await ms.save(_make_item(id="match", content=_SEMANTIC_QUERY, importance=0.2))
    await ms.save(_make_item(id="irrelevant", content="用户是程序员", importance=0.9))
    default = await ms.recall(user_id="u1", query=_SEMANTIC_QUERY)
    assert [m.id for m in default.items] == ["match", "irrelevant"]
    pref = await ms.recall(user_id="u1", query=_SEMANTIC_QUERY, hybrid_weights=(0.2, 0.6, 0.2))
    assert [m.id for m in pref.items] == ["irrelevant", "match"]


async def test_semantic_kind_filter_after_prefetch() -> None:
    """kind 组过滤在预取之后：语义命中 preference 被 kinds=[FACT] 滤出，只返回 fact。"""
    ms, _ = _semantic_ms()
    await ms.save(
        _make_item(id="pref", kind=KIND_PREFERENCE, content=_SEMANTIC_QUERY, importance=0.9)
    )
    await ms.save(_make_item(id="fact", kind=KIND_FACT, content="用户是程序员", importance=0.1))
    result = await ms.recall(user_id="u1", query=_SEMANTIC_QUERY, kinds=[KIND_FACT])
    assert [m.id for m in result.items] == ["fact"]


async def test_semantic_top_k_and_citation_scores() -> None:
    """语义模式 top_k 截断；每条 citation 的 score 为 float 非 None（区别于确定性）。"""
    ms, _ = _semantic_ms()
    for i in range(5):
        await ms.save(_make_item(id=f"m{i}", content=f"内容{i}", importance=0.5))
    result = await ms.recall(user_id="u1", query=_SEMANTIC_QUERY, top_k=2)
    assert len(result.items) == 2
    assert all(c.score is not None for c in result.sources)


async def test_semantic_requires_query() -> None:
    """带 embeddings 但 query 为空/空白 → 确定性 importance 排序（零回归）。"""
    ms, _ = _semantic_ms()
    await ms.save(_make_item(id="low", content=_SEMANTIC_QUERY, importance=0.1))
    await ms.save(_make_item(id="high", content="用户是程序员", importance=0.9))
    for query in (None, "", "   "):
        result = await ms.recall(user_id="u1", query=query)
        assert [m.id for m in result.items] == ["high", "low"]
        assert all(c.score is None for c in result.sources)


async def test_semantic_requires_embeddings() -> None:
    """无 embeddings（store 无 index）+ query → 确定性（回归护栏：有 query 也无语义）。"""
    ms = MemoryStore(InMemoryStore())
    await ms.save(_make_item(id="low", content=_SEMANTIC_QUERY, importance=0.1))
    await ms.save(_make_item(id="high", content="用户是程序员", importance=0.9))
    result = await ms.recall(user_id="u1", query=_SEMANTIC_QUERY)
    assert [m.id for m in result.items] == ["high", "low"]


async def test_recency_tiebreak_newer_first() -> None:
    """同 content（同 score、同 importance）：recency 打破平局 → 较新者靠前。"""
    ms, embedder = _semantic_ms()
    embedder.register("同好", _E0)
    now = datetime.now(UTC)
    await ms.save(
        _make_item(id="old", content="同好", importance=0.5, timestamp=now - timedelta(days=60))
    )
    await ms.save(_make_item(id="new", content="同好", importance=0.5, timestamp=now))
    result = await ms.recall(user_id="u1", query="同好")
    assert [m.id for m in result.items] == ["new", "old"]


async def test_semantic_recall_uses_fetch_factor_limit() -> None:
    """asearch limit 契约：语义 limit=top_k*fetch_factor 且传 query；
    确定性 limit=top_k*2 无 query。"""
    embedder = _MapEmbedder()
    embedder.register(_SEMANTIC_QUERY, _E0)
    store = _RecordingStore(index={"dims": _DIMS, "embed": embedder, "fields": ["content"]})
    ms = MemoryStore(store)
    for i in range(5):
        await ms.save(_make_item(id=f"m{i}", content=f"内容{i}", importance=0.5))
    await ms.recall(user_id="u1", query=_SEMANTIC_QUERY, top_k=2)
    assert store.last_query == _SEMANTIC_QUERY
    assert store.last_limit == 2 * 4  # top_k * recall_fetch_factor
    await ms.recall(user_id="u1", top_k=3)
    assert store.last_query is None
    assert store.last_limit == 3 * 2  # v2.0 确定性 limit


# —— 图级：load_context / recall_memory 传 query 接线 ——


async def test_load_context_preload_preference_importance_dominant(build_graph, run_graph) -> None:
    """load_context 偏好预加载走 preference_weights：importance 主导（身份先验），
    无关高 importance 偏好排在精确命中低 importance 前。"""
    embedder = _MapEmbedder()
    embedder.register(_SEMANTIC_QUERY, _E0)
    injected = _semantic_store(embedder)
    await MemoryStore(injected).save(
        _make_item(id="p1", kind=KIND_PREFERENCE, content=_SEMANTIC_QUERY, importance=0.2)
    )
    await MemoryStore(injected).save(
        _make_item(id="p2", kind=KIND_PREFERENCE, content="用户是程序员", importance=0.9)
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), store=injected)
    response, _ = await run_graph(graph, text=_SEMANTIC_QUERY, user_id="u1")
    assert response is not None
    snap = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    ctx = snap.values.get("memory_context") or []
    # content 权重（语义主导）会给 [p1, p2]；preference 权重 importance 主导 → 反转。
    assert [m.id for m in ctx] == ["p2", "p1"]


async def test_load_context_preload_semantic_decides_tie(build_graph, run_graph) -> None:
    """preference 权重下 importance 相同 → 语义决胜：精确命中偏好前置。"""
    embedder = _MapEmbedder()
    embedder.register(_SEMANTIC_QUERY, _E0)
    injected = _semantic_store(embedder)
    await MemoryStore(injected).save(
        _make_item(id="p1", kind=KIND_PREFERENCE, content=_SEMANTIC_QUERY, importance=0.5)
    )
    await MemoryStore(injected).save(
        _make_item(id="p2", kind=KIND_PREFERENCE, content="用户是程序员", importance=0.5)
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), store=injected)
    response, _ = await run_graph(graph, text=_SEMANTIC_QUERY, user_id="u1")
    assert response is not None
    snap = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    ctx = snap.values.get("memory_context") or []
    assert [m.id for m in ctx] == ["p1", "p2"]


async def test_recall_memory_passes_query_on_tool_branch(build_graph, run_graph) -> None:
    """recall_memory 传的是**改写后的检索查询**（T2）：TOOL_USE 分支 fact 语义命中。

    v6.0 起 query 来源从 `state["input"]` 变为 `state["retrieval_query"]`（主改写），
    故脚本里的查询理解消息给出与注册向量一致的改写文本 —— 断言即「召回用的是改写结果」。
    """
    embedder = _MapEmbedder()
    embedder.register("12加7等于多少", _E0)
    injected = _semantic_store(embedder)
    await MemoryStore(injected).save(
        _make_item(id="f1", kind=KIND_FACT, content="12加7等于多少", importance=0.2)
    )
    await MemoryStore(injected).save(
        _make_item(id="f2", kind=KIND_FACT, content="用户是程序员", importance=0.9)
    )
    calc = make_fake_tool("calc", content="19")
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="test")
            ),
            understand_message("12加7等于多少"),
            AIMessage(content="", tool_calls=[{"name": "calc", "args": {}, "id": "c1"}]),
            fake_text_message("结果是 19"),
        ],
        tools=[calc],
        store=injected,
    )
    response, _ = await run_graph(graph, text="12加7等于多少", user_id="u1")
    assert response is not None
    snap = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    ctx = snap.values.get("memory_context") or []
    # f1 语义精确命中（改写查询与注册向量一致）→ 压在 importance 更高的 f2 之前。
    assert [m.id for m in ctx] == ["f1", "f2"]


async def test_graph_semantic_falls_back_without_embeddings(build_graph, run_graph) -> None:
    """无 embeddings 图内零回归：有 query 仍走 importance 确定性排序。"""
    injected = InMemoryStore()  # 无 index → store_has_embeddings False。
    await MemoryStore(injected).save(
        _make_item(id="f_high", kind=KIND_FACT, content="用户是程序员", importance=0.9)
    )
    await MemoryStore(injected).save(
        _make_item(id="f_low", kind=KIND_FACT, content=_SEMANTIC_QUERY, importance=0.1)
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), store=injected)
    response, _ = await run_graph(graph, text=_SEMANTIC_QUERY, user_id="u1")
    assert response is not None
    snap = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    ctx = snap.values.get("memory_context") or []
    assert [m.id for m in ctx] == ["f_high", "f_low"]


# —— v6.0 T2：多变体查询融合（`variant_queries`）——


class _QueryRecordingStore(InMemoryStore):
    """记录**每次** asearch 的 query（验证多变体并发检索与单查询零回归）。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.queries: list[str | None] = []

    async def asearch(self, namespace_prefix, /, **kwargs):  # type: ignore[no-untyped-def]
        self.queries.append(kwargs.get("query"))
        return await super().asearch(namespace_prefix, **kwargs)


async def test_recall_with_variant_queries_hits_union() -> None:
    """`variant_queries` 触发并发多查询 + RRF 融合：只被变体命中的条目也进入结果。

    WHY 用「有变体 vs 无变体」对照断言（而非固定分数）：受控 embedder 的余弦由预置向量
    决定，但「哪一条各自独占命中」要写死分数才能断言 —— 对照实验直接证明**变体检索是
    结果集变化的原因**，比钉死数值更能说明行为，也不受排序细节影响。
    """
    embedder = _MapEmbedder()
    embedder.register("出差报销", _E0)
    embedder.register("差旅补贴标准", _E1)
    store = _QueryRecordingStore(index={"dims": _DIMS, "embed": embedder, "fields": ["content"]})
    ms = MemoryStore(store)
    await ms.save(_make_item(id="f1", kind=KIND_FACT, content="差旅补贴标准", importance=0.3))
    await ms.save(_make_item(id="f2", kind=KIND_FACT, content="出差报销制度", importance=0.3))

    baseline = await ms.recall(user_id="u1", kinds=[KIND_FACT], top_k=5, query="出差报销")
    store.queries.clear()
    with_variants = await ms.recall(
        user_id="u1", kinds=[KIND_FACT], top_k=5, query="出差报销", variant_queries=["差旅补贴标准"]
    )

    # 主查询 + 变体各检索一次（并发两路）。
    assert store.queries == ["出差报销", "差旅补贴标准"]
    # 变体路把「主查询命中不到的那条」拉进了结果集。
    assert {m.id for m in with_variants.items} >= {m.id for m in baseline.items}
    assert {m.id for m in with_variants.items} == {"f1", "f2"}
    # 代表分数是各路的**最高原始语义分**（融合分不冒充相似度）。
    assert all(c.score is not None for c in with_variants.sources)


async def test_recall_without_variants_uses_single_query() -> None:
    """未传 `variant_queries` → 单查询路径（零回归：不并发、不融合）。"""
    embedder = _MapEmbedder()
    embedder.register("差旅补贴标准", _E0)
    store = _QueryRecordingStore(index={"dims": _DIMS, "embed": embedder, "fields": ["content"]})
    ms = MemoryStore(store)
    await ms.save(_make_item(id="f1", kind=KIND_FACT, content="差旅补贴标准", importance=0.3))
    await ms.save(_make_item(id="f2", kind=KIND_FACT, content="用户是程序员", importance=0.9))

    result = await ms.recall(user_id="u1", kinds=[KIND_FACT], top_k=5, query="差旅补贴标准")

    assert store.queries == ["差旅补贴标准"]
    assert [m.id for m in result.items] == ["f1", "f2"]


async def test_recall_variants_ignored_in_deterministic_mode() -> None:
    """无 embeddings（确定性模式）不看查询：变体参数不产生额外检索（该模式与查询无关）。"""
    store = _QueryRecordingStore()  # 无 index
    ms = MemoryStore(store)
    await ms.save(_make_item(id="f1", kind=KIND_FACT, content="差旅补贴标准", importance=0.3))

    await ms.recall(
        user_id="u1", kinds=[KIND_FACT], top_k=5, query="出差报销", variant_queries=["差旅补贴标准"]
    )

    assert store.queries == [None]  # 确定性模式 asearch 不带 query
