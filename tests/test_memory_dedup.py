"""V3-M3 保存去重（content-hash 精确层 + 语义近似层 upsert）测试（离线，无 numpy / 无 DB）。

覆盖：`normalize_content_hash` 纯函数、L1 精确层（完全重复/归一化变体/空 hash 跳过）、
L2 语义层（高分合并/低分新写/阈值边界/kind 隔离/无 embeddings 降级）、
D4 合并规则（原 id 保留/importance=max/content 取更长/recency 刷新/来源更新）、
`content_hash` 向后兼容、persist 接线（dedup 启用合并 / 禁用退回 v2.0 / 落库指纹）。

WHY 用 `_MapEmbedder` 而非 `tests.fakes.FakeEmbeddings`：FakeEmbeddings 的 hash 向量
对「不同文本」的 cosine 不可解析预测（实测两段中文短语可达 0.75），无法可靠构造
「高于/低于阈值」的断言；`_MapEmbedder` 按文本映射预置单位向量，未注册文本返回
与 e0 正交的默认向量（cosine 恰 0），score 精确可控、断言确定。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from agent.core.config import DedupConfig
from agent.memory import (
    KIND_FACT,
    KIND_PREFERENCE,
    LONG_TERM_NAMESPACE,
    RELATION_EXACT,
    RELATION_NOT_DUPLICATE,
    RELATION_OVERLAP,
    CandidateVerdict,
    MemoryItem,
    MemoryStore,
    normalize_content_hash,
    save_conversation_memory,
)
from agent.memory.models import MemoryExtraction

_DIMS = 8
# 注册向量沿 e0；默认向量沿 e1（与 e0 正交 → 未注册文本 cosine 恰 0）。
_E0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_THRESHOLD = DedupConfig().semantic_threshold


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
    """构造带语义索引的 InMemoryStore（镜像 build_memory_backends 的 index 配置）。"""
    return InMemoryStore(index={"dims": _DIMS, "embed": embedder, "fields": ["content"]})


def _unit_vector_at(cosine: float) -> list[float]:
    """沿 e0 方向、与 e0 夹角余弦为 `cosine` 的单位向量：InMemoryStore 返回的 score 即该值。"""
    v = [0.0] * _DIMS
    v[0] = cosine
    v[1] = math.sqrt(1.0 - cosine**2)
    return v


def _make_item(
    *,
    id: str,
    content: str,
    kind: str = KIND_FACT,
    importance: float = 0.5,
    timestamp: datetime | None = None,
    session_id: str = "s1",
    user_id: str = "u1",
    content_hash: str | None = None,
) -> MemoryItem:
    """构造 MemoryItem；`content_hash` 缺省由内容归一化指纹算出（"" 则显式跳过 L1）。"""
    return MemoryItem(
        id=id,
        kind=kind,
        content=content,
        session_id=session_id,
        user_id=user_id,
        timestamp=timestamp or datetime.now(UTC),
        provenance="test",
        importance=importance,
        content_hash=normalize_content_hash(content) if content_hash is None else content_hash,
    )


class _StubExtractor:
    """固定返回给定抽取列表的 stub 抽取器（可空）；记录 config 供透传断言。"""

    def __init__(self, extractions: Sequence[MemoryExtraction]) -> None:
        self._extractions = list(extractions)
        self.configs: list[dict | None] = []

    async def extract(
        self, messages: Sequence[BaseMessage], *, config: dict | None = None
    ) -> list[MemoryExtraction]:
        self.configs.append(config)
        return list(self._extractions)


async def _count_items(store: MemoryStore, *, user_id: str = "u1") -> int:
    """当前用户命名空间的条目数（演示规模用 top-50 召回近似）。"""
    return len((await store.recall(user_id=user_id, top_k=50)).items)


# —— 单元：normalize_content_hash ——


def test_normalize_content_hash_ignores_whitespace_punct_case() -> None:
    """空白/全角标点/大小写差异归一到同一指纹（L1 精确层零误并前提）。"""
    assert normalize_content_hash("我喜欢喝美式咖啡。") == normalize_content_hash(
        "我喜欢喝美式咖啡"
    )
    assert normalize_content_hash(" User  is  a developer ") == normalize_content_hash(
        "user is a DEVELOPER"
    )
    assert normalize_content_hash("苹果，香蕉；梨！") == normalize_content_hash("苹果香蕉梨")


def test_normalize_content_hash_distinguishes_different_text() -> None:
    """不同内容指纹不同（SHA-256 冲突可忽略）；否定/换词不误并。"""
    assert normalize_content_hash("我喜欢喝美式咖啡") != normalize_content_hash("我讨厌喝美式咖啡")
    assert normalize_content_hash("用户是程序员") != normalize_content_hash("用户是设计师")
    assert normalize_content_hash("") == normalize_content_hash("")


# —— L1 精确层 ——


async def test_upsert_exact_duplicate_merges() -> None:
    """完全重复内容 → merged：原 id 复用、importance 取 max、条数不涨。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡", importance=0.3))
    outcome = await store.upsert(
        _make_item(id="dup", content="用户喜欢喝美式咖啡", importance=0.9),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "merged"
    assert outcome.item.id == "base"  # 原 id 保留（citation source_id 稳定）
    assert outcome.item.importance == 0.9  # importance = max(旧, 新)
    assert await _count_items(store) == 1


async def test_upsert_normalized_variant_merges() -> None:
    """仅标点/空白差异（归一化后同内容）→ L1 精确命中合并。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    outcome = await store.upsert(
        _make_item(id="var", content="用户喜欢喝美式咖啡。"),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "merged"
    assert outcome.item.id == "base"
    assert await _count_items(store) == 1


async def test_upsert_empty_content_hash_skips_exact_layer() -> None:
    """content_hash=""（旧数据/直接调用）→ L1 跳过；无 embeddings 时按新事实落盘。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    outcome = await store.upsert(
        _make_item(id="empty-hash", content="用户喜欢喝美式咖啡", content_hash=""),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "inserted"
    assert await _count_items(store) == 2


# —— L2 语义层 ——

_SEMANTIC_BASE = "用户喜欢喝美式咖啡"
_SEMANTIC_PARAPHRASE = "用户每天早上爱喝黑咖啡"


def _semantic_ms(paraphrase_cosine: float) -> tuple[MemoryStore, _MapEmbedder]:
    """构造语义 store：base 注册 e0，改写文本注册为与 e0 夹角余弦=`paraphrase_cosine`。"""
    embedder = _MapEmbedder()
    embedder.register(_SEMANTIC_BASE, _E0)
    embedder.register(_SEMANTIC_PARAPHRASE, _unit_vector_at(paraphrase_cosine))
    return MemoryStore(_semantic_store(embedder)), embedder


async def test_upsert_semantic_high_score_merges() -> None:
    """同 kind 语义改写 cosine≥阈值 → merged（同事实不同表述），保留原 id。"""
    ms, _ = _semantic_ms(0.95)
    await ms.save(_make_item(id="base", content=_SEMANTIC_BASE))
    outcome = await ms.upsert(
        _make_item(id="para", content=_SEMANTIC_PARAPHRASE),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "merged"
    assert outcome.item.id == "base"
    assert await _count_items(ms) == 1


async def test_upsert_semantic_low_score_inserts() -> None:
    """同 kind 语义改写 cosine<阈值 → 视为近似但不同 → 新写（误并被阈值拦住）。"""
    ms, _ = _semantic_ms(0.85)
    await ms.save(_make_item(id="base", content=_SEMANTIC_BASE))
    outcome = await ms.upsert(
        _make_item(id="para", content=_SEMANTIC_PARAPHRASE),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "inserted"
    assert await _count_items(ms) == 2


async def test_upsert_semantic_threshold_boundary() -> None:
    """阈值边界：≥0.92 merged、<0.92 inserted（默认保守阈值两侧各验一遍）。"""
    for cosine, expect in ((0.921, "merged"), (0.91, "inserted")):
        ms, _ = _semantic_ms(cosine)
        await ms.save(_make_item(id="base", content=_SEMANTIC_BASE))
        outcome = await ms.upsert(
            _make_item(id="para", content=_SEMANTIC_PARAPHRASE),
            semantic_threshold=_THRESHOLD,
        )
        assert outcome.action == expect, f"cosine={cosine} 期望 {expect}"


async def test_upsert_semantic_kind_mismatch_inserts() -> None:
    """kind 不同（fact 改写撞上 preference）→ filter 隔离，绝不跨 kind 合并。"""
    embedder = _MapEmbedder()
    embedder.register(_SEMANTIC_BASE, _E0)
    embedder.register(_SEMANTIC_PARAPHRASE, _unit_vector_at(0.95))
    ms = MemoryStore(_semantic_store(embedder))
    # 存量是 preference，改写来的 fact 即使 cosine 高分也不并入。
    await ms.save(_make_item(id="pref", kind=KIND_PREFERENCE, content=_SEMANTIC_BASE))
    outcome = await ms.upsert(
        _make_item(id="fact", kind=KIND_FACT, content=_SEMANTIC_PARAPHRASE),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "inserted"
    assert await _count_items(ms) == 2


async def test_upsert_semantic_unavailable_falls_back_to_exact_only() -> None:
    """无 embeddings（store 无 index）→ L2 跳过，语义改写按新事实落盘（D3 降级）。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content=_SEMANTIC_BASE))
    outcome = await store.upsert(
        _make_item(id="para", content=_SEMANTIC_PARAPHRASE),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "inserted"
    assert await _count_items(store) == 2


# —— find_dedup_candidates（去重候选召回，D4-2 判定器输入） ——


async def test_find_candidates_exact_hit_returns_exact_only() -> None:
    """L1 content-hash 精确命中 → (exact, [])：不经 L2、不省语义查询。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    exact, semantic = await store.find_dedup_candidates(
        _make_item(id="dup", content="用户喜欢喝美式咖啡")
    )
    assert exact is not None and exact.id == "base"
    assert semantic == []


async def test_find_candidates_semantic_returns_topk() -> None:
    """L1 未命中且 store 配 embeddings → (None, semantic top-K)，按余弦降序、带分数。"""
    ms, _ = _semantic_ms(0.95)
    await ms.save(_make_item(id="base", content=_SEMANTIC_BASE))
    exact, semantic = await ms.find_dedup_candidates(
        _make_item(id="para", content=_SEMANTIC_PARAPHRASE)
    )
    assert exact is None
    assert len(semantic) == 1
    item, score = semantic[0]
    assert item.id == "base"
    assert score == pytest.approx(0.95)


async def test_find_candidates_no_embeddings_returns_empty() -> None:
    """store 未配 embeddings（无 index）→ L2 跳过，(None, [])（D3 降级，仅 L1）。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    exact, semantic = await store.find_dedup_candidates(
        _make_item(id="para", content=_SEMANTIC_PARAPHRASE)
    )
    assert exact is None
    assert semantic == []


async def test_find_candidates_empty_hash_skips_exact() -> None:
    """content_hash="" → L1 跳过（旧数据/直接调用）；无 embeddings → 空候选。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    exact, semantic = await store.find_dedup_candidates(
        _make_item(id="dup", content="用户喜欢喝美式咖啡", content_hash="")
    )
    assert exact is None
    assert semantic == []


# —— D4 合并规则 ——


async def test_merge_content_longer_wins() -> None:
    """合并规则：content 取更长者（近似合并不丢信息）；incoming 更长则替换。

    WHY 直测 `merge` 而非走 upsert：content 长度差异意味着归一化指纹不同，
    普通 store（无 embeddings）上 L1/L2 都不会命中 → 必为 inserted；
    content 取长规则是 `merge` 内部行为，直接调公开方法（同 test_memory_semantic
    直测 `_hybrid_rerank` 的先例）才能隔离观测。
    """
    store = MemoryStore(InMemoryStore())
    namespace = ("u1", LONG_TERM_NAMESPACE)
    existing = _make_item(id="base", content="用户喜欢喝美式咖啡")
    # incoming 更长 → 合并结果 content 用 incoming，原 id 保留。
    longer = "用户喜欢喝美式咖啡，每天早上都要来一杯"
    outcome = await store.merge(namespace, existing, _make_item(id="long", content=longer))
    assert outcome.action == "merged"
    assert outcome.item.id == "base"
    assert outcome.item.content == longer
    # 反向：存量更长 → 保留存量 content（稳定性），id 仍不变。
    shorter = "用户喜欢喝美式咖啡"
    out2 = await store.merge(namespace, existing, _make_item(id="short", content=shorter))
    assert out2.item.id == "base"
    assert out2.item.content == existing.content


async def test_merge_importance_takes_max() -> None:
    """importance = max(旧, 新)：低→高升档、高→低不退化。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡", importance=0.9))
    outcome = await store.upsert(
        _make_item(id="low", content="用户喜欢喝美式咖啡", importance=0.2),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.item.importance == 0.9  # 存量高 → 保留
    outcome2 = await store.upsert(
        _make_item(id="high", content="用户喜欢喝美式咖啡", importance=1.0),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome2.item.importance == 1.0  # 新值高 → 升档


async def test_merge_refreshes_timestamp_to_newer() -> None:
    """合并刷新 timestamp（recency 更新）为两条里较新者。"""
    store = MemoryStore(InMemoryStore())
    old = datetime.now(UTC) - timedelta(days=5)
    new = datetime.now(UTC)
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡", timestamp=old))
    outcome = await store.upsert(
        _make_item(id="fresh", content="用户喜欢喝美式咖啡", timestamp=new),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.item.timestamp == new
    assert outcome.item.timestamp > old


async def test_merge_records_recent_source() -> None:
    """provenance/session_id 记录最近来源（incoming 的会话与来源）。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    outcome = await store.upsert(
        _make_item(id="later", content="用户喜欢喝美式咖啡", session_id="s2"),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.item.session_id == "s2"
    assert outcome.item.provenance == "test"


# —— 向后兼容 ——


def test_memory_item_validates_without_content_hash() -> None:
    """旧数据行（无 content_hash 键）model_validate 默认空串，向后兼容。"""
    raw = _make_item(id="old", content="用户喜欢喝美式咖啡").model_dump(mode="json")
    raw.pop("content_hash")
    item = MemoryItem.model_validate(raw)
    assert item.content_hash == ""


# —— persist 接线 ——


async def test_save_conversation_memory_dedup_merges_duplicates() -> None:
    """抽取到两条同内容事实 + dedup 启用 → 第二次 upsert 合并，条数不涨。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor(
        [
            MemoryExtraction(kind=KIND_FACT, content="用户喜欢喝美式咖啡", importance=0.7),
            MemoryExtraction(kind=KIND_FACT, content="用户喜欢喝美式咖啡。", importance=0.8),
        ]
    )
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
        dedup=DedupConfig(),
    )
    result = await store.recall(user_id="u1")
    assert len(result.items) == 1
    assert result.items[0].importance == 0.8  # max(0.7, 0.8)


async def test_save_conversation_memory_dedup_disabled_inserts() -> None:
    """dedup.enabled=False → 退回 v2.0 逐条 save，同内容各存一条（零回归）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor(
        [
            MemoryExtraction(kind=KIND_FACT, content="用户喜欢喝美式咖啡", importance=0.7),
            MemoryExtraction(kind=KIND_FACT, content="用户喜欢喝美式咖啡", importance=0.7),
        ]
    )
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
        dedup=DedupConfig(enabled=False),
    )
    assert await _count_items(store) == 2


async def test_save_conversation_memory_stores_content_hash() -> None:
    """落库条目携带归一化指纹（为未来再启用去重留哈希）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor(
        [MemoryExtraction(kind=KIND_FACT, content="用户喜欢喝美式咖啡", importance=0.7)]
    )
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )
    result = await store.recall(user_id="u1")
    assert result.items[0].content_hash == normalize_content_hash("用户喜欢喝美式咖啡")


# —— D4-2 判定器决策（MemoryRelationJudge stub 注入固定 verdicts，锁定三分类策略） ——


class _StubJudge:
    """固定返回给定 verdicts 的 stub 判定器（模拟 LLM 三分类结果，不触发真实调用）。"""

    def __init__(self, verdicts: Sequence[CandidateVerdict]) -> None:
        self._verdicts = list(verdicts)
        self.judged: list[list[MemoryItem]] = []  # 记录每轮被判定的候选，供调用侧断言
        self.configs: list[dict | None] = []  # 记录收到的 config，供观测透传断言（O3）

    async def judge(
        self,
        item: MemoryItem,
        candidates: Sequence[MemoryItem],
        *,
        config: dict | None = None,
    ) -> list[CandidateVerdict]:
        self.judged.append(list(candidates))
        self.configs.append(config)
        return list(self._verdicts)


async def _save_with_judge(
    ms: MemoryStore,
    verdicts: Sequence[CandidateVerdict],
    *,
    content: str,
    judge: bool = True,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> _StubJudge | None:
    """经 `save_conversation_memory`（dedup 启用）驱动一次带去重保存并返回 stub judge。

    `judge=False` 时不注入判定器 → 走 upsert 阈值路径（回归锁定），返回 None。
    `callbacks` / `metadata` 供观测透传断言（O3）。
    """
    extractor = _StubExtractor([MemoryExtraction(kind=KIND_FACT, content=content, importance=0.7)])
    stub = _StubJudge(verdicts) if judge else None
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=ms,
        dedup=DedupConfig(),
        judge=stub,
        callbacks=callbacks,
        metadata=metadata,
    )
    return stub


def _judge_semantic_store(base_text: str, *, incoming: str, cosine: float) -> MemoryStore:
    """语义 store：存量文本注册 e0、待保存文本注册与 e0 夹角余弦=cosine 的向量。"""
    embedder = _MapEmbedder()
    embedder.register(base_text, _E0)
    embedder.register(incoming, _unit_vector_at(cosine))
    return MemoryStore(_semantic_store(embedder))


def _composite_store() -> MemoryStore:
    """牛奶/咖啡双候选 store：待保存「牛奶和咖啡」对两候选余弦都高（组合场景）。

    牛奶沿 e0；咖啡沿 [0.5, √0.75]（与牛奶余弦 0.5，非重复基线由判定器定论）；
    组合向量 [0.8, 0.6]：与牛奶余弦 0.8、与咖啡余弦 0.9196——两条都进 top-K 候选。
    """
    embedder = _MapEmbedder()
    embedder.register("用户喜欢喝牛奶", _E0)
    coffee = [0.0] * _DIMS
    coffee[0], coffee[1] = 0.5, math.sqrt(0.75)
    embedder.register("用户喜欢喝咖啡", coffee)
    composite = [0.0] * _DIMS
    composite[0], composite[1] = 0.8, 0.6
    embedder.register("用户喜欢喝牛奶和咖啡", composite)
    return MemoryStore(_semantic_store(embedder))


def _exact_overlap_store() -> MemoryStore:
    """双候选 store：待保存「用户是男生，身高175厘米」按余弦降序命中 男生(0.96)→身高(0.8)。

    男生沿 e0；身高沿 [0.6, 0.8]；组合向量 [0.96, 0.28] 与男生余弦 0.96、与身高余弦 0.8，
    故 asearch 顺序固定为 候选1=男生、候选2=身高（verdict.index 因此确定）。
    """
    embedder = _MapEmbedder()
    embedder.register("用户是男生", _E0)
    height = [0.0] * _DIMS
    height[0], height[1] = 0.6, 0.8
    embedder.register("用户身高175厘米", height)
    both = [0.0] * _DIMS
    both[0], both[1] = 0.96, 0.28
    embedder.register("用户是男生，身高175厘米", both)
    return MemoryStore(_semantic_store(embedder))


async def test_save_judge_single_exact_merges() -> None:
    """判定器判单候选 exact_duplicate（同义改写）→ 合并进该候选，id 保留、条数不涨。"""
    ms = _judge_semantic_store("用户喜欢喝美式咖啡", incoming="用户喜欢喝黑咖啡", cosine=0.95)
    await ms.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    await _save_with_judge(
        ms,
        [CandidateVerdict(index=1, relation=RELATION_EXACT, reason="同义改写")],
        content="用户喜欢喝黑咖啡",
    )
    result = await ms.recall(user_id="u1")
    assert len(result.items) == 1
    assert result.items[0].id == "base"


async def test_save_judge_receives_observation_config() -> None:
    """判定器（带外链路第二个 LLM 调用）经 config 收到 callbacks+metadata（O3）。"""
    ms = _judge_semantic_store("用户喜欢喝美式咖啡", incoming="用户喜欢喝黑咖啡", cosine=0.95)
    await ms.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    handler = UsageMetadataCallbackHandler()
    stub = await _save_with_judge(
        ms,
        [CandidateVerdict(index=1, relation=RELATION_EXACT, reason="同义改写")],
        content="用户喜欢喝黑咖啡",
        callbacks=[handler],
        metadata={"langfuse_session_id": "s1"},
    )
    assert stub is not None
    assert stub.configs == [{"callbacks": [handler], "metadata": {"langfuse_session_id": "s1"}}]


async def test_save_judge_default_config_is_none() -> None:
    """未接线时判定器收到 config=None（与接线前逐字同路，零回归判据）。"""
    ms = _judge_semantic_store("用户喜欢喝美式咖啡", incoming="用户喜欢喝黑咖啡", cosine=0.95)
    await ms.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    stub = await _save_with_judge(
        ms,
        [CandidateVerdict(index=1, relation=RELATION_EXACT, reason="同义改写")],
        content="用户喜欢喝黑咖啡",
    )
    assert stub is not None
    assert stub.configs == [None]


async def test_save_judge_single_overlap_merges_new_info() -> None:
    """判定器判单候选 overlap_merge（补充新信息）→ 合并吸收，content 取更完整版本。"""
    ms = _judge_semantic_store(
        "用户下周一去北京出差", incoming="用户下周一去北京出差，周三返回", cosine=0.95
    )
    await ms.save(_make_item(id="base", content="用户下周一去北京出差"))
    incoming = "用户下周一去北京出差，周三返回"
    await _save_with_judge(
        ms,
        [CandidateVerdict(index=1, relation=RELATION_OVERLAP, reason="补充返回时间")],
        content=incoming,
    )
    result = await ms.recall(user_id="u1")
    assert len(result.items) == 1
    assert result.items[0].id == "base"
    assert result.items[0].content == incoming  # 更长者保留（新信息不丢失）


async def test_save_judge_all_not_duplicate_inserts() -> None:
    """判定器判候选 not_duplicate（不同对象/观点相反）→ 新写，绝不误并。"""
    ms = _judge_semantic_store("用户喜欢喝咖啡", incoming="用户不喜欢喝咖啡", cosine=0.95)
    await ms.save(_make_item(id="base", content="用户喜欢喝咖啡"))
    await _save_with_judge(
        ms,
        [
            CandidateVerdict(
                index=1, relation=RELATION_NOT_DUPLICATE, reason="观点相反（否定词翻转）"
            )
        ],
        content="用户不喜欢喝咖啡",
    )
    result = await ms.recall(user_id="u1")
    assert len(result.items) == 2  # 喜欢 + 不喜欢 两条共存


async def test_save_judge_composite_two_overlaps_inserts() -> None:
    """组合场景：待存「牛奶和咖啡」同时命中两条 overlap → 新写，不腐蚀任一既有事实。"""
    ms = _composite_store()
    await ms.save(_make_item(id="milk", content="用户喜欢喝牛奶"))
    await ms.save(_make_item(id="coffee", content="用户喜欢喝咖啡"))
    await _save_with_judge(
        ms,
        [
            CandidateVerdict(index=1, relation=RELATION_OVERLAP, reason="组合之一"),
            CandidateVerdict(index=2, relation=RELATION_OVERLAP, reason="组合之二"),
        ],
        content="用户喜欢喝牛奶和咖啡",
    )
    result = await ms.recall(user_id="u1", top_k=50)
    # 三条共存：牛奶 + 咖啡 + 组合新写；既有两条内容原样保留（非破坏）。
    assert len(result.items) == 3
    assert {m.content for m in result.items} == {
        "用户喜欢喝牛奶",
        "用户喜欢喝咖啡",
        "用户喜欢喝牛奶和咖啡",
    }


async def test_save_judge_exact_preferred_over_overlap() -> None:
    """exact 与 overlap 并存 → 合并进最高分 exact（确定性重复优先，候选按余弦降序）。"""
    ms = _exact_overlap_store()
    await ms.save(_make_item(id="gender", content="用户是男生"))
    await ms.save(_make_item(id="height", content="用户身高175厘米"))
    await _save_with_judge(
        ms,
        [
            CandidateVerdict(index=1, relation=RELATION_EXACT, reason="同一事实"),
            CandidateVerdict(index=2, relation=RELATION_OVERLAP, reason="补充性别"),
        ],
        content="用户是男生，身高175厘米",
    )
    result = await ms.recall(user_id="u1", top_k=50)
    assert len(result.items) == 2  # 男生被合并、身高保留
    by_id = {m.id: m for m in result.items}
    assert by_id["gender"].content == "用户是男生，身高175厘米"
    assert "height" in by_id


async def test_save_judge_out_of_order_verdicts_merges_highest_score_exact() -> None:
    """模型乱序输出 verdict（先 index=2 后 index=1）→ 合并仍进最高分 exact（index=1, 0.96）。

    WHY 回归锁定：`exacts` 按 verdict 输出顺序构建，若不加分数降序重排，
    `exacts[0]` 会错误取到 身高(0.8) 而污染 `gender`；显式排序保证语义最相近者胜出。
    """
    ms = _exact_overlap_store()
    await ms.save(_make_item(id="gender", content="用户是男生"))
    await ms.save(_make_item(id="height", content="用户身高175厘米"))
    await _save_with_judge(
        ms,
        [
            CandidateVerdict(index=2, relation=RELATION_EXACT, reason="同事实"),
            CandidateVerdict(index=1, relation=RELATION_EXACT, reason="同事实"),
        ],
        content="用户是男生，身高175厘米",
    )
    result = await ms.recall(user_id="u1", top_k=50)
    assert len(result.items) == 2
    by_id = {m.id: m for m in result.items}
    assert by_id["gender"].content == "用户是男生，身高175厘米"  # 最高分 exact 吸收合并
    assert by_id["height"].content == "用户身高175厘米"  # 未被误并


async def test_save_judge_failure_falls_back_to_threshold() -> None:
    """judge 返回空（调用失败/空结果）→ 退回阈值路径：高分合并、低分新写（零回归）。"""
    # 高分：余弦 0.95 ≥ 阈值 → 合并（判定器缺席时语义层仍生效）。
    ms = _judge_semantic_store("用户喜欢喝美式咖啡", incoming="用户喜欢喝黑咖啡", cosine=0.95)
    await ms.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    await _save_with_judge(ms, [], content="用户喜欢喝黑咖啡")
    result = await ms.recall(user_id="u1")
    assert len(result.items) == 1
    assert result.items[0].id == "base"
    # 低分：余弦 0.85 < 阈值 → 新写。
    ms2 = _judge_semantic_store("用户喜欢喝美式咖啡", incoming="用户喜欢喝黑咖啡", cosine=0.85)
    await ms2.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    await _save_with_judge(ms2, [], content="用户喜欢喝黑咖啡")
    assert len((await ms2.recall(user_id="u1")).items) == 2


async def test_save_no_judge_uses_threshold_path() -> None:
    """未注入 judge → 走 upsert 阈值路径（与 V3-M3 行为一致，回归锁定）。"""
    ms = _judge_semantic_store("用户喜欢喝美式咖啡", incoming="用户喜欢喝黑咖啡", cosine=0.95)
    await ms.save(_make_item(id="base", content="用户喜欢喝美式咖啡"))
    await _save_with_judge(ms, [], content="用户喜欢喝黑咖啡", judge=False)
    result = await ms.recall(user_id="u1")
    assert len(result.items) == 1
    assert result.items[0].id == "base"
