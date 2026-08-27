"""长期记忆 langgraph Store 适配层。

业务代码只允许经 `MemoryStore` 访问记忆；
存储实现可换 `InMemoryStore`（dev）/ `PostgresStore`（prod），对业务侧无感。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from langgraph.store.base import BaseStore, SearchItem

from agent.memory.models import MemoryItem, MemoryRecallResult, SaveOutcome
from agent.share.models import Citation, MemoryRecallConfig

logger = logging.getLogger(__name__)

# 长期记忆命名空间后缀：(user_id, LONG_TERM_NAMESPACE) 按 user 隔离、跨会话共享。
LONG_TERM_NAMESPACE = "long_term"

# 记忆 kind 常量（P4-2 抽取器 / P4-3 召回按 kind 过滤复用，禁止散落字符串字面量）。
KIND_FACT = "fact"
KIND_EPISODE = "episode"
KIND_PREFERENCE = "preference"
# 召回兜底归类：抽取器输出的非规范 kind统一归入 other 组，
# 保证未知 kind 条目结构上仍可召回（kinds 含 KIND_OTHER 时命中），且不污染定向召回。
KIND_OTHER = "other"

# 抽取器规范 kind 集合
KNOWN_KINDS = frozenset({KIND_FACT, KIND_EPISODE, KIND_PREFERENCE})

# 语义候选召回上限：同 kind 取 top-N 相似记忆交给判定器/阈值决策。
# 仅作「候选召回」而非「合并决策」——有判定器时由判定器按事实三分类决定合并/新写，
# 无判定器时退回语义阈值（upsert）
DEDUP_SEMANTIC_FETCH_LIMIT = 5


def _kind_group(kind: str) -> str:
    """把任意 kind 归入可召回的组：规范三类保持原样，其余统一归入 KIND_OTHER。"""
    return kind if kind in KNOWN_KINDS else KIND_OTHER


def store_has_embeddings(store: BaseStore) -> bool:
    """store 实例是否配置了 embeddings（语义检索可用信号，D3 降级判定）。

    langgraph InMemoryStore / PostgresStore 配 index 后经 `_ensure_index_config`
    暴露 `embeddings` 属性，未配则为 None。RAG 用 Qdrant 无此概念，故该判定只存在于记忆侧。
    """
    return getattr(store, "embeddings", None) is not None


def _aware_utc(ts: datetime) -> datetime:
    """归一为 UTC-aware：naive 视为 UTC（手工构造/旧数据可能无时区），保证跨时区/naive 可比。

    WHY 独立 helper：recency 年龄计算与确定性排序 tie-break 都需要「可比时间」；
    Python 3.11+ 对 naive/aware 混合 datetime 直接比较会抛 TypeError。
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _clamp01(value: float) -> float:
    """clamp 到 [0,1]：cosine 相似度可为负（语义相反），混合信号统一归到非负区间。"""
    return max(0.0, min(1.0, value))


def _recency_signal(ts: datetime, *, half_life_days: float) -> float:
    """近因信号：1/(1 + age_days/half_life)，越近越高；未来时间防御 clamp 到 [0,1]。"""
    aware = _aware_utc(ts)
    age_seconds = max(0.0, (datetime.now(UTC) - aware).total_seconds())
    age_days = age_seconds / 86_400.0
    return 1.0 / (1.0 + age_days / half_life_days)


def _rerank_signals(
    item: MemoryItem, raw_score: float | None, *, half_life_days: float
) -> tuple[float, float, float]:
    """提取三路混合重排信号（score / importance / recency），顺序与权重元组一一对应。

    WHY 可插拔信号形式：未来加 lexical(BM25) 信号 = 本函数返回四元组 +
    `MemoryRecallConfig` 权重元组类型注解/默认扩为四元组，调用方零改动。
    score 缺失（asearch 兜底填充条目）按 0 计；负数 clamp。
    """
    score = _clamp01(raw_score) if raw_score is not None else 0.0
    return (
        score,
        item.importance,
        _recency_signal(item.timestamp, half_life_days=half_life_days),
    )


class MemoryStore:
    """langgraph Store 适配：InMemoryStore(dev) / PostgresStore(prod) 均可。

    命名空间 `(user_id, "long_term")`、key=memory_id、value=`MemoryItem` 的 JSON dump。
    `save` 无去重新写；`upsert` 带确定性两层去重：
    L1 content-hash 精确层（零误并）→ L2 语义近似层（cosine ≥ 阈值，需 embeddings）→
    未命中新写；L2 不可用时自动降级为仅 L1。
    `find_dedup_candidates` 暴露去重候选（L1 精确命中 / L2 top-K），供上层判定器
    （`MemoryRelationJudge`，见 persist._save_deduped）在阈值之外按事实三分类决策；
    `merge` 是两条候选的合并原语（保留原 id）。
    `recall` 双模式：语义（query 非空且 store 配 embeddings）→ 混合重排
    （score/importance/recency）；
    确定性（无 query 或 embeddings 不可用）→ importance 降序（零回归）。
    """

    def __init__(
        self,
        store: BaseStore,
        *,
        recall_config: MemoryRecallConfig | None = None,
    ) -> None:
        """构造适配层。`recall_config` 缺省用 spec 默认值，保证 `MemoryStore(store)` 零改动可用。"""
        self._store = store
        self._recall_config = recall_config or MemoryRecallConfig()

    async def save(self, item: MemoryItem) -> None:
        """保存一条长期记忆。失败向上抛，由带外调用方（P4-2）负责降级与日志。"""
        namespace = (item.user_id, LONG_TERM_NAMESPACE)
        # mode="json"：datetime → ISO 字符串，保证 InMemoryStore / PostgresStore 均可 JSON 序列化。
        await self._store.aput(namespace, item.id, item.model_dump(mode="json"))

    async def find_dedup_candidates(
        self, item: MemoryItem
    ) -> tuple[MemoryItem | None, list[tuple[MemoryItem, float | None]]]:
        """取去重候选：L1 content-hash 精确命中（零误并）或 L2 同 kind 语义 top-K。

        返回 `(exact, semantic)`：L1 命中时 exact 为唯一条目、semantic 为空
        （不经 LLM、不取 L2，省一次向量查询）；否则 exact=None，semantic 为
        `asearch(query, filter={"kind"}, limit=DEDUP_SEMANTIC_FETCH_LIMIT)` 的解析结果
        """
        namespace = (item.user_id, LONG_TERM_NAMESPACE)
        if item.content_hash:  # L1 精确层：内容指纹完全一致 → 同一事实
            pairs = self._parse_hits(
                await self._store.asearch(
                    namespace, filter={"content_hash": item.content_hash}, limit=1
                )
            )
            if pairs:
                return pairs[0][0], []
        if store_has_embeddings(self._store):  # L2 语义层（D3：未配 embeddings 自动跳过）
            hits = await self._store.asearch(
                namespace,
                query=item.content,
                filter={"kind": item.kind},
                limit=DEDUP_SEMANTIC_FETCH_LIMIT,
            )
            return None, self._parse_hits(hits)
        return None, []

    async def upsert(self, item: MemoryItem, *, semantic_threshold: float) -> SaveOutcome:
        """带去重的保存：L1 content-hash 精确 → L2 语义近似 → 未命中新写。

        L1：归一化后 hash 精确命中同一条事实（零误并、零向量成本）→ 合并。
        L2：embeddings 可用时同 kind 语义检索，最高 cosine ≥ `semantic_threshold`
        视为「同事实不同表述」→ 合并；否则视为近似但不同的条目 → 新写。
        本方法是**确定性阈值路径**（无判定器时的降级 / 直接调用默认）；
        判定器路径（`MemoryRelationJudge`，见 persist._save_deduped）复用
        `find_dedup_candidates` 取候选后按事实三分类决策。
        合并保留原 memory_id（历史 citation 的 source_id 仍可命中），字段按 D4 合并规则。
        `content_hash` 为空（旧数据/直接调用）时跳过 L1，语义可用仍走 L2。
        """
        exact, semantic = await self.find_dedup_candidates(item)
        if exact is not None:  # L1 精确层
            return await self.merge((item.user_id, LONG_TERM_NAMESPACE), exact, item)
        scored = [p for p in semantic if p[1] is not None]  # L2 语义层（D3：无 embeddings 为空）
        if scored and scored[0][1] >= semantic_threshold:  # asearch 已降序 → scored[0] 即最高分
            return await self.merge((item.user_id, LONG_TERM_NAMESPACE), scored[0][0], item)
        await self.save(item)  # 两层均未命中 → 新写一条
        return SaveOutcome(action="inserted", item=item)

    async def merge(
        self,
        namespace: tuple[str, ...],
        existing: MemoryItem,
        incoming: MemoryItem,
    ) -> SaveOutcome:
        """按 D4 合并规则把 incoming 并进 existing（保留原 id，落库并返回合并结果）。

        内容取更完整版本（更长者），故近似合并也不丢失信息；
        importance 取 max（事实重要性先验不退化）；timestamp 取新者刷新 recency；
        provenance/session_id 记录最近来源。
        """
        use_incoming = len(incoming.content) >= len(existing.content)
        merged = existing.model_copy(
            update={
                "content": incoming.content if use_incoming else existing.content,
                # 与落库 content 对齐：谁的内容留下，谁的指纹留下；incoming 缺省时保旧。
                "content_hash": incoming.content_hash or existing.content_hash,
                "importance": max(existing.importance, incoming.importance),
                # recency 刷新：比较用 `_aware_utc` 归一，防御 naive/aware 混合。
                "timestamp": max(existing.timestamp, incoming.timestamp, key=_aware_utc),
                "session_id": incoming.session_id,
                "provenance": incoming.provenance,
            }
        )
        await self._store.aput(namespace, existing.id, merged.model_dump(mode="json"))
        return SaveOutcome(action="merged", item=merged)

    async def recall(
        self,
        *,
        user_id: str,
        top_k: int = 5,
        kinds: Sequence[str] | None = None,
        query: str | None = None,
        hybrid_weights: tuple[float, float, float] | None = None,
    ) -> MemoryRecallResult:
        """双模式召回：语义（query 非空且 embeddings 可用）或确定性。

        WHY 双模式：语义列是增量能力，未配 embedding / 无 query 时必须保持 v2.0 行为（零回归）。
        语义模式：asearch(query, limit=top_k*fetch_factor) 预取放大 → kind 组过滤 →
        混合重排（score/importance/recency）→ top_k；确定性模式：limit=top_k*2 →
        kind 过滤 → importance 降序 + 时间倒序兜底。
        `hybrid_weights` 覆盖本次调用的混合权重：缺省用 `content_weights`；
        偏好预加载等调用方传 `preference_weights` 以区分召回职责（确定性模式忽略）。
        """
        namespace = (user_id, LONG_TERM_NAMESPACE)
        norm_query = (query or "").strip()
        # store.embeddings 为 None 当且仅当装配时 EmbeddingsFactory 不可用
        semantic = bool(norm_query) and store_has_embeddings(self._store)

        if semantic:
            limit = top_k * self._recall_config.recall_fetch_factor
            hits = await self._store.asearch(namespace, query=norm_query, limit=limit)
            weights = (
                hybrid_weights
                if hybrid_weights is not None
                else self._recall_config.content_weights
            )
            ranked = self._hybrid_rerank(
                self._filter_pairs_by_kinds(self._parse_hits(hits), kinds),
                weights=weights,
            )
        else:
            hits = await self._store.asearch(namespace, limit=top_k * 2)
            ranked = self._deterministic_rank(
                self._filter_pairs_by_kinds(self._parse_hits(hits), kinds)
            )

        items = [m for m, _score in ranked[:top_k]]
        sources = [
            Citation(source_id=m.id, source_title=m.kind, snippet=m.content, score=score)
            for m, score in ranked[:top_k]
        ]
        return MemoryRecallResult(items=items, sources=sources)

    def _parse_hits(self, hits: Sequence[SearchItem]) -> list[tuple[MemoryItem, float | None]]:
        """把 asearch 结果解析为 (item, raw_score) 对；脏数据跳过记 warning（双模式共用）"""
        pairs: list[tuple[MemoryItem, float | None]] = []
        for hit in hits:
            try:
                pairs.append((MemoryItem.model_validate(hit.value), hit.score))
            except Exception:
                # 脏数据（非法 value）跳过不中断召回，避免单条坏数据污染整轮。
                logger.warning("跳过损坏的记忆条目 key=%s", hit.key, exc_info=True)
        return pairs

    def _filter_pairs_by_kinds(
        self,
        pairs: Sequence[tuple[MemoryItem, float | None]],
        kinds: Sequence[str] | None,
    ) -> list[tuple[MemoryItem, float | None]]:
        """按 kind「组」过滤（_kind_group 归组；kinds 为空时全量返回）。双模式共用，
        语义模式在重排前过滤。

        未知 kind 统一归入 other 组：kinds 显式含 KIND_OTHER 时未知 kind 条目才被召回。
        """
        if not kinds:
            return list(pairs)
        allowed_groups = {_kind_group(k) for k in kinds}
        return [p for p in pairs if _kind_group(p[0].kind) in allowed_groups]

    def _deterministic_rank(
        self, pairs: Sequence[tuple[MemoryItem, float | None]]
    ) -> list[tuple[MemoryItem, float | None]]:
        """v2.0 确定性排序：importance 降序 + 时间倒序兜底（原实现迁移到 pair 结构，顺序不变）。

        key 用 `_aware_utc` 归一，仅防御 naive/aware 混合比较崩溃；
        全 aware 数据下顺序与 v2.0 完全一致。
        """
        pairs = list(pairs)
        pairs.sort(
            key=lambda p: (p[0].importance, _aware_utc(p[0].timestamp)),
            reverse=True,
        )
        return pairs

    def _hybrid_rerank(
        self,
        pairs: Sequence[tuple[MemoryItem, float | None]],
        *,
        weights: tuple[float, float, float],
    ) -> list[tuple[MemoryItem, float | None]]:
        """语义混合重排：hybrid = w_score*score + w_importance*importance + w_recency*recency，
        按 hybrid 降序。

        `weights` 由 `recall` 显式传入（content / preference 两套职责权重），
        本方法不感知 kind 语义；`recency_half_life_days` 取自配置。
        返回 `(item, raw_score)`——第二个元素是**原始语义分数**（供 Citation.score 回填），
        不是 hybrid_score；排序才是 hybrid 结果。
        tie-break：hybrid 相同 → importance 降序 → 时间倒序（与确定性模式一致的确定性次序）。
        """
        half_life = self._recall_config.recency_half_life_days
        scored: list[tuple[MemoryItem, float, float | None]] = []
        for item, raw in pairs:
            score_sig, importance_sig, recency_sig = _rerank_signals(
                item, raw, half_life_days=half_life
            )
            hybrid = weights[0] * score_sig + weights[1] * importance_sig + weights[2] * recency_sig
            scored.append((item, hybrid, raw))
        scored.sort(
            key=lambda t: (t[1], t[0].importance, _aware_utc(t[0].timestamp)),
            reverse=True,
        )
        return [(item, raw) for item, _hybrid, raw in scored]
