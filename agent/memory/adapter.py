"""长期记忆 langgraph Store 适配层。

业务代码只允许经 `MemoryStore` 访问记忆；
存储实现可换 `InMemoryStore`（dev）/ `PostgresStore`（prod），对业务侧无感。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langgraph.store.base import BaseStore

from agent.memory.models import MemoryItem, MemoryRecallResult
from agent.share.models import Citation

logger = logging.getLogger(__name__)

# 长期记忆命名空间后缀：(user_id, LONG_TERM_NAMESPACE) 按 user 隔离、跨会话共享。
LONG_TERM_NAMESPACE = "long_term"

# 记忆 kind 常量（P4-2 抽取器 / P4-3 召回按 kind 过滤复用，禁止散落字符串字面量）。
KIND_FACT = "fact"
KIND_EPISODE = "episode"
KIND_PREFERENCE = "preference"


class MemoryStore:
    """langgraph Store 适配：InMemoryStore(dev) / PostgresStore(prod) 均可。

    命名空间 `(user_id, "long_term")`、key=memory_id、value=`MemoryItem` 的 JSON dump。
    `recall` 为 v2.0 确定性召回（§3.4）：kind 过滤 + importance 降序 + top_k（时间倒序兜底）；
    语义检索（`search(query=...)`）留 §9 开放项，配 embeddings 后接入。
    """

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    async def save(self, item: MemoryItem) -> None:
        """保存一条长期记忆。失败向上抛，由带外调用方（P4-2）负责降级与日志。"""
        namespace = (item.user_id, LONG_TERM_NAMESPACE)
        # mode="json"：datetime → ISO 字符串，保证 InMemoryStore / PostgresStore 均可 JSON 序列化。
        await self._store.aput(namespace, item.id, item.model_dump(mode="json"))

    async def recall(
        self,
        *,
        user_id: str,
        top_k: int = 5,
        kinds: Sequence[str] | None = None,
    ) -> MemoryRecallResult:
        """确定性召回：kind 过滤 + importance 降序 + top_k（时间倒序兜底），带来源引用。"""
        namespace = (user_id, LONG_TERM_NAMESPACE)
        hits = await self._store.asearch(namespace, limit=top_k * 2)
        items: list[MemoryItem] = []
        for hit in hits:
            try:
                items.append(MemoryItem.model_validate(hit.value))
            except Exception:
                # 脏数据（非法 value）跳过不中断召回，避免单条坏数据污染整轮。
                logger.warning("跳过损坏的记忆条目 key=%s", hit.key, exc_info=True)
        if kinds:
            items = [m for m in items if m.kind in kinds]
        items.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        items = items[:top_k]
        sources = [
            Citation(source_id=m.id, source_title=m.kind, snippet=m.content) for m in items
        ]
        return MemoryRecallResult(items=items, sources=sources)
