"""记忆访问统一抽象。

记忆只允许通过 `MemoryStore` 接口访问（见 CLAUDE.md 记忆约定）；
生产默认实现为 Memory MCP 服务，本地开发用 InMemory 实现兜底（P4 补齐）。
"""

from __future__ import annotations

from typing import Protocol

from agent.memory.models import MemoryItem, MemoryRecallResult


class MemoryStore(Protocol):
    """长期记忆存储契约。"""

    async def save(self, item: MemoryItem) -> None: ...

    async def recall(
        self,
        query: str,
        *,
        session_id: str,
        user_id: str,
        top_k: int = 5,
    ) -> MemoryRecallResult: ...
