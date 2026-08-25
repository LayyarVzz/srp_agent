"""会话元数据管理核心（仓库之上的业务层）。

会话**消息**由 LangGraph checkpointer 按 `thread_id == session_id` 承载；
`SessionManager` 只维护「会话身份」元数据（session_id↔user_id 映射 + created_at），
具体持久化委托 `SessionRepository`（SQLAlchemy `sessions` 表，见 repository.py），
与 checkpointer 是「元数据 vs 消息」的互补关系，不重复存消息。

WHY 分层 manager/repository：
- **业务与存储解耦**：归属强校验、label 护栏、错误码映射放在 Manager；SQL/方言、
  过期过滤、建表/连接池放在 Repository——测试可注入假仓库，存储实现可独立演进。
- **owner 校验**：`resolve`/`delete` 经仓库的 user_id 过滤天然校验归属——checkpointer
  的 thread_id 默认不隔离 user，谁拿到 id 谁能续跑该线程，归属校验必须在入口层经
  本类完成（跨 user 统一 not_found，防枚举）。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from agent.core.config import SessionBehaviorConfig
from agent.errors import (
    SESSION_ERROR_INVALID_ID,
    SESSION_ERROR_NOT_FOUND,
    SessionError,
)
from agent.session.models import SessionContext
from agent.session.repository import SessionRepository

logger = logging.getLogger(__name__)


class SessionManager:
    """会话元数据业务层（等同 MemoryStore 之于记忆，只记身份不记消息）。

    与旧版差异：不再直接面对 langgraph BaseStore——TTL 生命周期与过期语义
    由 Repository 的 `expires_at` 承载，本类只负责校验与错误码映射。
    """

    def __init__(
        self,
        repository: SessionRepository,
        *,
        config: SessionBehaviorConfig | None = None,
    ) -> None:
        self._repository = repository
        self._config = config or SessionBehaviorConfig()

    async def create(self, *, user_id: str) -> SessionContext:
        """创建一条会话元数据记录（发号 + 绑定归属）。

        session_id 由本方法生成（uuid4 hex），调用方无需也不应自行指定；
        `thread_id == session_id` 契约在入口层把 session_id 传入 graph config。
        TTL 由 config.ttl_minutes 折算为 expires_at 交给仓库落库。
        """
        self._validate_label(user_id, what="user_id")
        ctx = SessionContext(
            session_id=uuid4().hex,
            user_id=user_id,
            created_at=datetime.now(UTC),
        )
        await self._repository.create(ctx, ttl_minutes=self._config.ttl_minutes)
        return ctx

    async def resolve(self, *, user_id: str, session_id: str) -> SessionContext:
        """解析会话并强校验归属：未命中 / 不属于该 user 一律报 not_found。

        WHY 不区分「不存在」与「存在但不属于你」：仓库已按 user 过滤，返回
        not_found 不泄露「该 session 是否存在」（防跨用户枚举）。
        """
        self._validate_label(user_id, what="user_id")
        self._validate_label(session_id, what="session_id")
        ctx = await self._repository.get(user_id=user_id, session_id=session_id)
        if ctx is None:
            raise SessionError(SESSION_ERROR_NOT_FOUND, f"会话不存在: {session_id}")
        return ctx

    async def list(self, *, user_id: str, limit: int = 20) -> list[SessionContext]:
        """列出该用户的会话元数据，按 created_at 降序（仓库保证顺序与过期过滤）。"""
        self._validate_label(user_id, what="user_id")
        return await self._repository.list(user_id=user_id, limit=limit)

    async def delete(self, *, user_id: str, session_id: str) -> None:
        """删除一条会话元数据（归属校验：未命中 / 越权 / 已过期 → not_found）。

        仅删元数据：checkpointer 线程随会话清理依赖持久化 checkpointer，列为开放项。
        """
        self._validate_label(user_id, what="user_id")
        self._validate_label(session_id, what="session_id")
        deleted = await self._repository.delete(user_id=user_id, session_id=session_id)
        if not deleted:
            raise SessionError(SESSION_ERROR_NOT_FOUND, f"会话不存在: {session_id}")

    # —— 私有辅助 ——

    def _validate_label(self, value: str, *, what: str) -> None:
        """会话标识校验：非空且不含 "."（对齐旧 BaseStore namespace 规则，沿用护栏）。"""
        if not value or "." in value:
            raise SessionError(
                SESSION_ERROR_INVALID_ID, f"{what} 非法（非空且不能含点号）: {value!r}"
            )
