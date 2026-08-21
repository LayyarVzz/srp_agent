"""会话元数据 langgraph Store 适配层（会话管理核心）。

会话**消息**由 LangGraph checkpointer 按 `thread_id == session_id` 承载；
`SessionManager` 只在 langgraph `BaseStore` 上登记「会话身份」元数据
（session_id↔user_id 映射 + created_at + meta），与 checkpointer 是「元数据 vs 消息」
的互补关系，不重复存消息。

WHY 用 Store 而非进程内注册表：
- **单一事实源**：与长期记忆共用同一套 BaseStore（dev=InMemoryStore / prod=PostgresStore），
  prod 换 store 实例即跨进程共享、TTL 生效，业务零改动；进程内 dict 与 checkpointer 是
  双份事实易漂移。
- **owner 校验**：命名空间 `(user_id, SESSIONS_NAMESPACE)` 按 user 隔离，`resolve` 天然校验
  归属——checkpointer 的 thread_id 默认不隔离 user，谁拿到 id 谁能续跑该线程，归属校验
  必须在入口层经本类完成。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from langgraph.store.base import BaseStore, Item

from agent.core.config import SessionBehaviorConfig
from agent.errors import (
    SESSION_ERROR_INVALID_ID,
    SESSION_ERROR_INVALID_STATE,
    SESSION_ERROR_NOT_FOUND,
    SessionError,
)
from agent.session.models import SessionContext

logger = logging.getLogger(__name__)

# 会话元数据命名空间后缀：(user_id, SESSIONS_NAMESPACE) 按 user 隔离、跨会话共享。
SESSIONS_NAMESPACE = "sessions"


class SessionManager:
    """langgraph BaseStore 之上的会话元数据薄适配（等同 MemoryStore 之于记忆）。

    只记会话身份与作用域；消息本体由 checkpointer 承载（thread_id == session_id）。
    """

    def __init__(
        self,
        store: BaseStore,
        *,
        config: SessionBehaviorConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or SessionBehaviorConfig()
        self._ttl_warned = False  # TTL 后端不支持时仅告警一次，避免每轮刷屏

    async def create(self, *, user_id: str) -> SessionContext:
        """创建一条会话元数据记录（发号 + 绑定归属）。

        session_id 由本方法生成（uuid4 hex），调用方无需也不应自行指定；
        `thread_id == session_id` 契约在入口层把 session_id 传入 graph config。
        """
        self._validate_label(user_id, what="user_id")
        ctx = SessionContext(
            session_id=uuid4().hex,
            user_id=user_id,
            created_at=datetime.now(UTC),
        )
        await self._store.aput(
            self._namespace(user_id),
            ctx.session_id,
            ctx.model_dump(mode="json"),
            ttl=self._effective_ttl(),
        )
        return ctx

    async def resolve(self, *, user_id: str, session_id: str) -> SessionContext:
        """解析会话并强校验归属：未命中 / 不属于该 user 一律报 not_found。

        WHY 不区分「不存在」与「存在但不属于你」：命名空间已按 user 隔离，
        返回 not_found 不泄露「该 session 是否存在」（防跨用户枚举）。
        """
        self._validate_label(user_id, what="user_id")
        self._validate_label(session_id, what="session_id")
        item = await self._store.aget(self._namespace(user_id), session_id)
        if item is None:
            raise SessionError(SESSION_ERROR_NOT_FOUND, f"会话不存在: {session_id}")
        return self._context_from_item(item)

    async def list(self, *, user_id: str, limit: int = 20) -> list[SessionContext]:
        """列出该用户的会话元数据，按 created_at 降序（asearch 顺序不保证）。

        脏记录（非法 value）跳过并告警，不中断整列——与记忆召回端一致。
        """
        self._validate_label(user_id, what="user_id")
        hits = await self._store.asearch(self._namespace(user_id), limit=limit)
        contexts: list[SessionContext] = []
        for hit in hits:
            try:
                contexts.append(SessionContext.model_validate(hit.value))
            except Exception:
                logger.warning("跳过损坏的会话元数据 key=%s", hit.key, exc_info=True)
        contexts.sort(key=lambda c: c.created_at, reverse=True)
        return contexts

    async def delete(self, *, user_id: str, session_id: str) -> None:
        """删除一条会话元数据（先 resolve 确认归属，越权/未命中 → not_found）。

        仅删元数据：checkpointer 线程随会话清理依赖持久化 checkpointer，列为开放项。
        """
        self._validate_label(user_id, what="user_id")
        self._validate_label(session_id, what="session_id")
        await self.resolve(user_id=user_id, session_id=session_id)
        await self._store.adelete(self._namespace(user_id), session_id)

    # —— 私有辅助 ——

    def _namespace(self, user_id: str) -> tuple[str, ...]:
        return (user_id, SESSIONS_NAMESPACE)

    def _validate_label(self, value: str, *, what: str) -> None:
        """命名空间 label 校验（对齐 BaseStore 规则）：非空且不含 "."。"""
        if not value or "." in value:
            raise SessionError(
                SESSION_ERROR_INVALID_ID, f"{what} 非法（非空且不能含点号）: {value!r}"
            )

    def _effective_ttl(self) -> float | None:
        """按 store 能力返回 TTL 秒数；不支持 TTL 的 store（InMemoryStore）返回 None。

        WHY 传 None：`aput(ttl=None)` 永不触发 NotImplementedError；配置了 TTL 但
        后端不支持时仅告警一次，不中断会话创建。
        """
        if self._config.ttl_minutes is None:
            return None
        if not self._store.supports_ttl:
            if not self._ttl_warned:
                logger.warning(
                    "会话元数据 TTL=%d 分钟已配置，但当前 store 不支持 TTL（%s），已忽略",
                    self._config.ttl_minutes,
                    type(self._store).__name__,
                )
                self._ttl_warned = True
            return None
        return float(self._config.ttl_minutes) * 60.0

    def _context_from_item(self, item: Item) -> SessionContext:
        try:
            return SessionContext.model_validate(item.value)
        except Exception as exc:
            raise SessionError(
                SESSION_ERROR_INVALID_STATE, f"会话元数据记录损坏 key={item.key}"
            ) from exc
