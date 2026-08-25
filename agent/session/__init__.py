"""会话与作用域管理。

`SessionManager` 在独立 `sessions` 表（SQLAlchemy，见 repository.py）上维护会话
元数据（session_id↔user_id 映射 + created_at + expires_at），消息本体由 checkpointer
按 `thread_id == session_id` 承载（见 agent/core/graph.py 的 load_context）。
"""

from agent.session.factory import SessionBackend, build_session_backend
from agent.session.manager import SessionManager
from agent.session.models import SessionContext
from agent.session.repository import (
    SessionRepository,
    SQLAlchemySessionRepository,
    build_session_repository,
)

__all__ = [
    "SQLAlchemySessionRepository",
    "SessionBackend",
    "SessionContext",
    "SessionManager",
    "SessionRepository",
    "build_session_backend",
    "build_session_repository",
]
