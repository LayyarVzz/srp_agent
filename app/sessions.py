from __future__ import annotations

from datetime import UTC, datetime

from .errors import InteractionError
from .schemas import SessionContext

_sessions: dict[str, SessionContext] = {}


def require_user_id(user_id: str | None) -> str:
    if not user_id or not user_id.strip() or "." in user_id:
        raise InteractionError(
            code="auth.identity_required",
            message="缺少或非法的 X-User-Id 请求头",
            status_code=401,
        )
    return user_id.strip()


def create_session(user_id: str) -> SessionContext:
    session = SessionContext(
        user_id=user_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    _sessions[session.session_id] = session
    return session


def list_sessions(user_id: str, limit: int = 20) -> list[SessionContext]:
    sessions = [session for session in _sessions.values() if session.user_id == user_id]
    sessions.sort(key=lambda session: session.created_at, reverse=True)
    return sessions[:limit]


def get_or_create_session(session_id: str | None, user_id: str) -> SessionContext:
    if not session_id:
        return create_session(user_id)
    if "." in session_id:
        raise InteractionError(
            code="session_error.invalid_id",
            message="session_id 不合法",
            status_code=400,
        )
    session = _sessions.get(session_id)
    if session is None or session.user_id != user_id:
        raise InteractionError(
            code="session_error.not_found",
            message=f"会话不存在: {session_id}",
            status_code=404,
        )
    return session


def delete_session(session_id: str, user_id: str) -> None:
    session = get_or_create_session(session_id, user_id)
    _sessions.pop(session.session_id, None)
