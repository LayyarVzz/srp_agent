"""会话路由：发号 / 列表 / 删除（仅元数据），消息本体由 checkpointer 承载。

归属强校验经 `SessionManager.resolve/delete`（跨用户统一 404 `session_error.not_found`，
防枚举）；`require_user_id` 同时供 chat 路由复用（MVP 身份约定，api.md §2.1）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Response

from agent.runtime import AgentRuntime
from app.deps import get_runtime
from app.errors import AUTH_IDENTITY_REQUIRED, APIError
from app.models import SessionCreated, SessionListResponse

router = APIRouter(tags=["sessions"])

RuntimeDep = Annotated[AgentRuntime, Depends(get_runtime)]
UserHeader = Annotated[str | None, Header()]


def require_user_id(x_user_id: str | None) -> str:
    """校验 X-User-Id（MVP 身份约定）；缺失/为空 → 401 `auth.identity_required`。

    仅查缺失/空：`.` 等非法字符由 `SessionManager._validate_label` 校验
    （400 `session_error.invalid_id`），避免两层重复实现。
    """
    if not x_user_id or not x_user_id.strip():
        raise APIError(AUTH_IDENTITY_REQUIRED, "缺少 X-User-Id 请求头", status_code=401)
    return x_user_id.strip()


@router.post("/sessions", status_code=201, response_model=SessionCreated)
async def create_session(
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
) -> SessionCreated:
    """创建会话：服务端 uuid4 发号（`thread_id == session_id` 契约，api.md §3.1）。"""
    user_id = require_user_id(x_user_id)
    ctx = await runtime.sessions.create(user_id=user_id)
    return SessionCreated.from_context(ctx)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> SessionListResponse:
    """会话列表（按 created_at 降序）。"""
    user_id = require_user_id(x_user_id)
    sessions = await runtime.sessions.list(user_id=user_id, limit=limit)
    return SessionListResponse(sessions=sessions)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
) -> Response:
    """删除会话（仅元数据；checkpointer 线程清理为扩展项）。"""
    user_id = require_user_id(x_user_id)
    await runtime.sessions.delete(user_id=user_id, session_id=session_id)
    return Response(status_code=204)
