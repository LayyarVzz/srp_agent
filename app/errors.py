"""app 层统一异常与错误信封。

错误响应统一为：`{ok: false, error: {code, message, path}}`（除 FastAPI 422 校验错误外）。
三层错误映射：

- `APIError`（app 层，如 `asr.*` / `auth.*`）→ 异常自带 HTTP 状态码；
- `SessionError`（归属/存在性）→ 404 `session_error.not_found` / 400 `session_error.invalid_id`；
- `AgentError` 与未预期异常 → 500 `internal_error`（日志留痕，不泄露内部细节）。

错误码集中声明。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.errors import SESSION_ERROR_NOT_FOUND, AgentError, SessionError

logger = logging.getLogger(__name__)

# —— 错误码常量 ——
AUTH_IDENTITY_REQUIRED = "auth.identity_required"  # 401：缺少/非法 X-User-Id
INTERNAL_ERROR = "internal_error"  # 500 兜底：未预期异常（不向客户端泄露细节）


class APIError(Exception):
    """app 层可返回给前端的统一异常（携带 HTTP 状态码 + 错误码）。

    用于 ASR 等业务边界错误（`asr.*`）；路由层 / 服务层抛出，异常处理器统一转信封。
    """

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def error_body(code: str, message: str, path: str) -> dict[str, object]:
    """构造统一错误信封。"""
    return {"ok": False, "error": {"code": code, "message": message, "path": path}}


def register_exception_handlers(app: FastAPI) -> None:
    """挂接异常处理器：app 层异常 / 会话错误 / Agent 错误 / 未预期异常 → 信封。

    顺序无关（FastAPI 按异常类型的 MRO 匹配，互不派生时各自精确命中）。
    """

    @app.exception_handler(APIError)
    async def on_api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, request.url.path),
        )

    @app.exception_handler(SessionError)
    async def on_session_error(request: Request, exc: SessionError) -> JSONResponse:
        # 归属/存在性错误 → 404；非法 id 等 → 400（api.md §6.1）。
        status = 404 if exc.code == SESSION_ERROR_NOT_FOUND else 400
        return JSONResponse(
            status_code=status,
            content=error_body(exc.code, exc.message, request.url.path),
        )

    @app.exception_handler(AgentError)
    async def on_agent_error(request: Request, exc: AgentError) -> JSONResponse:
        # Agent 模块异常兜底：500，日志留痕，不向客户端泄露内部细节。
        logger.error("Agent 错误 code=%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=500,
            content=error_body(INTERNAL_ERROR, "服务内部错误", request.url.path),
        )

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未预期异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content=error_body(INTERNAL_ERROR, "服务内部错误", request.url.path),
        )
