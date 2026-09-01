from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from .logging_service import record_event


class InteractionError(Exception):
    """交互层可返回给前端的统一异常。"""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


async def interaction_error_handler(request: Request, exc: InteractionError) -> JSONResponse:
    record_event(
        "interaction.error",
        payload={
            "code": exc.code,
            "message": exc.message,
            "path": request.url.path,
        },
        level="ERROR",
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "path": request.url.path,
            },
        },
    )
