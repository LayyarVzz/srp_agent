"""请求级关联上下文（X-Request-Id 注入 / 透传 + 一轮交互的会话绑定）。

WHY 独立模块（而非塞进 `app/main.py`）：两件事同属「请求作用域」——
① ASGI 中间件：入口层一次绑定，使**全链路**（路由 / agent / MCP / 第三方库）的日志
都带同一个 `trace_id`；② `turn_context`：把 `session_id` / `user_id` 绑到一轮交互上，
使结构化事件能按会话/用户聚合。二者被「路由层」与「中间件」共用，放 app 层可避免
`shared/logging.py` 依赖 ASGI（底座必须能被纯 stdlib 环境 import）。

安全口径：`X-Request-Id` / `X-User-Id` 都是**外部输入**，只做净化后用于日志关联与回写，
绝不参与鉴权判定（鉴权仍由 `require_user_id` + `SessionManager.resolve` 承担）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final
from uuid import uuid4

from shared.logging import (
    EVENT_LOGGER_NAME,
    bind_context,
    current_trace_id,
    current_user_id,
    log_event,
    log_request_finished,
    unbind_context,
)

logger = logging.getLogger(EVENT_LOGGER_NAME)

# 关联 id 的请求头（响应回写同名头：调用方可直接拿 trace_id 去 grep 全链路日志）。
REQUEST_ID_HEADER: Final = "x-request-id"
REQUEST_ID_HEADER_OUT: Final = "X-Request-Id"
USER_ID_HEADER: Final = "x-user-id"

# trace_id 前缀：一眼区分「调用方传入」与「服务端生成」（排查时不必翻文档）。
TRACE_ID_PREFIX: Final = "req_"

# 事件字段名常量（禁止散落字面量；与 §C3 结构化事件字段同名同义）。
FIELD_METHOD: Final = "method"
FIELD_PATH: Final = "path"
FIELD_SOURCE: Final = "source"
FIELD_STATUS_CODE: Final = "status_code"
FIELD_SESSION_CREATED: Final = "session_created"

# 交互终态：与 `finished_reason` / SSE error 帧同源（业务失败与传输失败必须可区分）。
STATUS_COMPLETED: Final = "completed"
STATUS_ERROR: Final = "error"
# SSE error 帧对应的错误码（流内异常：图运行中断 / 会话错误 / ASR 边界错误）。
CODE_STREAM_ERROR: Final = "stream_error"

# HTTP 状态码阈值：< 400 视为成功（与 `request.finished` 的 status 口径一致）。
_HTTP_ERROR_THRESHOLD: Final = 400
# 未收到响应起始帧（客户端中途断开）时的兜底状态码：不得把「无响应」记成成功。
_STATUS_UNKNOWN: Final = 500


def new_trace_id() -> str:
    """生成服务端 trace_id（调用方未透传时使用）。"""
    return f"{TRACE_ID_PREFIX}{uuid4().hex[:16]}"


@contextmanager
def turn_context(*, session_id: str | None, user_id: str | None) -> Iterator[str]:
    """把一轮交互的会话/用户绑到日志上下文，`yield` 当前 trace_id。

    WHY contextmanager 而非裸 bind/unbind：调用点若中途抛异常（图运行失败、
    客户端断开），只有 `finally` 里的复位能保证**不串号** —— 上一个用户的
    `user_id` 顺着 ContextVar 继承进下一个请求的日志，比丢字段严重得多。

    trace_id 由中间件绑定；此处只补会话/用户维度，并原样返回 trace_id 供调用方
    记 `request.finished`（不必再读一次 ContextVar，语义更直白）。
    """
    trace_id = current_trace_id() or new_trace_id()
    tokens = bind_context(trace_id=trace_id, session_id=session_id, user_id=user_id)
    try:
        yield trace_id
    finally:
        unbind_context(tokens)


def finish_turn(
    *,
    status: str,
    duration_ms: int,
    code: str | None = None,
    **fields: Any,
) -> None:
    """记一轮交互的收尾事件（`request.finished`）—— 交互层与 HTTP 层共用同一口径。

    WHY 共用函数而非各自拼字段：`trace_id`/`duration_ms`/`status` 三个键必须同名同义，
    否则「同一轮交互在事件表里两条记录、字段对不上」，聚合统计直接失效。
    """
    log_request_finished(
        logger,
        status=status,
        duration_ms=duration_ms,
        code=code,
        fields=fields,
    )


def elapsed_ms(started: float) -> int:
    """端到端耗时（毫秒，整型）：从 `time.perf_counter()` 基线起算。"""
    return int((time.perf_counter() - started) * 1000)


class RequestContextMiddleware:
    """纯 ASGI 中间件：注入/透传 `X-Request-Id`，绑定日志上下文，回写响应头。

    WHY 纯 ASGI 而非 `BaseHTTPMiddleware`：后者把响应体包成流并额外起任务，
    对 SSE 实时性与「客户端断开时 finally 的执行时机」都有影响；本项目 SSE 是主链路，
    不引入多余的缓冲层。

    行为：
    - 请求头 `X-Request-Id` 有值 → 清洗后沿用（跨服务/前端串联）；否则服务端生成；
    - 绑定 `trace_id` + `X-User-Id`（会话维度由路由层解析出 session 后经 `turn_context` 补绑）；
    - 记 `request.received`（method/path/source，**不含请求体**）；
    - 响应回写 `X-Request-Id`；
    - 请求结束（含异常）后记 `request.finished`（状态码 + 端到端耗时）并复位上下文。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            # lifespan / websocket 直通：本中间件只管 HTTP 请求作用域。
            await self.app(scope, receive, send)
            return

        headers = _header_map(scope)
        trace_id = headers.get(REQUEST_ID_HEADER) or new_trace_id()
        tokens = bind_context(
            trace_id=trace_id,
            # 会话头是可选扩展（存在则先绑上，让 received 事件也带会话维度）；
            # 权威会话 id 仍由路由层解析后经 turn_context 覆盖。
            session_id=headers.get("x-session-id"),
            user_id=headers.get(USER_ID_HEADER) or current_user_id(),
        )
        started = time.perf_counter()
        status_code = _STATUS_UNKNOWN
        try:
            log_event(
                logger,
                "request.received",
                status="received",
                fields={
                    FIELD_METHOD: scope.get("method"),
                    FIELD_PATH: scope.get("path"),
                    FIELD_SOURCE: "http",
                },
            )
            status_code = await self._run(scope, receive, send, trace_id=trace_id)
        finally:
            elapsed = elapsed_ms(started)
            ok = status_code < _HTTP_ERROR_THRESHOLD
            finish_turn(
                status=STATUS_COMPLETED if ok else STATUS_ERROR,
                duration_ms=elapsed,
                code=None if ok else f"status_{status_code}",
                **{
                    FIELD_STATUS_CODE: status_code,
                    FIELD_METHOD: scope.get("method"),
                    FIELD_PATH: scope.get("path"),
                },
            )
            unbind_context(tokens)

    async def _run(self, scope: dict[str, Any], receive: Any, send: Any, *, trace_id: str) -> int:
        """驱动下游 ASGI 调用，截下响应起始帧以回写 `X-Request-Id` 并记状态码。"""
        status_code = _STATUS_UNKNOWN

        async def _send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", _STATUS_UNKNOWN))
                raw = list(message.get("headers") or [])
                if not any(k.decode("latin-1").lower() == REQUEST_ID_HEADER for k, _ in raw):
                    raw.append(
                        (REQUEST_ID_HEADER_OUT.encode("latin-1"), trace_id.encode("latin-1"))
                    )
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, _send_wrapper)
        return status_code


def _header_map(scope: dict[str, Any]) -> dict[str, str]:
    """把 ASGI 的 `[(bytes, bytes)]` 头列表转成小写键 dict（重复键取首个）。"""
    headers: dict[str, str] = {}
    for key, value in scope.get("headers") or []:
        name = key.decode("latin-1").lower()
        if name not in headers:
            headers[name] = value.decode("latin-1")
    return headers


__all__ = [
    "CODE_STREAM_ERROR",
    "FIELD_METHOD",
    "FIELD_PATH",
    "FIELD_SESSION_CREATED",
    "FIELD_SOURCE",
    "FIELD_STATUS_CODE",
    "REQUEST_ID_HEADER",
    "REQUEST_ID_HEADER_OUT",
    "STATUS_COMPLETED",
    "STATUS_ERROR",
    "TRACE_ID_PREFIX",
    "USER_ID_HEADER",
    "RequestContextMiddleware",
    "elapsed_ms",
    "finish_turn",
    "new_trace_id",
    "turn_context",
]
