"""A2A JSON-RPC 2.0 编解码与错误码（纯逻辑，可单测）。

职责：请求解码（容错 → 结构化 A2AProtocolError）、响应/错误响应构造、
method 常量与支持集。method 分发本身在 app/a2a/routes.py（薄路由）完成。

错误语义（dev-version5.0.md §8.1）：JSON-RPC 2.0 标准码承载协议层错误，
`a2a.*` 内部码放 error.data.code 承载业务语义，两层并存、可路由。
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

# —— method 常量（A2A 核心子集）——
A2A_METHOD_MESSAGE_SEND = "message/send"
A2A_METHOD_MESSAGE_STREAM = "message/stream"
A2A_METHOD_TASK_GET = "task/get"
A2A_METHOD_TASK_CANCEL = "task/cancel"

SUPPORTED_METHODS = frozenset(
    {
        A2A_METHOD_MESSAGE_SEND,
        A2A_METHOD_MESSAGE_STREAM,
        A2A_METHOD_TASK_GET,
        A2A_METHOD_TASK_CANCEL,
    }
)

# —— JSON-RPC 2.0 标准错误码 ——
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

# —— server-defined 错误码区段（-32000 ~ -32099）——
JSONRPC_TASK_NOT_FOUND = -32001
JSONRPC_TASK_NOT_CANCELABLE = -32002

# —— a2a.* 内部码（放 error.data.code；与根 CLAUDE.md 错误码纪律同风格）——
A2A_ERROR_INVALID_REQUEST = "a2a.invalid_request"  # 参数/peer 身份非法
A2A_ERROR_METHOD_NOT_SUPPORTED = "a2a.method_not_supported"  # method 未实现
A2A_ERROR_TASK_NOT_FOUND = "a2a.task_not_found"  # task_id 未命中注册表
A2A_ERROR_TASK_NOT_CANCELABLE = "a2a.task_not_cancelable"  # 任务已终态
A2A_ERROR_INTERNAL = "a2a.internal_error"  # 未预期异常兜底


class A2AProtocolError(Exception):
    """A2A 处理错误：同时携带 JSON-RPC 标准码与 a2a.* 内部码。

    agent/a2a 纯逻辑层与 routes 层统一抛出本异常，路由层捕获后
    经 `error_response` 转 JSON-RPC error（app 层零业务映射逻辑）。
    """

    def __init__(self, *, jsonrpc_code: int, a2a_code: str, message: str) -> None:
        super().__init__(message)
        self.jsonrpc_code = jsonrpc_code
        self.a2a_code = a2a_code
        self.message = message


def invalid_request(message: str) -> A2AProtocolError:
    """构造参数/身份类错误（a2a.invalid_request）。"""
    return A2AProtocolError(
        jsonrpc_code=JSONRPC_INVALID_REQUEST, a2a_code=A2A_ERROR_INVALID_REQUEST, message=message
    )


def task_not_found(task_id: str) -> A2AProtocolError:
    """构造任务未命中错误（a2a.task_not_found）。"""
    return A2AProtocolError(
        jsonrpc_code=JSONRPC_TASK_NOT_FOUND,
        a2a_code=A2A_ERROR_TASK_NOT_FOUND,
        message=f"任务不存在: {task_id}",
    )


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error 对象（data.code 承载 a2a.* 内部码）。"""

    code: int
    message: str
    data: dict[str, object] | None = None


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求（A2A 全部走 POST /a2a 单端点）。"""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, object] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 响应（result 与 error 互斥，error 优先）。"""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, object] | None = None
    error: JsonRpcError | None = None


def decode_request(raw: bytes | str) -> JsonRpcRequest:
    """解码 JSON-RPC 请求；解析失败/结构非法统一抛 A2AProtocolError。

    解析错误（非法 JSON）无法回带 id → 响应 id 恒为 None（JSON-RPC 规范）。
    """
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise A2AProtocolError(
            jsonrpc_code=JSONRPC_PARSE_ERROR,
            a2a_code=A2A_ERROR_INVALID_REQUEST,
            message=f"请求体不是合法 JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise invalid_request("请求体必须是 JSON 对象")
    if payload.get("jsonrpc") != "2.0":
        raise invalid_request("jsonrpc 字段必须为 \"2.0\"")
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise invalid_request("缺少 method 字段")
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise invalid_request("params 必须是对象")
    request_id = payload.get("id")
    if request_id is not None and not isinstance(request_id, (str, int)):
        raise invalid_request("id 必须为字符串/整数/ null")
    return JsonRpcRequest(
        jsonrpc="2.0",
        id=request_id,
        method=method,
        params=params,
    )


def result_response(request_id: str | int | None, result: dict[str, object]) -> JsonRpcResponse:
    """构造成功响应信封。"""
    return JsonRpcResponse(id=request_id, result=result)


def error_response(request_id: str | int | None, exc: A2AProtocolError) -> JsonRpcResponse:
    """构造错误响应信封：标准码入 error.code，a2a.* 内部码入 error.data.code。"""
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(
            code=exc.jsonrpc_code,
            message=exc.message,
            data={"code": exc.a2a_code},
        ),
    )
