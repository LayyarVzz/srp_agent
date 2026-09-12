"""A2A 协议适配模块（T5 Server 纯逻辑层，app/a2a 只做 HTTP 转发）。

核心子集：AgentCard 发现 / message/send / message/stream(SSE) / task/get /
task/cancel / text Part；能力矩阵显式裁剪见 models.py 模块注释。
"""

from agent.a2a.mapper import peer_user_id, resolve_peer, task_frame, task_from_response
from agent.a2a.models import (
    A2A_AGENT_DESCRIPTION,
    A2A_AGENT_NAME,
    A2A_AGENT_VERSION,
    A2AMessage,
    A2APart,
    A2APeer,
    A2ASkill,
    A2AStatusUpdate,
    A2ATask,
    A2ATaskState,
    AgentCard,
    build_agent_card,
    is_terminal,
)
from agent.a2a.protocol import (
    A2A_ERROR_INTERNAL,
    A2A_ERROR_INVALID_REQUEST,
    A2A_ERROR_METHOD_NOT_SUPPORTED,
    A2A_ERROR_TASK_NOT_CANCELABLE,
    A2A_ERROR_TASK_NOT_FOUND,
    A2A_METHOD_MESSAGE_SEND,
    A2A_METHOD_MESSAGE_STREAM,
    A2A_METHOD_TASK_CANCEL,
    A2A_METHOD_TASK_GET,
    SUPPORTED_METHODS,
    A2AProtocolError,
    JsonRpcRequest,
    JsonRpcResponse,
    decode_request,
    error_response,
    result_response,
)
from agent.a2a.registry import A2ATaskRegistry

__all__ = [
    "A2A_AGENT_DESCRIPTION",
    "A2A_AGENT_NAME",
    "A2A_AGENT_VERSION",
    "A2A_ERROR_INTERNAL",
    "A2A_ERROR_INVALID_REQUEST",
    "A2A_ERROR_METHOD_NOT_SUPPORTED",
    "A2A_ERROR_TASK_NOT_CANCELABLE",
    "A2A_ERROR_TASK_NOT_FOUND",
    "A2A_METHOD_MESSAGE_SEND",
    "A2A_METHOD_MESSAGE_STREAM",
    "A2A_METHOD_TASK_CANCEL",
    "A2A_METHOD_TASK_GET",
    "SUPPORTED_METHODS",
    "A2AMessage",
    "A2APart",
    "A2APeer",
    "A2AProtocolError",
    "A2ASkill",
    "A2AStatusUpdate",
    "A2ATask",
    "A2ATaskRegistry",
    "A2ATaskState",
    "AgentCard",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "build_agent_card",
    "decode_request",
    "error_response",
    "is_terminal",
    "peer_user_id",
    "resolve_peer",
    "result_response",
    "task_frame",
    "task_from_response",
]
