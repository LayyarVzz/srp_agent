"""错误分类与护栏。

`AgentError` 携带可路由的 `code` 字段；已落地 LLM 调用与会话元数据（SessionError）两类，
其余分类（ToolError / GuardrailError / RetryPolicy）在后续阶段补全。
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorRecord(BaseModel):
    """存放入 LangGraph state 的结构化错误信息（可被 checkpointer 序列化）。

    WHY 与 `AgentError` 区分：`AgentError` 是 Exception 实例，无法可靠地经
    msgpack checkpointer 序列化；凡要落进图状态的错误一律用本模型（code 可路由）。
    """

    code: str
    message: str


class AgentError(Exception):
    """Agent 模块统一的错误基类。

    `code` 为可路由的错误码（如 ``llm_error.request``），供上层据此选择降级路径；
    `message` 为可读描述，必须避免包含密钥等敏感信息。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LLMError(AgentError):
    """LLM 调用相关错误（鉴权 / 构造 / 请求 / 输出校验）。"""


class SessionError(AgentError):
    """会话元数据访问错误（归属校验 / 非法 id / 记录损坏）。"""


# —— 错误码常量（集中声明，禁止散落的字符串字面量）——
LLM_ERROR_AUTH = "llm_error.auth"
LLM_ERROR_CONSTRUCTION = "llm_error.construction"
LLM_ERROR_REQUEST = "llm_error.request"
LLM_ERROR_INVALID_OUTPUT = "llm_error.invalid_output"
SESSION_ERROR_NOT_FOUND = "session_error.not_found"  # resolve 未命中（含跨用户访问）
SESSION_ERROR_INVALID_ID = "session_error.invalid_id"
SESSION_ERROR_INVALID_STATE = "session_error.invalid_state"  # 元数据记录损坏

__all__ = [
    "LLM_ERROR_AUTH",
    "LLM_ERROR_CONSTRUCTION",
    "LLM_ERROR_INVALID_OUTPUT",
    "LLM_ERROR_REQUEST",
    "SESSION_ERROR_INVALID_ID",
    "SESSION_ERROR_INVALID_STATE",
    "SESSION_ERROR_NOT_FOUND",
    "AgentError",
    "ErrorRecord",
    "LLMError",
    "SessionError",
]
