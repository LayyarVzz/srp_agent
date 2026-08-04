"""错误分类与护栏。

先落地最小底座：`AgentError` 携带可路由的 `code` 字段，
完整的错误分类（ToolError / GuardrailError / SessionError / RetryPolicy）在 P6 阶段补全。
"""

from __future__ import annotations


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


# —— 错误码常量（集中声明，禁止散落的字符串字面量）——
LLM_ERROR_AUTH = "llm_error.auth"
LLM_ERROR_CONSTRUCTION = "llm_error.construction"
LLM_ERROR_REQUEST = "llm_error.request"
LLM_ERROR_INVALID_OUTPUT = "llm_error.invalid_output"

__all__ = [
    "LLM_ERROR_AUTH",
    "LLM_ERROR_CONSTRUCTION",
    "LLM_ERROR_INVALID_OUTPUT",
    "LLM_ERROR_REQUEST",
    "AgentError",
    "LLMError",
]
