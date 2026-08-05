"""工具响应/适配模型与错误码。

WHY 本模块只承载「响应与轨迹」契约：
工具执行本身由 LangChain `ToolNode` 承担，适配层把 `AIMessage.tool_calls` +
`ToolMessage` 翻译成这些模型；不在此定义工具声明/注册表（BaseTool 承担声明）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.share.models import Citation

# —— 工具错误码（tool_error.* 命名空间）——
TOOL_ERROR_EXECUTION = "tool_error.execution"  # 工具执行失败（含参数校验/运行时异常）
TOOL_ERROR_UNKNOWN_TOOL = "tool_error.unknown_tool"  # 模型幻觉出未注册工具名


class ToolError(BaseModel):
    """结构化工具错误（成功/失败均须返回结构化错误码，禁止让请求直接失败）。"""

    code: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    """工具执行结果统一封装：成功/失败均带结构化错误码。"""

    tool_name: str
    ok: bool
    data: dict[str, object] | None = None
    error: ToolError | None = None
    citations: list[Citation] = Field(default_factory=list)
    duration_ms: int = 0


class ToolCallRecord(BaseModel):
    """一次工具调用轨迹记录（供 tool_trace 回溯）。

    `status` 仅取值 `"ok"` / `"error"`（由适配层填写）；工具选择阶段的
    「未选中/无需工具」不再产生独立记录（模型直接回答时 tool_trace 为空）。
    """

    tool_name: str
    arguments: dict[str, object]
    status: str  # ok / error
    result: ToolResult | None = None
