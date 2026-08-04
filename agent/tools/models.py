"""工具契约数据模型与错误码（P1 类型级预留，无执行逻辑）。

WHY P1 先立契约：`build_agent_graph` 的依赖注入需要带类型的 `ToolRegistry`，
真实本地/MCP 工具在 P2 补齐；本模块只描述「工具长什么样、怎么调」。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.errors import AgentError
from agent.share.models import Citation

# —— 工具错误码（tool_error.* 命名空间，与 P6 错误分类兼容）——
TOOL_ERROR_NO_TOOL = "tool_error.no_tool"
TOOL_ERROR_NOT_IMPLEMENTED = "tool_error.not_implemented"
TOOL_ERROR_UNKNOWN_TOOL = "tool_error.unknown_tool"


class ToolSpec(BaseModel):
    """工具声明：输入 JSON Schema + 可 JSON 序列化的输出契约（工具契约）。"""

    name: str
    description: str
    input_schema: dict[str, object]
    source: Literal["local", "mcp"]


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
    """一次工具调用轨迹记录（供 tool_trace 回溯）。"""

    tool_name: str
    arguments: dict[str, object]
    status: str  # ok / error / no_tool_matched ...
    result: ToolResult | None = None


class UnknownToolError(AgentError):
    """注册表未找到指定工具时抛出的进程内异常（不入 checkpointed state）。"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(TOOL_ERROR_UNKNOWN_TOOL, f"未知工具: {tool_name}")
