"""工具响应/适配模型与 MCP 接入工厂。

工具执行统一由 LangChain `ToolNode` 承担；本包提供响应模型（ToolResult/ToolCallRecord）
与 `build_tools_from_mcp` 装配工厂。
"""

from agent.tools.factory import build_tools_from_mcp
from agent.tools.models import (
    TOOL_ERROR_EXECUTION,
    TOOL_ERROR_UNKNOWN_TOOL,
    ToolCallRecord,
    ToolError,
    ToolResult,
)

__all__ = [
    "TOOL_ERROR_EXECUTION",
    "TOOL_ERROR_UNKNOWN_TOOL",
    "ToolCallRecord",
    "ToolError",
    "ToolResult",
    "build_tools_from_mcp",
]
