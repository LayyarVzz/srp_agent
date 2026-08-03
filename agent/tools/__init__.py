"""工具注册表与执行：统一分发本地工具与 MCP 工具（P1 类型级预留）。"""

from agent.tools.models import (
    TOOL_ERROR_NO_TOOL,
    TOOL_ERROR_NOT_IMPLEMENTED,
    TOOL_ERROR_UNKNOWN_TOOL,
    ToolCallRecord,
    ToolError,
    ToolResult,
    ToolSpec,
    UnknownToolError,
)
from agent.tools.registry import InMemoryToolRegistry, Tool, ToolRegistry

__all__ = [
    "TOOL_ERROR_NOT_IMPLEMENTED",
    "TOOL_ERROR_NO_TOOL",
    "TOOL_ERROR_UNKNOWN_TOOL",
    "InMemoryToolRegistry",
    "Tool",
    "ToolCallRecord",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "UnknownToolError",
]
