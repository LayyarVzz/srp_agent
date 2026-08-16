"""tools_mcp FastMCP 服务包：统一注册为 MCP 工具。

所有工具经 MCP 服务接入 Agent
"""

from services.tools_mcp.server import mcp

__all__ = ["mcp"]
