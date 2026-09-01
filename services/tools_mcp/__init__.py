"""tools_mcp FastMCP 服务包：统一注册为 MCP 工具 + 客户端连接配置导出。

所有工具经 MCP 服务接入 Agent；连接配置（stdio / streamable-http）与
RAG（services/rag_mcp/client_config.py）对称地收敛在本包统一导出。
"""

from services.tools_mcp.client_config import (
    TOOLS_MCP_SERVER_MODULE,
    TOOLS_MCP_SERVER_NAME,
    build_tools_mcp_http_connection,
    build_tools_mcp_stdio_connection,
)
from services.tools_mcp.server import mcp

__all__ = [
    "TOOLS_MCP_SERVER_MODULE",
    "TOOLS_MCP_SERVER_NAME",
    "build_tools_mcp_http_connection",
    "build_tools_mcp_stdio_connection",
    "mcp",
]
