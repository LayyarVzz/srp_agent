"""a2a_mcp FastMCP 服务包：远端 A2A 智能体封装为 MCP 工具 + 客户端连接配置导出。

「A2A Client = MCP 登记的智能体协作工具」（dev-version5.0.md D5.3）：
Agent 经 MultiServerMCPClient 加载 `a2a_call_agent` 为普通 BaseTool，
跨实例协作与本地工具在图内同构（零改图）。
"""

from services.a2a_mcp.client import A2AAgentResult, A2AClient, A2AClientError
from services.a2a_mcp.client_config import (
    A2A_MCP_SERVER_MODULE,
    A2A_MCP_SERVER_NAME,
    build_a2a_mcp_http_connection,
    build_a2a_mcp_stdio_connection,
)
from services.a2a_mcp.config import A2AMCPRuntimeSettings, MCPTransport
from services.a2a_mcp.server import mcp

__all__ = [
    "A2A_MCP_SERVER_MODULE",
    "A2A_MCP_SERVER_NAME",
    "A2AAgentResult",
    "A2AClient",
    "A2AClientError",
    "A2AMCPRuntimeSettings",
    "MCPTransport",
    "build_a2a_mcp_http_connection",
    "build_a2a_mcp_stdio_connection",
    "mcp",
]
