"""lark_mcp FastMCP 服务包：飞书工具面（T4），lark-cli 子进程封装为 MCP 工具。

所有工具经 MCP 服务接入 Agent（CLAUDE.md：MCP 是 Agent 唯一的外部工具边界）；
连接配置与门控探测原语（resolve_lark_cli_command）在本包统一导出，
agent 侧（runtime）据此做「配置启用 + 命令探测成功」门控登记（零回归降级）。
"""

from services.lark_mcp.cli import (
    LarkCliError,
    LarkCliRunner,
    resolve_lark_cli_command,
)
from services.lark_mcp.client_config import (
    LARK_MCP_SERVER_MODULE,
    LARK_MCP_SERVER_NAME,
    build_lark_mcp_stdio_connection,
)
from services.lark_mcp.server import mcp

__all__ = [
    "LARK_MCP_SERVER_MODULE",
    "LARK_MCP_SERVER_NAME",
    "LarkCliError",
    "LarkCliRunner",
    "build_lark_mcp_stdio_connection",
    "mcp",
    "resolve_lark_cli_command",
]
