"""tools_mcp 服务入口。

默认 stdio（本地开发、demo stdio 子进程、进程内测试不变）；配置
MCP_TRANSPORT=streamable-http（及 MCP_HOST/MCP_PORT/MCP_STREAMABLE_HTTP_PATH）
"""

from __future__ import annotations

from services.tools_mcp.config import build_run_params
from services.tools_mcp.server import mcp
from settings import configure_logging, get_settings

if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings)
    mcp.run(**build_run_params(settings))
