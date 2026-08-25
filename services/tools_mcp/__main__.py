"""tools_mcp 服务入口。

默认 stdio（本地开发、demo stdio 子进程、进程内测试不变）；配置
MCP_TRANSPORT=streamable-http（及 MCP_HOST/MCP_PORT/MCP_STREAMABLE_HTTP_PATH）
即以 HTTP 运行。配置经本地 `MCPRuntimeSettings` 读取，**不依赖根 settings.py /
agent 框架**，可独立打包部署（见 services/tools_mcp/README.md）。
"""

from __future__ import annotations

from services.tools_mcp.config import (
    MCPRuntimeSettings,
    build_run_params,
    configure_logging,
)
from services.tools_mcp.server import mcp

if __name__ == "__main__":
    settings = MCPRuntimeSettings()
    configure_logging(settings)
    mcp.run(**build_run_params(settings))
