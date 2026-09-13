"""a2a_mcp 服务入口。

默认 stdio（本地开发、demo stdio 子进程、进程内测试不变）；配置
MCP_TRANSPORT=streamable-http（及 MCP_HOST/MCP_PORT/MCP_STREAMABLE_HTTP_PATH）
即以 HTTP 运行。配置经本地 `A2AMCPRuntimeSettings` 读取，**不依赖根
settings.py / agent 框架**，可独立打包部署。

远端智能体注册表：env `A2A_MCP_AGENTS`（JSON `{"<agent_id>": "<base_url>"}`）；
注册表为空时服务无可调用工具（Agent 侧门控亦不登记，零回归）。
"""

from __future__ import annotations

from services.a2a_mcp.config import (
    A2AMCPRuntimeSettings,
    build_run_params,
    configure_logging,
)
from services.a2a_mcp.server import mcp

if __name__ == "__main__":
    settings = A2AMCPRuntimeSettings()
    configure_logging(settings)
    mcp.run(**build_run_params(settings))
