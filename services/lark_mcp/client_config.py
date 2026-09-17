"""lark_mcp 客户端连接配置（镜像 services/tools_mcp/client_config.py）。

WHY 独立模块：连接参数收敛在服务侧统一导出，agent 侧（runtime）只做门控判断
后选择 builder，不再散落构造逻辑；与 RAG / tools_mcp 对称。
本模块只依赖 stdlib，不依赖项目根 `settings.py`。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_mcp_adapters.sessions import StdioConnection

# 服务名常量单一来源在 shared/lark（agent 侧拦截器据此识别飞书工具，且 agent 不 import
# services）；本模块只保留「模块名」常量，避免两处同名字符串各自漂移。
LARK_MCP_SERVER_MODULE = "services.lark_mcp"


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_lark_mcp_stdio_connection() -> StdioConnection:
    """构造连接 lark_mcp 的 stdio 配置（以子进程自动拉起本服务）。

    WHY 强制 MCP_TRANSPORT=stdio：子进程会继承 .env / 环境里的
    MCP_TRANSPORT=streamable-http，误启 HTTP 服务并抢端口导致 Connection closed
    （tools_mcp 同款先例）。
    """
    env = dict(os.environ)
    env["MCP_TRANSPORT"] = "stdio"
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", LARK_MCP_SERVER_MODULE],
        "cwd": str(_get_repo_root()),
        "env": env,
    }
