"""tools_mcp 客户端连接配置（镜像 services/rag_mcp/client_config.py）。

WHY 独立模块：连接参数（stdio 子进程命令 / HTTP 地址）收敛在服务侧统一导出，
agent 侧（runtime）只按传输方式选择 builder，不再散落构造逻辑；与 RAG 对称。
本模块只依赖本包 `config`（MCPRuntimeSettings），不依赖项目根 `settings.py`。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_mcp_adapters.sessions import StdioConnection

TOOLS_MCP_SERVER_NAME = "tools_mcp"
TOOLS_MCP_SERVER_MODULE = "services.tools_mcp"


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_tools_mcp_stdio_connection() -> StdioConnection:
    """构造连接 tools_mcp 的 stdio 配置（以子进程自动拉起本服务）。

    WHY 强制 MCP_TRANSPORT=stdio：子进程会继承 .env / 环境里的
    MCP_TRANSPORT=streamable-http，误启 HTTP 服务并抢端口导致 Connection closed。
    """
    env = dict(os.environ)
    env["MCP_TRANSPORT"] = "stdio"
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", TOOLS_MCP_SERVER_MODULE],
        "cwd": str(_get_repo_root()),
        "env": env,
    }


def build_tools_mcp_http_connection(*, host: str, port: int, path: str) -> dict[str, str]:
    """构造连接 tools_mcp 的 streamable-http 配置（远端 / 容器部署形态）。

    WHY 与 RAG 的差异：RAG 仅 stdio；tools_mcp 支持双传输，HTTP 地址由运行环境
    （MCP_HOST / MCP_PORT / MCP_STREAMABLE_HTTP_PATH）经参数显式传入，本函数纯构造。
    """
    return {"transport": "streamable_http", "url": f"http://{host}:{port}{path}"}
