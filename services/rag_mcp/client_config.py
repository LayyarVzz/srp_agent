"""RAG MCP客户端连接配置。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_mcp_adapters.sessions import StdioConnection

RAG_MCP_SERVER_NAME = "rag"
RAG_MCP_SERVER_MODULE = "services.rag_mcp.server"


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_rag_mcp_stdio_connection() -> StdioConnection:
    """构造连接RAG MCP Server的stdio配置。"""
    env = dict(os.environ)
    env["FASTMCP_CHECK_FOR_UPDATES"] = "off"

    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [
            "-m",
            RAG_MCP_SERVER_MODULE,
        ],
        "cwd": str(_get_repo_root()),
        "env": env,
    }
