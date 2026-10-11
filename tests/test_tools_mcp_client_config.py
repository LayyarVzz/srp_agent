"""services/tools_mcp/client_config.py 单测：客户端连接配置构造（镜像 RAG client_config）。"""

from __future__ import annotations

import sys

from services.tools_mcp.client_config import (
    TOOLS_MCP_SERVER_MODULE,
    TOOLS_MCP_SERVER_NAME,
    build_tools_mcp_http_connection,
    build_tools_mcp_stdio_connection,
)


def test_stdio_connection_forces_stdio_transport(monkeypatch) -> None:
    """stdio 子进程强制 MCP_TRANSPORT=stdio，防止继承 HTTP 配置误启服务抢端口。"""
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    conn = build_tools_mcp_stdio_connection()
    assert conn["transport"] == "stdio"
    assert conn["command"] == sys.executable
    assert conn["args"] == ["-m", TOOLS_MCP_SERVER_MODULE]
    assert conn["env"]["MCP_TRANSPORT"] == "stdio"


def test_http_connection_builds_url() -> None:
    """streamable-http 连接：host/port/path 拼出 url，transport 键用下划线（客户端约定）。"""
    conn = build_tools_mcp_http_connection(host="127.0.0.1", port=8100, path="/mcp")
    assert conn == {"transport": "streamable_http", "url": "http://127.0.0.1:8100/mcp"}


def test_server_name_constants() -> None:
    """服务名/模块名常量（与 RAG client_config 同构）。"""
    assert TOOLS_MCP_SERVER_NAME == "tools_mcp"
    assert TOOLS_MCP_SERVER_MODULE == "services.tools_mcp"
