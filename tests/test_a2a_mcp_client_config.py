"""services/a2a_mcp/client_config.py 单测：客户端连接配置构造（镜像 tools_mcp）。"""

from __future__ import annotations

import sys

from services.a2a_mcp.client_config import (
    A2A_MCP_SERVER_MODULE,
    A2A_MCP_SERVER_NAME,
    build_a2a_mcp_http_connection,
    build_a2a_mcp_stdio_connection,
)


def test_stdio_connection_forces_stdio_transport(monkeypatch) -> None:
    """stdio 子进程强制 MCP_TRANSPORT=stdio：父进程已选择管道接入，子进程须与之一致。"""
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    conn = build_a2a_mcp_stdio_connection()
    assert conn["transport"] == "stdio"
    assert conn["command"] == sys.executable
    assert conn["args"] == ["-m", A2A_MCP_SERVER_MODULE]
    assert conn["env"]["MCP_TRANSPORT"] == "stdio"


def test_http_connection_builds_url() -> None:
    """streamable-http 连接：host/port/path 拼出 url，transport 键用下划线（客户端约定）。"""
    conn = build_a2a_mcp_http_connection(host="a2a_mcp", port=8102, path="/mcp")
    assert conn == {"transport": "streamable_http", "url": "http://a2a_mcp:8102/mcp"}


def test_server_name_constants() -> None:
    """服务名/模块名常量（MultiServerMCPClient 的 key 用下划线命名）。"""
    assert A2A_MCP_SERVER_NAME == "a2a_mcp"
    assert A2A_MCP_SERVER_MODULE == "services.a2a_mcp"
