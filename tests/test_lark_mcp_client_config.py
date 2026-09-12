"""services/lark_mcp/client_config.py 单测：客户端连接配置构造（镜像 tools_mcp 同名测试）。"""

from __future__ import annotations

import sys

from services.lark_mcp.client_config import (
    LARK_MCP_SERVER_MODULE,
    LARK_MCP_SERVER_NAME,
    build_lark_mcp_stdio_connection,
)


def test_stdio_connection_forces_stdio_transport(monkeypatch) -> None:
    """stdio 子进程强制 MCP_TRANSPORT=stdio，防止继承 HTTP 配置误启服务抢端口。"""
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    conn = build_lark_mcp_stdio_connection()
    assert conn["transport"] == "stdio"
    assert conn["command"] == sys.executable
    assert conn["args"] == ["-m", LARK_MCP_SERVER_MODULE]
    assert conn["env"]["MCP_TRANSPORT"] == "stdio"


def test_stdio_connection_cwd_is_repo_root() -> None:
    """cwd 固定仓库根：stdio 子进程以 `python -m services.lark_mcp` 自仓库根拉起。"""
    conn = build_lark_mcp_stdio_connection()
    assert conn["cwd"].endswith("srp_agent") or conn["cwd"].endswith("srp_agent\\")


def test_server_name_constants() -> None:
    """服务名/模块名常量（与 rag / tools_mcp 的 client_config 同构）。"""
    assert LARK_MCP_SERVER_NAME == "lark_mcp"
    assert LARK_MCP_SERVER_MODULE == "services.lark_mcp"
