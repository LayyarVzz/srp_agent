"""services/tools_mcp/config.py 纯函数单测：配置 → mcp.run 参数映射。"""

from __future__ import annotations

import pytest

from services.tools_mcp.config import MCPRuntimeSettings, MCPTransport, build_run_params


@pytest.mark.parametrize(
    ("transport", "expected"),
    [
        (
            MCPTransport.STDIO,
            {"transport": "stdio"},
        ),
        (
            MCPTransport.STREAMABLE_HTTP,
            {
                "transport": "streamable-http",
                "host": "127.0.0.1",
                "port": 8100,
                "path": "/mcp",
                "stateless_http": True,
            },
        ),
    ],
)
def test_build_run_params(transport: MCPTransport, expected: dict) -> None:
    """stdio 分支不注入 http 参数（避免透传给 run_stdio_async 触发 TypeError）；
    streamable-http 分支完整透传 host/port/path/stateless_http。"""
    assert build_run_params(MCPRuntimeSettings(mcp_transport=transport)) == expected


def test_build_run_params_custom_http_fields() -> None:
    """自定义 host/port/path 覆盖时按配置透传。"""
    settings = MCPRuntimeSettings(
        mcp_transport=MCPTransport.STREAMABLE_HTTP,
        mcp_host="0.0.0.0",  # noqa: S104
        mcp_port=9000,
        mcp_streamable_http_path="/tools",
    )
    assert build_run_params(settings) == {
        "transport": "streamable-http",
        "host": "0.0.0.0",  # noqa: S104
        "port": 9000,
        "path": "/tools",
        "stateless_http": True,
    }


def test_mcp_transport_values() -> None:
    """服务端传输值用连字符，与客户端连接键（下划线）区分。"""
    assert MCPTransport.STDIO.value == "stdio"
    assert MCPTransport.STREAMABLE_HTTP.value == "streamable-http"
