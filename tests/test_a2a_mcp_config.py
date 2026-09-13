"""services/a2a_mcp/config.py 纯函数单测：配置 → mcp.run 参数映射（镜像 tools_mcp）。"""

from __future__ import annotations

import pytest

from services.a2a_mcp.config import A2AMCPRuntimeSettings, MCPTransport, build_run_params


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
                "port": 8102,
                "path": "/mcp",
                "stateless_http": True,
            },
        ),
    ],
)
def test_build_run_params(transport: MCPTransport, expected: dict) -> None:
    """stdio 分支不注入 http 参数（避免透传给 run_stdio_async 触发 TypeError）；
    streamable-http 分支完整透传 host/port/path/stateless_http（端口 8102 避让）。

    WHY 显式传 host/port：本服务与 tools_mcp 共享 `MCP_*` 环境变量（stdio 子进程
    继承 .env 属预期行为），断言默认值须隔离环境差异。
    """
    settings = A2AMCPRuntimeSettings(
        mcp_transport=transport, mcp_host="127.0.0.1", mcp_port=8102
    )
    assert build_run_params(settings) == expected


def test_registry_and_call_behavior_defaults() -> None:
    """注册表缺省为空（空表 → Agent 侧门控不登记，零回归）；预算/轮询/截断有保守默认。"""
    settings = A2AMCPRuntimeSettings()
    assert settings.a2a_mcp_agents == {}
    assert settings.a2a_request_timeout_s == 120.0
    assert settings.a2a_poll_interval_s == 1.0
    assert settings.a2a_output_max_chars == 10_000


def test_registry_parses_json_env(monkeypatch) -> None:
    """A2A_MCP_AGENTS 环境变量以 JSON 解析为注册表（与服务侧配置同源）。"""
    monkeypatch.setenv("A2A_MCP_AGENTS", '{"b": "http://127.0.0.1:8002"}')
    settings = A2AMCPRuntimeSettings()
    assert settings.a2a_mcp_agents == {"b": "http://127.0.0.1:8002"}


def test_mcp_transport_values() -> None:
    """服务端传输值用连字符，与客户端连接键（下划线）区分。"""
    assert MCPTransport.STDIO.value == "stdio"
    assert MCPTransport.STREAMABLE_HTTP.value == "streamable-http"
