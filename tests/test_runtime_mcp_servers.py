"""agent/runtime.py::_build_mcp_servers 门控矩阵测试（T4 零回归语义）。

不拉起真实子进程：命令探测经 monkeypatch 打桩；RuntimeSettings 以
`_env_file=None` 构造隔离 .env（沿用 test_settings.py 的隔离策略）。
"""

from __future__ import annotations

import pytest

from agent.core.config import AgentFrameworkConfig, LarkToolsConfig
from agent.runtime import _build_mcp_servers
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings

FAKE_CLI = "/fake/path/lark-cli"


@pytest.fixture
def settings() -> RuntimeSettings:
    """隔离 .env 的运行环境（stdio 默认，避免本机 .env 的 streamable-http 干扰）。"""
    return RuntimeSettings(_env_file=None)


@pytest.fixture
def cfg() -> AgentFrameworkConfig:
    return AgentFrameworkConfig.get_default()


@pytest.fixture
def probe_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """命令探测成功（返回可执行路径）。"""
    monkeypatch.setattr("agent.runtime.resolve_lark_cli_command", lambda override: FAKE_CLI)


@pytest.fixture
def probe_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """命令探测失败（无 CLI 环境）。"""
    monkeypatch.setattr("agent.runtime.resolve_lark_cli_command", lambda override: None)


def test_lark_registered_when_enabled_and_probed(settings, cfg, probe_ok) -> None:
    """默认配置 + 探测成功 → lark_mcp 登记（stdio 子进程形态），既有服务不受影响。"""
    servers = _build_mcp_servers(settings, cfg)
    assert set(servers) == {"rag", "tools_mcp", "lark_mcp"}
    lark_conn = servers["lark_mcp"]
    assert lark_conn["transport"] == "stdio"
    assert lark_conn["args"] == ["-m", "services.lark_mcp"]


def test_lark_skipped_when_probe_fails(settings, cfg, probe_fail) -> None:
    """无 lark-cli 环境 → 服务不登记，Agent 以既有工具集照常服务（V5-M2 零回归）。"""
    servers = _build_mcp_servers(settings, cfg)
    assert set(servers) == {"rag", "tools_mcp"}


def test_lark_skipped_when_framework_disabled(settings, probe_ok) -> None:
    """框架配置禁用（cfg.lark.enabled=False）→ 不登记（代码级门控）。"""
    cfg = AgentFrameworkConfig.get_default()
    cfg.lark = LarkToolsConfig(enabled=False)
    assert "lark_mcp" not in _build_mcp_servers(settings, cfg)


def test_lark_skipped_when_env_disabled(settings, cfg, probe_ok) -> None:
    """环境项禁用（LARK_CLI_ENABLED=false）→ 不登记（环境级门控）。"""
    env_settings = settings.model_copy(update={"lark_cli_enabled": False})
    assert "lark_mcp" not in _build_mcp_servers(env_settings, cfg)


def test_lark_skipped_when_env_off_overrides_probe(settings, cfg, probe_ok) -> None:
    """双门控取交集：环境关闭优先于探测成功。"""
    env_settings = settings.model_copy(update={"lark_cli_enabled": False})
    assert "lark_mcp" not in _build_mcp_servers(env_settings, cfg)


def test_lark_default_config_enabled() -> None:
    """AgentFrameworkConfig 聚合根默认含 lark 子配置且启用（doc §8.4）。"""
    assert AgentFrameworkConfig.get_default().lark.enabled is True


def test_existing_services_unconditional(settings, cfg, probe_fail) -> None:
    """探测失败只影响 lark_mcp：rag 恒登记，tools_mcp 按 stdio 默认登记（既有路径零回归）。"""
    servers = _build_mcp_servers(settings, cfg)
    assert servers["rag"]["transport"] == "stdio"
    assert servers["tools_mcp"]["transport"] == "stdio"


# —— HTTP 形态（lark_mcp 独立容器，v5.1 §8.1 / plan-docker §2.2）——


@pytest.fixture
def http_settings(settings) -> RuntimeSettings:
    """容器部署形态：lark 走 streamable-http，指向 compose 服务名。

    WHY 用枚举值而非字符串：`model_copy(update=...)` **绕过校验**，传字符串会让
    运行时 `is MCPTransport.STREAMABLE_HTTP` 判定失效（真实 env 路径经
    pydantic-settings 校验，恒为枚举成员）。
    """
    return settings.model_copy(
        update={
            "lark_mcp_transport": MCPTransport.STREAMABLE_HTTP,
            "lark_mcp_host": "lark_mcp",
            "lark_mcp_port": 8101,
        }
    )


def test_lark_http_registered_without_local_probe(http_settings, cfg, probe_fail) -> None:
    """HTTP 形态不探测本地 CLI：lark-cli 在另一容器，探测必失败但登记照常（B3 修复）。"""
    servers = _build_mcp_servers(http_settings, cfg)
    assert set(servers) == {"rag", "tools_mcp", "lark_mcp"}
    lark_conn = servers["lark_mcp"]
    assert lark_conn["transport"] == "streamable_http"
    assert lark_conn["url"] == "http://lark_mcp:8101/mcp"
    # tools_mcp 传输与本项**刻意分离**：lark 走 HTTP 不代表 tools_mcp 也走 HTTP。
    assert servers["tools_mcp"]["transport"] == "stdio"


def test_lark_http_still_gated_by_environment(http_settings, cfg, probe_ok) -> None:
    """HTTP 形态下双门控仍生效：环境关闭 → 不登记（探测成功也不登记）。"""
    env_settings = http_settings.model_copy(update={"lark_cli_enabled": False})
    assert "lark_mcp" not in _build_mcp_servers(env_settings, cfg)


def test_lark_http_still_gated_by_framework_config(http_settings, probe_ok) -> None:
    """HTTP 形态下框架配置门控仍生效：cfg.lark.enabled=False → 不登记。"""
    cfg = AgentFrameworkConfig.get_default()
    cfg.lark = LarkToolsConfig(enabled=False)
    assert "lark_mcp" not in _build_mcp_servers(http_settings, cfg)


def test_lark_stdio_semantics_unchanged(settings, cfg, probe_ok) -> None:
    """默认（stdio）路径逐字未变：仍是子进程形态，且探测成功才登记。"""
    servers = _build_mcp_servers(settings, cfg)
    assert servers["lark_mcp"]["transport"] == "stdio"
    assert servers["lark_mcp"]["args"] == ["-m", "services.lark_mcp"]
