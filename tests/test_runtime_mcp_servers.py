"""agent/runtime.py::_build_mcp_servers 门控矩阵测试（T4 零回归语义）。

不拉起真实子进程：命令探测经 monkeypatch 打桩；RuntimeSettings 以
`_env_file=None` 构造隔离 .env（沿用 test_settings.py 的隔离策略）。
"""

from __future__ import annotations

import pytest

from agent.core.config import AgentFrameworkConfig, LarkToolsConfig
from agent.runtime import _build_mcp_servers
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
