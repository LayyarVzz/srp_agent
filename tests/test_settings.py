"""settings.py —— 根级运行环境配置单测（离线）。"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from agent.core.config import LLMProvider
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """每个用例前后清空 get_settings 缓存，避免 env 变更被缓存掩盖。"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults() -> None:
    s = RuntimeSettings()
    assert s.app_name == "srp-agent"
    assert s.environment == "dev"
    assert s.llm_provider is LLMProvider.DEEPSEEK
    assert s.api_host == "0.0.0.0"  # noqa: S104
    assert s.api_port == 8000
    assert s.log_level == "INFO"


def test_env_override(monkeypatch) -> None:
    """进程环境变量可覆盖代码默认值（BaseSettings 行为）。"""
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = RuntimeSettings()
    assert s.llm_provider is LLMProvider.QWEN
    assert s.api_port == 9000
    assert s.log_level == "DEBUG"


def test_mcp_defaults() -> None:
    """tools_mcp 默认走 stdio，回环监听，端口避开 API_PORT=8000。"""
    s = RuntimeSettings(_env_file=None)  # 隔离 .env，只验证代码默认值
    assert s.mcp_transport is MCPTransport.STDIO
    assert s.mcp_host == "127.0.0.1"
    assert s.mcp_port == 8100
    assert s.mcp_streamable_http_path == "/mcp"
    assert s.mcp_stateless_http is True


def test_mcp_env_override(monkeypatch) -> None:
    """MCP_* 环境变量可切换为 streamable-http 并覆盖 host/port。"""
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")  # noqa: S104
    monkeypatch.setenv("MCP_PORT", "9000")
    s = RuntimeSettings()
    assert s.mcp_transport is MCPTransport.STREAMABLE_HTTP
    assert s.mcp_host == "0.0.0.0"  # noqa: S104
    assert s.mcp_port == 9000


def test_get_settings_cached_singleton() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_api_key_secret_masked() -> None:
    s = RuntimeSettings(llm_api_key=SecretStr("sk-very-secret"))
    assert "sk-very-secret" not in str(s.llm_api_key)


def test_database_url_default_none(monkeypatch) -> None:
    """未配置 DATABASE_URL → database_url 为 None（回退 InMemoryStore + MemorySaver）。"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = RuntimeSettings(_env_file=None)
    assert s.database_url is None


def test_database_url_env_override(monkeypatch) -> None:
    """DATABASE_URL 环境变量 → SecretStr，repr 脱敏（明文不进日志）。"""
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/srp_agent")
    s = RuntimeSettings(_env_file=None)
    assert s.database_url is not None
    assert (
        s.database_url.get_secret_value()
        == "postgresql://postgres:postgres@localhost:5432/srp_agent"
    )
    assert "postgres:postgres" not in str(s.database_url)  # SecretStr 掩码
