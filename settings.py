"""项目根级运行环境配置。

本模块统一管理**运行环境**相关配置：密钥、服务端口、外部服务地址、日志级别；

值来源优先级：进程环境变量 > `.env` 文件 > 代码默认值。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.core.config import LLMProvider
from services.tools_mcp.config import MCPTransport


class RuntimeSettings(BaseSettings):
    """运行环境配置（BaseSettings，自动读取 `.env` 与环境变量）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    app_name: str = "srp-agent"

    # —— LLM 运行选择 + 密钥（值由 .env 提供；行为参数见 agent/core/config.py）——
    llm_provider: LLMProvider = LLMProvider.DEEPSEEK
    llm_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    llm_base_url: str | None = None  # 覆盖预设
    llm_model: str | None = None  # 覆盖预设

    # —— Embedding（OpenAI-compatible 文本向量化）——
    embedding_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    embedding_base_url: str | None = None
    embedding_model: str | None = None

    # —— FastAPI 服务端口 ——
    api_host: str = "0.0.0.0"  # noqa: S104  # 开发默认监听全部接口，部署时按需收紧
    api_port: int = 8000 

    # —— tools_mcp 服务运行方式（stdio 默认；streamable-http 供端口/远程部署）——
    mcp_transport: MCPTransport = MCPTransport.STDIO
    mcp_host: str = "127.0.0.1"  # 云服务器/外部访问须设 0.0.0.0（默认回环更安全）
    mcp_port: int = 8100  # 避开 API_PORT=8000
    mcp_streamable_http_path: str = "/mcp"  # 与 fastmcp 默认一致，客户端连接地址即该路径
    mcp_stateless_http: bool = True  # 工具纯函数无会话 → 默认无状态，支持水平扩展


    # —— 日志（运行期级别）——
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    """进程内单例；测试需要隔离环境时可 `get_settings.cache_clear()` 或直接构造。"""
    return RuntimeSettings()


def configure_logging(settings: RuntimeSettings | None = None) -> None:
    """集中配置 root logger 级别（入口层调用一次）。"""
    level = (settings or get_settings()).log_level
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
