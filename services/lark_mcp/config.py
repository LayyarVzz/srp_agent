"""lark_mcp 服务运行参数与本地运行环境构造。

将 MCP 运行环境配置（`MCP_*` 环境变量）与 lark-cli 子进程参数（`LARK_CLI_*`）
映射为 `mcp.run(**kwargs)` 参数与 CLI 调用行为。
本模块**不依赖项目根 `settings.py`**（也即不连带加载 agent 框架），可独立打包部署；
配置统一经本地 `LarkMCPRuntimeSettings` 读取（env `MCP_*` / `LARK_CLI_*` / `LOG_LEVEL`，
与全仓库 `.env` 兼容）。

lark-cli 凭据说明：token 存于 CLI 自身的 OS 钥匙串（Windows 凭据管理器等），
本服务进程不接触明文 token；鉴权引导见 services/lark_mcp/README.md。
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPTransport(StrEnum):
    """lark_mcp 运行传输方式（值须与 fastmcp `run(transport=...)` 对齐）。

    注意：fastmcp 服务端传输值用连字符 `streamable-http`；langchain-mcp-adapters
    客户端连接配置里的键是下划线 `streamable_http`。
    """

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"


class LarkMCPRuntimeSettings(BaseSettings):
    """lark_mcp 本地运行环境（独立于根 settings.py）。

    WHY 字段名复用根 `RuntimeSettings` 的 `mcp_*` 命名：pydantic-settings 大小写
    不敏感匹配环境变量（`mcp_transport` ↔ `MCP_TRANSPORT`），与全仓库 `.env` 完全兼容。
    `LARK_CLI_COMMAND` / `LARK_CLI_TIMEOUT_S` 仅作用于 lark-cli 子进程调用，
    与根 settings.py 的同名环境项保持一致（stdio 子进程继承同一份 .env）。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mcp_transport: MCPTransport = MCPTransport.STDIO
    mcp_host: str = "127.0.0.1"  # 云服务器/外部访问须设 0.0.0.0（默认回环更安全）
    mcp_port: int = 8101  # 避开 API_PORT=8000 与 tools_mcp 默认 8100
    mcp_streamable_http_path: str = "/mcp"  # 与 fastmcp 默认一致，客户端连接地址即该路径
    mcp_stateless_http: bool = True  # 工具无会话状态 → 默认无状态，支持水平扩展
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # —— lark-cli 子进程参数 ——
    lark_cli_command: str | None = None  # LARK_CLI_COMMAND 显式覆盖；空则按 PATH 探测
    lark_cli_timeout_s: float = Field(default=30.0, ge=1.0)  # 单次 CLI 调用超时（网络 IO）
    lark_output_max_chars: int = Field(default=10_000, ge=1)  # 工具输出长度上限（护栏）


def build_run_params(settings: LarkMCPRuntimeSettings) -> dict[str, Any]:
    """把本地运行环境配置映射为 `mcp.run(**kwargs)` 参数。

    WHY 与 tools_mcp 同构：fastmcp 在 stdio 下会把多余 kwargs 透传给
    `run_stdio_async`（其签名仅 show_banner/log_level/stateless），传 host/port/path
    会 TypeError，因此 http 系列参数只能出现在非 stdio 分支。
    """
    params: dict[str, Any] = {"transport": settings.mcp_transport.value}
    if settings.mcp_transport is not MCPTransport.STDIO:
        params.update(
            host=settings.mcp_host,
            port=settings.mcp_port,
            path=settings.mcp_streamable_http_path,
            stateless_http=settings.mcp_stateless_http,
        )
    return params


def configure_logging(settings: LarkMCPRuntimeSettings) -> None:
    """集中配置 lark_mcp 的 root logger 级别（入口层调用一次）。"""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
