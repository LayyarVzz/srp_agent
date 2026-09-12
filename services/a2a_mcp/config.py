"""a2a_mcp 服务运行参数与本地运行环境构造（镜像 services/tools_mcp/config.py）。

将 MCP 运行环境配置（`MCP_*` 环境变量）与 A2A 远端注册表（`A2A_MCP_AGENTS`）
映射为 `mcp.run(**kwargs)` 参数与客户端行为。本模块**不依赖项目根 `settings.py`**
（也即不连带加载 agent 框架），可独立打包部署；配置统一经本地
`A2AMCPRuntimeSettings` 读取（env `MCP_*` / `A2A_MCP_*` / `LOG_LEVEL`）。

`A2A_MCP_AGENTS` 是远端智能体注册表：JSON 对象 `{"<agent_id>": "<base_url>"}`；
注册表为空时本服务无可注册工具（Agent 侧门控也不登记，零回归）。
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPTransport(StrEnum):
    """a2a_mcp 运行传输方式（值须与 fastmcp `run(transport=...)` 对齐）。

    每个服务独立定义本枚举（镜像 tools_mcp/lark_mcp 惯例）：服务端 fastmcp
    传输值用连字符 `streamable-http`；langchain-mcp-adapters 客户端连接配置
    里的键是下划线 `streamable_http`。
    """

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"


class A2AMCPRuntimeSettings(BaseSettings):
    """a2a_mcp 本地运行环境（独立于根 settings.py，与 tools_mcp 同构）。

    WHY 字段名复用 `mcp_*` / `a2a_mcp_*` 命名：pydantic-settings 大小写不敏感
    匹配环境变量，与全仓库 `.env`、compose 完全兼容；Agent 侧门控在根
    settings 读同名 `A2A_MCP_AGENTS`，两边约定同名 env、代码解耦。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mcp_transport: MCPTransport = MCPTransport.STDIO
    mcp_host: str = "127.0.0.1"  # 云服务器/外部访问须设 0.0.0.0（默认回环更安全）
    mcp_port: int = 8102  # 避开 API_PORT=8000 / tools_mcp 8100 / lark_mcp 8101
    mcp_streamable_http_path: str = "/mcp"
    mcp_stateless_http: bool = True
    # —— A2A 远端注册表与调用行为 ——
    a2a_mcp_agents: dict[str, str] = Field(default_factory=dict)  # {agent_id: base_url}
    # 本服务调用远端时声明的 peer 身份（远端须已登记该 peer，否则 a2a.invalid_request）。
    a2a_mcp_peer_id: str = "srp-agent"
    a2a_request_timeout_s: float = Field(default=120.0, ge=1)  # 单次任务总预算（远端执行可能较慢）
    a2a_poll_interval_s: float = Field(default=1.0, ge=0.05)  # task/get 轮询间隔
    a2a_output_max_chars: int = Field(default=10_000, ge=1)  # 工具输出长度上限（服务侧截断）
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def build_run_params(settings: A2AMCPRuntimeSettings) -> dict[str, Any]:
    """把本地运行环境配置映射为 `mcp.run(**kwargs)` 参数（镜像 tools_mcp）。

    fastmcp 在 stdio 下会把多余 kwargs 透传给 `run_stdio_async`（其签名仅
    show_banner/log_level/stateless），传 host/port/path 会 TypeError，
    因此 http 系列参数只能出现在非 stdio 分支。
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


def configure_logging(settings: A2AMCPRuntimeSettings) -> None:
    """集中配置 a2a_mcp 的 root logger 级别（入口层调用一次）。"""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
