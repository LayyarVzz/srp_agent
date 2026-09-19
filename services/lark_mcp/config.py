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

from pydantic import Field, SecretStr
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

    # —— 飞书共享应用与绑定链路（v5.1，dev-version5.1.md §9）——
    # 应用**仅作 OAuth 客户端**（无 bot 身份）：app_id/secret 只用于设备码授权、
    # 换码与刷新；工具调用恒为 user 身份，身份来自各自绑定的 UAT。
    lark_app_id: str | None = None
    lark_app_secret: SecretStr = Field(default_factory=lambda: SecretStr(""))
    # 令牌加密密钥（Fernet 派生）：未配置 → 绑定功能 fail-closed 关闭，绝不落明文 token。
    lark_token_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    # 绑定记录 / 设备码待定态所在库（镜像 sessions 表裁决：无 DSN → SQLite memory）。
    database_url: SecretStr | None = None
    # 绑定域专用 DSN（**优先于 `database_url`**）。
    #
    # WHY 需要独立项：根 `DATABASE_URL` 是**全局裁决**（同时驱动 memory / checkpointer /
    # sessions），而绑定域的约束与之并不一致 ——
    #   - api 侧用 `DATABASE_URL` 时必须连 Postgres（langgraph 无 SQLite Store）；
    #   - 绑定域只需「跨进程稳定」的库，本机无 Postgres 时**文件型 SQLite**即可跑通。
    # 不拆分则本地联调只能二选一：要么为绑定域单独起 Postgres，要么接受 api 启动失败。
    # ⚠️ 必须指向**文件型**（`sqlite+aiosqlite:///...`）：绑定域的状态要跨**多次 MCP
    # 工具调用**共享（每次调用是一个新子进程），`:memory:` 会让待定态在调用间蒸发。
    lark_database_url: SecretStr | None = None
    # 认证族域名（默认飞书；Lark 品牌 = accounts.larksuite.com / open.larksuite.com）。
    lark_accounts_base_url: str = "https://accounts.feishu.cn"
    lark_open_base_url: str = "https://open.feishu.cn"
    # 配置态单用户凭据（仅无绑定功能的本地/单机形态使用，见 credentials.py）。
    lark_user_access_token: SecretStr = Field(default_factory=lambda: SecretStr(""))
    lark_binding_enabled: bool = True  # 绑定总开关；密钥缺省时仍自动关闭（fail-closed）
    lark_token_refresh_skew_s: int = Field(default=300, ge=0)  # 提前刷新窗口
    lark_oauth_timeout_s: float = Field(default=15.0, ge=1.0)  # OAuth HTTP 调用超时
    lark_device_flow_poll_max_s: float = Field(default=600.0, ge=1.0)  # 设备码轮询总上限

    @property
    def binding_database_url(self) -> str | None:
        """绑定域实际使用的 DSN：`LARK_DATABASE_URL` 优先，缺省回退 `DATABASE_URL`。

        WHY 收敛成单点属性而非在装配处写 `or`：DSN 裁决属「存储语义」的一部分
        （与 §7 的加密 / 乐观锁同级），必须一处声明 —— 消费方（server 装配）只取值，
        不重复判别，否则后续再加一个 DSN 来源就会漏改调用点。
        """
        preferred = self.lark_database_url or self.database_url
        return preferred.get_secret_value() if preferred else None


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
