"""项目根级运行环境配置。

本模块统一管理**运行环境**相关配置：密钥、服务端口、外部服务地址、日志级别；

值来源优先级：进程环境变量 > `.env` 文件 > 代码默认值。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.core.config import LLMProvider
from services.tools_mcp.config import MCPTransport

# 观测环境标记的合法形态（与 langfuse SDK 同规则）。WHY 用 `re` + field_validator 而非
# `Field(pattern=...)`：pydantic-core 的正则引擎不支持 lookahead，`(?!langfuse)` 会直接
# 让类定义期报 SchemaError（配置校验反而把服务搞挂）。
_LANGFUSE_ENV_RE: Final = re.compile(r"(?!langfuse)[a-z0-9_-]{1,40}")


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

    # —— FastAPI 服务端口 ——
    api_host: str = "0.0.0.0"  # noqa: S104  # 开发默认监听全部接口，部署时按需收紧
    api_port: int = 8000

    # —— CORS（FastAPI 交互服务）：dev 默认前端 dev server 地址，部署时显式配置 ——
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # —— tools_mcp 服务运行方式（stdio 默认；streamable-http 供端口/远程部署）——
    mcp_transport: MCPTransport = MCPTransport.STDIO
    mcp_host: str = "127.0.0.1"  # 云服务器/外部访问须设 0.0.0.0（默认回环更安全）
    mcp_port: int = 8100  # 避开 API_PORT=8000
    mcp_streamable_http_path: str = "/mcp"  # 与 fastmcp 默认一致，客户端连接地址即该路径
    mcp_stateless_http: bool = True  # 工具纯函数无会话 → 默认无状态，支持水平扩展

    # —— 飞书 lark-cli（T4）：环境门控 + 命令覆盖；CLI 超时/截断等服务侧项见
    # services/lark_mcp/config.py。命令探测失败时 lark_mcp 不登记，Agent 照常服务（零回归）——
    lark_cli_enabled: bool = True
    lark_cli_command: str | None = None  # LARK_CLI_COMMAND 显式覆盖；空则探测 PATH
    # —— lark_mcp 客户端连接（容器/远程形态）：与 tools_mcp 的 `mcp_*` **刻意分离** ——
    # WHY 独立三项而非复用 `MCP_TRANSPORT/HOST/PORT`：`MCP_*` 是 tools_mcp 的地址
    # （端口 8100），lark_mcp 在独立容器里监听 8101；共用一个环境项会让 api 把
    # lark 请求打到 tools_mcp 上（或反之），表现为「连上了但工具列表不对」。
    # 命名与 `A2A_MCP_*` 同构：一个远端 MCP 服务 = 一组独立连接项。
    lark_mcp_transport: MCPTransport = MCPTransport.STDIO  # streamable-http = 连远端容器
    lark_mcp_host: str = "127.0.0.1"  # 容器部署 = compose 服务名 lark_mcp
    lark_mcp_port: int = 8101  # 避开 API 8000 / tools_mcp 8100 / a2a_mcp 8102
    # —— A2A 智能体互联（T5 Server 入站；T6 Client 环境项见下）——
    # 环境变量约定 A2A_ENABLED / A2A_PEER_MAP（pydantic-settings 大小写不敏感匹配）。
    a2a_enabled: bool = True  # 关闭时所有入站 A2A 请求拒绝（AgentCard 发现不受影响）
    # peer 注册表：JSON 对象 {"<peer_id>": "<user_id|null>"}；user_id 缺省落匿名命名空间。
    a2a_peer_map: dict[str, str] = Field(default_factory=dict)

    # —— A2A Client（T6：a2a_mcp 服务装配；注册表非空才登记，空表零回归）——
    a2a_mcp_enabled: bool = True  # 总开关；与注册表非空同时满足才登记 a2a_mcp 服务
    # 远端智能体注册表：env A2A_MCP_AGENTS（JSON {"<agent_id>": "<base_url>"}），
    # 服务侧（services/a2a_mcp/config.py）读同名 env，两边约定同名、代码解耦。
    a2a_mcp_agents: dict[str, str] = Field(default_factory=dict)
    a2a_mcp_host: str = "127.0.0.1"  # streamable-http 形态的 a2a_mcp 地址（容器部署 = 服务名）
    a2a_mcp_port: int = 8102  # 避开 API_PORT=8000 / tools_mcp 8100 / lark_mcp 8101

    # —— embedding（与 RAG 对齐；语义召回开关）——
    embedding_enabled: bool = False  # 置 true 启用语义召回/去重
    embedding_model: str = ""  # 与 RAG 同一模型名
    embedding_dims: int = Field(default=0, ge=0)  # 与模型一致；enabled 时必须 > 0
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))

    # —— Postgres 记忆后端（生产持久化；缺省则不启用，回退 InMemoryStore + MemorySaver）——
    # checkpointer（会话/短期）与 Store（长期/会话元数据）共用此库，各自独立连接池与表。
    database_url: SecretStr | None = Field(
        default=None,
        description="postgresql://user:pass@host:port/db",
    )

    # —— 讯飞语音听写（IAT）：语音链路配置统一由本模块管理，禁止在业务代码内嵌 / 手动读 .env ——
    # 环境变量约定 XF_IAT_APP_ID / XF_IAT_API_KEY / XF_IAT_API_SECRET（pydantic-settings
    # 大小写不敏感匹配）；未配置时语音接口返回 asr.missing_credentials，text 链路不受影响。
    xf_iat_app_id: SecretStr = Field(default_factory=lambda: SecretStr(""))
    xf_iat_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    xf_iat_api_secret: SecretStr = Field(default_factory=lambda: SecretStr(""))
    xf_iat_url: str = "wss://iat-api.xfyun.cn/v2/iat"  # 可覆盖默认端点

    # —— 观测（Langfuse Cloud；SDK 直连，**默认关闭**）——
    # WHY 默认关闭：观测是外挂能力，不得成为对话链路的前置条件 —— 未配置即图 config
    # 零观测回调（逐字零回归），Langfuse 侧不可达时 Agent 照常回答（fail-open）。
    # env 名与 SDK **同名**（日志/文档/trace 三处口径一致）；字段名与 env 名不同者必须显式
    # 声明 `AliasChoices` —— pydantic 的默认映射是按**字段名**大写（`langfuse_enabled`
    # 会去找 LANGFUSE_ENABLED，而 SDK 认的是 LANGFUSE_TRACING_ENABLED），不声明就永远读不到。
    # env 的读写**只发生在 shared/observability.py 这一个装配点**，业务代码不自行读 env。
    langfuse_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("LANGFUSE_TRACING_ENABLED", "langfuse_enabled")
    )
    langfuse_public_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    # 项目密钥：SecretStr（不写日志、不进 ToolMessage、不回前端）；不参与任何事件字段。
    langfuse_secret_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    # Cloud 区域端点（EU 默认；US/JP/HIPAA 见 .env.example）。只提供 BASE_URL：
    # `LANGFUSE_HOST` 是废弃旧名，且两者同设时 SDK **静默取 BASE_URL**（易误判）。
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # 环境标记：缺省对齐 ENVIRONMENT。SDK 要求 ^(?!langfuse)[a-z0-9_-]+$ 且 ≤40 字符，
    # 不合规会被 SDK 丢弃并告警（静默掉一个标记）→ 此处启动即校验，拼错不静默。
    langfuse_environment: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_TRACING_ENVIRONMENT", "langfuse_environment"),
    )
    # 版本标记：缺省 app_name@ENVIRONMENT。**显式给值**的理由：不设时 SDK 会回退读
    # 常见 CI release 变量（get_common_release_envs）→ 本地 trace 被 CI 变量串味。
    langfuse_release: str | None = None
    # 采样率：SDK 越界直接抛 ValueError（启动即失败，不静默）→ 此处同边界校验。
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    # SDK 全部 API 请求超时（秒）。env 名是 LANGFUSE_TIMEOUT（SDK 口径），非 _TIMEOUT_S。
    langfuse_timeout_s: int = Field(
        default=5, ge=1, validation_alias=AliasChoices("LANGFUSE_TIMEOUT", "langfuse_timeout_s")
    )
    # aclose() 里 flush 的上限（秒）：退出路径不得因云端不可达而挂住进程。
    # 本项目自有旋钮（SDK 的 FLUSH_AT/FLUSH_INTERVAL 是批量参数，语义不同，刻意不投影）。
    langfuse_flush_timeout_s: float = Field(default=5.0, gt=0.0)

    # —— 日志（运行期级别与形态；格式化/脱敏/关联标识统一由 shared.logging 承担）——
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # LOG_FORMAT：text（默认，人眼可读 + trace 前缀）| json（一行一条，供日志采集）。
    # 未知值由 shared.logging 回退 text —— 日志形态拼错不得让服务起不来。
    log_format: Literal["text", "json"] = "text"

    @field_validator("langfuse_environment", "langfuse_release", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """空串等同未配置。

        WHY 必须有：compose 里 `${LANGFUSE_TRACING_ENVIRONMENT:-}` 展开就是**空串**，
        而空串不是「配了一个空环境名」——不归一化会让容器在宿主没配 .env 时启动即校验失败
        （默认关闭的观测反而把服务搞挂，方向完全反了）。
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("langfuse_environment")
    @classmethod
    def _check_langfuse_environment(cls, value: str | None) -> str | None:
        """校验观测环境标记（与 SDK 同规则，但提前到启动期 fail-fast）。

        WHY 不交给 SDK：SDK 对不合规值只**丢弃并告警**——trace 照发但少了环境标记，
        表现为「云端看到一堆 default 环境的 trace、分不清是哪套部署」，属静默失真。
        """
        if value is None:
            return None
        if not _LANGFUSE_ENV_RE.fullmatch(value):
            msg = (
                f"LANGFUSE_TRACING_ENVIRONMENT 不合法：{value!r}"
                "（只允许 [a-z0-9_-]、≤40 字符、不得以 langfuse 开头）"
            )
            raise ValueError(msg)
        return value


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    """进程内单例；测试需要隔离环境时可 `get_settings.cache_clear()` 或直接构造。"""
    return RuntimeSettings()
