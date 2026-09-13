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
