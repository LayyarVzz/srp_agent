"""观测底座（Phase D）：Langfuse SDK 直连的**唯一装配点**。

WHY 放 `shared/` 而非 `agent/`：与 `shared/embeddings.py` 同构 —— 配置模型 + 工厂 + 唯一
装配点，供 api（图调用）及未来消费方共用；观测不是 Agent 私有能力，也不该散落在图节点里。

**本轮不做 OTel 装配**（plan §5.3）：不设任何 `OTEL_*`、不自建 collector、不装
`openinference-*` 之类 instrumentation、不写 `start_span` 手埋点。注意 Langfuse SDK 内部
**就是** OTel（自建 TracerProvider + OTLP-HTTP exporter 上报到
`{base_url}/api/public/otel/v1/traces`）—— 那是它的实现细节，端点由 `base_url` 决定，
与本项目的「不装配 OTel」并不矛盾。

五条硬约束（plan §5.2，逐条落在本模块）：
1. **env 投影只发生在这里**：SDK 在客户端首次构造时读环境变量（且 env 有 lru_cache 语义），
   故装配时必须先把 settings 值投影进 `os.environ`；业务代码**禁止**自行读 env。
2. **非阻塞 + fail-open**：只经 langchain **回调**接入（不在图节点内手动 `start_span` 包 LLM
   调用，那会给对话链路引入同步网络点）；上报失败只记日志、绝不冒泡进对话链路。
3. **异步优先**：复用 langchain 的 async 回调钩子，不自研收集器。
4. **脱敏走 SDK 原生 `mask=`**：送云端的是 prompt / 回答 / 工具入参与结果**原文**，
   必须掩码；掩码键复用 `shared.logging` 的同一套口径（飞书 token 绝不进 trace）。
5. **数据出境**：Langfuse Cloud 是境外 SaaS，开启即代表接受把对话样本发往所选区域。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, Final

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from settings import RuntimeSettings
from shared.logging import MASKED, is_sensitive_field, mask_text

logger = logging.getLogger(__name__)

# Cloud 区域端点默认值（EU；US/JP/HIPAA 见 .env.example）。只提供 BASE_URL：
# `LANGFUSE_HOST` 是 SDK 的废弃旧名，且两者同设时 SDK 静默取 BASE_URL（易误判）。
DEFAULT_BASE_URL: Final = "https://cloud.langfuse.com"

# 云端 trace 展示名（固定值：按入口筛 trace 时不必记调用点的字面量）。
TRACE_NAME: Final = "agent.chat"

# 带外记忆保存的 trace 名（O3）：与对话 trace `agent.chat` 分开呈现，云端按 trace name
# 即可区分「对话本体」与「带外抽取/判重」——后者靠归因键与对话 trace 对齐，不混入成本口径。
OUT_OF_BAND_TRACE_NAME: Final = "memory.out_of_band_save"

# —— langchain metadata → Langfuse 归因键（**SDK 契约名，禁止改写**）——
# SDK v4 在**根 run** 上读这些键并自行 `propagate_attributes(...)`
# （`langfuse/langchain/CallbackHandler.py` 的 `_parse_langfuse_trace_attributes`），
# 这正是官方「Option 1（最简单）」路径：不必把 `graph.astream(...)` 这个异步生成器
# 裹进同步上下文管理器。
_META_SESSION: Final = "langfuse_session_id"
_META_USER: Final = "langfuse_user_id"
_META_TRACE_NAME: Final = "langfuse_trace_name"

# 本地 trace_id 的关联键：**故意不加 `langfuse_` 前缀** —— 它不是给 SDK 的归因指令，
# 而是作为普通 span metadata 落库，供「本地事件表 ↔ 云端 trace」互查。
# WHY 需要它：云端的 trace id 由 SDK 自己生成，与本项目 C2 的 `X-Request-Id`
# （`req_` + 16 字节 hex，含前缀、不是 W3C 32 位 hex）**没有天然映射** ——
# 不额外带一个键，两边就只剩 session/user/时间可用，拿一条 request id 跳不过去。
_META_REQUEST_TRACE: Final = "request_trace_id"

# mask 函数自身失败时的兜底（与 SDK 内部占位符同文案）：宁可丢内容也不泄漏。
FULLY_MASKED: Final = "<fully masked due to failed mask function>"

# 需投影进 `os.environ` 的项：`{SDK env 名: ObservabilityConfig 字段名}`。
# **必须与 SDK 同名**（日志、文档、trace 三处口径一致）；名字由
# `tests/test_observability.py` 对着 SDK 自己的常量表逐个校验（拼错即红）。
# 刻意**不投影**：`LANGFUSE_HOST`（废弃旧名）、`LANGFUSE_DEBUG`（SDK 会再调一次
# `logging.basicConfig`，与 shared.logging 的统一 formatter 冲突）、`LANGFUSE_FLUSH_AT` /
# `LANGFUSE_FLUSH_INTERVAL`（SDK 未设时沿用 OTel 的批量参数，属其内部行为）、
# 以及任何 `OTEL_*`（本轮不做 OTel 装配）。
_ENV_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("LANGFUSE_PUBLIC_KEY", "public_key"),
    ("LANGFUSE_SECRET_KEY", "secret_key"),
    ("LANGFUSE_BASE_URL", "base_url"),
    ("LANGFUSE_TRACING_ENABLED", "enabled"),
    ("LANGFUSE_TRACING_ENVIRONMENT", "environment"),
    ("LANGFUSE_RELEASE", "release"),
    ("LANGFUSE_SAMPLE_RATE", "sample_rate"),
    ("LANGFUSE_TIMEOUT", "timeout_s"),
)

# 每请求新建的 langchain handler 工厂（可注入 fake：CI 无凭据、无外网，禁止真连云端）。
HandlerFactory = Callable[["ObservabilityConfig"], BaseCallbackHandler]
# SDK 客户端工厂（可注入 fake：行为用例不构造真客户端、不产生后台导出线程）。
ClientFactory = Callable[["ObservabilityConfig"], Any]


def _json_safe(value: Any) -> Any:
    """把掩码结果收敛到 JSON 可序列化类型（SDK 对 mask 返回值有此硬要求）。

    WHY 需要：`mask_text` 只处理字符串、`_mask_structure` 只深走 dict/list —— 其余类型
    原样放回，而 `datetime` / 自定义对象一旦进 OTel 属性就会在序列化环节失败
    （表现为整条 span 异常）。未知类型统一 `str()`：既保持可读，又不让观测因数据类型而断。
    """
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    return str(value)


def _mask_structure(value: Any) -> Any:
    """结构化掩码：**按键名**打码 + 对字符串走 C1 的正则兜底。

    WHY 不能只用 `shared.logging.mask_value`：它是按**值**逐个施正则的，而字段名规则
    依赖「键与值出现在同一段文本里」—— 一旦 payload 是 `{"user_access_token": "u-…"}`，
    值单独拎出来就没有任何字段名可依（只剩 `sk-`/`Bearer`/JWT 这类形态兜底能命中），
    **键级凭据会整批漏过去**。故此处补上键名判定（判定点与日志侧同一个
    `is_sensitive_field`，口径不分叉）。
    超掩码是刻意选择的方向：宁可把一个名叫 `token` 的普通字段打掉（安全），
    也不要放一个真 token 出去（不可逆）。
    """
    if isinstance(value, Mapping):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            masked[name] = MASKED if is_sensitive_field(name) else _mask_structure(item)
        return masked
    if isinstance(value, list | tuple | set | frozenset):
        return [_mask_structure(item) for item in value]
    if isinstance(value, str):
        return mask_text(value)
    return value


def mask_payload(*, data: Any = None, **kwargs: object) -> Any:
    """SDK 掩码协议实现：`mask(*, data, **kwargs)`，返回值必须 JSON 可序列化。

    作用面：SDK 创建 span 时的 `input` / `output` / `metadata`（LangChain handler 的 span
    正是经 `start_observation()` / `update()` 创建 → **对全链路生效**）。
    掩码键与正则复用 §C1 的同一套口径（`is_sensitive_field` / `mask_text`）——日志脱敏
    与云端脱敏分叉的那一侧就是泄漏点，飞书 token 绝不进 trace（CLAUDE.md 安全约束）。
    兜底语义与 SDK 一致：本函数自身若抛错，返回整车掩码占位，绝不把未掩码数据放出去。
    """
    try:
        return _json_safe(_mask_structure(data))
    except Exception as exc:  # 掩码失败 = 安全事件 → 丢弃内容，不冒泡
        logger.warning("观测掩码失败，按最严处理（整条内容掩码）：%s", exc)
        return FULLY_MASKED


# 传给 SDK 的是**模块级函数**而非绑定方法：SDK 以 `mask(data=data)` 关键字调用，
# 用未绑定函数可避免实例方法签名歧义（少一个「为什么这里要 self」的坑）。
_MASK_FUNCTION: Final = mask_payload


def _default_handler_factory(config: ObservabilityConfig) -> BaseCallbackHandler:
    """构造真的 Langfuse langchain handler（**每请求新建**）。

    WHY 每请求新建：官方文档对复用同一 handler 明确提示并发环境需谨慎
    （`last_trace_id` 等实例状态、内部 per-run 状态字典）；本项目是并发多会话
    （`thread_id == session_id`）→ 每请求一个。构造极轻（只取 SDK 单例 client）。

    WHY 传 `public_key`：多项目场景下不带 key 会拿到「禁用的客户端」以防串号；
    显式带上即确定性绑定到本项目的客户端（该客户端在装配期已注册为 SDK 单例）。

    WHY 延迟 import：`langfuse` 会连带拉起 OTel 栈，放模块顶层会让**未开观测**的
    进程也付这份导入代价（本项目默认关闭，导入期不该有观测开销）。
    """
    from langfuse.langchain import CallbackHandler

    return CallbackHandler(public_key=config.public_key.get_secret_value())


def _default_client_factory(config: ObservabilityConfig) -> Any:
    """构造真的 SDK 客户端（进程级唯一；SDK 侧亦有按 public_key 的单例语义）。

    全部参数**显式传**（不依赖 SDK 读 env 的时机与字符串比较）：`tracing_enabled=True`
    的参数侧语义是「不因 env 里出现非 "false" 的杂值而意外开/关」——SDK 侧
    `tracing_enabled = 参数 and env != "false"`，参数是确定性那一半。
    """
    from langfuse import Langfuse

    return Langfuse(
        public_key=config.public_key.get_secret_value(),
        secret_key=config.secret_key.get_secret_value(),
        base_url=config.base_url,
        environment=config.environment,
        release=config.release,
        sample_rate=config.sample_rate,
        timeout=config.timeout_s,
        tracing_enabled=True,
        mask=_MASK_FUNCTION,
    )


class ObservabilityConfig(BaseModel):
    """观测配置（由 `RuntimeSettings` 单向投影；观测侧的唯一配置入口）。

    `from_settings` 返回 `None` = **未配置**（关闭观测），这是「零回归」的唯一判据：
    关闭时既不建客户端、也不写 env、更不给图挂回调。
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    public_key: SecretStr = SecretStr("")
    secret_key: SecretStr = SecretStr("")
    base_url: str = DEFAULT_BASE_URL
    environment: str = "dev"
    release: str = ""
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout_s: int = Field(default=5, ge=1)
    flush_timeout_s: float = Field(default=5.0, gt=0.0)

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> ObservabilityConfig | None:
        """从 `RuntimeSettings` 构造；未启用或缺凭据 → `None`。

        WHY 缺凭据也判为「未配置」：SDK 对缺 key 的行为是**禁用客户端 + 告警**（不抛错），
        于是「开了开关但没配 key」会得到一个静默不工作的观测面；本项目直接归到
        「未配置」，语义与零回调一致，且不给对话链路挂一堆空转 handler。
        """
        if not settings.langfuse_enabled:
            return None
        if not settings.langfuse_public_key.get_secret_value() or (
            not settings.langfuse_secret_key.get_secret_value()
        ):
            logger.warning(
                "观测已开启但缺少 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY → 保持关闭"
                "（零回调；不落到 SDK 的「禁用客户端」静默态）"
            )
            return None
        environment = settings.langfuse_environment or settings.environment
        return cls(
            enabled=True,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            # 空串（compose 的 `${VAR:-}` 展开）视为未配置 → 回默认区域端点。
            base_url=settings.langfuse_base_url.strip() or DEFAULT_BASE_URL,
            environment=environment,
            # 始终显式给 release：不设时 SDK 会回退读常见 CI release 变量
            # （`get_common_release_envs`）→ 本地 trace 被 CI 变量串味。
            release=settings.langfuse_release or f"{settings.app_name}@{environment}",
            sample_rate=settings.langfuse_sample_rate,
            timeout_s=settings.langfuse_timeout_s,
            flush_timeout_s=settings.langfuse_flush_timeout_s,
        )

    def env_projection(self) -> dict[str, str]:
        """要写进 `os.environ` 的 `{env 名: 字符串值}`（名字见 `_ENV_KEYS`）。"""
        values: dict[str, str] = {}
        for env_name, attr in _ENV_KEYS:
            value = getattr(self, attr)
            if isinstance(value, SecretStr):
                values[env_name] = value.get_secret_value()
            elif isinstance(value, bool):
                values[env_name] = "true" if value else "false"
            else:
                values[env_name] = str(value)
        return values

    def describe(self) -> str:
        """一行可日志的装配摘要（**不含任何密钥**）。"""
        return (
            f"base_url={self.base_url} environment={self.environment} "
            f"release={self.release} sample_rate={self.sample_rate}"
        )


class Observability:
    """观测句柄：持有 SDK 客户端，产出 langchain 回调与归因元数据。

    生命周期：`install_observability` 装配（进程级单例）→ `AgentRuntime` 持有 →
    `aclose()` flush + shutdown。关闭后再取回调得空列表（fail-safe：不会拿到绑在
    已关闭客户端上的 handler）。
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        client: Any = None,
        handler_factory: HandlerFactory | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._handler_factory = handler_factory or _default_handler_factory
        self._closed = False

    # —— 装配 ——

    @classmethod
    def from_config(
        cls,
        config: ObservabilityConfig,
        *,
        handler_factory: HandlerFactory | None = None,
        client_factory: ClientFactory | None = None,
    ) -> Observability:
        """按已解析配置装配（投影 env → 建客户端）。

        `handler_factory` / `client_factory` 供测试注入 fake（CI 无凭据、无外网：
        真 handler 一旦产生 span 就会走后台导出线程去连云端）。
        """
        _project_env(config)
        build = client_factory or _default_client_factory
        client = build(config)
        logger.info("观测已装配：%s", config.describe())
        return cls(config, client=client, handler_factory=handler_factory)

    @classmethod
    def from_settings(
        cls,
        settings: RuntimeSettings,
        *,
        handler_factory: HandlerFactory | None = None,
        client_factory: ClientFactory | None = None,
    ) -> Observability | None:
        """未配置 → `None`（零回调）；已配置 → 装配句柄。"""
        config = ObservabilityConfig.from_settings(settings)
        if config is None:
            return None
        return cls.from_config(
            config, handler_factory=handler_factory, client_factory=client_factory
        )

    # —— 契约（plan §5.2）——

    @property
    def config(self) -> ObservabilityConfig:
        """当前观测配置（只读冻结）。"""
        return self._config

    @property
    def closed(self) -> bool:
        """是否已 aclose（关闭后不再产出回调）。"""
        return self._closed

    def langchain_callbacks(self) -> list[BaseCallbackHandler]:
        """本次请求要挂到 graph config 上的回调（每调用一次即新建 handler）。

        关闭/已 aclose → `[]`（调用方据此**不写** `callbacks` 键，保持逐字零回归）。
        """
        if self._closed:
            return []
        return [self._handler_factory(self._config)]

    def trace_metadata(
        self,
        *,
        session_id: str,
        user_id: str,
        trace_id: str | None = None,
        name: str = TRACE_NAME,
    ) -> dict[str, object]:
        """归因元数据：会话 / 用户 / trace 名（+ 本地 trace_id 关联键）。

        `thread_id == session_id` 契约直接成为 Langfuse 的 `session_id`；`user_id` 与
        记忆 / 飞书同一身份口径 —— 三处身份一致才谈得上「按人回溯」。
        `name` 为 trace 展示名：对话链路用默认 `agent.chat`；带外保存（O3）传
        `memory.out_of_band_save`，其余归因键保持与对话 trace 同值以便对齐。
        """
        metadata: dict[str, object] = {
            _META_SESSION: session_id,
            _META_USER: user_id,
            _META_TRACE_NAME: name,
        }
        if trace_id:
            metadata[_META_REQUEST_TRACE] = trace_id
        return metadata

    def mask(self, *, data: Any = None, **kwargs: object) -> Any:
        """掩码入口（与传给 SDK 的 `mask=` 同一实现；此处保留实例方法以满足契约）。"""
        return mask_payload(data=data, **kwargs)

    async def aclose(self) -> None:
        """flush（带超时）+ shutdown；幂等，**任何失败只记日志**。

        WHY 显式 flush：`shutdown()` 会 flush，但容器 SIGTERM 下 `atexit` 不保证执行，
        退出前必须主动把缓冲区里的 span 推出去（SDK docstring：flush 只保证「已送达 API」，
        服务端入库是**异步**的，可能 15–30s 后才可查询 —— 验收脚本不得 flush 后立刻断言）。
        WHY 带超时：云端不可达时退出路径不得挂住进程（观测是外挂能力，不能拖住关停）。
        """
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._client.flush), timeout=self._config.flush_timeout_s
            )
        except Exception as exc:  # 超时/网络/取消之外的任何异常都不冒泡
            logger.warning("观测 flush 未完成（不影响对话，span 可能丢弃）：%s", exc)
        try:
            self._client.shutdown()
        except Exception as exc:
            logger.warning("观测客户端关闭失败：%s", exc)


def _project_env(config: ObservabilityConfig) -> None:
    """把配置投影进 `os.environ` —— **本模块是全项目唯一写观测 env 的地方**。

    WHY 必须写 env 而不只是传构造参数：SDK 在**客户端首次构造**时读环境变量，且 env 读取
    带 `lru_cache` 语义（同一进程内改动不再生效）；任何绕过本函数的 `get_client()` 调用
    （第三方集成、SDK 内部）都会按 env 兜底 —— 投影一次即让两条路径同值。
    注意密钥随之进入进程环境（与 `.env` → 环境变量的既有口径一致），日志侧不受影响。
    """
    for env_name, value in config.env_projection().items():
        os.environ[env_name] = value


# 进程级单例：观测是**进程级**能力（一个客户端、一个后台导出线程），
# 由 `AgentRuntime.create` 装配、`AgentRuntime.aclose` 关闭。
_installed: Observability | None = None


def install_observability(
    settings: RuntimeSettings, *, handler_factory: HandlerFactory | None = None
) -> Observability | None:
    """装配/复用进程级观测句柄；未配置 → `None`。

    已关闭的旧句柄**不复用**（shutdown 后的客户端不再接收 span，复用等于静默失去观测），
    此时按当前 settings 重新装配 —— 与「改配置需重启进程」并不冲突：重启后就是新进程。
    """
    global _installed
    if _installed is not None and not _installed.closed:
        return _installed
    _installed = Observability.from_settings(settings, handler_factory=handler_factory)
    return _installed


def reset_observability() -> None:
    """丢弃进程级句柄（**测试隔离用**；生产中观测随进程存活，不重置）。

    不关闭 SDK 客户端 —— 句柄关闭的显式路径是 `Observability.aclose()`（由 runtime 驱动）。
    """
    global _installed
    _installed = None


__all__ = [
    "DEFAULT_BASE_URL",
    "FULLY_MASKED",
    "OUT_OF_BAND_TRACE_NAME",
    "TRACE_NAME",
    "ClientFactory",
    "HandlerFactory",
    "Observability",
    "ObservabilityConfig",
    "install_observability",
    "mask_payload",
    "reset_observability",
]
