"""根级共享日志底座（api / agent / services 共用，**单点配置**）。

WHY 放 `shared/`：与 `shared/embeddings.py` 同构 —— 「配置模型 + 唯一配置入口」，
供 API 进程与四个 MCP 服务进程共用同一套 formatter / 脱敏规则 / 事件白名单；
`settings.py` 与各服务 `config.py` 只做「读 env → 转 LoggingConfig → 调 configure_logging」，
不再各自 `logging.basicConfig`（格式必须一致，否则 `trace_id` 在服务之间就断了）。

设计四件套（对应 docs/plan-docker-observability.md §4 Phase C）：

1. **统一 formatter**：`text`（人眼可读、带 `trace=… session=… user=…` 关联前缀，
   可直接 `grep trace_id` 串起 api→agent→MCP 全链路）/ `json`（一行一条 JSON，
   供容器日志采集；键序固定 = 字段顺序，便于肉眼比对）；
2. **ContextVar 关联标识**：`trace_id` / `session_id` / `user_id` 经 `Filter` 注入**每一条**
   记录（含第三方库的日志），无需在调用点手写前缀；
3. **脱敏 Filter**：密钥 / Bearer / token / 手机号 / 邮箱正则兜底（安全硬约束：
   任何路径都不得把凭据写进日志）；脱敏在 formatter **之外**完成，两种格式同时受保护；
4. **LogEvent 结构化事件**（Pydantic 白名单）：事件名固定、字段固定 ——
   消息体只记 `len` + `sha256[:8]`（`summarize_text`），禁止原文进日志。

职责边界：本模块只管「格式化 / 关联 / 脱敏 / 事件契约」，**不做持久化** ——
落库（`interaction_events`）由消费方经 `subscribe_events` 注册监听器实现，
故本模块零 DB 依赖、可被任意进程安全导入（含 docker 内无 Postgres 的 MCP 容器）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

# —— 日志级别（与 settings / 各服务 config 的 Literal 口径一致）——
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# —— 环境变量键名（**仅本模块读 env**：配置唯一入口是 configure_logging）——
LOG_LEVEL_ENV: Final = "LOG_LEVEL"
LOG_FORMAT_ENV: Final = "LOG_FORMAT"
LOG_SERVICE_ENV: Final = "LOG_SERVICE"

# —— text 格式模板：关联前缀紧贴级别，一眼看出「哪条链路、哪个会话、哪个用户」——
# 反注入：字段值经 `_safe_token` 净化（只留字母数字与 `-_.@:`），用户 id 里的
# 伪造换行/`[` 无法伪装出新的日志行（伪造日志行是真实攻击面，不是理论洁癖）。
TEXT_FORMAT: Final = (
    "%(asctime)s %(levelname)-7s [%(name)s]%(ctx)s %(message)s%(durations)s%(event)s"
)
# 缺省格式：只换 formatter，模板口径不变（`fmt` 参数仅在测试/特化场景使用）。
TEXT_DATEFMT: Final = "%Y-%m-%d %H:%M:%S"

# —— 脱敏占位符 ——
MASKED: Final = "***"

# —— 事件白名单字段名（LogEvent 的模型字段；extra 里出现同名键视为实现错误）——
_EVENT_FIELD_NAMES: Final = frozenset(
    {
        "event",
        "level",
        "service",
        "trace_id",
        "session_id",
        "user_id",
        "status",
        "code",
        "duration_ms",
        "tool_name",
        "fields",
    }
)

# 关联标识字段名：formatter 从 record 上按这些名字取值（由 Filter 统一注入）。
_CONTEXT_FIELD_NAMES: Final = ("trace_id", "session_id", "user_id")

# 可从 `fields` 提升为事件顶层字段的名字（见 log_event 的 WHY）。
_PROMOTABLE_FIELDS: Final = (
    "trace_id",
    "session_id",
    "user_id",
    "status",
    "code",
    "duration_ms",
    "tool_name",
)

# 关联标识约束：request id / session id / user id 均为短串；超长/带控制字符者
# 直接丢弃（污染日志的输入不该被原样写进去）。48 字符覆盖 UUID + 前缀。
_CONTEXT_ID_MAX_CHARS: Final = 48

_TAG_RE: Final = re.compile(r"[^A-Za-z0-9_.@:-]")

# —— 脱敏规则（有序：先特化后一般；`(?!)` 组回填 key，保留可读性）——

# ① 已知字段名 + 值：JSON（`"k": "v"`）与 key=value（`k=v` / `k: v`）两种书写。
#    值边界刻意保守（`[^\s"',;)}\]]*`）：宁可少吞几个字符，也不跨键吃掉正文。
_FIELD_MASK_RE: Final = re.compile(
    r"(?i)(?P<key>"
    r"api[_-]?key|authorization|proxy[_-]?authorization|access[_-]?token|"
    r"refresh[_-]?token|user[_-]?access[_-]?token|tenant[_-]?access[_-]?token|"
    r"lark[_-]?token[_-]?key|client[_-]?secret|app[_-]?secret|secret|password|passwd|"
    r"credential|private[_-]?key|session[_-]?key"
    r")(?P<sep>[\"']?\s*[:=]\s*[\"']?)"
    # 认证方案前缀（`Bearer ` / `Basic `）随值一起吞掉：否则空格会把值截成
    # 「Bearer」，真正的 token 反而留在日志里（最危险的半脱敏）。
    r"(?P<scheme>(?:bearer|basic)\s+)?"
    r"(?P<value>[^\s\"',;)}\]]*)"
)


def _mask_field_match(match: re.Match[str]) -> str:
    key, sep, value = match.group("key"), match.group("sep"), match.group("value")
    # 空值（`"api_key": ""`）保持原样：打码空值只会产生噪音。
    if not value:
        return match.group(0)
    return f"{key}{sep}{match.group('scheme') or ''}{MASKED}"


# ② 具体凭据形态（无字段名可依时兜底）。
_SHAPE_MASK_RES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("token", re.compile(r"(?i)\b(?:sk|pk|rk|gh[pous]|xox[baprs])[-_][A-Za-z0-9_-]{8,}")),
    ("bearer", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")),
    # 邮箱 / 中国大陆手机号：脱敏是合规底线（日志不得沉淀可识别个人信息）。
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
)

# 标准 LogRecord 自有属性名：计算「调用方额外携带的字段」时须排除。
_STANDARD_RECORD_ATTRS: Final[frozenset[str]] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}

# 事件：结构化事件的统一契约 —— 事件名是**产品级接口**（日志检索、事件表、
# 指标三者同名同义），故以常量声明，禁止在调用点散落字符串字面量。
EVENT_REQUEST_RECEIVED: Final = "request.received"
EVENT_INTENT_CLASSIFIED: Final = "intent.classified"
EVENT_TOOL_CALLED: Final = "tool.called"
EVENT_ANSWER_GENERATED: Final = "answer.generated"
EVENT_MEMORY_SAVED: Final = "memory.saved"
EVENT_REQUEST_FINISHED: Final = "request.finished"

# 事件类日志的统一 logger 名：事件是「带结构化字段的日志行」，跨进程同名，
# 便于按 `logger == srp_agent.event` 过滤出全部结构化事件（与业务日志区分）。
EVENT_LOGGER_NAME: Final = "srp_agent.event"


class LogFormat(StrEnum):
    """日志输出格式：text = 人眼可读（默认，带关联前缀）；json = 一行一条 JSON。"""

    TEXT = "text"
    JSON = "json"


class ServiceName(StrEnum):
    """进程身份（事件字段与 `service` 标签的取值域）。

    用于 `event.service`：一条日志/事件属于哪个进程，是排查的第一步
    （「api 说有、MCP 说没有」比「不知道谁说的」有用得多）。
    """

    API = "api"
    TOOLS_MCP = "tools_mcp"
    LARK_MCP = "lark_mcp"
    A2A_MCP = "a2a_mcp"


class LoggingConfig(BaseModel):
    """日志配置（env `LOG_LEVEL` / `LOG_FORMAT` / `LOG_SERVICE`，装配层经 `from_env` 注入）。

    WHY 独立配置模型而非直接吃 `RuntimeSettings`：MCP 服务（`services/*/config.py`）与
    api 的 settings 是两套 BaseSettings（刻意解耦、可独立打包），本底座若依赖其中之一
    就会把服务镜像拖上整棵依赖树。两者都只做「读 env → LoggingConfig → configure_logging」。
    """

    service: str = "srp-agent"  # 进程身份：api / tools_mcp / lark_mcp / a2a_mcp
    level: LogLevel = "INFO"
    log_format: LogFormat = LogFormat.TEXT
    mask_enabled: bool = True  # 脱敏总开关；关闭仅供「确认脱敏确实在生效」的对照实验

    @field_validator("log_format", mode="before")
    @classmethod
    def _validate_log_format(cls, value: Any) -> Any:
        """格式值规整：未知值回退 text（日志配置拼错不得让进程起不来）。"""
        return _coerce_format(value)

    @classmethod
    def from_settings(cls, settings: Any, *, service: str | None = None) -> LoggingConfig:
        """由运行配置投影（`RuntimeSettings` 或各 MCP 服务 settings）。

        `service` 优先取显式入参（入口层最清楚自己是谁）；缺省回退 env `LOG_SERVICE`，
        再回退配置对象里的 `app_name`（api 侧用），最后是默认值。
        level / format 同样「配置对象优先、env 兜底」—— BaseSettings 已把 env 合并进对象，
        故两个来源天然一致（直接构造配置对象时仍能读到 env）。
        """
        return cls(
            service=(
                service
                or os.getenv(LOG_SERVICE_ENV)
                or str(getattr(settings, "app_name", "") or "")
                or "srp-agent"
            ),
            level=getattr(settings, "log_level", None) or os.getenv(LOG_LEVEL_ENV, "INFO"),
            log_format=getattr(settings, "log_format", None) or os.getenv(LOG_FORMAT_ENV, ""),
        )


class LogEvent(BaseModel):
    """结构化事件（**白名单字段**：模型没声明的键进不来，扩字段必须改这里）。

    事件是「可观测 + 可落库」的统一载体：`fields` 承载事件特有属性（如 `tool_name`
    的实参摘要、`answer` 的长度与哈希），`summary()` 输出**扁平** dict（与
    `interaction_events` 表的列 + payload 一一对应，避免两条口径）。

    `extra="forbid"`：没声明的字段直接报错而非静默丢弃 —— 日志结构漂移是排查事故时
    最贵的坑（以为记了、其实没记），宁可在开发期就炸出来。
    """

    model_config = ConfigDict(extra="forbid")

    event: str
    # 事件唯一标识：由生成端（本进程）分配，写入事件表作为主键。用 uuid4 而非自增：
    # 多 worker / 多副本各自无协调地写同一张表，是「多 worker 语义正确性」的前提。
    id: str = Field(default_factory=lambda: str(uuid4()))
    level: LogLevel = "INFO"
    service: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    status: str | None = None
    code: str | None = None
    duration_ms: int | None = None
    tool_name: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """扁平摘要：`fields` 平铺进顶层（同名冲突以 fields 为准，事件属性更具体）。

        不含 `id`（时间语义之外的实现细节）：摘要供「按语义 grep / 聚合」，id 只在
        落库时作为主键使用；把它塞进摘要会让每个消费方都要跳过它。
        """
        data: dict[str, Any] = {
            "event": self.event,
            "level": self.level,
            "service": self.service,
            "timestamp": self.ts.isoformat(),
        }
        for name in ("trace_id", "session_id", "user_id", "status", "code", "tool_name"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        data.update(self.fields)
        return data


# 事件监听器（落库等消费方注册；**进程内**，与配置生命周期一致）。
EventListener = Any  # Callable[[LogEvent], None]，用 Any 避免 typing 噪音（单点使用）

_listeners: list[EventListener] = []
# 监听器重入哨兵：监听器内部再打日志时不得二次分发（否则失败路径自我递归）。
_in_listener: ContextVar[bool] = ContextVar("srp_log_in_listener", default=False)


# —— 关联标识（ContextVar：跨 async 任务自动继承，无需层层传参）——

_trace_id: ContextVar[str | None] = ContextVar("srp_trace_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("srp_session_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("srp_user_id", default=None)


def current_trace_id() -> str | None:
    """当前上下文的链路 id（无则 None；供 X-Request-Id 回写响应头）。"""
    return _trace_id.get()


def current_session_id() -> str | None:
    """当前上下文的会话 id。"""
    return _session_id.get()


def current_user_id() -> str | None:
    """当前上下文的用户 id。"""
    return _user_id.get()


def bind_context(
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[Any, ...]:
    """绑定关联标识，返回 token 元组供 `unbind_context` 还原（**成对使用**）。

    语义是「设置/清除」而非「合并」：显式传 None = 清除该维度 —— 请求之间必须彻底
    复位，否则上一个用户的 `user_id` 会跟着 ContextVar 继承到下一个请求的日志里
    （跨用户串号，比丢字段严重得多）。
    """
    return (
        _trace_id.set(_sanitize_context_id(trace_id)),
        _session_id.set(_sanitize_context_id(session_id)),
        _user_id.set(_sanitize_context_id(user_id)),
    )


def unbind_context(tokens: tuple[Any, ...]) -> None:
    """还原 `bind_context` 返回的 token（逆序 reset，异常路径也安全）。"""
    for var, token in zip((_trace_id, _session_id, _user_id), tokens, strict=True):
        var.reset(token)


def clear_context() -> None:
    """清空全部关联标识（无 token 的场景：进程收尾 / 测试夹具）。"""
    _trace_id.set(None)
    _session_id.set(None)
    _user_id.set(None)


# —— 脱敏 ——


def mask_text(text: str) -> str:
    """按正则规则脱敏一段文本（密钥 / Bearer / token / 邮箱 / 手机号）。

    WHY 正则兜底而非「只在已知字段上打码」：日志里的敏感值来源不可穷举
    （异常消息里内嵌的 URL、第三方库直接打的 header、模型回显的内容），
    必须在**出站路径**统一兜底；字段名规则的值为空串时不动（避免把键名打坏）。
    """
    masked = _FIELD_MASK_RE.sub(_mask_field_match, text)
    for _, pattern in _SHAPE_MASK_RES:
        masked = pattern.sub(MASKED, masked)
    return masked


def mask_value(value: Any) -> Any:
    """递归脱敏任意 JSON 结构（dict / list / str 深走；其余类型原样返回）。

    WHY tuple 必须保持 tuple：`logging` 的 `%` 拼装要求 `record.args` 是 tuple
    （list 会直接抛 `TypeError: not all arguments converted`）—— 脱敏改写把
    `("a", 1)` 变成 `["a", 1]` 就会让**每一条带参日志**炸掉。故按容器类型分别还原。
    """
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, dict):
        return {key: mask_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(mask_value(item) for item in value)
    if isinstance(value, list):
        return [mask_value(item) for item in value]
    return value


def _sanitize_context_id(value: str | None) -> str | None:
    """关联标识净化：去控制字符、限长；净化后为空则视为未提供。"""
    if not value:
        return None
    cleaned = _TAG_RE.sub("", str(value))[:_CONTEXT_ID_MAX_CHARS]
    return cleaned or None


# —— 消息体摘要（消息原文禁止进日志）——


def summarize_text(text: Any, *, head: int = 8) -> str:
    """把任意文本摘成 `len=.. sha256=..`（**只记长度与哈希前缀，不记原文**）。

    这是本项目「消息体可观测但不可还原」的统一口径：既能按 sha256 前缀把
    「同一句话」在 api → agent → MCP 之间对齐，又不把用户原文沉淀进日志/事件表
    （CLAUDE.md 安全约束：禁止会话明文内容中的敏感字段落盘）。
    """
    raw = "" if text is None else str(text)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:head]
    return f"len={len(raw)} sha256={digest}"


# —— formatter 视图 ——


class _LogView:
    """formatter 取值用的只读视图：LogRecord + 关联标识 + 事件/耗时缺省值。

    WHY 不直接往 LogRecord 上挂属性：`%(duration_ms)s` 这类占位符在**未携带**该字段的
    记录上会 KeyError（logging 的 `record.__dict__` 查找语义），而日志绝不因缺字段失败。
    视图统一把缺失值折成 None，模板再渲染成空串。
    """

    __slots__ = ("asctime", "code", "duration_ms", "event", "extra", "levelname", "name", "service")

    def __init__(self, record: logging.LogRecord) -> None:
        self.name = record.name
        self.levelname = record.levelname
        self.asctime = _format_ts(record.created)
        self.event = getattr(record, "event", None)
        self.duration_ms = getattr(record, "duration_ms", None)
        self.code = getattr(record, "code", None)
        self.service = getattr(record, "_service", None) or ""
        self.extra = _extra_fields(record)


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """取调用方经 `extra=` 携带的自定义字段（排除标准属性与关联标识）。"""
    extra: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _STANDARD_RECORD_ATTRS or key in _CONTEXT_FIELD_NAMES:
            continue
        if key.startswith("_"):
            continue
        extra[key] = value
    return extra


def _format_ts(created: float) -> str:
    """时间戳（本地时区，含毫秒）：跨服务按同一机器时区对齐，肉眼可直接比较。"""
    return datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _join_context(view: _LogView, record: logging.LogRecord) -> str:
    """渲染关联前缀 `[trace=… session=… user=…]`；全空则整体省略（不产生空括号噪音）。"""
    parts = [
        f"{name.split('_')[0]}={_safe_token(getattr(record, name, None))}"
        for name in _CONTEXT_FIELD_NAMES
        if getattr(record, name, None)
    ]
    return f" [{' '.join(parts)}]" if parts else ""


def _safe_token(value: Any) -> str:
    """标签值净化：只保留 `[A-Za-z0-9-_.]`（防伪造换行/括号污染日志行结构）。"""
    return _TAG_RE.sub("", str(value))


class ContextFilter(logging.Filter):
    """给每条记录补齐 service 与关联标识（**Filter 早于 formatter 执行**）。

    过滤：把 `extra=` 里的关联标识收敛为净化值（外部传入的 header 值不可信）；
    补齐：`_service` / record 上的 trace/session/user —— 第三方库（uvicorn / httpx /
    langchain）打的日志同样带上，这正是「一条 trace_id 串起全链路」的前提。
    """

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record._service = self.service  # 自定义字段，formatter 按名读取
        for name in _CONTEXT_FIELD_NAMES:
            var = {"trace_id": _trace_id, "session_id": _session_id, "user_id": _user_id}[name]
            value = getattr(record, name, None) or var.get()
            setattr(record, name, _sanitize_context_id(value))
        return True


class SensitiveFilter(logging.Filter):
    """脱敏过滤：改写 `record.msg` / `record.args` / `extra`（格式化之后无法补救）。

    WHY 必须在 Filter 阶段做：formatter 拿到的是已渲染文本，改它就得为每种格式各写
    一遍；此处改写源字段，text/json 两种 formatter 同时受保护，且 `record.args`
    改写后 `getMessage()` 的 `%s` 拼装自然带掩码。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_text(record.msg)
        if record.args:
            record.args = mask_value(record.args)  # type: ignore[assignment]
        for key, value in list(record.__dict__.items()):
            if key in _STANDARD_RECORD_ATTRS or key in _CONTEXT_FIELD_NAMES:
                continue
            if key.startswith("_"):
                continue
            record.__dict__[key] = mask_value(value)
        return True


class TextFormatter(logging.Formatter):
    """text 格式：人眼可读 + 关联前缀；事件/耗时按需追加（无则不留空占位）。

    实现取向：**预计算片段 + 字符串拼接**而非纯 `%` 模板 —— `asctime`/关联标识/事件
    都是「有则显示」的可选段落，用模板占位符会在缺字段时留下多余分隔符（`[] () -`），
    排版立刻变脏；这里按段落拼接，任一维度缺失都不影响其余段落。
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt="%(message)s", datefmt=datefmt or TEXT_DATEFMT)
        self._fmt = fmt or TEXT_FORMAT

    def format(self, record: logging.LogRecord) -> str:
        view = _LogView(record)
        # 逐段拼装：时间 级别 [logger][关联] 消息 (耗时) event=…
        segments = [
            view.asctime,
            f"{record.levelname:<7}",
            f"[{record.name}]",
        ]
        ctx = _join_context(view, record)
        if ctx:
            segments.append(ctx.lstrip())
        segments.append(super().format(record))
        if view.duration_ms is not None:
            segments.append(f"({view.duration_ms}ms)")
        if view.event:
            segments.append(f"event={view.event}")
            if view.code:
                segments.append(f"code={view.code}")
        return " ".join(segment for segment in segments if segment)

    def formatException(self, ei: Any) -> str:  # 覆写 stdlib 命名
        """异常 traceback：缩进一层并去掉重复的 `Traceback` 噪音，贴齐日志块。"""
        return "\n" + super().formatException(ei)


class JsonFormatter(logging.Formatter):
    """json 格式：一行一条 JSON（键序固定），供容器日志采集与 grep。

    字段：`ts/level/logger/service/trace_id/session_id/user_id/event/code/duration_ms/
    message/extra`（None 值省略）；异常走 `exc`；`extra` 里的键以调用方为准。
    """

    def format(self, record: logging.LogRecord) -> str:
        view = _LogView(record)
        payload: dict[str, Any] = {
            "ts": view.asctime,
            "level": record.levelname,
            "logger": record.name,
        }
        if view.service:
            payload["service"] = view.service
        for name in _CONTEXT_FIELD_NAMES:
            value = getattr(record, name, None)
            if value:
                payload[name] = value
        for name, value in (("event", view.event), ("code", view.code)):
            if value:
                payload[name] = value
        if view.duration_ms is not None:
            payload["duration_ms"] = view.duration_ms
        payload["message"] = mask_text(record.getMessage())
        if view.extra:
            payload["extra"] = view.extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# —— 配置入口 ——


# —— 输出流 ——


class _DynamicStdout:
    """动态 stdout 代理：每次写入时解析 `sys.stdout`（而非在配置时绑死对象）。

    WHY 不直接持有 `sys.stdout`：`logging.StreamHandler` 会把流对象存成字段，一旦
    运行期替换了 `sys.stdout`（测试捕获、服务热重载、容器内日志重定向），日志会继续
    写向**旧的**流 —— 表现为「日志凭空消失」或绕过采集。代理只转发需要的两个方法。
    """

    def write(self, message: str) -> int:
        return sys.stdout.write(message)

    def flush(self) -> None:
        sys.stdout.flush()


def configure_logging(
    config: LoggingConfig | None = None,
    *,
    fmt: str | None = None,
    force: bool = False,
    stream: Any = None,
) -> LoggingConfig:
    """进程级单点配置：装 formatter / 过滤器到 root（幂等，重复调用不叠加 handler）。

    参数：
    - `config`：缺省 `LoggingConfig()`（service=srp-agent / INFO / text）；
    - `fmt`：显式覆盖 text 模板（测试与特化场景；json 格式忽略）；
    - `force=True`：先摘除 root 现有 handler（**容器/测试**：清掉 uvicorn 等
      预设 handler 造成的双份输出）；
    - `stream`：输出流（缺省 **动态解析 sys.stdout** —— 容器内一切走 stdout，
      聚合交 compose；动态解析而非绑死对象，是为了让输出跟随运行期的 stdout 重定向，
      否则日志会绕过捕获/重定向直接写到进程原始 stdout）。

    返回生效的 `LoggingConfig`，便于入口层记录「以什么形态起的服务」。
    """
    cfg = config or LoggingConfig()
    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    level = logging.getLevelNamesMapping()[cfg.level]
    root.setLevel(level)

    handler = logging.StreamHandler(stream or _DynamicStdout())
    handler.setLevel(level)
    handler.setFormatter(
        JsonFormatter() if cfg.log_format is LogFormat.JSON else TextFormatter(fmt=fmt)
    )
    # 过滤器装到 handler 上（而非 logger）：只作用于本 handler 的输出，
    # 不会因为根 logger 上还有第三方 handler 而重复改写同一 record。
    handler.addFilter(ContextFilter(cfg.service))
    if cfg.mask_enabled:
        handler.addFilter(SensitiveFilter())

    # 幂等判定：已有同名同格式 handler（如重复 configure）则替换为最新配置，
    # 而不是叠加 —— 叠加会让每条日志打两遍（容器里立刻可见）。
    for existing in list(root.handlers):
        if getattr(existing, "_srp_logging", False):
            root.removeHandler(existing)
    handler._srp_logging = True  # type: ignore[attr-defined]  # 标记本底座自装 handler
    root.addHandler(handler)

    # uvicorn / fastmcp 的 access/error logger 自带 handler 时会绕过 root 配置
    # （表现为「应用日志有 trace_id、访问日志没有」）→ 统一交给 root 处理。
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastmcp"):
        lib_logger = logging.getLogger(name)
        lib_logger.handlers = []
        lib_logger.propagate = True
        lib_logger.setLevel(level)
    return cfg


# —— 结构化事件 ——


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: LogLevel = "INFO",
    status: str | None = None,
    code: str | None = None,
    duration_ms: int | None = None,
    tool_name: str | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    message: str = "",
    fields: dict[str, Any] | None = None,
) -> LogEvent:
    """发一条结构化事件（白名单字段 + 可选人类可读 message），并分发给监听器。

    关联标识缺省取当前 ContextVar（中间件/装配点绑定一次，全链路自动带上）；
    显式传参用于「不在请求上下文里」的场景（如装配期、带外任务收尾）。

    `fields` 里的**事件标准字段**（trace_id/session_id/user_id/status/code/duration_ms/
    tool_name）会被提升为事件顶层字段，而不是塞进 `fields` 子字典 —— 这样事件表列
    与事件属性始终一一对应（否则同一语义会有「顶层列」与「payload 里的键」两套口径，
    聚合统计必然对不上）。便捷函数（`log_memory_saved` 等）因此不必逐个声明这些参数。

    监听器（落库）失败**只记日志、不上抛** —— 可观测性绝不反噬主链路。
    """
    extra_fields = dict(fields or {})
    promoted = {name: extra_fields.pop(name) for name in _PROMOTABLE_FIELDS if name in extra_fields}
    payload = LogEvent(
        event=event,
        level=level,
        service=_service_name(),
        trace_id=_sanitize_context_id(trace_id or promoted.get("trace_id")) or _trace_id.get(),
        session_id=_sanitize_context_id(session_id or promoted.get("session_id"))
        or _session_id.get(),
        user_id=_sanitize_context_id(user_id or promoted.get("user_id")) or _user_id.get(),
        status=status or promoted.get("status"),
        code=code or promoted.get("code"),
        duration_ms=duration_ms if duration_ms is not None else promoted.get("duration_ms"),
        tool_name=tool_name or promoted.get("tool_name"),
        fields=extra_fields,
    )
    extra: dict[str, Any] = {"event": event}
    if code:
        extra["code"] = code
    if duration_ms is not None:
        extra["duration_ms"] = duration_ms
    logger.log(logging.getLevelNamesMapping()[level], message, extra=extra)
    _dispatch(payload)
    return payload


def subscribe_events(listener: EventListener) -> None:
    """注册事件监听器（落库/指标等消费方；同一对象重复注册只生效一次）。"""
    if listener not in _listeners:
        _listeners.append(listener)


def unsubscribe_events(listener: EventListener) -> None:
    """注销事件监听器（幂等；未注册时无操作）。"""
    if listener in _listeners:
        _listeners.remove(listener)


def listener_count() -> int:
    """当前已注册的监听器数量（测试断言「订阅未泄漏」；事件监听是**进程级全局**状态，
    泄漏会让后一个用例的事件写进前一个用例的库，表现为「莫名多出别的事件」）。"""
    return len(_listeners)


def _service_name() -> str:
    """当前进程的 service 名（取首个自装 handler 的配置，未配置则为空串）。"""
    for handler in logging.getLogger().handlers:
        if getattr(handler, "_srp_logging", False):
            for flt in handler.filters:
                if isinstance(flt, ContextFilter):
                    return flt.service
    return ""


def _dispatch(event: LogEvent) -> None:
    """把事件分发给监听器；监听器抛错只告警（且不触发二次分发）。"""
    if not _listeners or _in_listener.get():
        return
    token = _in_listener.set(True)
    try:
        for listener in list(_listeners):
            try:
                listener(event)
            except Exception as exc:  # 可观测性不得反噬主链路
                logging.getLogger(__name__).warning("日志事件监听器失败：%s", exc)
    finally:
        _in_listener.reset(token)


def _coerce_format(value: Any) -> LogFormat:
    """格式值规整：接受 `LogFormat` / 字符串；未知值回退 text（不因拼错而静默丢日志）。"""
    if isinstance(value, LogFormat):
        return value
    try:
        return LogFormat(str(value).strip().lower())
    except ValueError:
        return LogFormat.TEXT


# —— 事件便捷函数（调用点一行、字段口径统一；ToolCallRecord 等语义见 Phase C §4）——


def log_request_received(logger: logging.Logger, *, source: str, **fields: Any) -> LogEvent:
    """`request.received`：请求到达（method/path/source；不含任何请求体原文）。"""
    return log_event(
        logger, EVENT_REQUEST_RECEIVED, status="received", fields={"source": source, **fields}
    )


def log_intent_classified(
    logger: logging.Logger, *, intent: str, confidence: float, **fields: Any
) -> LogEvent:
    """`intent.classified`：意图判定结果（intent + 置信度 + 判定依据摘要）。"""
    return log_event(
        logger,
        EVENT_INTENT_CLASSIFIED,
        status=intent,
        fields={"intent": intent, "confidence": round(float(confidence), 4), **fields},
    )


def log_tool_called(
    logger: logging.Logger,
    *,
    tool_name: str,
    status: str,
    duration_ms: int | None = None,
    code: str | None = None,
    **fields: Any,
) -> LogEvent:
    """`tool.called`：一次工具调用（耗时/成败/错误码；**实参不落原文**，由调用方摘要）。"""
    level: LogLevel = "INFO" if status == "ok" else "WARNING"
    return log_event(
        logger,
        EVENT_TOOL_CALLED,
        level=level,
        status=status,
        code=code,
        duration_ms=duration_ms,
        tool_name=tool_name,
        fields=fields,
    )


def log_answer_generated(
    logger: logging.Logger,
    *,
    answer: str,
    finished_reason: str,
    duration_ms: int | None = None,
    **fields: Any,
) -> LogEvent:
    """`answer.generated`：回答生成完成（**只记 len + sha256**；reason + 引用数）。"""
    return log_event(
        logger,
        EVENT_ANSWER_GENERATED,
        status=finished_reason,
        duration_ms=duration_ms,
        fields={"answer": summarize_text(answer), "finished_reason": finished_reason, **fields},
    )


def log_memory_saved(
    logger: logging.Logger,
    *,
    action: str,
    kind: str,
    memory_id: str,
    **fields: Any,
) -> LogEvent:
    """`memory.saved`：一条长期记忆落库（action=inserted/merged + kind + id）。"""
    return log_event(
        logger,
        EVENT_MEMORY_SAVED,
        status=action,
        fields={"action": action, "kind": kind, "memory_id": memory_id, **fields},
    )


def log_request_finished(
    logger: logging.Logger,
    *,
    status: str,
    duration_ms: int,
    code: str | None = None,
    **fields: Any,
) -> LogEvent:
    """`request.finished`：一轮交互收尾（终态 + 端到端耗时 + token 用量）。"""
    return log_event(
        logger,
        EVENT_REQUEST_FINISHED,
        level="INFO" if code is None else "ERROR",
        status=status,
        code=code,
        duration_ms=duration_ms,
        fields=fields,
    )


@dataclass(frozen=True)
class EventSnapshot:
    """测试/调试用的「已分发事件」快照条目（不参与生产路径）。"""

    event: str
    level: str
    service: str
    message: str
    fields: dict[str, Any]


class RecordingListener:
    """内存事件监听器（**测试专用**：断言事件序列/字段，零 DB 依赖）。"""

    def __init__(self) -> None:
        self.events: list[LogEvent] = []

    def __call__(self, event: LogEvent) -> None:
        self.events.append(event)

    def names(self) -> list[str]:
        """已记录的事件名序列（按分发顺序）。"""
        return [e.event for e in self.events]

    def find(self, name: str) -> list[LogEvent]:
        """按事件名筛选。"""
        return [e for e in self.events if e.event == name]

    def install(self) -> RecordingListener:
        """注册自身为监听器并返回（链式：`RecordingListener().install()`）。"""
        subscribe_events(self)
        return self

    def uninstall(self) -> None:
        """注销自身。"""
        unsubscribe_events(self)


__all__ = [
    "EVENT_ANSWER_GENERATED",
    "EVENT_INTENT_CLASSIFIED",
    "EVENT_LOGGER_NAME",
    "EVENT_MEMORY_SAVED",
    "EVENT_REQUEST_FINISHED",
    "EVENT_REQUEST_RECEIVED",
    "EVENT_TOOL_CALLED",
    "LOG_FORMAT_ENV",
    "LOG_LEVEL_ENV",
    "LOG_SERVICE_ENV",
    "MASKED",
    "TEXT_FORMAT",
    "ContextFilter",
    "EventSnapshot",
    "JsonFormatter",
    "LogEvent",
    "LogFormat",
    "LogLevel",
    "LoggingConfig",
    "RecordingListener",
    "SensitiveFilter",
    "ServiceName",
    "TextFormatter",
    "bind_context",
    "clear_context",
    "configure_logging",
    "current_session_id",
    "current_trace_id",
    "current_user_id",
    "log_answer_generated",
    "log_event",
    "log_intent_classified",
    "log_memory_saved",
    "log_request_finished",
    "log_request_received",
    "log_tool_called",
    "mask_text",
    "mask_value",
    "subscribe_events",
    "summarize_text",
    "unbind_context",
    "unsubscribe_events",
]
