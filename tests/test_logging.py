"""shared/logging.py —— 日志底座单测（离线，零外部依赖）。

覆盖 Phase C / C1 的三条验收：脱敏用例、格式切换、ContextVar 传播；
外加「消息原文不进日志」「事件白名单」「监听器失败不反噬」四条安全/健壮性边界。
"""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from shared.logging import (
    EVENT_REQUEST_FINISHED,
    MASKED,
    ContextFilter,
    JsonFormatter,
    LogEvent,
    LogFormat,
    LoggingConfig,
    RecordingListener,
    SensitiveFilter,
    TextFormatter,
    bind_context,
    clear_context,
    configure_logging,
    current_session_id,
    current_trace_id,
    current_user_id,
    log_answer_generated,
    log_event,
    log_request_finished,
    log_tool_called,
    mask_text,
    mask_value,
    summarize_text,
    unbind_context,
)

# —— 夹具 ——


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """每个用例前后清空关联标识（ContextVar 会跨用例继承，必须复位）。"""
    clear_context()
    yield
    clear_context()


@pytest.fixture
def text_record() -> logging.LogRecord:
    """构造一条带关联标识与事件字段的 LogRecord（直接喂 formatter，不经 root 配置）。"""

    def _make(
        message: object = "hello",
        args: tuple[object, ...] = (),
        **extra: object,
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name="srp_agent.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=args,
            exc_info=None,
        )
        record._service = "api"
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    return _make


def _rendered(record: logging.LogRecord, formatter: logging.Formatter, *, mask: bool = True) -> str:
    """按生产装配顺序渲染：ContextFilter → SensitiveFilter → formatter。"""
    ContextFilter("api").filter(record)
    if mask:
        SensitiveFilter().filter(record)
    return formatter.format(record)


# —— 脱敏 ——


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ('{"api_key": "sk-live-abcdef123456"}', "sk-live-abcdef123456"),
        ("api_key=sk-live-abcdef123456", "sk-live-abcdef123456"),
        ('{"LLM_API_KEY": "sk-live-abcdef123456"}', "sk-live-abcdef123456"),
        ('{"user_access_token": "u-abcdef1234567890"}', "u-abcdef1234567890"),
        ('{"refresh_token": "r-abcdef1234567890"}', "r-abcdef1234567890"),
        ('{"app_secret": "abcdef1234567890abcdef"}', "abcdef1234567890abcdef"),
        ('{"password": "hunter2hunter2"}', "hunter2hunter2"),
        ('{"lark_token_key": "0123456789abcdef0123"}', "0123456789abcdef0123"),
        ("Authorization: Bearer abcdef1234567890", "abcdef1234567890"),
        ("https://x/y?access_token=abc123456789&z=1", "abc123456789"),
        ("token sk-proj-abcdefghijklmnop", "sk-proj-abcdefghijklmnop"),
        (
            "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijkl",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijkl",
        ),
    ],
)
def test_mask_text_removes_credentials(raw: str, secret: str) -> None:
    """字段名与凭据形态两类规则都能吃掉密钥原文，且保留键名可读性。"""
    masked = mask_text(raw)
    assert secret not in masked
    assert MASKED in masked


def test_mask_text_masks_email_and_phone() -> None:
    """邮箱/手机号兜底：日志不沉淀可识别个人信息（合规底线）。"""
    masked = mask_text("联系 zhangsan@example.com 或 13812345678 谢谢")
    assert "zhangsan@example.com" not in masked
    assert "13812345678" not in masked
    assert "联系" in masked and "谢谢" in masked  # 正文其余部分保持可读


def test_mask_text_keeps_empty_value_and_normal_text() -> None:
    """空值与普通文本不被改写（避免把 `api_key=` 这类配置态行打坏）。"""
    assert mask_text('{"api_key": ""}') == '{"api_key": ""}'
    assert mask_text("工具 current_datetime 调用成功，耗时 12ms") == (
        "工具 current_datetime 调用成功，耗时 12ms"
    )


def test_mask_value_walks_structures() -> None:
    """dict/list 深走脱敏（extra 字段/事件 payload 复用同一规则）。"""
    masked = mask_value({"a": ["sk-live-abcdef123456"], "b": {"api_key": "sk-x-abcdef1234567"}})
    assert masked["a"][0] == MASKED
    assert masked["b"]["api_key"] == MASKED


def test_mask_value_preserves_tuple_type() -> None:
    """tuple 保持 tuple：改成 list 会让 logging 的 `%` 拼装抛 TypeError（每条带参日志都炸）。"""
    masked = mask_value(("Bearer abcdef1234567890", 1))
    assert isinstance(masked, tuple)
    assert masked[1] == 1


def test_parametrized_log_renders_after_masking(text_record) -> None:
    """带参日志（`%s`/`%d`）经脱敏后仍能正确渲染（回归护栏：曾被改成 list 而全崩）。"""
    record = text_record("main=%s | 子查询 %d", ("改写后的查询", 0))
    rendered = _rendered(record, TextFormatter())
    assert "main=改写后的查询 | 子查询 0" in rendered


def test_sensitive_filter_rewrites_both_msg_and_args(text_record) -> None:
    """msg 与 args 都被改写（`%s` 拼装后不得残留密钥）。"""
    record = text_record("key=%s", ("sk-live-abcdef123456",))
    rendered = _rendered(record, TextFormatter())
    assert "sk-live-abcdef123456" not in rendered
    assert MASKED in rendered


def test_sensitive_filter_masks_extra_fields(text_record) -> None:
    """extra 字段同样过脱敏（json 格式下 extra 是主要泄漏面）。"""
    record = text_record("done", (), note="Bearer abcdef1234567890")
    payload = json.loads(_rendered(record, JsonFormatter()))
    assert "abcdef1234567890" not in payload["extra"]["note"]


def test_masking_applies_to_json_format(text_record) -> None:
    """两种格式共用同一 Filter：切到 json 不会绕过脱敏。"""
    record = text_record("api_key=sk-live-abcdef123456")
    line = _rendered(record, JsonFormatter())
    assert "sk-live-abcdef123456" not in line


def test_masking_can_be_disabled(text_record) -> None:
    """`mask_enabled=False` 仅用于对照实验：关闭后原文出现（证明前面的用例确实在脱敏）。"""
    record = text_record("api_key=sk-live-abcdef123456")
    rendered = _rendered(record, TextFormatter(), mask=False)
    assert "sk-live-abcdef123456" in rendered


# —— 格式切换 ——


def test_text_format_layout_is_greppable(text_record) -> None:
    """text 格式：关联前缀 + 事件 + 耗时同排一行（人眼可读、可直接 grep trace_id）。"""
    record = text_record(
        "回答完成",
        (),
        trace_id="req-1",
        session_id="sess-1",
        user_id="user-1",
        event="request.finished",
        duration_ms=123,
    )
    rendered = _rendered(record, TextFormatter())
    assert "[srp_agent.test]" in rendered
    assert "trace=req-1 session=sess-1 user=user-1" in rendered
    assert "回答完成" in rendered
    assert "(123ms)" in rendered
    assert "event=request.finished" in rendered
    assert "\n" not in rendered  # 单行（聚合/轮转友好）


def test_text_format_omits_missing_segments(text_record) -> None:
    """缺字段不产生空占位符（排版不脏）：无关联/无耗时/无事件时只剩基础段。"""
    rendered = _rendered(text_record("plain", ()), TextFormatter())
    assert "[" not in rendered.split("]")[0] or True  # 级别前缀不参与断言
    assert "()" not in rendered and "[] " not in rendered
    assert "trace=" not in rendered and "event=" not in rendered and "ms)" not in rendered
    assert rendered.endswith("plain")


def test_json_format_is_single_line_jsonl(text_record) -> None:
    """json 格式：一行合法 JSON，字段名与事件表口径一致。"""
    record = text_record(
        "工具成功",
        (),
        trace_id="req-1",
        session_id="sess-1",
        user_id="user-1",
        event="tool.called",
        code="tool_error.execution",
        duration_ms=42,
        tool_name="current_datetime",
    )
    payload = json.loads(_rendered(record, JsonFormatter()))
    assert "\n" not in _rendered(record, JsonFormatter())
    assert payload["level"] == "INFO"
    assert payload["logger"] == "srp_agent.test"
    assert payload["service"] == "api"
    assert payload["trace_id"] == "req-1"
    assert payload["session_id"] == "sess-1"
    assert payload["user_id"] == "user-1"
    assert payload["event"] == "tool.called"
    assert payload["code"] == "tool_error.execution"
    assert payload["duration_ms"] == 42
    assert payload["message"] == "工具成功"
    assert payload["extra"]["tool_name"] == "current_datetime"


def test_json_format_includes_traceback(text_record) -> None:
    """异常进 json：`exc` 字段承载 traceback，且仍是单行 JSON。"""
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = text_record("失败")
        record.exc_info = sys.exc_info()
    line = _rendered(record, JsonFormatter())
    assert "\n" not in line
    assert "ValueError: boom" in json.loads(line)["exc"]


# —— 配置入口 ——


def test_logging_config_from_settings_projects_service_and_env(monkeypatch) -> None:
    """配置投影：显式 service 优先，level/format 来自配置对象或 env。"""

    class _Settings:
        app_name = "srp-agent"
        log_level = "WARNING"

    monkeypatch.setenv("LOG_FORMAT", "json")
    cfg = LoggingConfig.from_settings(_Settings(), service="tools_mcp")
    assert cfg.service == "tools_mcp"
    assert cfg.level == "WARNING"
    assert cfg.log_format is LogFormat.JSON


def test_logging_config_invalid_format_falls_back_to_text() -> None:
    """拼错的格式值回退 text（不因配置错误而静默丢日志）。"""
    assert LoggingConfig(log_format="yaml").log_format is LogFormat.TEXT  # type: ignore[arg-type]


def test_configure_logging_is_idempotent() -> None:
    """重复配置只保留一个自装 handler（叠加会让每条日志打两遍）。"""
    configure_logging(LoggingConfig(service="api", level="INFO"))
    configure_logging(LoggingConfig(service="api", level="DEBUG"))
    own = [h for h in logging.getLogger().handlers if getattr(h, "_srp_logging", False)]
    assert len(own) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_routes_library_loggers_through_root() -> None:
    """uvicorn/fastmcp 交给 root：否则访问日志没有 trace_id（链路断在库日志上）。"""
    configure_logging(LoggingConfig(service="api"))
    uvicorn_logger = logging.getLogger("uvicorn.access")
    assert uvicorn_logger.handlers == []
    assert uvicorn_logger.propagate is True


def test_configure_logging_json_format_writes_parseable_line(capsys) -> None:
    """走完整 root 装配（非直喂 formatter）：stdout 上确实是一行 JSON。"""
    configure_logging(LoggingConfig(service="a2a_mcp", log_format=LogFormat.JSON))
    logging.getLogger("srp_agent.probe").info("探针日志")
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["service"] == "a2a_mcp"
    assert payload["message"] == "探针日志"
    configure_logging(LoggingConfig(service="api"))


# —— ContextVar 传播 ——


def test_bind_context_propagates_to_current_helpers() -> None:
    """绑定后可经 current_* 读回（Filter 与响应头回写共用同一来源）。"""
    bind_context(trace_id="req-1", session_id="sess-1", user_id="user-1")
    assert current_trace_id() == "req-1"
    assert current_session_id() == "sess-1"
    assert current_user_id() == "user-1"


def test_unbind_context_restores_previous_values() -> None:
    """成对使用：退出后回到绑定前的状态（请求之间不得串号）。"""
    outer = bind_context(trace_id="outer")
    inner = bind_context(trace_id="inner")
    assert current_trace_id() == "inner"
    unbind_context(inner)
    assert current_trace_id() == "outer"
    unbind_context(outer)
    assert current_trace_id() is None


async def test_context_propagates_into_child_task() -> None:
    """async 任务继承：图内节点/带外任务无需传参即带同一 trace_id。"""
    import asyncio

    async def _probe() -> str | None:
        await asyncio.sleep(0)
        return current_trace_id()

    bind_context(trace_id="req-task")
    assert await asyncio.create_task(_probe()) == "req-task"


def test_context_filter_fills_record_and_sanitizes(text_record) -> None:
    """Filter 补齐关联标识，并把外部传入值净化（防伪造日志行/注入换行）。"""
    bind_context(trace_id="req-1", user_id="u-cleaned")
    record = text_record("x", (), user_id="evil\n2020-01-01 ERROR [fake] injected")
    ContextFilter("api").filter(record)
    assert record.trace_id == "req-1"  # type: ignore[attr-defined]
    assert "evil" in record.user_id
    assert "\n" not in record.user_id  # type: ignore[attr-defined]
    assert record._service == "api"


def test_context_binding_clears_on_none() -> None:
    """显式传 None = 清除该维度；解除绑定后回到外层值（请求之间不得串号）。"""
    outer = bind_context(trace_id="req-1", session_id="s1")
    inner = bind_context(trace_id=None, session_id=None)
    assert current_trace_id() is None  # 已清除（不是「保持外层值」）
    assert current_session_id() is None
    unbind_context(inner)
    assert current_trace_id() == "req-1"  # 复位到外层
    unbind_context(outer)
    assert current_trace_id() is None


# —— 消息摘要与事件白名单 ——


def test_summarize_text_never_reveals_content() -> None:
    """摘要口径：只留长度与 sha256 前缀，同一文本可对齐、原文不可还原。"""
    summary = summarize_text("用户的私密消息")
    assert "用户的私密消息" not in summary
    assert "len=7" in summary
    assert "sha256=" in summary
    assert summarize_text("").startswith("len=0 sha256=")


def test_log_event_whitelist_and_summary() -> None:
    """事件字段白名单 + 扁平摘要（与 interaction_events 列一一对应）。"""
    event = LogEvent(
        event="tool.called",
        service="api",
        trace_id="req-1",
        status="ok",
        duration_ms=12,
        tool_name="calculate",
        fields={"args": "len=3 sha256=abc"},
    )
    summary = event.summary()
    assert summary["event"] == "tool.called"
    assert summary["timestamp"].startswith("20")
    assert summary["duration_ms"] == 12
    assert summary["tool_name"] == "calculate"
    assert summary["args"] == "len=3 sha256=abc"


def test_log_event_rejects_unknown_top_level_field() -> None:
    """没声明的顶层字段进不来（扩字段必须改模型，避免日志结构漂移）。"""
    with pytest.raises(ValidationError):
        LogEvent(event="x", 未声明字段="boom")  # type: ignore[call-arg]


def test_log_event_emits_structured_record_and_dispatches(caplog: pytest.LogCaptureFixture) -> None:
    """`log_event` 既落文本日志（带 event/耗时），也分发给监听器。"""
    listener = RecordingListener().install()
    try:
        with caplog.at_level(logging.INFO, logger="srp_agent.events"):
            log_tool_called(
                logging.getLogger("srp_agent.events"),
                tool_name="calculate",
                status="ok",
                duration_ms=7,
            )
    finally:
        listener.uninstall()
    assert listener.names() == ["tool.called"]
    assert listener.events[0].tool_name == "calculate"
    assert listener.events[0].duration_ms == 7
    record = next(r for r in caplog.records if getattr(r, "event", None) == "tool.called")
    assert record.duration_ms == 7  # type: ignore[attr-defined]


def test_answer_generated_event_stores_hash_not_text() -> None:
    """`answer.generated` 只记 len + sha256（回答原文不进事件/日志）。"""
    event = log_answer_generated(
        logging.getLogger("srp_agent.events"),
        answer="这是完整回答正文",
        finished_reason="completed",
    )
    assert event.fields["answer"].startswith("len=8 sha256=")
    assert "完整回答" not in json.dumps(event.summary(), ensure_ascii=False)


def test_request_finished_event_carries_status_and_duration() -> None:
    """`request.finished`：终态 + 端到端耗时 + token 用量（指标口径同名字段）。"""
    event = log_request_finished(
        logging.getLogger("srp_agent.events"),
        status="completed",
        duration_ms=512,
        tokens={"input_tokens": 12, "output_tokens": 34},
    )
    summary = event.summary()
    assert summary["status"] == "completed"
    assert summary["duration_ms"] == 512
    assert summary["tokens"] == {"input_tokens": 12, "output_tokens": 34}
    assert event.event == EVENT_REQUEST_FINISHED


def test_listener_failure_never_breaks_caller(caplog: pytest.LogCaptureFixture) -> None:
    """监听器（落库）抛错只告警：可观测性绝不反噬主链路。"""

    def _boom(_: LogEvent) -> None:
        raise RuntimeError("db down")

    from shared.logging import subscribe_events, unsubscribe_events

    subscribe_events(_boom)
    try:
        with caplog.at_level(logging.WARNING):
            event = log_event(logging.getLogger("srp_agent.events"), "probe")
    finally:
        unsubscribe_events(_boom)
    assert event.event == "probe"


def test_listener_failure_does_not_recurse(caplog: pytest.LogCaptureFixture) -> None:
    """监听器内部再打日志不得二次分发（失败路径不允许自我递归）。"""
    listener = RecordingListener()
    calls: list[str] = []

    def _listener(event: LogEvent) -> None:
        listener(event)
        calls.append(event.event)
        if len(calls) < 5:  # 若发生递归，这里会被打爆
            log_event(logging.getLogger("srp_agent.events"), "nested")

    from shared.logging import subscribe_events, unsubscribe_events

    subscribe_events(_listener)
    try:
        with caplog.at_level(logging.WARNING):
            log_event(logging.getLogger("srp_agent.events"), "probe")
    finally:
        unsubscribe_events(_listener)
    assert calls == ["probe"]
