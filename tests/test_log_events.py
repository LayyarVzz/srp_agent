"""结构化事件单测（Phase C / C3）。

覆盖两条验收：
① 事件链完整：`request.received → intent.classified → tool.called → answer.generated
   → memory.saved → request.finished`（顺序 + 关键字段），全部取自 `LogEvent` 白名单；
② 安全边界：**消息体只记 len + sha256** —— 日志/事件里没有用户原文、没有密钥、
   没有飞书 token（CLAUDE.md 安全约束，与 §C1 脱敏同口径）。

事件流经 ASGI 应用真实触发（fake LLM + 假工具），不做手工造事件的自证式断言。
日志断言用 `capfd`（进程 fd 级捕获）：日志经 stdout 输出，pytest 的全局捕获层不保证
被 `capsys`（Python 级重定向）看到，用 fd 级捕获才是「容器里 docker logs 看到什么」的
同一口径。
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import pytest
from httpx import ASGITransport, AsyncClient

from agent.intent.models import Intent
from agent.memory.models import MemoryExtraction
from agent.memory.persist import save_conversation_memory
from app.request_context import REQUEST_ID_HEADER_OUT
from shared.logging import (
    EVENT_ANSWER_GENERATED,
    EVENT_INTENT_CLASSIFIED,
    EVENT_MEMORY_SAVED,
    EVENT_REQUEST_FINISHED,
    EVENT_REQUEST_RECEIVED,
    EVENT_TOOL_CALLED,
    LogEvent,
    RecordingListener,
    clear_context,
)
from tests.conftest import make_fake_tool, tool_call_messages

HEADERS = {"X-User-Id": "demo-user"}
CHAT_URL = "/api/v1/interactions/text"

# 测试用敏感串（构造为明确的假值，避免被误认成真实凭据）。
FAKE_SECRET = "sk-live-abcdef1234567890"


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """用例前后清空关联标识。"""
    clear_context()
    yield
    clear_context()


@pytest.fixture
def listener() -> RecordingListener:
    """挂一个内存事件监听器（断言事件序列/字段，零 DB）。"""
    recorder = RecordingListener().install()
    yield recorder
    recorder.uninstall()


def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _tool_echo_messages(reply: str) -> list[Any]:
    """工具型一轮的消息序列（意图 + 查询理解 + 工具调用 + 终答）。"""
    return tool_call_messages([[{"name": "current_datetime", "args": {}, "id": "call_1"}]], reply)


async def test_event_chain_is_complete_and_ordered(
    api_app_factory: Any, listener: RecordingListener
) -> None:
    """一轮工具型对话产出完整事件链，且顺序与真实执行顺序一致。"""
    app, runtime = await api_app_factory(
        _tool_echo_messages("现在是十点"), tools=[make_fake_tool("current_datetime")]
    )
    try:
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL,
                headers={**HEADERS, "X-Request-Id": "req-chain"},
                json={"text": "现在几点"},
            )
        assert resp.status_code == 200
    finally:
        await runtime.aclose()

    names = listener.names()
    for expected in (
        EVENT_REQUEST_RECEIVED,
        EVENT_INTENT_CLASSIFIED,
        EVENT_TOOL_CALLED,
        EVENT_ANSWER_GENERATED,
        EVENT_REQUEST_FINISHED,
    ):
        assert expected in names, f"缺事件 {expected}：{names}"
    assert names.index(EVENT_REQUEST_RECEIVED) < names.index(EVENT_INTENT_CLASSIFIED)
    assert names.index(EVENT_INTENT_CLASSIFIED) < names.index(EVENT_TOOL_CALLED)
    assert names.index(EVENT_TOOL_CALLED) < names.index(EVENT_ANSWER_GENERATED)
    assert names.index(EVENT_ANSWER_GENERATED) < names.index(EVENT_REQUEST_FINISHED)


async def test_tool_called_event_fields(api_app_factory: Any, listener: RecordingListener) -> None:
    """`tool.called`：工具名/成败/耗时齐全，批量口径（batch_size）可见。"""
    app, runtime = await api_app_factory(
        _tool_echo_messages("现在是十点"), tools=[make_fake_tool("current_datetime")]
    )
    try:
        async with _client(app) as client:
            await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
    finally:
        await runtime.aclose()

    events = listener.find(EVENT_TOOL_CALLED)
    assert len(events) == 1
    event = events[0]
    assert event.tool_name == "current_datetime"
    assert event.status == "ok"
    assert event.duration_ms is not None and event.duration_ms >= 0
    assert event.fields["batch_size"] == 1
    assert event.fields["args"] == ""


async def test_intent_classified_event_fields(
    api_app_factory: Any, listener: RecordingListener
) -> None:
    """`intent.classified`：意图 + 置信度（判定理由是摘要，不是原文）。"""
    app, runtime = await api_app_factory(_tool_echo_messages("现在是十点"))
    try:
        async with _client(app) as client:
            await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
    finally:
        await runtime.aclose()

    event = listener.find(EVENT_INTENT_CLASSIFIED)[0]
    assert event.status == "tool_use"
    assert event.fields["intent"] == str(Intent.TOOL_USE)
    assert event.fields["confidence"] == 0.95
    assert event.fields["reason"].startswith("len=")  # 理由只留摘要


async def test_answer_generated_event_has_no_original_text(
    api_app_factory: Any, listener: RecordingListener
) -> None:
    """`answer.generated` 只记 len + sha256；用户原文与回答原文都不进事件。

    断言对象是「事件摘要 vs 实际回答」而非硬编码长度：回答可能来自 `call_model`
    直接产出（本用例即此路径），长度由模型消息决定，测试不该钉死它。
    """
    user_text = "我的手机号是 13812345678，请记住"
    app, runtime = await api_app_factory(_tool_echo_messages("好的，已记住这一点"))
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": user_text})
        answer = resp.json()["answer"]
    finally:
        await runtime.aclose()

    event = listener.find(EVENT_ANSWER_GENERATED)[0]
    assert event.fields["answer"].startswith(f"len={len(answer)} sha256=")
    dumped = json.dumps(event.summary(), ensure_ascii=False)
    assert answer not in dumped
    assert user_text not in dumped
    assert "13812345678" not in dumped  # 个人信息不得沉淀进事件表


def test_long_tool_arguments_are_summarized() -> None:
    """长实参只留摘要（用户原话/检索 query 属会话明文，禁止落盘）。"""
    from agent.core.graph import _ARG_INLINE_MAX_CHARS, _summarize_arguments

    long_query = "帮我查一下最近三个月所有和飞书授权有关的工单记录并汇总给我" * 3
    assert len(long_query) > _ARG_INLINE_MAX_CHARS
    summary = _summarize_arguments({"query": long_query, "top_k": 5})
    assert long_query not in summary
    assert f"query[len={len(long_query)} sha256=" in summary
    assert "top_k=5" in summary  # 短标量原样保留（调试价值高）


async def test_memory_saved_event_records_action_without_content() -> None:
    """`memory.saved`：动作/类型/id/会话/用户齐全，内容只留摘要（记忆常含个人信息）。"""
    from langgraph.store.memory import InMemoryStore

    from agent.memory.adapter import MemoryStore

    listener = RecordingListener().install()
    content = "用户偏好：每天早上 8 点提醒我看日程"

    class _Extractor:
        """最小抽取器替身：直接返回一条已判定的偏好记忆。"""

        async def extract(self, messages: Any) -> list[MemoryExtraction]:
            return [
                MemoryExtraction(
                    kind="preference",
                    content=content,
                    importance=0.8,
                    category="preference",
                    worth_score=0.9,
                    worth_reason="稳定偏好",
                )
            ]

    store = MemoryStore(InMemoryStore())
    try:
        await save_conversation_memory(
            [],
            session_id="sess-mem",
            user_id="user-mem",
            extractor=_Extractor(),  # type: ignore[arg-type]
            store=store,
        )
    finally:
        listener.uninstall()

    event = listener.find(EVENT_MEMORY_SAVED)[0]
    assert event.status == "inserted"
    assert event.fields["kind"] == "preference"
    assert event.fields["memory_id"]
    # 会话/用户是**事件顶层字段**（fields 里的同名字段会被提升，见 log_event）。
    assert event.session_id == "sess-mem"
    assert event.user_id == "user-mem"
    assert event.fields["content"].startswith(f"len={len(content)} sha256=")
    assert content not in json.dumps(event.summary(), ensure_ascii=False)


async def test_events_and_log_lines_never_leak_secrets(
    api_app_factory: Any,
    listener: RecordingListener,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """端到端安全断言：事件字段与 stdout 日志里都不得出现密钥/飞书 token。

    用工具**失败**路径触发异常日志（异常消息里内嵌凭据是最现实的泄漏面）。
    """
    failing = make_fake_tool(
        "current_datetime", fail_with=RuntimeError(f"调用失败 api_key={FAKE_SECRET}")
    )
    app, runtime = await api_app_factory(_tool_echo_messages("降级回答"), tools=[failing])
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
        assert resp.status_code == 200
    finally:
        await runtime.aclose()

    captured = capfd.readouterr()
    assert FAKE_SECRET not in captured.out
    assert FAKE_SECRET not in captured.err
    for event in listener.events:
        assert FAKE_SECRET not in json.dumps(event.summary(), ensure_ascii=False)
    assert listener.find(EVENT_TOOL_CALLED)[0].status == "error"


async def test_failed_tool_event_carries_error_code(
    api_app_factory: Any, listener: RecordingListener
) -> None:
    """工具失败事件带结构化错误码（与 tool_error.* 同源，供告警按码聚合）。"""
    failing = make_fake_tool("current_datetime", fail_with=RuntimeError("boom"))
    app, runtime = await api_app_factory(_tool_echo_messages("降级回答"), tools=[failing])
    try:
        async with _client(app) as client:
            await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
    finally:
        await runtime.aclose()

    event = listener.find(EVENT_TOOL_CALLED)[0]
    assert event.status == "error"
    assert event.code == "tool_error.execution"
    assert event.level == "WARNING"  # 失败事件提到 WARNING，便于按级别过滤


async def test_event_logger_lines_carry_event_field(
    api_app_factory: Any, capfd: pytest.CaptureFixture[str]
) -> None:
    """事件同时落 stdout 日志行（带 event= 字段）：容器里 `grep event=` 即可取事件流。"""
    app, runtime = await api_app_factory(_tool_echo_messages("现在是十点"))
    try:
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL,
                headers={**HEADERS, "X-Request-Id": "req-line"},
                json={"text": "现在几点"},
            )
        assert resp.headers[REQUEST_ID_HEADER_OUT] == "req-line"
    finally:
        await runtime.aclose()

    lines = [line for line in capfd.readouterr().out.splitlines() if "event=" in line]
    assert any("event=intent.classified" in line for line in lines)
    assert any("event=request.finished" in line for line in lines)
    # 事件行同样带关联前缀（同一条 trace 里能看到「日志 + 事件」两种形态）。
    assert any("trace=req-line" in line and "event=" in line for line in lines)


async def test_json_format_events_are_parseable(
    api_app_factory: Any, capfd: pytest.CaptureFixture[str]
) -> None:
    """`LOG_FORMAT=json` 下事件行是合法 JSONL（采集端可直接解析）。

    WHY 先建应用再切格式：`create_app` 自己会按 LOG_* 配一次日志（默认 text），
    顺序颠倒的话本用例会被它覆盖回 text —— 测的就不是 JSON 形态了。
    """
    from shared.logging import LogFormat, LoggingConfig, ServiceName, configure_logging

    app, runtime = await api_app_factory(_tool_echo_messages("现在是十点"))
    configure_logging(LoggingConfig(service=ServiceName.API, log_format=LogFormat.JSON))
    try:
        async with _client(app) as client:
            await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
    finally:
        await runtime.aclose()

    payloads = [
        json.loads(line.strip())  # 解析失败即测试失败（JSONL 契约）
        for line in capfd.readouterr().out.splitlines()
        if line.strip().startswith("{")
    ]
    events = [p for p in payloads if "event" in p]
    assert {p["event"] for p in events} >= {EVENT_INTENT_CLASSIFIED, EVENT_REQUEST_FINISHED}
    assert all(p["service"] == "api" for p in payloads)
    assert all(FAKE_SECRET not in json.dumps(p, ensure_ascii=False) for p in payloads)
    # 事件字段与列同名同义：session_id 是顶层键（不是嵌在 payload 里）。
    finished = next(p for p in events if p["event"] == EVENT_REQUEST_FINISHED)
    assert finished["session_id"]
    assert finished["duration_ms"] >= 0


def test_log_event_event_names_are_constants() -> None:
    """事件名以常量声明（禁止散落字面量 —— 与日志检索/事件表/指标三处同名同义）。"""
    import inspect

    from agent.core import graph
    from agent.memory import persist as memory_persist

    for module in (graph, memory_persist):
        source = inspect.getsource(module)
        for literal in (
            '"tool.called"',
            '"answer.generated"',
            '"memory.saved"',
            '"intent.classified"',
            "'tool.called'",
            "'answer.generated'",
        ):
            assert literal not in source, f"事件名应为常量：{literal}"


def test_log_event_promotes_standard_fields_from_fields_dict() -> None:
    """`fields` 里的标准字段被提升到顶层（事件表列与事件属性保持一份口径）。"""
    from shared.logging import log_memory_saved

    event = log_memory_saved(
        logging.getLogger("srp_agent.event"),
        action="inserted",
        kind="fact",
        memory_id="m1",
        session_id="s1",
        user_id="u1",
    )
    assert event.session_id == "s1"
    assert event.user_id == "u1"
    assert "session_id" not in event.fields  # 不重复留在 payload 里
    assert event.summary()["session_id"] == "s1"


def test_recording_listener_snapshot_shape() -> None:
    """`LogEvent.summary()` 是扁平 dict（事件表列 + payload 一份口径）。"""
    event = LogEvent(event="probe", service="api", fields={"a": 1}, duration_ms=3)
    summary = event.summary()
    assert summary["a"] == 1
    assert summary["duration_ms"] == 3
    assert summary["service"] == "api"


def test_event_logger_does_not_use_business_logger() -> None:
    """事件用独立 logger（`srp_agent.event`）：便于只过滤事件流，不掺业务日志。"""
    from agent.core import graph
    from agent.memory import persist as memory_persist
    from shared.logging import EVENT_LOGGER_NAME

    assert graph.event_logger.name == EVENT_LOGGER_NAME
    assert memory_persist.event_logger.name == EVENT_LOGGER_NAME
    assert logging.getLogger(EVENT_LOGGER_NAME).name != graph.logger.name


# —— C4：耗时真实计时 + LLM token 累计 ——


async def test_tool_duration_is_measured_not_zero(
    api_app_factory: Any, listener: RecordingListener
) -> None:
    """工具耗时不再恒为 0（历史上 `ToolResult.duration_ms` 存在但无人写入）。

    用带 `delay_s` 的假工具制造可测量耗时：断言记录、事件、tool_trace 三处口径一致。
    """
    slow = make_fake_tool("current_datetime", content="ok", delay_s=0.03)
    app, runtime = await api_app_factory(_tool_echo_messages("现在是十点"), tools=[slow])
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
        trace = resp.json()["tool_trace"]
    finally:
        await runtime.aclose()

    assert trace, "tool_trace 为空"
    assert trace[0]["result"]["duration_ms"] >= 20  # 计时确实覆盖了工具执行
    event = listener.find(EVENT_TOOL_CALLED)[0]
    assert event.duration_ms is not None and event.duration_ms >= 20


async def test_request_finished_carries_tokens_and_duration(
    api_app_factory: Any, listener: RecordingListener
) -> None:
    """agent 侧 `request.finished` 带 token 累计与端到端耗时（指标口径）。

    WHY 用 source 区分两条 `request.finished`：api 中间件记的是 **HTTP** 请求口径，
    agent runtime 记的是**一轮交互**口径（含 token 用量）。同名事件靠 `source` 区分，
    混在一起聚合会把「HTTP 往返」与「Agent 一轮」当成同一个指标。
    """
    app, runtime = await api_app_factory(_tool_echo_messages("现在是十点"))
    try:
        async with _client(app) as client:
            await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
    finally:
        await runtime.aclose()

    agent_events = [
        e for e in listener.find(EVENT_REQUEST_FINISHED) if e.fields.get("source") == "agent"
    ]
    assert agent_events, "缺 agent 侧 request.finished"
    event = agent_events[-1]
    assert event.duration_ms is not None and event.duration_ms >= 0
    assert event.status == "completed"
    tokens = event.fields["tokens"]
    assert tokens["total_tokens"] > 0  # 这一轮确实调过 LLM
    assert tokens["input_tokens"] > 0 and tokens["output_tokens"] > 0
    assert tokens["total_tokens"] == tokens["input_tokens"] + tokens["output_tokens"]
    assert tokens["models"], "缺多模型明细（归因需要）"


def test_request_finished_tokens_empty_when_no_llm_call() -> None:
    """未调用 LLM 的一轮：tokens 为空表（如实反映「没花钱」，而不是伪造 0 计数）。"""
    from langchain_core.callbacks.usage import UsageMetadataCallbackHandler

    from agent.runtime import _token_usage

    assert _token_usage(UsageMetadataCallbackHandler()) == {}


def test_token_usage_folds_multiple_models() -> None:
    """多模型 usage 折叠：顶层合计 + models 明细（合计供看板、明细供归因）。"""

    class _Handler:
        usage_metadata: ClassVar[dict[str, dict[str, int]]] = {
            "model-a": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "model-b": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        }

    from agent.runtime import _token_usage

    folded = _token_usage(_Handler())  # type: ignore[arg-type]
    assert folded["input_tokens"] == 11
    assert folded["output_tokens"] == 7
    assert folded["total_tokens"] == 18
    assert set(folded["models"]) == {"model-a", "model-b"}


def test_token_usage_never_raises_on_broken_handler() -> None:
    """指标读取失败不得反噬主链路（异常吞掉返回空表）。"""

    class _Broken:
        @property
        def usage_metadata(self) -> dict[str, Any]:
            raise RuntimeError("boom")

    from agent.runtime import _token_usage

    assert _token_usage(_Broken()) == {}  # type: ignore[arg-type]


async def test_failed_turn_records_error_status(
    api_app_factory: Any, listener: RecordingListener
) -> None:
    """图运行中断：`request.finished` 落 status=error + 错误类型（失败轮次可被检索）。"""
    app, runtime = await api_app_factory(_tool_echo_messages("现在是十点"))

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("graph down")
        yield  # pragma: no cover  # 保持 async generator 语义

    original = runtime.graph.astream
    runtime.graph.astream = _boom  # type: ignore[method-assign]
    try:
        # raise_app_exceptions=False：异常由 app 的处理器转 500 信封，测试要的是「状态码 +
        # 事件」两件事，而不是让 ASGI 传输把异常直接抛给调用方。
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "现在几点"})
        assert resp.status_code == 500
    finally:
        runtime.graph.astream = original  # type: ignore[method-assign]
        await runtime.aclose()

    agent_events = [
        e for e in listener.find(EVENT_REQUEST_FINISHED) if e.fields.get("source") == "agent"
    ]
    assert agent_events and agent_events[-1].status == "error"
    assert agent_events[-1].code == "RuntimeError"
