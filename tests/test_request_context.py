"""请求级日志上下文单测（Phase C / C2）：X-Request-Id 注入透传 + 4 服务入口统一口径。

覆盖三条验收：
① 中间件回写 `X-Request-Id`（调用方传入则沿用、缺失则服务端生成），且该 id 出现在
   请求内产生的日志行里（「一条 trace_id 可 grep 出 api→agent→MCP 全链路」的入口半边）；
② `session_id` / `user_id` 随交互绑进事件，且请求结束后**彻底复位**（不跨请求串号）；
③ api 与 3 个 MCP 服务的日志配置同源（同一 formatter / 脱敏 / 关联标识口径）。

MCP 侧「全链路」的另一半（工具子进程内同样带 trace_id）需真起进程，属 §5.x 联调口径，
不在离线单测断言；此处只钉住「配置同源」这一确定性前提。
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from agent.intent.models import Intent
from app.request_context import (
    REQUEST_ID_HEADER_OUT,
    RequestContextMiddleware,
    elapsed_ms,
    new_trace_id,
    turn_context,
)
from services.a2a_mcp.config import A2AMCPRuntimeSettings
from services.a2a_mcp.config import build_logging_config as a2a_logging
from services.lark_mcp.config import LarkMCPRuntimeSettings
from services.lark_mcp.config import build_logging_config as lark_logging
from services.tools_mcp.config import MCPRuntimeSettings
from services.tools_mcp.config import build_logging_config as tools_logging
from settings import RuntimeSettings
from shared.logging import (
    EVENT_REQUEST_FINISHED,
    EVENT_REQUEST_RECEIVED,
    LogFormat,
    LoggingConfig,
    RecordingListener,
    ServiceName,
    clear_context,
    configure_logging,
    current_session_id,
    current_trace_id,
    current_user_id,
)
from tests.conftest import chat_turn_messages

HEADERS = {"X-User-Id": "demo-user"}
CHAT_URL = "/api/v1/interactions/text"
STREAM_URL = "/api/v1/interactions/text/stream"


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """用例前后清空关联标识（ContextVar 跨用例继承）。"""
    clear_context()
    yield
    clear_context()


@pytest.fixture(autouse=True)
def _event_listener() -> Any:
    """每个用例挂一个内存事件监听器（断言结构化事件，零 DB）。"""
    listener = RecordingListener().install()
    yield listener
    listener.uninstall()


def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# —— 中间件：注入 / 透传 / 回写 ——


async def test_middleware_echoes_inbound_request_id(api_app_factory: Any) -> None:
    """调用方透传的 X-Request-Id 原样回写（跨服务/前端串联的基础）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    try:
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL, headers={**HEADERS, "X-Request-Id": "req-caller-1"}, json={"text": "hi"}
            )
        assert resp.status_code == 200
        assert resp.headers[REQUEST_ID_HEADER_OUT] == "req-caller-1"
    finally:
        await runtime.aclose()


async def test_middleware_generates_request_id_when_absent(api_app_factory: Any) -> None:
    """未透传时服务端生成 trace_id 并回写（前端可直接拿去对日志）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "hi"})
        assert resp.status_code == 200
        trace_id = resp.headers[REQUEST_ID_HEADER_OUT]
        assert trace_id.startswith("req_") and len(trace_id) > 8
    finally:
        await runtime.aclose()


async def test_trace_id_appears_in_log_lines(
    api_app_factory: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """trace_id 出现在该请求产生的日志行里（text 格式可直接 grep）。"""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    configure_logging(LoggingConfig(service=ServiceName.API, level="INFO"))
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    try:
        async with _client(app) as client:
            await client.post(
                CHAT_URL, headers={**HEADERS, "X-Request-Id": "req-grep-me"}, json={"text": "hi"}
            )
    finally:
        await runtime.aclose()
    out = capsys.readouterr().out
    assert "req-grep-me" in out
    assert "trace=req-grep-me" in out  # 关联前缀形态（不是被当成普通文本）

    # 延迟 import：`logging` 标准库名与项目 `shared.logging` 同名，显式别名避免歧义。
    from logging import getLogger

    getLogger("srp_agent.request").info("探针：本条也应带 trace")
    assert "探针：本条也应带 trace" in capsys.readouterr().out


async def test_structured_events_carry_trace_session_user(
    api_app_factory: Any, _event_listener: RecordingListener
) -> None:
    """请求事件带 trace_id；交互事件额外带 session_id / user_id（事件表聚合的前提）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    try:
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL, headers={**HEADERS, "X-Request-Id": "req-evt-1"}, json={"text": "hi"}
            )
        session_id = resp.json()["session_id"]
    finally:
        await runtime.aclose()

    received = _event_listener.find(EVENT_REQUEST_RECEIVED)
    assert received and received[0].trace_id == "req-evt-1"
    finished = [e for e in _event_listener.find(EVENT_REQUEST_FINISHED) if e.session_id]
    assert finished, "缺 request.finished（带会话维度）事件"
    assert finished[-1].session_id == session_id
    assert finished[-1].user_id == "demo-user"
    assert finished[-1].duration_ms is not None and finished[-1].duration_ms >= 0


async def test_context_is_reset_after_request(
    api_app_factory: Any, _event_listener: RecordingListener
) -> None:
    """请求结束后关联标识彻底复位（否则下一请求的日志会串上一个用户的身份）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    try:
        async with _client(app) as client:
            await client.post(CHAT_URL, headers=HEADERS, json={"text": "hi"})
        assert current_trace_id() is None
        assert current_session_id() is None
        assert current_user_id() is None
    finally:
        await runtime.aclose()


async def test_sse_stream_events_carry_session(api_app_factory: Any) -> None:
    """SSE 流式同样带会话维度（生成器由独立任务驱动，故流内重绑上下文）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    listener = RecordingListener().install()
    try:
        async with _client(app) as client:
            resp = await client.post(
                STREAM_URL, headers={**HEADERS, "X-Request-Id": "req-sse-1"}, json={"text": "hi"}
            )
        assert resp.status_code == 200
    finally:
        listener.uninstall()
        await runtime.aclose()
    finished = [e for e in listener.find(EVENT_REQUEST_FINISHED) if e.session_id]
    assert finished, "SSE 流内未记带会话维度的 request.finished"
    assert finished[-1].trace_id == "req-sse-1"


async def test_middleware_records_error_status(api_app_factory: Any) -> None:
    """4xx 也记 request.finished（status=error + 状态码码），且不串到下一个请求。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    listener = RecordingListener().install()
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, json={"text": "hi"})  # 缺 X-User-Id → 401
        assert resp.status_code == 401
    finally:
        listener.uninstall()
        await runtime.aclose()
    finished = listener.find(EVENT_REQUEST_FINISHED)
    assert finished and finished[-1].status == "error"
    assert finished[-1].code == "status_401"
    assert current_trace_id() is None


def test_middleware_passes_through_non_http_scope() -> None:
    """非 HTTP scope（lifespan / websocket）直通，不绑上下文也不记事件。"""
    seen: list[dict] = []

    async def _app(scope: dict, receive: Any, send: Any) -> None:
        seen.append(scope)

    middleware = RequestContextMiddleware(_app)
    import asyncio

    asyncio.run(middleware({"type": "lifespan"}, None, None))
    assert seen == [{"type": "lifespan"}]
    assert current_trace_id() is None


# —— turn_context ——


def test_turn_context_binds_and_restores() -> None:
    """一轮交互绑定会话/用户，退出后还原（异常路径同样还原）。"""
    with turn_context(session_id="s1", user_id="u1") as trace_id:
        assert trace_id.startswith("req_")
        assert current_session_id() == "s1"
        assert current_user_id() == "u1"
    assert current_session_id() is None
    assert current_user_id() is None

    with pytest.raises(RuntimeError), turn_context(session_id="s2", user_id="u2"):
        raise RuntimeError("boom")
    assert current_session_id() is None


def test_turn_context_reuses_inbound_trace_id() -> None:
    """已有 trace_id（中间件绑定）时沿用，不另生成（否则链路被截断）。"""
    from shared.logging import bind_context, unbind_context

    tokens = bind_context(trace_id="req-inbound")
    try:
        with turn_context(session_id="s1", user_id="u1") as trace_id:
            assert trace_id == "req-inbound"
        assert current_trace_id() == "req-inbound"  # 退出 turn 后仍保留请求级 trace
    finally:
        unbind_context(tokens)


def test_new_trace_id_is_unique_and_prefixed() -> None:
    """生成的 trace_id 带前缀且互不重复。"""
    ids = {new_trace_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("req_") for i in ids)


def test_elapsed_ms_measures_wall_clock() -> None:
    """耗时口径为毫秒整数（duration_ms 类型契约，供事件表列与指标复用）。"""
    import time

    started = time.perf_counter()
    assert elapsed_ms(started) >= 0


# —— 4 个进程的日志配置同源 ——


def test_all_service_entrypoints_share_logging_config() -> None:
    """api + 3 个 MCP 服务的日志配置同源：service 可区分、其余口径一致。"""
    api = LoggingConfig.from_settings(RuntimeSettings(_env_file=None), service=ServiceName.API)
    tools = tools_logging(MCPRuntimeSettings(_env_file=None))
    lark = lark_logging(LarkMCPRuntimeSettings(_env_file=None))
    a2a = a2a_logging(A2AMCPRuntimeSettings(_env_file=None))

    assert [c.service for c in (api, tools, lark, a2a)] == [
        ServiceName.API,
        ServiceName.TOOLS_MCP,
        ServiceName.LARK_MCP,
        ServiceName.A2A_MCP,
    ]
    # 形态与脱敏口径必须逐字一致（不一致就意味着某一路日志没法用同一套采集规则）。
    assert {c.log_format for c in (api, tools, lark, a2a)} == {LogFormat.TEXT}
    assert {c.level for c in (api, tools, lark, a2a)} == {"INFO"}
    assert {c.mask_enabled for c in (api, tools, lark, a2a)} == {True}


def test_mcp_logging_config_follows_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP 服务跟随 LOG_FORMAT / LOG_LEVEL 环境项（容器内统一切形态）。"""
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    cfg = tools_logging(MCPRuntimeSettings(_env_file=None))
    assert cfg.log_format is LogFormat.JSON
    assert cfg.level == "DEBUG"


def test_mcp_config_module_has_no_basicconfig() -> None:
    """MCP 侧不再自行 basicConfig：格式统一由 shared.logging 承担。"""
    import inspect

    from services.a2a_mcp import config as a2a_config
    from services.lark_mcp import config as lark_config
    from services.tools_mcp import config as tools_config

    for module in (tools_config, lark_config, a2a_config):
        # 断言「没有调用」而非「文本里没有这个词」：docstring 会提到它（说明为何不这么写）。
        assert "logging.basicConfig(" not in inspect.getsource(module)
        assert hasattr(module, "build_logging_config")


def test_shared_logging_has_no_heavy_imports() -> None:
    """日志底座可被纯 stdlib 容器安全导入：不拖 fastapi / agent 框架 / langchain。"""
    import inspect

    import shared.logging as module

    source = inspect.getsource(module)
    for forbidden in ("fastapi", "langchain", "sqlalchemy", "agent."):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source
    # 事件 logger 名在底座声明（api 与 MCP 侧同名，便于按 logger 过滤结构化事件）。
    from shared.logging import EVENT_LOGGER_NAME

    assert logging.getLogger(EVENT_LOGGER_NAME).name == "srp_agent.event"
