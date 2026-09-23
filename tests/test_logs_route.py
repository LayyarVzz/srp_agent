"""日志/事件路由单测（Phase C / C5）：`/api/v1/logs/recent` 与 `/logs/trace/{id}` 查库。

**端到端验收**是本文件的核心：真发一次 HTTP 请求 → 结构化事件经带外 sink 落库 →
接口读回同一条链路。这条链路验证的正是 C5 想解决的问题（进程内 deque 在多 worker
下「一半请求查得到、一半查不到」）。

`api_app_factory` 已注入隔离事件库（内存 SQLite）与未启动的 sink；本文件用
`_start_observability(app)` 显式驱动「生产 lifespan 的观测启动段」（ASGITransport
不触发 lifespan），保证测的是同一段启动逻辑而不是另写一套。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from agent.intent.models import Intent
from shared.events_store import EventRepository, SQLAlchemyEventRepository, build_event_repository
from shared.logging import (
    EVENT_INTENT_CLASSIFIED,
    EVENT_REQUEST_FINISHED,
    EVENT_REQUEST_RECEIVED,
    clear_context,
    subscribe_events,
)
from tests.conftest import chat_turn_messages, tool_call_messages

HEADERS = {"X-User-Id": "demo-user"}
CHAT_URL = "/api/v1/interactions/text"
LOGS_URL = "/api/v1/logs/recent"


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """用例前后清空关联标识（避免上一个用例的 user_id 串到本用例事件里）。"""
    clear_context()
    yield
    clear_context()


def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _repository(app: Any) -> EventRepository:
    """取应用注入的事件仓库（与 sink 写库的是同一实例）。"""
    repo: EventRepository = app.state.event_repository
    return repo


def _start_observability(app: Any) -> None:
    """启动观测写入：订阅 sink + 启线程（等价于生产 lifespan 的启动段）。"""
    subscribe_events(app.state.event_sink.send)
    app.state.event_sink.start()


async def _wait_for_events(repository: EventRepository, *, at_least: int = 1) -> None:
    """等待 sink 把事件写进库（带外写是异步的，测试不能假设入队即已落库）。"""
    for _ in range(50):
        if await repository.count() >= at_least:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"等待事件落库超时（当前 {await repository.count()} 条）")


async def _wait_for_trace_finished(repository: EventRepository, trace_id: str) -> None:
    """等**指定 trace** 的 `request.finished` 落库：它是「这条链路已完整写下」的终点标志。

    WHY 不等条数：一次测试里会发多个 HTTP 请求（拿会话 + 查日志），条数是含混的 ——
    等条数会与「另一个请求的事件交错」赛跑（表现为偶发多出一条别的 trace 的事件）。
    """
    for _ in range(50):
        records = await repository.list_by_trace(trace_id=trace_id)
        if any(r.event == EVENT_REQUEST_FINISHED for r in records):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"等待 {trace_id} 的 request.finished 落库超时")


async def test_recent_returns_events_of_real_request(api_app_factory: Any) -> None:
    """真发一次请求 → 事件落库 → `/logs/recent` 读回（HTTP 口径与 Agent 口径都在）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    repository = _repository(app)
    _start_observability(app)
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "你好"})
            assert resp.status_code == 200
            trace_id = resp.headers["X-Request-Id"]
            await _wait_for_trace_finished(repository, trace_id)
            logs = await client.get(LOGS_URL, headers=HEADERS)
        assert logs.status_code == 200
    finally:
        await runtime.aclose()

    # 本次对话的事件全在同一 trace 下（`/logs/recent` 的过滤口径即 user，不带 trace）。
    body = logs.json()
    assert body["ok"] is True
    assert body["limit"] == 50
    events = [e for e in body["events"] if e["trace_id"] == trace_id]
    names = [e["event"] for e in events]
    assert EVENT_REQUEST_RECEIVED in names
    assert EVENT_REQUEST_FINISHED in names
    assert all(e["user_id"] == "demo-user" for e in events)
    assert all(e["service"] == "api" for e in events)
    # 会话可回溯：入口中间件先于「自动建会话」发生，故 `request.received` 允许无 session_id，
    # 但请求收尾与图内事件必须带上（否则一条链路无法归到某个会话）。
    finished = next(e for e in events if e["event"] == EVENT_REQUEST_FINISHED)
    assert finished["session_id"]
    assert any(e["session_id"] for e in events)
    # 耗时是可聚合的列（不是塞在 payload 里的字符串）。
    assert finished["duration_ms"] >= 0
    assert finished["status"] == "completed"


async def test_all_events_after_received_carry_session(api_app_factory: Any) -> None:
    """已存在会话的一轮：`request.received` **之后**的事件都带 session_id。

    `request.received` 是入口第一件事（那时还不知道权威会话 id，除非调用方带
    `X-Session-Id` 头），除此之外整条链路必须可归到会话 —— 否则「按会话排查」断链。
    """
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    repository = _repository(app)
    _start_observability(app)
    try:
        async with _client(app) as client:
            created = await client.post("/api/v1/sessions", headers=HEADERS)
            session_id = created.json()["session_id"]
            resp = await client.post(
                CHAT_URL,
                headers={**HEADERS, "X-Request-Id": "req-sess", "X-Session-Id": session_id},
                json={"text": "你好", "session_id": session_id},
            )
            await _wait_for_trace_finished(repository, "req-sess")
            records = await repository.list_by_trace(trace_id="req-sess")
    finally:
        await runtime.aclose()

    assert records, "该 trace 无事件"
    after_received = [r for r in records if r.event != EVENT_REQUEST_RECEIVED]
    missing = [r.event for r in after_received if r.session_id != session_id]
    assert missing == [], f"这些事件未带 session_id：{missing}"
    # 带上 `X-Session-Id` 头时，入口事件也能直接归到会话。
    received = next(r for r in records if r.event == EVENT_REQUEST_RECEIVED)
    assert received.session_id == session_id


async def test_trace_endpoint_returns_ordered_chain(api_app_factory: Any) -> None:
    """`/logs/trace/{trace_id}` 返回一条链路的事件且**升序**（真实发生顺序）。"""
    messages = tool_call_messages([[{"name": "calc", "args": {}, "id": "c1"}]], "算好了")
    app, runtime = await api_app_factory(messages)
    repository = _repository(app)
    _start_observability(app)
    try:
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL, headers={**HEADERS, "X-Request-Id": "req-trace"}, json={"text": "算一下"}
            )
            trace_id = resp.headers["X-Request-Id"]
            await _wait_for_trace_finished(repository, trace_id)
            trace = await client.get(f"/api/v1/logs/trace/{trace_id}", headers=HEADERS)
        assert trace.status_code == 200
    finally:
        await runtime.aclose()

    body = trace.json()
    assert body["trace_id"] == "req-trace"
    names = [e["event"] for e in body["events"]]
    assert names[0] == EVENT_REQUEST_RECEIVED
    assert names[-1] == EVENT_REQUEST_FINISHED
    assert names.index(EVENT_INTENT_CLASSIFIED) < names.index(EVENT_REQUEST_FINISHED)
    assert all(e["trace_id"] == "req-trace" for e in body["events"])


async def test_trace_endpoint_unknown_id_is_empty(api_app_factory: Any) -> None:
    """未知 trace_id → 200 + 空列表（排查口不该把「没有」报成错误）。"""
    app, runtime = await api_app_factory([])
    try:
        async with _client(app) as client:
            resp = await client.get("/api/v1/logs/trace/req_nonexistent", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["events"] == []
    finally:
        await runtime.aclose()


async def test_recent_is_isolated_by_user(api_app_factory: Any) -> None:
    """归属隔离：另一个用户查不到 demo-user 的事件（接口层不得跨用户泄漏）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    repository = _repository(app)
    _start_observability(app)
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "你好"})
            await _wait_for_trace_finished(repository, resp.headers["X-Request-Id"])
            mine = await client.get(LOGS_URL, headers=HEADERS)
            others = await client.get(LOGS_URL, headers={"X-User-Id": "someone-else"})
        assert mine.json()["events"], "自己的事件应可见"
        assert others.json()["events"] == [], "其他用户不得看到别人的事件"
    finally:
        await runtime.aclose()


async def test_recent_requires_user_header(api_app_factory: Any) -> None:
    """缺 `X-User-Id` → 401（与 sessions/chat 同一身份约定，防匿名枚举事件）。"""
    app, runtime = await api_app_factory([])
    try:
        async with _client(app) as client:
            resp = await client.get(LOGS_URL)
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "auth.identity_required"
    finally:
        await runtime.aclose()


async def test_recent_limit_is_bounded(api_app_factory: Any) -> None:
    """`limit` 参数越界 → 422（契约由 FastAPI 校验，路由不手写边界）。"""
    app, runtime = await api_app_factory([])
    try:
        async with _client(app) as client:
            too_big = await client.get(LOGS_URL, headers=HEADERS, params={"limit": 999})
            too_small = await client.get(LOGS_URL, headers=HEADERS, params={"limit": 0})
        assert too_big.status_code == 422
        assert too_small.status_code == 422
    finally:
        await runtime.aclose()


async def test_recent_respects_limit(api_app_factory: Any) -> None:
    """`limit` 生效且回显（前端可对账「我请求的上限是否被采纳」）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    repository = _repository(app)
    _start_observability(app)
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "你好"})
            await _wait_for_trace_finished(repository, resp.headers["X-Request-Id"])
            limit_resp = await client.get(LOGS_URL, headers=HEADERS, params={"limit": 2})
        assert limit_resp.json()["limit"] == 2
        assert len(limit_resp.json()["events"]) == 2
    finally:
        await runtime.aclose()


async def test_recent_orders_newest_first(api_app_factory: Any) -> None:
    """最新事件排最前（排查时第一屏就是刚发生的事）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好"))
    repository = _repository(app)
    _start_observability(app)
    try:
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "你好"})
            await _wait_for_trace_finished(repository, resp.headers["X-Request-Id"])
            events = (await client.get(LOGS_URL, headers=HEADERS)).json()["events"]
        timestamps = [e["ts"] for e in events]
        assert timestamps == sorted(timestamps, reverse=True)
    finally:
        await runtime.aclose()


def test_dead_logging_service_is_removed() -> None:
    """`app/logging_service.py` 死代码必须已被删除（C5 清理项）。

    WHY 用文件存在性断言：该模块全仓库零调用，任何「先留着」的改动都会让「日志/事件
    唯一口径」重新分叉（进程内 deque 与事件表两套读源，多 worker 下必然不一致）。
    """
    from pathlib import Path

    assert not Path("app/logging_service.py").exists()


def test_events_repository_is_sqlalchemy_impl() -> None:
    """生产装配产出 SQLAlchemy 实现（dev 也走同一实现，只换 DSN/方言）。"""
    repo = build_event_repository(None)
    assert isinstance(repo, SQLAlchemyEventRepository)
