"""A2A HTTP 端点离线测试（fake LLM 注入，ASGITransport 直测）。

覆盖：AgentCard 发现、message/send、message/stream SSE 帧序、task/get、
task/cancel、peer 身份/参数/method/解析错误的 JSON-RPC 错误信封、A2A 关闭语义。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Any

from httpx import ASGITransport, AsyncClient

from agent.a2a.models import A2APeer
from agent.intent.models import Intent
from tests.conftest import chat_turn_messages

PEER = "agent-b"
PEER_HEADERS = {"X-A2a-Peer-Id": PEER}
A2A_URL = "/a2a"
CARD_URL = "/.well-known/agent.json"


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """解析 SSE 帧序列：`event: X` + `data: {...}`，返回 [(event, data), ...]。"""
    frames: list[tuple[str, dict[str, Any]]] = []
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        ev = re.search(r"^event: (.+)$", frame, re.M)
        data = re.search(r"^data: (.+)$", frame, re.M)
        assert ev is not None and data is not None, f"非法 SSE 帧: {frame!r}"
        frames.append((ev.group(1), json.loads(data.group(1))))
    return frames


def rpc_request(method: str, params: dict[str, Any] | None = None, request_id: Any = "r1") -> dict:
    """构造 JSON-RPC 2.0 请求体。"""
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def message_params(text: str) -> dict[str, Any]:
    """构造 message/send / message/stream 入参（user 角色 + 单 text part）。"""
    return {"message": {"role": "user", "parts": [{"kind": "text", "text": text}]}}


async def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _app_with_peer(api_app_factory: Any) -> tuple[Any, Any]:
    """构造应用 + 注册 agent-b peer 的 runtime（测试负责 aclose）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "peer 你好"))
    runtime.cfg.a2a.peers[PEER] = A2APeer(id=PEER)
    return app, runtime


def _err_code(body: dict[str, Any]) -> str | None:
    error = body.get("error") or {}
    data = error.get("data") or {}
    code = data.get("code")
    return code if isinstance(code, str) else None


# —— AgentCard 发现 ——


async def test_agent_card_discovery(api_app_factory: Any) -> None:
    """GET /.well-known/agent.json：url 动态指向本实例 /a2a，能力矩阵如实声明。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.get(CARD_URL)
        assert resp.status_code == 200
        card = resp.json()
        assert card["url"] == "http://testserver/a2a"
        assert card["protocolVersion"] == "0.3.0"
        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"]["pushNotifications"] is False
        assert len(card["skills"]) == 1
    finally:
        await runtime.aclose()


async def test_agent_card_available_when_a2a_disabled(api_app_factory: Any) -> None:
    """A2A 关闭时发现不受影响（关闭只拒绝方法调用）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "peer 你好"))
    runtime.cfg.a2a.enabled = False
    try:
        async with await _client(app) as client:
            resp = await client.get(CARD_URL)
        assert resp.status_code == 200
    finally:
        await runtime.aclose()


# —— message/send ——


async def test_message_send_ok(api_app_factory: Any) -> None:
    """message/send：同步返回终态 Task（completed + reply text Part），并登记注册表。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL,
                headers=PEER_HEADERS,
                json=rpc_request("message/send", message_params("你好")),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == "r1"
        result = body["result"]
        assert result["state"] == "completed"
        assert result["id"] == result["session_id"]  # task↔session 一一对应
        assert result["message"]["role"] == "agent"
        assert result["message"]["parts"][0]["text"] == "peer 你好"
        assert result["finished_reason"] == "completed"
        # 终态登记：同 task_id 可 task/get 复查
        async with await _client(app) as client:
            resp2 = await client.post(
                A2A_URL, headers=PEER_HEADERS, json=rpc_request("task/get", {"id": result["id"]})
            )
        assert resp2.json()["result"]["id"] == result["id"]
    finally:
        await runtime.aclose()


async def test_message_send_requires_peer_header(api_app_factory: Any) -> None:
    """缺少 X-A2a-Peer-Id → a2a.invalid_request。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL, json=rpc_request("message/send", message_params("你好"))
            )
        assert resp.status_code == 200  # 协议错误走 JSON-RPC 信封，HTTP 恒 200
        body = resp.json()
        assert body["result"] is None
        assert _err_code(body) == "a2a.invalid_request"
    finally:
        await runtime.aclose()


async def test_message_send_unregistered_peer(api_app_factory: Any) -> None:
    """未登记 peer → a2a.invalid_request（防开放匿名滥用）。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL,
                headers={"X-A2a-Peer-Id": "nobody"},
                json=rpc_request("message/send", message_params("你好")),
            )
        assert _err_code(resp.json()) == "a2a.invalid_request"
    finally:
        await runtime.aclose()


async def test_message_send_empty_text(api_app_factory: Any) -> None:
    """message 缺少非空 text part → a2a.invalid_request。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL,
                headers=PEER_HEADERS,
                json=rpc_request("message/send", {"message": {}}),
            )
        assert _err_code(resp.json()) == "a2a.invalid_request"
    finally:
        await runtime.aclose()


async def test_message_send_a2a_disabled(api_app_factory: Any) -> None:
    """A2A 入站关闭 → 已登记 peer 也被拒（零回归开关）。"""
    app, runtime = await _app_with_peer(api_app_factory)
    runtime.cfg.a2a.enabled = False
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL,
                headers=PEER_HEADERS,
                json=rpc_request("message/send", message_params("你好")),
            )
        assert _err_code(resp.json()) == "a2a.invalid_request"
    finally:
        await runtime.aclose()


# —— 协议层错误 ——


async def test_method_not_supported(api_app_factory: Any) -> None:
    """未知 method → -32601 + a2a.method_not_supported。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL, headers=PEER_HEADERS, json=rpc_request("no/such/method")
            )
        body = resp.json()
        assert body["error"]["code"] == -32601
        assert _err_code(body) == "a2a.method_not_supported"
    finally:
        await runtime.aclose()


async def test_parse_error(api_app_factory: Any) -> None:
    """非法 JSON → -32700，id 回带 None（无法解析出请求 id）。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(A2A_URL, headers=PEER_HEADERS, content="not json")
        body = resp.json()
        assert body["error"]["code"] == -32700
        assert body["id"] is None
    finally:
        await runtime.aclose()


async def test_invalid_jsonrpc_version(api_app_factory: Any) -> None:
    """jsonrpc 字段非 "2.0" → -32600。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL, json={"jsonrpc": "1.0", "id": "r1", "method": "task/get", "params": {}}
            )
        assert resp.json()["error"]["code"] == -32600
    finally:
        await runtime.aclose()


# —— task/get / task/cancel ——


async def test_task_get_unknown(api_app_factory: Any) -> None:
    """未命中 task_id → a2a.task_not_found。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL, headers=PEER_HEADERS, json=rpc_request("task/get", {"id": "nope"})
            )
        assert _err_code(resp.json()) == "a2a.task_not_found"
    finally:
        await runtime.aclose()


async def test_task_get_requires_id(api_app_factory: Any) -> None:
    """task/get 缺 params.id → a2a.invalid_request。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL, headers=PEER_HEADERS, json=rpc_request("task/get", {})
            )
        assert _err_code(resp.json()) == "a2a.invalid_request"
    finally:
        await runtime.aclose()


async def test_task_cancel_working_task(api_app_factory: Any) -> None:
    """task/cancel：取消进行中任务（注册表中绑定了真实执行句柄）→ canceled。"""

    app, runtime = await _app_with_peer(api_app_factory)
    try:
        registry = runtime.a2a_tasks
        registry.register("t-work")
        registry.mark_working("t-work")

        async def _forever() -> None:
            await asyncio.Event().wait()

        handle = asyncio.get_running_loop().create_task(_forever())
        registry.attach_handle("t-work", handle)
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL, headers=PEER_HEADERS, json=rpc_request("task/cancel", {"id": "t-work"})
            )
        assert resp.json()["result"]["state"] == "canceled"
        with contextlib.suppress(asyncio.CancelledError):
            await handle  # 等待取消落地（cancelled() 需任务真正结束）
        assert handle.cancelled()
    finally:
        await runtime.aclose()


async def test_task_cancel_completed_task(api_app_factory: Any) -> None:
    """取消已终态任务 → a2a.task_not_cancelable。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            send = await client.post(
                A2A_URL,
                headers=PEER_HEADERS,
                json=rpc_request("message/send", message_params("你好")),
            )
            task_id = send.json()["result"]["id"]
            resp = await client.post(
                A2A_URL, headers=PEER_HEADERS, json=rpc_request("task/cancel", {"id": task_id})
            )
        assert _err_code(resp.json()) == "a2a.task_not_cancelable"
    finally:
        await runtime.aclose()


# —— message/stream（SSE）——


async def test_message_stream_frames(api_app_factory: Any) -> None:
    """SSE 帧序：受理帧（宣告 task_id）→ 过程帧（status/token）→ 终态 Task 帧。"""
    app, runtime = await _app_with_peer(api_app_factory)
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL,
                headers=PEER_HEADERS,
                json=rpc_request("message/stream", message_params("你好")),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = parse_sse(resp.text)
        assert all(event == "message" for event, _ in frames)
        # 首帧：受理（status-update，message 文本）
        first = frames[0][1]
        assert first["result"]["kind"] == "status-update"
        assert first["result"]["status"]["message"] == "任务已受理"
        task_id = first["result"]["taskId"]
        # 末帧：终态 Task（无 kind 字段）
        last = frames[-1][1]
        assert "kind" not in last["result"]
        assert last["result"]["id"] == task_id
        assert last["result"]["state"] == "completed"
        assert last["result"]["message"]["parts"][0]["text"] == "peer 你好"
        # 中段至少包含一个 token 过程帧（回答增量预览）
        assert any(f[1]["result"]["status"]["delta"] for f in frames[1:-1])
        # 任务终态已登记注册表（task/get 可复查）
        assert runtime.a2a_tasks.get(task_id).state.value == "completed"
    finally:
        await runtime.aclose()


async def test_message_stream_internal_error_frame(api_app_factory: Any) -> None:
    """流中未预期异常 → JSON-RPC error 帧（a2a.internal_error），HTTP 仍 200。"""
    app, runtime = await _app_with_peer(api_app_factory)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        async def _gen() -> Any:
            raise RuntimeError("boom")
            yield None  # pragma: no cover

        return _gen()

    runtime.run_a2a_task_stream = _boom  # type: ignore[method-assign]
    try:
        async with await _client(app) as client:
            resp = await client.post(
                A2A_URL,
                headers=PEER_HEADERS,
                json=rpc_request("message/stream", message_params("你好")),
            )
        assert resp.status_code == 200
        frames = parse_sse(resp.text)
        assert len(frames) == 1
        body = frames[0][1]
        assert body["error"]["code"] == -32603
        assert _err_code(body) == "a2a.internal_error"
    finally:
        await runtime.aclose()
