"""services/a2a_mcp/client.py 测试：进程内 fake A2A server（ASGITransport/MockTransport，零网络）。

覆盖：AgentCard 发现、message/send 即终态、非终态轮询、超时、远端协议错误、
failed 任务、SSE 流式聚合、连接失败归一。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from services.a2a_mcp.client import A2AClient, A2AClientError

BASE_URL = "http://fake-a2a"


def _task(state: str, reply: str = "", *, task_id: str = "t1", error: str | None = None) -> dict:
    """构造终态/过程 Task（Server 子集 wire 形态）。"""
    task: dict[str, Any] = {
        "id": task_id,
        "session_id": task_id,
        "state": state,
        "created_at": "2026-01-01T00:00:00Z",
    }
    if reply:
        task["message"] = {"role": "agent", "parts": [{"kind": "text", "text": reply}]}
    if error:
        task["error"] = error
    return task


def _frame(result: dict[str, Any], request_id: Any = 1) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result, "error": None}


def _error_payload(code: int, message: str, a2a_code: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": None,
        "error": {"code": code, "message": message, "data": {"code": a2a_code}},
    }


def build_fake_a2a_app(script: dict[str, Any], capture: dict[str, Any] | None = None) -> FastAPI:
    """按脚本行为的 fake A2A server：send 依次返回 / poll 依次返回 / 固定 error / SSE 帧。

    `capture` 非空时记录最近一次 /a2a 请求的 peer 身份头（供请求契约断言）。
    """
    app = FastAPI()
    state = {"send_idx": 0, "poll_idx": 0}

    @app.get("/.well-known/agent.json")
    async def card() -> dict[str, Any]:
        return {"name": "fake-agent", "url": f"{BASE_URL}/a2a", "capabilities": {"streaming": True}}

    @app.post("/a2a")
    async def a2a(request: Request) -> Any:
        if capture is not None:
            capture["peer_id"] = request.headers.get("x-a2a-peer-id")
        body = await request.json()
        if script.get("error") is not None:
            return JSONResponse(script["error"])
        method = body.get("method")
        request_id = body.get("id")
        if method == "message/send":
            seq: list[dict[str, Any]] = script["send"]
            idx = min(state["send_idx"], len(seq) - 1)
            state["send_idx"] += 1
            return JSONResponse(_frame(seq[idx], request_id))
        if method == "task/get":
            seq = script["poll"]
            idx = min(state["poll_idx"], len(seq) - 1)
            state["poll_idx"] += 1
            return JSONResponse(_frame(seq[idx], request_id))
        if method == "message/stream":
            frames: list[dict[str, Any]] = script["sse"]

            async def generate() -> Any:
                for frame in frames:
                    yield f"event: message\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        raise NotImplementedError(method)  # pragma: no cover

    return app


def _client(app: FastAPI, **kwargs: Any) -> A2AClient:
    """构造指向进程内 fake server 的客户端（ASGITransport，零真实网络）。"""
    defaults: dict[str, Any] = {"request_timeout_s": 5.0, "poll_interval_s": 0.01}
    defaults.update(kwargs)
    return A2AClient(transport=httpx.ASGITransport(app=app), **defaults)


# —— AgentCard 发现 ——


async def test_get_agent_card() -> None:
    client = _client(build_fake_a2a_app({}))
    card = await client.get_agent_card(BASE_URL)
    assert card["name"] == "fake-agent"
    assert card["url"] == f"{BASE_URL}/a2a"


async def test_get_agent_card_unreachable() -> None:
    """连接失败归一为 A2AClientError（MockTransport 模拟 ConnectError）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = A2AClient(transport=httpx.MockTransport(handler))
    with pytest.raises(A2AClientError, match="AgentCard"):
        await client.get_agent_card(BASE_URL)


# —— message/send（同步 + 轮询）——


async def test_call_agent_immediate_completed() -> None:
    """message/send 直接返回终态：reply 提取自 text part，无需轮询。"""
    app = build_fake_a2a_app({"send": [_task("completed", "远端回答")]})
    result = await _client(app).call_agent(base_url=BASE_URL, agent_id="b", message="你好")
    assert result.agent_id == "b"
    assert result.task_id == "t1"
    assert result.state == "completed"
    assert result.reply == "远端回答"


async def test_call_agent_sends_peer_header() -> None:
    """peer 身份随请求头声明（远端按注册表校验调用方，缺头 → a2a.invalid_request）。"""
    seen: dict[str, Any] = {}
    app = build_fake_a2a_app({"send": [_task("completed", "ok")]}, capture=seen)
    client = A2AClient(
        transport=httpx.ASGITransport(app=app), request_timeout_s=5.0, peer_id="agent-a"
    )
    await client.call_agent(base_url=BASE_URL, agent_id="b", message="你好")
    assert seen["peer_id"] == "agent-a"


async def test_call_agent_polls_until_terminal() -> None:
    """非终态（working）→ task/get 轮询至 completed。"""
    app = build_fake_a2a_app(
        {
            "send": [_task("working")],
            "poll": [_task("working"), _task("completed", "轮询后的回答")],
        }
    )
    result = await _client(app).call_agent(base_url=BASE_URL, agent_id="b", message="你好")
    assert result.state == "completed"
    assert result.reply == "轮询后的回答"


async def test_call_agent_poll_timeout() -> None:
    """持续 working → 总预算耗尽 → a2a 超时错误。"""
    app = build_fake_a2a_app({"send": [_task("working")], "poll": [_task("working")]})
    client = _client(app, request_timeout_s=0.2, poll_interval_s=0.05)
    with pytest.raises(A2AClientError, match="超时"):
        await client.call_agent(base_url=BASE_URL, agent_id="b", message="你好")


async def test_call_agent_remote_protocol_error() -> None:
    """远端 JSON-RPC error 信封 → A2AClientError（data.code 透出 a2a.* 码）。"""
    app = build_fake_a2a_app(
        {"error": _error_payload(-32600, "未登记的 peer", "a2a.invalid_request")}
    )
    with pytest.raises(A2AClientError, match=r"a2a.invalid_request"):
        await _client(app).call_agent(base_url=BASE_URL, agent_id="b", message="你好")


async def test_call_agent_failed_task() -> None:
    """远端任务 failed → A2AClientError（error 文案透出，Agent 侧归一执行失败）。"""
    app = build_fake_a2a_app({"send": [_task("failed", error="图运行异常")]})
    with pytest.raises(A2AClientError, match="图运行异常"):
        await _client(app).call_agent(base_url=BASE_URL, agent_id="b", message="你好")


# —— message/stream（SSE）——


async def test_call_agent_stream_aggregates() -> None:
    """SSE 聚合：过程帧收进度摘要，终态 Task 帧取 reply 与 task_id。"""
    app = build_fake_a2a_app(
        {
            "sse": [
                _frame(
                    {
                        "kind": "status-update",
                        "taskId": "t9",
                        "contextId": "t9",
                        "status": {"state": "working", "message": "任务已受理", "delta": None},
                    }
                ),
                _frame(
                    {
                        "kind": "status-update",
                        "taskId": "t9",
                        "contextId": "t9",
                        "status": {"state": "working", "message": None, "delta": "远"},
                    }
                ),
                _frame(_task("completed", "远端流式回答", task_id="t9")),
            ]
        }
    )
    result = await _client(app).call_agent_stream(base_url=BASE_URL, agent_id="b", message="你好")
    assert result.task_id == "t9"
    assert result.state == "completed"
    assert result.reply == "远端流式回答"
    assert result.progress == ["任务已受理"]  # 纯 delta 帧不产进度文本


async def test_call_agent_stream_error_frame() -> None:
    """SSE 流内 JSON-RPC error 帧 → A2AClientError。"""
    app = build_fake_a2a_app({"sse": [_error_payload(-32603, "boom", "a2a.internal_error")]})
    with pytest.raises(A2AClientError, match=r"a2a.internal_error"):
        await _client(app).call_agent_stream(base_url=BASE_URL, agent_id="b", message="你好")
