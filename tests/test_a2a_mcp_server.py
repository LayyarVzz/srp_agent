"""services/a2a_mcp FastMCP 服务测试（进程内 Client + 离线 transport 注入，零网络）。

覆盖：a2a_call_agent 正常调用（同步/流式）、未知 agent_id 错误、输出截断。
fake 远端复用 tests/test_a2a_mcp_client.py 的脚本化 fake server。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_async

from services.a2a_mcp import server as a2a_server
from services.a2a_mcp.client import A2AClient
from services.a2a_mcp.config import A2AMCPRuntimeSettings
from services.a2a_mcp.server import mcp
from tests.test_a2a_mcp_client import _task, build_fake_a2a_app

FAKE_BASE_URL = "http://fake-a2a"


def _patch_remote(monkeypatch: Any, script: dict[str, Any], **settings_kwargs: Any) -> None:
    """把 server 的注册表指向 fake 远端（ASGITransport，零真实网络）。"""
    app = build_fake_a2a_app(script)
    defaults: dict[str, Any] = {"a2a_mcp_agents": {"b": FAKE_BASE_URL}}
    defaults.update(settings_kwargs)
    settings = A2AMCPRuntimeSettings(**defaults)
    offline_client = A2AClient(
        request_timeout_s=settings.a2a_request_timeout_s,
        poll_interval_s=0.01,
        transport=httpx.ASGITransport(app=app),
    )
    monkeypatch.setattr(a2a_server, "_get_settings", lambda: settings)
    monkeypatch.setattr(a2a_server, "_build_client", lambda _settings: offline_client)


@pytest.fixture
async def tool_client(monkeypatch: Any) -> Client:
    """同进程 fastmcp 客户端 + fake 远端（send 即终态）。"""
    _patch_remote(monkeypatch, {"send": [_task("completed", "远端回答")]})
    async with Client(mcp) as client:
        yield client


async def test_a2a_call_agent_ok(tool_client: Client) -> None:
    """正常调用：结构化 JSON 输出（agent_id/task_id/state/reply）。"""
    res = await tool_client.call_tool(
        "a2a_call_agent", {"agent_id": "b", "message": "帮我查一下"}
    )
    assert res.is_error is False
    payload = json.loads(res.content[0].text)
    assert payload == {
        "agent_id": "b",
        "task_id": "t1",
        "state": "completed",
        "reply": "远端回答",
        "progress": [],
    }


async def test_a2a_call_agent_unknown_agent_id(tool_client: Client) -> None:
    """注册表缺失 agent_id → 工具错误结果（ValueError 归一）。"""
    res = await tool_client.call_tool(
        "a2a_call_agent",
        {"agent_id": "nobody", "message": "你好"},
        raise_on_error=False,
    )
    assert res.is_error is True


async def test_a2a_call_agent_stream(monkeypatch: Any) -> None:
    """stream=True：走 SSE，progress 承载过程帧文本。"""
    _patch_remote(
        monkeypatch,
        {
            "sse": [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "kind": "status-update",
                        "taskId": "t2",
                        "contextId": "t2",
                        "status": {"state": "working", "message": "任务已受理", "delta": None},
                    },
                    "error": None,
                },
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": _task("completed", "流式回答", task_id="t2"),
                    "error": None,
                },
            ]
        },
    )
    async with Client(mcp) as client:
        res = await client.call_tool(
            "a2a_call_agent", {"agent_id": "b", "message": "你好", "stream": True}
        )
    assert res.is_error is False
    payload = json.loads(res.content[0].text)
    assert payload["task_id"] == "t2"
    assert payload["reply"] == "流式回答"
    assert payload["progress"] == ["任务已受理"]


async def test_a2a_call_agent_truncates_long_reply(monkeypatch: Any) -> None:
    """超长 reply 服务侧截断（JSON 结构完整，结构化字段不受影响）。"""
    _patch_remote(
        monkeypatch,
        {"send": [_task("completed", "x" * 50)]},
        a2a_output_max_chars=10,
    )
    async with Client(mcp) as client:
        res = await client.call_tool("a2a_call_agent", {"agent_id": "b", "message": "你好"})
    payload = json.loads(res.content[0].text)
    assert len(payload["reply"]) == 10 + len("…（已截断）")
    assert payload["state"] == "completed"


# —— streamable-http 传输：验证远程部署形态（进程内真实 HTTP，动态空闲端口）——


async def test_streamable_http_e2e(monkeypatch: Any) -> None:
    """以 streamable-http 起真实端点：工具已注册、health 可探（注册表空 → 调用报错）。"""
    monkeypatch.setattr(a2a_server, "_settings", A2AMCPRuntimeSettings(a2a_mcp_agents={}))
    async with run_server_async(mcp, transport="streamable-http") as url:
        async with Client(StreamableHttpTransport(url)) as client:
            tools = await client.list_tools()
            assert {t.name for t in tools} == {"a2a_call_agent"}
            res = await client.call_tool(
                "a2a_call_agent",
                {"agent_id": "b", "message": "你好"},
                raise_on_error=False,
            )
            assert res.is_error is True  # 注册表为空 → 未注册智能体错误
