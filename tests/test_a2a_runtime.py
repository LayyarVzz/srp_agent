"""AgentRuntime A2A 入站编排测试（fake LLM 注入，离线驱动真图）。

覆盖：任务完成（session_id 即 task_id）、peer 命名空间隔离、未登记/禁用/关闭拒绝、
显式 user_id 映射、流式事件序（started → status/token → done）。
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.a2a.models import A2APeer
from agent.a2a.protocol import A2A_ERROR_INVALID_REQUEST, A2AProtocolError
from agent.core.config import AgentFrameworkConfig
from agent.intent.models import Intent
from agent.runtime import AgentRuntime, _build_mcp_servers
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings
from tests.conftest import chat_turn_messages

PEER_ID = "agent-b"
PEER_HEADERS_USER = "a2a:agent-b"


async def _make_runtime(
    api_runtime_factory: Any,
    *,
    peers: list[A2APeer] | None = None,
    enabled: bool = True,
) -> AgentRuntime:
    """构造注入 fake LLM 的 runtime，并按需注册入站 peer（默认 agent-b）。"""
    runtime = await api_runtime_factory(chat_turn_messages(Intent.CHAT, "peer 你好"))
    runtime.cfg.a2a.enabled = enabled
    for peer in peers or [A2APeer(id=PEER_ID)]:
        runtime.cfg.a2a.peers[peer.id] = peer
    return runtime


async def test_run_a2a_task_completes(api_runtime_factory: Any) -> None:
    """任务完成：session_id 即 task_id；会话登记在 peer 命名空间下。"""
    runtime = await _make_runtime(api_runtime_factory)
    try:
        resp = await runtime.run_a2a_task(peer_id=PEER_ID, text="你好")
        assert resp.reply == "peer 你好"
        assert resp.session_id
        # task↔session 一一对应：以 peer 虚拟用户可解析该会话（归属校验通过）
        ctx = await runtime.sessions.resolve(user_id=PEER_HEADERS_USER, session_id=resp.session_id)
        assert ctx.session_id == resp.session_id
    finally:
        await runtime.aclose()


async def test_peer_namespace_isolation(api_runtime_factory: Any) -> None:
    """peer 会话与人类用户会话按 user_id 隔离（匿名命名空间不混入人类数据）。"""
    runtime = await _make_runtime(api_runtime_factory)
    try:
        await runtime.run_a2a_task(peer_id=PEER_ID, text="你好")
        assert len(await runtime.sessions.list(user_id=PEER_HEADERS_USER)) == 1
        assert await runtime.sessions.list(user_id="demo-user") == []
    finally:
        await runtime.aclose()


async def test_explicit_user_id_mapping(api_runtime_factory: Any) -> None:
    """配置显式映射的 peer 以固定 user_id 入会话（可信 peer 身份）。"""
    runtime = await _make_runtime(
        api_runtime_factory, peers=[A2APeer(id="svc", user_id="svc-user")]
    )
    try:
        resp = await runtime.run_a2a_task(peer_id="svc", text="你好")
        ctx = await runtime.sessions.resolve(user_id="svc-user", session_id=resp.session_id)
        assert ctx.session_id == resp.session_id
    finally:
        await runtime.aclose()


async def test_unregistered_peer_rejected(api_runtime_factory: Any) -> None:
    """未登记 peer → a2a.invalid_request（防开放匿名滥用）。"""
    runtime = await _make_runtime(api_runtime_factory)
    try:
        with pytest.raises(A2AProtocolError) as exc_info:
            await runtime.run_a2a_task(peer_id="nobody", text="你好")
        assert exc_info.value.a2a_code == A2A_ERROR_INVALID_REQUEST
    finally:
        await runtime.aclose()


async def test_disabled_peer_rejected(api_runtime_factory: Any) -> None:
    """已登记但禁用的 peer 同样拒绝。"""
    runtime = await _make_runtime(
        api_runtime_factory, peers=[A2APeer(id=PEER_ID, enabled=False)]
    )
    try:
        with pytest.raises(A2AProtocolError):
            await runtime.run_a2a_task(peer_id=PEER_ID, text="你好")
    finally:
        await runtime.aclose()


async def test_a2a_disabled_rejects_all(api_runtime_factory: Any) -> None:
    """A2A 入站关闭时已登记 peer 也被拒（零回归开关）。"""
    runtime = await _make_runtime(api_runtime_factory, enabled=False)
    try:
        with pytest.raises(A2AProtocolError) as exc_info:
            await runtime.run_a2a_task(peer_id=PEER_ID, text="你好")
        assert exc_info.value.a2a_code == A2A_ERROR_INVALID_REQUEST
    finally:
        await runtime.aclose()


async def test_stream_events_order(api_runtime_factory: Any) -> None:
    """流式事件序：started 携带 task_id，done 最后且 session_id 一致。"""
    runtime = await _make_runtime(api_runtime_factory)
    try:
        events: list[tuple[str, Any]] = []
        async for event, payload in runtime.run_a2a_task_stream(peer_id=PEER_ID, text="你好"):
            events.append((event, payload))
        assert events[0][0] == "started"
        task_id = events[0][1]
        assert events[-1][0] == "done"
        response = events[-1][1]
        assert response.session_id == task_id
        assert response.reply == "peer 你好"
        kinds = [e for e, _ in events]
        assert kinds[0] == "started" and kinds[-1] == "done"
        assert "status" in kinds  # 过程状态轨迹存在（thinking 等）
    finally:
        await runtime.aclose()


async def test_registry_is_per_runtime(api_runtime_factory: Any) -> None:
    """a2a_tasks 注册表为 runtime 实例级（dataclass default_factory）。"""
    runtime_a = await _make_runtime(api_runtime_factory)
    runtime_b = await _make_runtime(api_runtime_factory)
    try:
        assert runtime_a.a2a_tasks is not runtime_b.a2a_tasks
    finally:
        await runtime_a.aclose()
        await runtime_b.aclose()


# —— T6 装配门控：a2a_mcp 仅在「启用 + 注册表非空」时登记（零回归）——


def test_build_mcp_servers_skips_a2a_when_registry_empty() -> None:
    """注册表为空 → a2a_mcp 不登记，其余服务不受影响（工具集与既有完全一致）。"""
    settings = RuntimeSettings(mcp_transport=MCPTransport.STDIO, a2a_mcp_agents={})
    servers = _build_mcp_servers(settings, AgentFrameworkConfig.get_default())
    assert "a2a_mcp" not in servers
    assert "tools_mcp" in servers and "rag" in servers


def test_build_mcp_servers_disabled_flag_blocks_a2a() -> None:
    """a2a_mcp_enabled=False 为总开关：注册表非空也不登记。"""
    settings = RuntimeSettings(
        mcp_transport=MCPTransport.STDIO,
        a2a_mcp_enabled=False,
        a2a_mcp_agents={"b": "http://127.0.0.1:8002"},
    )
    assert "a2a_mcp" not in _build_mcp_servers(settings, AgentFrameworkConfig.get_default())


def test_build_mcp_servers_registers_a2a_stdio() -> None:
    """stdio 形态：启用 + 注册表非空 → a2a_mcp 以子进程连接配置登记。"""
    settings = RuntimeSettings(
        mcp_transport=MCPTransport.STDIO,
        a2a_mcp_agents={"b": "http://127.0.0.1:8002"},
    )
    conn = _build_mcp_servers(settings, AgentFrameworkConfig.get_default())["a2a_mcp"]
    assert conn["transport"] == "stdio"


def test_build_mcp_servers_registers_a2a_http() -> None:
    """streamable-http 形态：a2a_mcp 用独立 host/port（容器部署 = 服务名）。"""
    settings = RuntimeSettings(
        mcp_transport=MCPTransport.STREAMABLE_HTTP,
        a2a_mcp_agents={"b": "http://x"},
        a2a_mcp_host="a2a_mcp",
        a2a_mcp_port=8102,
    )
    conn = _build_mcp_servers(settings, AgentFrameworkConfig.get_default())["a2a_mcp"]
    assert conn == {"transport": "streamable_http", "url": "http://a2a_mcp:8102/mcp"}
