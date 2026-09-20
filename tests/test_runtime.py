"""AgentRuntime 组合根测试：chat / chat_stream 编排（离线 fake LLM，不依赖真实 LLM/MCP）。

覆盖：
- chat_stream：status 事件先实时下发 + done 兜底完整 AgentResponse（会话编排挂接点）；
- chat（非流式）：与流式 done 语义一致；
- 澄清链路：意图低置信 → finished_reason=needs_clarification + clarification 载荷
  （前端 phase=clarify 的判定依据）。

WHY 不调用 AgentRuntime.create()：create 会拉起 MCP stdio 子进程并读环境配置，
单元测试直接构造 dataclass（注入 fake 图/后端）即可验证编排逻辑；extractor=None
使带外记忆保存跳过（尽力而为路径的挂接由 create 装配保证）。
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore

from agent.core.config import AgentFrameworkConfig
from agent.intent.models import Intent, IntentResult
from agent.memory import MemoryStore
from agent.response.models import (
    FINISHED_REASON_NEEDS_CLARIFICATION,
    AgentResponse,
    ClarifyResult,
)
from agent.runtime import AgentRuntime
from agent.session import build_session_backend
from tests.conftest import (
    chat_turn_messages,
    fake_structured_message,
    make_fake_tool,
    tool_call_messages,
    understand_message,
)


@pytest.fixture
def runtime_factory(build_graph: Any) -> Any:
    """构造注入 fake LLM 的 AgentRuntime（会话用 SQLite memory 后端，零配置）。"""

    async def _make(messages: list[Any], *, tools: list[Any] | None = None) -> AgentRuntime:
        graph = build_graph(messages, tools=tools)
        backend = await build_session_backend(None)
        return AgentRuntime(
            graph=graph,
            sessions=backend.manager,
            memory_store=MemoryStore(InMemoryStore()),
            cfg=AgentFrameworkConfig.get_default(),
            _session_backend=backend,  # 让 aclose() 关闭会话仓库连接池
        )

    return _make


async def test_chat_stream_emits_status_and_done(runtime_factory: Any) -> None:
    """chat_stream：status 事件先实时下发，done 最后兜底完整 AgentResponse。"""
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        events = [
            ev async for ev in runtime.chat_stream(user_id="u1", session_id="s1", text="你好")
        ]
        kinds = [kind for kind, _ in events]
        assert kinds[0] == "status"  # 先有过程状态（thinking）
        assert "done" in kinds  # 终态兜底
        assert events[-1][0] == "done"
        resp = events[-1][1]
        assert isinstance(resp, AgentResponse)
        assert resp.session_id == "s1"
        assert resp.reply == "你好！"
        assert resp.finished_reason == "completed"
    finally:
        await runtime.aclose()


async def test_chat_stream_emits_answer_tokens(runtime_factory: Any) -> None:
    """chat_stream：最终回答 token 增量实时下发——拼接 == reply、speaking 先于 token 且仅一次。"""
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好 世界！"))
    try:
        events = [
            ev async for ev in runtime.chat_stream(user_id="u1", session_id="s1", text="你好")
        ]
        tokens = [payload for kind, payload in events if kind == "token"]
        assert tokens  # 回答存在 token 增量（真流式：空白切分出多段）
        assert "".join(t.delta for t in tokens) == "你好 世界！"
        assert events[-1][0] == "done"
        assert events[-1][1].reply == "你好 世界！"
        # 顺序契约：speaking 恰一次（live 与 updates 去重），且先于首个 token。
        speaking_idx = [
            i
            for i, (kind, payload) in enumerate(events)
            if kind == "status" and payload.status == "speaking"
        ]
        assert len(speaking_idx) == 1
        first_token_idx = next(i for i, (kind, _) in enumerate(events) if kind == "token")
        assert speaking_idx[0] < first_token_idx
    finally:
        await runtime.aclose()


async def test_chat_stream_tool_answer_streams_tokens(runtime_factory: Any) -> None:
    """工具循环：模型 content-only 直答（call_model 内产生终答）也走 token 级流式。

    覆盖「现在几点了」类路径：工具执行后 call_model 直接作答 → content 增量在
    call_model 内实时外发，generate_answer 仅复用，不再是无增量的整段回答。
    """
    # tool_call_messages 已按 T2 口径在意图消息后带上查询理解消息（TOOL_USE 必过门控）。
    messages = tool_call_messages(
        [[{"name": "calc", "args": {}, "id": "call_1"}]], "现在是 14:30。"
    )
    runtime = await runtime_factory(messages, tools=[make_fake_tool("calc", content="14:30")])
    try:
        events = [
            ev async for ev in runtime.chat_stream(user_id="u1", session_id="s1", text="现在几点了")
        ]
        tokens = [payload for kind, payload in events if kind == "token"]
        assert tokens  # 工具循环终答存在 token 增量
        assert "".join(t.delta for t in tokens) == "现在是 14:30。"
        assert events[-1][0] == "done"
        resp = events[-1][1]
        assert resp.reply == "现在是 14:30。"
        # 顺序契约：speaking 恰一次（call_model live 与 generate updates 去重）、先于首个 token。
        speaking_idx = [
            i
            for i, (kind, payload) in enumerate(events)
            if kind == "status" and payload.status == "speaking"
        ]
        assert len(speaking_idx) == 1
        first_token_idx = next(i for i, (kind, _) in enumerate(events) if kind == "token")
        assert speaking_idx[0] < first_token_idx
        # 工具调用过程事件仍在（动画轨迹不因流式而缺失）。
        tool_kinds = [kind for kind, _ in events if kind == "tool"]
        assert tool_kinds
    finally:
        await runtime.aclose()


async def test_chat_stream_direct_tool_intent_answer_streams_tokens(
    runtime_factory: Any,
) -> None:
    """TOOL_USE 意图但模型直接作答（不经工具）→ 同样产生 token 增量。"""
    # v6.0 T2：「直接答」（非 CHAT 短输入）过确定性门控 → 意图消息后带一条查询理解。
    messages = chat_turn_messages(Intent.TOOL_USE, "直接作答文本", understand=True)
    runtime = await runtime_factory(messages)
    try:
        events = [
            ev async for ev in runtime.chat_stream(user_id="u1", session_id="s1", text="直接答")
        ]
        tokens = [payload for kind, payload in events if kind == "token"]
        assert tokens
        assert "".join(t.delta for t in tokens) == "直接作答文本"
        assert events[-1][0] == "done"
        assert events[-1][1].reply == "直接作答文本"
    finally:
        await runtime.aclose()


async def test_chat_returns_response(runtime_factory: Any) -> None:
    """chat（非流式）：直接返回 AgentResponse，与流式 done 语义一致。"""
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        resp = await runtime.chat(user_id="u1", session_id="s1", text="你好")
        assert isinstance(resp, AgentResponse)
        assert resp.reply == "你好！"
        assert resp.status_trace  # 轨迹非空（至少 thinking/speaking）
    finally:
        await runtime.aclose()


async def test_chat_clarify_flow(runtime_factory: Any) -> None:
    """澄清链路：意图低置信 → needs_clarification + clarification 载荷（phase=clarify 依据）。"""
    messages = [
        fake_structured_message(IntentResult(intent=Intent.CHAT, confidence=0.3, reason="模糊")),
        # v6.0 T2：「就是那个你懂的」（7 字、非寒暄）过门控 → 意图分类后补一条查询理解。
        understand_message(),
        fake_structured_message(
            ClarifyResult(question="你是想查询 A 还是 B？", options=["查 A", "查 B"])
        ),
    ]
    runtime = await runtime_factory(messages)
    try:
        resp = await runtime.chat(user_id="u1", session_id="s1", text="就是那个你懂的")
        assert resp.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
        assert resp.clarification is not None
        assert resp.clarification.question == "你是想查询 A 还是 B？"
        assert resp.clarification.options == ["查 A", "查 B"]
        assert resp.reply == resp.clarification.question
    finally:
        await runtime.aclose()
