"""图级集成：P4-2 带外保存落盘 + P4-3 长期记忆召回路径（预加载/按需召回）。

P4-2 验收「跨轮对话后 Store 出现抽取结果；回答零延迟」：
零延迟由 submit_memory_save 同步返回保证（fire-and-forget，不阻塞 astream）；
落盘由 wait_pending_saves 后 recall 断言。

P4-3 验收「用户偏好跨会话生效；引用/来源正确」：
load_context 预加载 preference → memory_context；TOOL_USE 分支经 recall_memory
注入 fact/episode；来源进 citations；memory_context 每轮普通覆盖不累积。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.store.memory import InMemoryStore

from agent.core.config import AgentFrameworkConfig, MemoryBehaviorConfig
from agent.core.graph import _MEMORY_BLOCK_HEADER, _render_memory_block, build_agent_graph
from agent.intent.models import Intent
from agent.memory import (
    KIND_EPISODE,
    KIND_FACT,
    KIND_PREFERENCE,
    MemoryStore,
    submit_memory_save,
    wait_pending_saves,
)
from agent.memory.models import MemoryExtraction, MemoryItem
from agent.response.status import Status
from tests.conftest import (
    RecordingFakeChatModel,
    chat_turn_messages,
    make_fake_tool,
    tool_call_messages,
)


class _StubExtractor:
    """固定返回一条抽取结果的 stub 抽取器。"""

    def __init__(self, extraction: MemoryExtraction) -> None:
        self._extraction = extraction

    async def extract(self, messages: Sequence[BaseMessage]) -> list[MemoryExtraction]:
        return [self._extraction]


async def test_after_graph_ends_memory_lands_in_store(build_graph, run_graph) -> None:
    """图跑完一轮对话后，带外保存把抽取结果落进注入的同一 Store 实例。"""
    injected = InMemoryStore()
    graph = build_graph(
        chat_turn_messages(Intent.CHAT, "你好，我是你的助手"),
        store=injected,
    )
    response, _ = await run_graph(graph, text="我叫小明，请记住")
    assert response is not None

    # 回答已下发（END），读最终 state 的完整 messages 提交带外保存。
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    store = MemoryStore(injected)  # 与图共用同一 BaseStore 实例
    submit_memory_save(
        state.values.get("messages") or [],
        session_id="s1",
        user_id="anonymous",
        extractor=_StubExtractor(
            MemoryExtraction(kind="fact", content="用户叫小明", importance=0.8)
        ),
        store=store,
    )
    await wait_pending_saves()

    result = await store.recall(user_id="anonymous")
    assert any(m.content == "用户叫小明" for m in result.items)


# —— P4-3 长期记忆召回路径 ——


def _memory_item(
    *,
    id: str,
    kind: str,
    content: str,
    user_id: str = "anonymous",
    importance: float = 0.5,
) -> MemoryItem:
    """构造一条种子长期记忆（provenance=seed，时间戳取当前时刻）。"""
    return MemoryItem(
        id=id,
        kind=kind,
        content=content,
        session_id="s-seed",
        user_id=user_id,
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=importance,
    )


async def _save(store: InMemoryStore, item: MemoryItem) -> None:
    """把种子记忆写入注入的 BaseStore（经 MemoryStore 适配层）。"""
    await MemoryStore(store).save(item)


async def _memory_context_of(graph: Any, session_id: str) -> list[MemoryItem]:
    """读 checkpointer 最终状态里的 memory_context。"""
    snap = await graph.aget_state({"configurable": {"thread_id": session_id}})
    return list(snap.values.get("memory_context") or [])


async def test_load_context_preloads_preferences_into_memory_context(
    build_graph, run_graph
) -> None:
    """CHAT 轮：load_context 预加载 preference → memory_context，来源进 citations。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", importance=0.9),
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), store=injected)
    response, _ = await run_graph(graph, text="你好")

    assert response is not None
    ctx = await _memory_context_of(graph, "s1")
    assert [m.id for m in ctx] == ["p1"]
    assert ctx[0].kind == KIND_PREFERENCE
    assert ctx[0].content == "用户偏好简洁回答"
    # 来源引用进入 response.citations；仅种 preference、fact/episode 召回为空
    # → 无 RETRIEVING（recall_memory 仅在有结果时下发状态）。
    assert any(c.source_id == "p1" for c in response.citations)
    assert Status.RETRIEVING not in [e.status for e in response.status_trace]


async def test_load_context_preload_off_when_disabled(build_graph, run_graph) -> None:
    """preload_profile=False：预加载关闭，memory_context/citations 均为空。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", importance=0.9),
    )
    cfg = AgentFrameworkConfig(memory=MemoryBehaviorConfig(preload_profile=False))
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), config=cfg, store=injected)
    response, _ = await run_graph(graph, text="你好")

    assert response is not None
    assert await _memory_context_of(graph, "s1") == []
    assert response.citations == []


async def test_recall_on_tool_branch_injects_facts_and_preserves_preload(
    build_graph, run_graph
) -> None:
    """TOOL_USE 轮：recall_memory 注入 fact/episode，并保留预加载的 preference。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", importance=0.6),
    )
    await _save(
        injected,
        _memory_item(id="f1", kind=KIND_FACT, content="用户是中学生", importance=0.9),
    )
    await _save(
        injected,
        _memory_item(id="e1", kind=KIND_EPISODE, content="昨晚九点入睡", importance=0.8),
    )
    calc = make_fake_tool("calc", content="19")
    graph = build_graph(
        tool_call_messages(
            [[{"name": "calc", "args": {"expression": "12+7"}, "id": "c1"}]], "12+7=19"
        ),
        tools=[calc],
        store=injected,
    )
    response, _ = await run_graph(graph, text="帮我计算 12+7")

    assert response is not None
    ctx = await _memory_context_of(graph, "s1")
    # 预加载 preference 在前，fact/episode 按 importance 降序追加。
    assert [m.id for m in ctx] == ["p1", "f1", "e1"]
    assert {m.kind for m in ctx} == {KIND_PREFERENCE, KIND_FACT, KIND_EPISODE}
    # 三类来源都进 citations；TOOL_USE 分支触发 RETRIEVING。
    assert {c.source_id for c in response.citations} == {"p1", "f1", "e1"}
    assert Status.RETRIEVING in [e.status for e in response.status_trace]


async def test_recall_triggered_on_chat_branch(build_graph, run_graph) -> None:
    """CHAT 轮也召回 fact（recall_memory 是共同上游）：注入 memory_context + 引用。

    v6.0 起输入需通过查询理解门控（纯寒暄「你好」按设计跳过主题召回，
    见 test_short_greeting_skips_topic_recall），故这里用一句知识型问题。
    """
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="f1", kind=KIND_FACT, content="用户是中学生", importance=0.9),
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "你好"), store=injected)
    response, _ = await run_graph(graph, text="我上次说我读几年级来着？")

    assert response is not None
    assert [m.id for m in await _memory_context_of(graph, "s1")] == ["f1"]
    assert {c.source_id for c in response.citations} == {"f1"}
    assert Status.RETRIEVING in [e.status for e in response.status_trace]


async def test_short_greeting_skips_topic_recall(build_graph, run_graph) -> None:
    """V6-M3：纯寒暄短输入被确定性门控跳过 → 不召回 fact/episode（不查 store）。

    WHY 这是精度收益而非功能缺失：拿「你好」去检索主题记忆只会带回噪声条目；
    preference 身份预加载不受影响（它不是主题检索，见 recall_memory 文档）。
    """
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="f1", kind=KIND_FACT, content="用户是中学生", importance=0.9),
    )
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", importance=0.9),
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "你好呀"), store=injected)
    response, _ = await run_graph(graph, text="你好")

    assert response is not None
    # 只剩身份预加载（preference），fact 未进上下文。
    assert [m.id for m in await _memory_context_of(graph, "s1")] == ["p1"]
    assert {c.source_id for c in response.citations} == {"p1"}
    # 主题召回未发生 → 无 RETRIEVING（与 v5.1「仅预加载」轮次的状态语义一致）。
    assert Status.RETRIEVING not in [e.status for e in response.status_trace]


async def test_memory_context_no_cross_turn_accumulation(build_graph, run_graph) -> None:
    """普通覆盖：同会话第二轮 memory_context 不累积（operator.add 会变 2 条）。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", importance=0.9),
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), store=injected)
    resp1, _ = await run_graph(graph, text="你好", session_id="s1")
    resp2, _ = await run_graph(graph, text="再来一次", session_id="s1")

    assert resp1 is not None and resp2 is not None
    assert [m.id for m in await _memory_context_of(graph, "s1")] == ["p1"]
    # 引用去重：同 source_id 不会跨轮重复追加。
    assert [c.source_id for c in resp2.citations].count("p1") == 1


async def test_preload_cross_session_same_user_persists(build_graph, run_graph) -> None:
    """跨会话同 user：两个 thread 各自预加载到同一 preference（Store 共享）。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", user_id="u1"),
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), store=injected)
    _, _ = await run_graph(graph, text="你好", session_id="t1", user_id="u1")
    _, _ = await run_graph(graph, text="又见面了", session_id="t2", user_id="u1")

    assert [m.id for m in await _memory_context_of(graph, "t1")] == ["p1"]
    assert [m.id for m in await _memory_context_of(graph, "t2")] == ["p1"]


async def test_recall_user_isolation_in_graph(build_graph, run_graph) -> None:
    """user 隔离：preference 属 u1，u1 会话命中、u2 会话 memory_context 为空。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", user_id="u1"),
    )
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"), store=injected)
    _, _ = await run_graph(graph, text="你好", session_id="t1", user_id="u1")
    _, _ = await run_graph(graph, text="你好", session_id="t2", user_id="u2")

    assert [m.id for m in await _memory_context_of(graph, "t1")] == ["p1"]
    assert await _memory_context_of(graph, "t2") == []


async def test_prompt_includes_memory_block(make_llm_service, run_graph) -> None:
    """TOOL_USE 轮：call_model 的 prompt 含不可信声明 + 记忆条目；classify 不含。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", importance=0.6),
    )
    await _save(
        injected,
        _memory_item(id="f1", kind=KIND_FACT, content="用户是中学生", importance=0.9),
    )
    service = make_llm_service(
        tool_call_messages([[{"name": "calc", "args": {}, "id": "c1"}]], "12+7=19"),
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(service, tools=[make_fake_tool("calc", content="19")], store=injected)
    _, _ = await run_graph(graph, text="帮我计算 12+7")

    prompts = service.chat_model.prompts
    assert prompts, "应记录至少一次模型调用"
    # prompts[0] = classify 结构化调用（不注入记忆）。
    classify_text = "".join(str(getattr(m, "content", "")) for m in prompts[0])
    assert _MEMORY_BLOCK_HEADER not in classify_text
    # prompts[-1] = 最后一次 call_model：含不可信声明与 preload/recall 条目。
    last_text = "".join(str(getattr(m, "content", "")) for m in prompts[-1])
    assert _MEMORY_BLOCK_HEADER in last_text
    assert "- [preference] 用户偏好简洁回答" in last_text
    assert "- [fact] 用户是中学生" in last_text


async def test_prompt_memory_block_on_chat_path(make_llm_service, run_graph) -> None:
    """CHAT 轮：generate_answer 的 prompt 含不可信声明 + 预加载 preference + 召回的 fact。"""
    injected = InMemoryStore()
    await _save(
        injected,
        _memory_item(id="p1", kind=KIND_PREFERENCE, content="用户偏好简洁回答", importance=0.9),
    )
    await _save(
        injected,
        _memory_item(id="f1", kind=KIND_FACT, content="用户是中学生", importance=0.8),
    )
    service = make_llm_service(
        chat_turn_messages(Intent.CHAT, "好的，记住了"),
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(service, store=injected)
    _, _ = await run_graph(graph, text="请简洁回答")

    prompts = service.chat_model.prompts
    last_text = "".join(str(getattr(m, "content", "")) for m in prompts[-1])
    assert _MEMORY_BLOCK_HEADER in last_text
    assert "- [preference] 用户偏好简洁回答" in last_text
    assert "- [fact] 用户是中学生" in last_text


async def test_render_memory_block_truncation_keeps_declaration() -> None:
    """截断：不可信声明头恒保留，内容按 max_chars 预算截断（尾部条目截断）。"""
    items = [
        _memory_item(id="m1", kind=KIND_FACT, content="用户是中学生" * 100),
        _memory_item(id="m2", kind=KIND_FACT, content="另一条事实"),
    ]
    block = _render_memory_block(items, max_chars=200)

    assert block is not None
    assert block.startswith(_MEMORY_BLOCK_HEADER)
    assert len(block) <= 200
    # 声明后的内容被截断（完整 bullet1 远超预算），声明未丢。
    assert "用户是中学生" in block
