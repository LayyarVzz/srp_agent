"""短期上下文管理（P3-4）：预算裁剪 / 滚动摘要 / 会话关键信息 / 遗忘策略。

覆盖（dev-version3.0.md §4/§6/§7.1）：
- 预算裁剪：`max_context_chars` 字符预算下限保护（保留最新），无 tokenizer 用字符近似。
- 滚动摘要：`trimmed_messages` 非空才触发（门控 no-op），单轮结构化调用产出摘要+关键信息；
  `short_term_summary` / `session_keyfacts` 随 checkpointer 持久化，被裁消息瞬态清空。
- 会话关键信息：结构化 `SessionKeyFact`，active 过滤 + `max_items` 截断（遗忘策略②确定性兜底）。
- 遗忘策略①：早于当前轮的已消费 ToolMessage 打引用桩（保留 id/结构），当前轮不打桩。
- 注入顺序：SYSTEM_PROMPT → 会话摘要 → 关键信息 → 长期记忆块 → 消息历史（不可信数据声明）。
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from agent.core.config import (
    AgentFrameworkConfig,
    AgentGraphConfig,
    KeyFactsConfig,
    MemoryBehaviorConfig,
    SummarizeConfig,
)
from agent.core.context import SessionKeyFact, ShortTermContext
from agent.core.graph import (
    _KEYFACTS_HEADER,
    _MEMORY_BLOCK_HEADER,
    _SUMMARY_HEADER,
    SYSTEM_PROMPT,
    build_agent_graph,
)
from agent.intent.models import Intent, IntentResult
from agent.memory import KIND_PREFERENCE, MemoryStore
from agent.memory.models import MemoryItem
from tests.conftest import (
    RecordingFakeChatModel,
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
    tool_call_messages,
    understand_message,
)


def _multi_turn_with_summaries(replies: list[str], summary: ShortTermContext) -> list:
    """rounds=1 下构造跨轮 fake 消息序列：t≥2 每次裁剪触发一次摘要结构化调用。

    WHY 单独构造：未裁剪轮消耗 3 条（classify + 查询理解 + generate），裁剪轮额外
    消耗 1 条摘要（summarize），序列长度与图消费顺序一一对应
    （v6.0 T2 在意图分类之后插入了 `understand_query` 的改写调用）。

    本 helper 的轮次输入是「第N问」，不命中寒暄词表且长度 ≥ `min_query_chars`，
    故每轮都过确定性门控、每轮都要一条查询理解消息（实测调用序：
    第 1 轮 classify→understand→generate；第 2/3 轮 summarize→classify→understand→generate）。
    """
    msgs: list = []
    for i, reply in enumerate(replies):
        if i >= 1:
            msgs.append(fake_structured_message(summary))
        msgs.append(
            fake_structured_message(
                IntentResult(intent=Intent.CHAT, confidence=0.95, reason="test")
            )
        )
        msgs.append(understand_message())
        msgs.append(fake_text_message(reply))
    return msgs


def _summary(text: str = "已压缩的旧轮目标：帮用户订下周去北京的机票") -> ShortTermContext:
    """一次摘要结构化输出：默认带 1 条 active goal 关键信息。"""
    return ShortTermContext(
        summary=text,
        keyfacts=[SessionKeyFact(content="用户想订下周去北京的机票", category="goal")],
    )


async def test_budget_trimming_trims_by_chars(build_graph, run_graph) -> None:
    """字符预算裁剪：超预算按旧→新裁掉更旧消息，预算下限内保留最新（rounds 不触发）。"""
    cfg = AgentFrameworkConfig(
        graph=AgentGraphConfig(trim_keep_recent_rounds=10, max_context_chars=120),
        memory=MemoryBehaviorConfig(summarize=SummarizeConfig(enabled=False)),
    )
    reply = "x" * 80
    # 输入「第N问」不命中寒暄词表（长度 3 ≥ min_query_chars=2）→ 每轮过确定性门控，
    # 故每轮需 3 条脚本消息（意图 / 查询理解 / 回答），而非门控跳过的 2 条
    # （对照：`test_rolling_summary_noop_when_no_trimming` 的「你好」命中寒暄表 → 2 条）。
    graph = build_graph(chat_turn_messages(Intent.CHAT, reply, understand=True) * 4, config=cfg)
    config = {"configurable": {"thread_id": "s1"}}
    for i in range(4):
        await run_graph(graph, text=f"第{i + 1}问", session_id="s1")

    state = await graph.aget_state(config)
    msgs = state.values["messages"]
    contents = [str(m.content) for m in msgs]
    total = sum(len(c) for c in contents)
    # 预算只约束「裁剪时点之前」的窗口（当轮回答在裁剪后追加），故总长 ≤ 预算 + 单条回答；
    # rounds=10 恒不触发轮数裁剪，能裁掉的旧轮只可能来自字符预算 → 证明预算裁剪生效。
    assert "第4问" in contents
    assert total <= 120 + 80
    assert contents.count("x" * 80) == 2  # 仅保留最近两轮回答（更旧两轮被裁）
    assert "第1问" not in contents
    assert "第2问" not in contents


async def test_rolling_summary_trigger_and_noop(build_graph, run_graph) -> None:
    """滚动摘要门控：未裁剪轮 no-op，裁剪轮触发并写入摘要/关键信息、清空瞬态。"""
    cfg = AgentFrameworkConfig(graph=AgentGraphConfig(trim_keep_recent_rounds=1))
    graph = build_graph(_multi_turn_with_summaries(["r1", "r2", "r3"], _summary()), config=cfg)
    config = {"configurable": {"thread_id": "s1"}}
    await run_graph(graph, text="第1问", session_id="s1")
    s1 = (await graph.aget_state(config)).values
    # 第 1 轮未裁剪 → 摘要未触发（字段从未写入）。
    assert not (s1.get("short_term_summary") or "").strip()
    assert s1.get("trimmed_messages") == []

    await run_graph(graph, text="第2问", session_id="s1")
    s2 = (await graph.aget_state(config)).values
    assert s2.get("short_term_summary") == _summary().summary
    assert s2.get("session_keyfacts") == _summary().keyfacts
    assert s2.get("trimmed_messages") == []  # 摘要消费后瞬态清空

    await run_graph(graph, text="第3问", session_id="s1")
    s3 = (await graph.aget_state(config)).values
    assert s3.get("short_term_summary") == _summary().summary  # 滚动重写（内容未变）

    s3_messages = s3["messages"]
    assert "第1问" not in [str(m.content) for m in s3_messages]  # 旧轮已被裁剪


async def test_rolling_summary_noop_when_no_trimming(make_llm_service, run_graph) -> None:
    """未裁剪时 summarize_history 零 LLM 调用：图仅 classify + generate 两次调用。"""
    svc = make_llm_service(
        chat_turn_messages(Intent.CHAT, "你好"), model_cls=RecordingFakeChatModel
    )
    graph = build_agent_graph(svc)
    response, _ = await run_graph(graph, text="你好")
    assert response is not None
    prompts = svc.chat_model.prompts
    assert len(prompts) == 2  # 仅 classify + generate，无摘要调用
    generate_prompt = prompts[-1]
    sys_parts = [m for m in generate_prompt if isinstance(m, SystemMessage)]
    # 无摘要/关键信息/记忆块时只注入 SYSTEM_PROMPT。
    assert [m.content for m in sys_parts] == [SYSTEM_PROMPT]


async def test_summarize_llm_failure_zero_regression(build_graph, run_graph) -> None:
    """摘要 LLM 失败（fake 序列耗尽）零回归：保留旧摘要、回答正常、异常不外抛。"""
    cfg = AgentFrameworkConfig(graph=AgentGraphConfig(trim_keep_recent_rounds=1))
    # 仅第 1 轮的 2 条消息；第 2 轮起 summarize/classify/generate 序列耗尽 → 各自降级。
    graph = build_graph(chat_turn_messages(Intent.CHAT, "回复1"), config=cfg)
    config = {"configurable": {"thread_id": "s1"}}
    responses = []
    for i in range(4):
        response, _ = await run_graph(graph, text=f"第{i + 1}问", session_id="s1")
        responses.append(response)
    assert all(r is not None for r in responses)
    s = (await graph.aget_state(config)).values
    assert not (s.get("short_term_summary") or "")  # 摘要失败 → 不写入
    assert s.get("trimmed_messages") == []  # 瞬态仍被清空，不残留


async def test_keyfacts_active_filter_and_truncation(build_graph, run_graph) -> None:
    """遗忘策略②确定性兜底：active=false 剔除后按 max_items 截断（保留 active 靠前者）。"""
    cfg = AgentFrameworkConfig(
        graph=AgentGraphConfig(trim_keep_recent_rounds=1),
        memory=MemoryBehaviorConfig(keyfacts=KeyFactsConfig(max_items=2)),
    )
    keyfacts = [
        SessionKeyFact(content="用户想订下周去北京的机票", category="goal"),
        SessionKeyFact(content="用户偏好靠窗座位", category="fact"),
        SessionKeyFact(content="用户明天上午有会", category="todo"),
        # 已达成/过期的旧目标：active=false → 确定性剔除。
        SessionKeyFact(content="旧目标：订上海机票", category="goal", active=False),
    ]
    graph = build_graph(
        _multi_turn_with_summaries(["r1", "r2"], _summary_with(keyfacts)), config=cfg
    )
    config = {"configurable": {"thread_id": "s1"}}
    await run_graph(graph, text="第1问", session_id="s1")
    await run_graph(graph, text="第2问", session_id="s1")

    s = (await graph.aget_state(config)).values
    stored = s.get("session_keyfacts") or []
    assert stored == keyfacts[:2]  # active 过滤（剔除第 4 条）后取前 2 条
    assert stored != keyfacts  # 确实发生了过滤/截断，而非原样保存


def _summary_with(keyfacts: list[SessionKeyFact]) -> ShortTermContext:
    """带指定关键信息列表的摘要结构化输出（test_keyfacts 专用）。"""
    return ShortTermContext(summary="已压缩的旧轮目标", keyfacts=keyfacts)


async def test_tool_output_stubbing_preserves_structure(build_graph, run_graph) -> None:
    """遗忘策略①：早于当前轮的已消费 ToolMessage 打引用桩（id 保留、可原位覆盖），
    当前轮工具结果不打桩。"""
    cfg = AgentFrameworkConfig(graph=AgentGraphConfig(trim_keep_recent_rounds=2))
    tools = [make_fake_tool("calc", content="19")]
    # 序列：turn1 消费 3 条（classify/call_model-tc/call_model-r1），
    # turn2 消费 4 条（summarize/classify/call_model-tc/call_model-r2）。
    msgs = [
        *tool_call_messages([[{"name": "calc", "args": {"expression": "12+7"}, "id": "c1"}]], "r1"),
        fake_structured_message(_summary()),
        *tool_call_messages([[{"name": "calc", "args": {"expression": "5*3"}, "id": "c2"}]], "r2"),
    ]
    graph = build_graph(msgs, config=cfg, tools=tools)
    config = {"configurable": {"thread_id": "s1"}}
    await run_graph(graph, text="第一问", session_id="s1")
    s1 = (await graph.aget_state(config)).values
    tool1 = next(m for m in s1["messages"] if isinstance(m, ToolMessage))
    assert tool1.tool_call_id == "c1" and tool1.content == "19"
    tool1_id = tool1.id  # add_messages 在首次合并时已为 ToolMessage 分配 id

    await run_graph(graph, text="第二问", session_id="s1")
    s2 = (await graph.aget_state(config)).values
    by_call = {m.tool_call_id: m for m in s2["messages"] if isinstance(m, ToolMessage)}
    stub = by_call["c1"]
    assert "[工具结果已消费 tool=calc; 引用 c1]" in stub.content  # 已打桩
    assert stub.id == tool1_id  # model_copy 保留原 id → add_messages 原位覆盖成功
    fresh = by_call["c2"]
    assert fresh.content == "19"  # 当前轮结果不打桩
    # 被裁的首轮 HumanMessage 已由 RemoveMessage 删除。
    assert "第一问" not in [str(m.content) for m in s2["messages"]]


async def test_build_prompt_injection_order(make_llm_service, run_graph) -> None:
    """注入顺序：SYSTEM_PROMPT → 会话摘要 → 关键信息 → 长期记忆块 → 消息历史。"""
    cfg = AgentFrameworkConfig(graph=AgentGraphConfig(trim_keep_recent_rounds=1))
    store = InMemoryStore()
    await MemoryStore(store).save(
        MemoryItem(
            id="m1",
            kind=KIND_PREFERENCE,
            content="用户喜欢简洁的回答",
            session_id="s-seed",
            user_id="alice",
            timestamp=datetime.now(UTC),
            provenance="seed",
            importance=0.9,
        )
    )
    svc = make_llm_service(
        _multi_turn_with_summaries(["r1", "r2", "r3"], _summary()),
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(svc, cfg, store=store)
    for i in range(3):
        await run_graph(graph, text=f"第{i + 1}问", session_id="s1", user_id="alice")

    prompts = svc.chat_model.prompts
    last = prompts[-1]  # 第 3 轮 generate_answer 的 prompt（含全部注入块）
    sys_parts = [m for m in last if isinstance(m, SystemMessage)]
    assert [m.content for m in sys_parts] == [
        SYSTEM_PROMPT,
        f"{_SUMMARY_HEADER}\n{_summary().summary}",
        f"{_KEYFACTS_HEADER}\n- [goal] 用户想订下周去北京的机票",
        f"{_MEMORY_BLOCK_HEADER}\n- [preference] 用户喜欢简洁的回答",
    ]
    # 消息历史紧随注入块之后（末条为当前轮 HumanMessage；其前是上一轮 AI 回复）。
    rest = last[len(sys_parts) :]
    assert isinstance(rest[-1], HumanMessage)
    assert rest[-1].content == "第3问"
