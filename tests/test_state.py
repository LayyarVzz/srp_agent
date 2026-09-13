"""agent/core/state —— AgentState 的 reducer 语义验证。

用微型 StateGraph 验证：累积型字段（messages/status_events/tool_calls）跨节点追加、
`tool_iterations` 普通覆盖不累加（工具循环上限计数依赖）。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from agent.core.state import AgentState
from agent.response.status import Status, StatusEvent
from agent.tools.models import ToolCallRecord


async def test_status_events_accumulate_across_nodes() -> None:
    """status_events 由 operator.add 累积：后写节点不覆盖前写节点。"""
    builder = StateGraph(AgentState)

    async def n1(state: AgentState) -> dict:
        return {"status": Status.THINKING, "status_events": [StatusEvent(status=Status.THINKING)]}

    async def n2(state: AgentState) -> dict:
        return {"status": Status.SPEAKING, "status_events": [StatusEvent(status=Status.SPEAKING)]}

    builder.add_node("n1", n1)
    builder.add_node("n2", n2)
    builder.set_entry_point("n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)
    graph = builder.compile()

    result = await graph.ainvoke({})
    assert [e.status for e in result["status_events"]] == [Status.THINKING, Status.SPEAKING]
    assert result["status"] == Status.SPEAKING


async def test_messages_accumulate_via_add_messages() -> None:
    """messages 经 add_messages 跨节点追加（与图内 load→generate 语义一致）。"""
    builder = StateGraph(AgentState)

    async def n1(state: AgentState) -> dict:
        return {"messages": [HumanMessage(content="你好")]}

    async def n2(state: AgentState) -> dict:
        return {"messages": [HumanMessage(content="再见")]}

    builder.add_node("n1", n1)
    builder.add_node("n2", n2)
    builder.set_entry_point("n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)
    graph = builder.compile()

    result = await graph.ainvoke({})
    assert [m.content for m in result["messages"]] == ["你好", "再见"]


async def test_tool_iterations_is_overwrite_field() -> None:
    """tool_iterations 必须普通覆盖（无 reducer）：循环上限计数依赖此语义。"""
    builder = StateGraph(AgentState)

    async def n1(state: AgentState) -> dict:
        return {"tool_iterations": 1}

    async def n2(state: AgentState) -> dict:
        return {"tool_iterations": 2}

    builder.add_node("n1", n1)
    builder.add_node("n2", n2)
    builder.set_entry_point("n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)
    graph = builder.compile()

    result = await graph.ainvoke({})
    assert result["tool_iterations"] == 2  # 覆盖而非 1+2=3


async def test_tool_calls_accumulate_via_operator_add() -> None:
    """tool_calls 经 operator.add 追加：适配层每轮新增记录不覆盖历史。"""
    builder = StateGraph(AgentState)

    async def n1(state: AgentState) -> dict:
        return {"tool_calls": [ToolCallRecord(tool_name="a", arguments={}, status="ok")]}

    async def n2(state: AgentState) -> dict:
        return {"tool_calls": [ToolCallRecord(tool_name="b", arguments={}, status="ok")]}

    builder.add_node("n1", n1)
    builder.add_node("n2", n2)
    builder.set_entry_point("n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)
    graph = builder.compile()

    result = await graph.ainvoke({})
    # operator.add 拼接两条新记录，而非覆盖为 1 条。
    assert [r.tool_name for r in result["tool_calls"]] == ["a", "b"]


async def test_subagent_results_accumulate_via_operator_add() -> None:
    """subagent_results 经 operator.add 追加：并行 Send 多实例各自回填不互相覆盖（T7）。"""
    from agent.core.models import SubagentResult

    builder = StateGraph(AgentState)

    async def branch_a(state: AgentState) -> dict:
        return {"subagent_results": [SubagentResult(step_index=0, ok=True, summary="A")]}

    async def branch_b(state: AgentState) -> dict:
        return {"subagent_results": [SubagentResult(step_index=1, ok=True, summary="B")]}

    builder.add_node("branch_a", branch_a)
    builder.add_node("branch_b", branch_b)
    builder.set_entry_point("branch_a")
    builder.add_edge("branch_a", "branch_b")
    builder.add_edge("branch_b", END)
    graph = builder.compile()

    result = await graph.ainvoke({})
    # 两条结果按序拼接，而非后写覆盖（join 依赖此语义聚合整批产出）。
    assert [r.step_index for r in result["subagent_results"]] == [0, 1]
