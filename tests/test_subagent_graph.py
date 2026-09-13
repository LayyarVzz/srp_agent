"""agent/core/subagent_graph —— 子代理 ReAct 子图离线单测（T7）。

覆盖（离线 fake LLM + 假工具，与 conftest 消费契约一致：一次 LLM 调用消费一条消息）：
- 工具步成功：call_model 请求工具 → ToolNode 执行 → 再答文本 → 正常收尾；
- 工具失败短路：任一 ToolMessage(status=error) → 立即 END（不重试）；
- 迭代上限收敛：达到 per_subagent_max_tool_calls 后未执行的 tool_calls 不再派发；
- 纯文本步：无工具调用直接收尾（tool_iterations 保持 0）；
- LLM 失败：LLMService 归一化的 LLMError → error 记入子图终态（供主图判步骤失败）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import SecretStr

from agent.core.config import LLMConfig, SubagentConfig
from agent.core.subagent_graph import (
    SUB_NODE_CALL_MODEL,
    SUB_NODE_DISPATCH_TOOL,
    build_subagent_graph,
)
from agent.llm import LLMService
from tests.conftest import StructuredFakeChatModel, make_fake_tool


def _tool_call_msg(name: str, call_id: str, args: dict | None = None) -> AIMessage:
    """构造单条 tool_calls 的 AIMessage（子代理工具步输出）。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


def _build_subgraph(
    messages: list[AIMessage],
    *,
    cfg: SubagentConfig | None = None,
    tools: list | None = None,
    model_cls: type[StructuredFakeChatModel] = StructuredFakeChatModel,
):
    """构建注入离线 fake 的子代理子图（独立编译，无 checkpointer）。"""
    service = LLMService(
        config=LLMConfig(api_key=SecretStr("sk-x")),
        chat_model=model_cls(messages=iter(messages)),
    )
    return build_subagent_graph(
        service,
        tools if tools is not None else [make_fake_tool("calc", content="19")],
        cfg or SubagentConfig(),
    )


async def test_tool_step_success_executes_and_answers() -> None:
    """工具步：请求工具 → 执行 → 文本作答收尾；迭代计数 1、无错误。"""
    graph = _build_subgraph([_tool_call_msg("calc", "c1"), AIMessage(content="计算结果 19")])
    result = await graph.ainvoke({"messages": [HumanMessage(content="计算 12+7")]})

    assert result["tool_iterations"] == 1
    assert result["error"] is None
    assert result["messages"][-1].content == "计算结果 19"
    # 轨迹含工具消息（ToolNode 产物）。
    assert any(isinstance(m, ToolMessage) and m.status == "success" for m in result["messages"])


async def test_tool_failure_short_circuits_to_end() -> None:
    """工具失败：error ToolMessage → 直接 END，不再回模型重试。"""
    bad = make_fake_tool("calc", fail_with=RuntimeError("boom"))
    graph = _build_subgraph([_tool_call_msg("calc", "c1")], tools=[bad])
    result = await graph.ainvoke({"messages": [HumanMessage(content="计算")]})

    assert result.get("tool_iterations") == 1
    last = result["messages"][-1]
    assert isinstance(last, ToolMessage) and last.status == "error"
    assert "boom" in str(last.content)


async def test_iteration_limit_stops_pending_tool_calls() -> None:
    """迭代上限：第 2 轮模型仍请求工具时不再派发（未执行的 tool_calls 收尾）。"""
    cfg = SubagentConfig(per_subagent_max_tool_calls=1)
    graph = _build_subgraph(
        [
            _tool_call_msg("calc", "c1"),
            _tool_call_msg("calc", "c2"),
            AIMessage(content="不会到达"),
        ],
        cfg=cfg,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="计算")]})

    # 仅 1 轮工具执行；第 2 轮的 tool_calls 收尾未派发（末条 AIMessage 无对应 ToolMessage）。
    assert result["tool_iterations"] == 1
    tool_call_ids = [
        tc.get("id")
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]
    executed_ids = [m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "c2" in tool_call_ids
    assert "c2" not in executed_ids


async def test_plain_text_step_ends_without_tools() -> None:
    """纯文本步（LLM 变换步）：无工具调用直接收尾，迭代计数 0。"""
    graph = _build_subgraph([AIMessage(content="直接作答内容")])
    result = await graph.ainvoke({"messages": [HumanMessage(content="总结一下")]})

    assert result.get("tool_iterations", 0) == 0
    assert result.get("error") is None
    assert result["messages"][-1].content == "直接作答内容"
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


async def test_llm_error_records_error_in_final_state() -> None:
    """LLM 失败：LLMService 归一化 LLMError → error 记入终态（主图据此判步骤失败）。"""

    class _RaiseModel(StructuredFakeChatModel):
        """首次调用即抛错的 fake（LLMService 归一化为 LLMError）。"""

        def _generate(self, messages: list, stop: list | None = None, **kwargs: Any):
            raise RuntimeError("llm down")

    graph = _build_subgraph([], model_cls=_RaiseModel)
    result = await graph.ainvoke({"messages": [HumanMessage(content="计算")]})

    assert result["error"] is not None
    assert "子代理模型调用失败" in result["error"]


def test_subgraph_node_names_are_declared_constants() -> None:
    """子图节点名集中声明：编译图中只含 sub_call_model / sub_dispatch_tool（END 除外）。"""
    graph = _build_subgraph([])
    nodes = set(graph.get_graph().nodes) - {"__end__", "__start__"}
    assert nodes == {SUB_NODE_CALL_MODEL, SUB_NODE_DISPATCH_TOOL}
