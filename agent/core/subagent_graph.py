"""并行子代理子图（T7，dev-version5.0.md §6.1）。

PLAN 中无依赖步骤经主图 `dispatch_subagents` 以 `Send` 扇出后，每个子代理步骤
由本模块编译的小型 ReAct 子图执行：`sub_call_model ⇄ sub_dispatch_tool`。

设计约束：
- **编译的子图，非手写循环**（CLAUDE.md 编排纪律）：分支/循环全部由 StateGraph 表达；
- **状态独立于主图**：子图自有 messages / 迭代计数，运行结束只把 `SubagentResult`
  回传主图（由主图包装节点 `run_subagent` 完成），防跨图状态污染；
- **与主图共享同一批 `BaseTool` / `LLMService` 实例**：工具生命周期不重复管理；
- **无 checkpointer**：子图运行是单步内瞬态过程，不产生持久化状态；
- 失败短路：子代理内任一工具失败或 LLM 失败立即结束（与 v4.0「单步失败 → 重规划」
  同语义，由主图 join 节点统一裁决），子代理自身不做重试。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.core.config import SubagentConfig
from agent.errors import LLMError
from agent.llm import LLMService

# 子图节点名常量（与主图节点名一样集中声明，禁止散落字符串字面量）。
SUB_NODE_CALL_MODEL = "sub_call_model"
SUB_NODE_DISPATCH_TOOL = "sub_dispatch_tool"


class SubagentState(TypedDict, total=False):
    """子代理子图状态（独立于主图：自有消息轨迹与迭代计数）。"""

    messages: Annotated[list[BaseMessage], add_messages]
    tool_iterations: int  # 工具迭代计数（上限 = SubagentConfig.per_subagent_max_tool_calls）
    error: str | None  # LLM 失败信息（瞬态；主图包装节点据此判定步骤失败）


def build_subagent_graph(
    llm: LLMService,
    tools: list[BaseTool],
    cfg: SubagentConfig,
) -> CompiledStateGraph:
    """编译子代理 ReAct 子图（每个主图实例编译一次，由 run_subagent 节点复用）。

    迭代上限：`per_subagent_max_tool_calls` 计「工具执行轮数」（与主图
    `tool_iterations` 同口径）——达到上限且模型仍请求工具时直接收尾，
    未执行的 tool_calls 不再派发（收敛护栏，防单子代理失控）。
    """
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def sub_call_model(state: SubagentState) -> dict[str, Any]:
        """子代理模型节点：以当前消息轨迹调用 LLM（不流式外发——并行分支的
        token 会互相交错，子代理产出经 SubagentResult 汇总后由主图统一播报）。
        """
        prompt = state.get("messages") or []
        try:
            resp = await llm.ainvoke_tools(tools, prompt)
        except LLMError as exc:
            # 失败不追加消息 → route_sub_model 见非 AIMessage 收尾；
            # error 经子图终态回传包装节点 → 步骤失败 → 主图重规划语义。
            return {"error": f"子代理模型调用失败: {exc}"}
        return {"messages": [resp], "error": None}

    async def sub_dispatch_tool(state: SubagentState) -> dict[str, Any]:
        """子代理工具节点：复用 ToolNode 并行执行末条 AIMessage 的全部 tool_calls。"""
        result = await tool_node.ainvoke({"messages": state.get("messages") or []})
        return {
            "messages": result["messages"],
            "tool_iterations": (state.get("tool_iterations") or 0) + 1,
        }

    def route_sub_model(state: SubagentState) -> str:
        """模型出口：请求工具且未达迭代上限 → 执行工具；否则收尾（直接作答/达上限）。"""
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            if (state.get("tool_iterations") or 0) < cfg.per_subagent_max_tool_calls:
                return SUB_NODE_DISPATCH_TOOL
            # 达上限仍请求工具：以当前产出收尾（未执行的 tool_calls 不派发）。
            return END
        return END

    def route_sub_tool(state: SubagentState) -> str:
        """工具出口：任一 ToolMessage 失败 → 立即收尾（步骤失败短路）；
        此前迭代的失败早已触发收尾，故全量检查与「本轮新增失败」等价。"""
        messages = state.get("messages") or []
        if any(isinstance(m, ToolMessage) and m.status == "error" for m in messages):
            return END
        return SUB_NODE_CALL_MODEL

    builder = StateGraph(SubagentState)
    builder.add_node(SUB_NODE_CALL_MODEL, sub_call_model)
    builder.add_node(SUB_NODE_DISPATCH_TOOL, sub_dispatch_tool)
    builder.set_entry_point(SUB_NODE_CALL_MODEL)
    builder.add_conditional_edges(
        SUB_NODE_CALL_MODEL,
        route_sub_model,
        {SUB_NODE_DISPATCH_TOOL: SUB_NODE_DISPATCH_TOOL, END: END},
    )
    builder.add_conditional_edges(
        SUB_NODE_DISPATCH_TOOL,
        route_sub_tool,
        {SUB_NODE_CALL_MODEL: SUB_NODE_CALL_MODEL, END: END},
    )
    # 无 checkpointer：子图为单步内瞬态执行，不持久化、不跨轮复用。
    return builder.compile()


__all__ = [
    "SUB_NODE_CALL_MODEL",
    "SUB_NODE_DISPATCH_TOOL",
    "SubagentState",
    "build_subagent_graph",
]
