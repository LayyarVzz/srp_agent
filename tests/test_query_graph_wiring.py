"""图 ↔ 工具执行期的查询理解接线测试（v6.0 T2 关键链路）。

WHY 单独立文件：`RetrievalQueryInterceptor` 从 `ToolRuntime.state` 读 `retrieval_query`，
而 `dispatch_tool` 必须把它**随输入透传给 ToolNode**（ToolNode 对普通 dict 输入原样返回）。
这条接线一旦断掉，拦截器会静默「读不到改写结果 → 原样放行」，表现为「改写白做」却
没有任何报错 —— 故必须显式锁住。
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from agent.core.state import NODE_UNDERSTAND_QUERY
from agent.intent.models import Intent, IntentResult
from tests.conftest import (
    AIMessage,
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    understand_message,
)


def _state_probe_tool(captured: list[dict[str, object]]) -> StructuredTool:
    """假工具：记录**工具执行期**可见的 `ToolRuntime.state`（拦截器读的就是它）。"""

    async def _run(*, runtime: object = None, **kwargs: object) -> str:
        state = getattr(runtime, "state", None)
        captured.append(dict(state) if isinstance(state, dict) else {"__missing__": True})
        return "ok"

    return StructuredTool.from_function(
        coroutine=_run,
        name="state_probe",
        description="记录执行期图状态的假工具",
        infer_schema=False,
    )


async def test_retrieval_query_reaches_tool_runtime_state(build_graph, run_graph) -> None:
    """查询理解产物必须出现在工具执行期的 runtime.state（拦截器据此覆盖检索实参）。"""
    captured: list[dict[str, object]] = []
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="test")
            ),
            understand_message("改写后的主查询"),
            AIMessage(content="", tool_calls=[{"name": "state_probe", "args": {}, "id": "c1"}]),
            fake_text_message("完成"),
        ],
        tools=[_state_probe_tool(captured)],
    )
    response, _ = await run_graph(graph, text="帮我查一下那个东西")

    assert response is not None
    assert captured, "工具未被调用"
    state = captured[0]
    assert state.get("retrieval_query") == "改写后的主查询"
    # 结构化对象也一并透传（拦截器据此取 HyDE 假设文档）。
    understanding = state.get("query_understanding")
    assert getattr(understanding, "main_query", None) == "改写后的主查询"


async def test_gate_skip_leaves_retrieval_query_none_in_tool_state(build_graph, run_graph) -> None:
    """门控跳过（寒暄类输入）→ runtime.state 的 retrieval_query 为 None（拦截器原样放行）。

    WHY 用寒暄输入：此时 `understand_query` 不调 LLM，但**节点仍执行**并写入
    `retrieval_query=None`；工具执行期读到的就是 None，等价 v5.1。
    """
    captured: list[dict[str, object]] = []
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="test")
            ),
            AIMessage(content="", tool_calls=[{"name": "state_probe", "args": {}, "id": "c1"}]),
            fake_text_message("完成"),
        ],
        tools=[_state_probe_tool(captured)],
    )
    response, _ = await run_graph(graph, text="谢谢")

    assert response is not None
    assert captured
    assert captured[0].get("retrieval_query") is None


async def test_understand_query_node_runs_before_recall(build_graph, run_graph) -> None:
    """图拓扑顺序：查询理解节点必须在召回之前（§2.3 D3 的顺序唯一解）。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的"))
    await run_graph(graph, text="你好")

    snapshot = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert snapshot.values.get("query_understanding") is not None
    # 顺序断言用状态而非事件：门控命中时 retrieval_query 为 None，但 query_understanding
    # 必须已被节点写入（未执行则该字段保持 load_context 重置后的 None）。
    understanding = snapshot.values["query_understanding"]
    assert understanding.retrieval_needed is False
    assert NODE_UNDERSTAND_QUERY == "understand_query"
