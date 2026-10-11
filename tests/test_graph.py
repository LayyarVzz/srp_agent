"""agent/core/graph —— 状态机端到端单测（离线 fake LLM + 假工具）。

覆盖：chat 路径、流式 updates、TOOL_USE 无工具直接回答、工具循环（执行/多轮/上限/
失败/未知工具/并行/模型直接回答/决策 LLM 降级）、工具结果以 ToolMessage 回流、
fallback 降级、classify 失败兜底、checkpointer 跨轮、trim 裁剪。
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from agent.core.config import (
    AgentFrameworkConfig,
    AgentGraphConfig,
    MemoryBehaviorConfig,
    SummarizeConfig,
)
from agent.core.graph import (
    _FALLBACK_DISCLAIMER_TEXT,
    _FALLBACK_GENERIC_TEXT,
    _FALLBACK_TOOL_ERROR_TEXT,
    build_agent_graph,
)
from agent.intent.models import Intent, IntentResult
from agent.response.models import AgentResponse
from agent.response.status import Status
from agent.tools.models import TOOL_ERROR_EXECUTION, TOOL_ERROR_UNKNOWN_TOOL
from tests.conftest import (
    RecordingFakeChatModel,
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
    tool_call_messages,
)


def _statuses_of(response: AgentResponse) -> list[Status]:
    """提取状态轨迹的状态序列。"""
    return [e.status for e in response.status_trace]


async def test_chat_path_end_to_end(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """纯聊天路径：意图结构化输出 CHAT → generate → 统一 AgentResponse。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, "你好！我是你的数字人助理。"))
    response, _chunks = await run_graph(graph, text="你好")

    assert isinstance(response, AgentResponse)
    assert response.reply == "你好！我是你的数字人助理。"
    assert response.finished_reason == "completed"
    assert response.session_id == "s1"
    assert _statuses_of(response) == [Status.THINKING, Status.THINKING, Status.SPEAKING]


async def test_missing_session_id_mirrors_thread(
    build_graph: Callable,
) -> None:
    """回归：input 缺 session_id 时兜底为真实 checkpointer 线程，而非固定 "default"。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, "你好"))
    config = {"configurable": {"thread_id": "anon-thread-9"}}
    async for _ in graph.astream({"input": "你好"}, config=config, stream_mode="updates"):
        pass

    state = graph.get_state(config)
    assert state.values["session_id"] == "anon-thread-9"  # 与线程对齐，非 "default"
    assert state.values["user_id"] == "anonymous"


async def test_stream_updates_yields_per_node_status(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """stream_mode="updates" 逐节点下发 status_events 增量（SSE 适配前提）。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, "回答"))
    _response, chunks = await run_graph(graph, text="你好")

    assert len(chunks) >= 3  # 至少 load_context / classify / generate 各一帧
    events = [
        event
        for chunk in chunks
        for node_updates in chunk.values()
        if node_updates  # 空更新的节点产出 {node: None}
        for event in node_updates.get("status_events", [])
    ]
    statuses = [e.status for e in events]
    assert Status.THINKING in statuses
    assert Status.SPEAKING in statuses


async def test_tool_use_without_tools_answers_directly(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """TOOL_USE 意图 + 无可用工具 → 模型直接回答（completed），不走 fallback。"""
    graph = build_graph(chat_turn_messages(Intent.TOOL_USE, "3+5=8", understand=True))
    response, _chunks = await run_graph(graph, text="帮我计算 3+5")

    assert response is not None
    assert response.finished_reason == "completed"
    assert response.reply == "3+5=8"
    assert _FALLBACK_DISCLAIMER_TEXT not in response.reply
    # 状态轨迹与 v5.1 一致：查询理解节点不新增状态值（无检索发生时也不下发 RETRIEVING）。
    assert _statuses_of(response) == [
        Status.THINKING,
        Status.THINKING,
        Status.USING_TOOL,
        Status.SPEAKING,
    ]


async def test_tool_use_single_tool_executes_and_answers(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """单工具执行：call_model 产 tool_calls → ToolNode 执行 → 回环后模型直接回答。"""
    rec: list[dict] = []
    calc = make_fake_tool("calc", content="19", recorder=rec)
    graph = build_graph(
        tool_call_messages(
            [[{"name": "calc", "args": {"expression": "12+7"}, "id": "c1"}]], "12+7=19"
        ),
        tools=[calc],
    )
    response, _chunks = await run_graph(graph, text="帮我计算 12+7")

    assert response is not None
    assert response.finished_reason == "completed"
    assert response.reply == "12+7=19"
    assert rec == [{"expression": "12+7"}]
    trace = response.tool_trace
    assert [(r.tool_name, r.status) for r in trace] == [("calc", "ok")]
    assert trace[0].result is not None and trace[0].result.ok


async def test_tool_loop_multiple_calls(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """多工具循环：calc → datetime，两条工具执行后模型给出终答。"""
    calc_rec: list[dict] = []
    now_rec: list[dict] = []
    calc = make_fake_tool("calc", recorder=calc_rec)
    now = make_fake_tool("datetime", recorder=now_rec)
    graph = build_graph(
        tool_call_messages(
            [
                [{"name": "calc", "args": {"expression": "3+5"}, "id": "c1"}],
                [{"name": "datetime", "args": {}, "id": "c2"}],
            ],
            "已计算",
        ),
        tools=[calc, now],
    )
    response, _chunks = await run_graph(graph, text="先算 3+5 再看时间")

    assert response is not None
    assert response.finished_reason == "completed"
    trace = response.tool_trace
    assert [(r.tool_name, r.status) for r in trace] == [("calc", "ok"), ("datetime", "ok")]
    assert calc_rec == [{"expression": "3+5"}]
    assert now_rec == [{}]


async def test_tool_iteration_cap_emits_tool_limit(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """达循环上限 → generate_answer 且 finished_reason=tool_limit（不再回环）。"""
    rec: list[dict] = []
    calc = make_fake_tool("calc", recorder=rec)
    cfg = AgentFrameworkConfig(graph=AgentGraphConfig(max_tool_iterations=2))
    graph = build_graph(
        tool_call_messages(
            [
                [{"name": "calc", "args": {"expression": "1+1"}, "id": "c1"}],
                [{"name": "calc", "args": {"expression": "2+2"}, "id": "c2"}],
            ],
            "上限回答",
        ),
        config=cfg,
        tools=[calc],
    )
    response, _chunks = await run_graph(graph, text="连续计算")

    assert response is not None
    assert response.finished_reason == "tool_limit"
    assert response.reply == "上限回答"
    assert len(response.tool_trace) == 2
    assert all(r.status == "ok" for r in response.tool_trace)
    assert len(rec) == 2  # 恰好两次，无第三次


async def test_tool_failure_falls_back(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """工具执行抛异常 → ToolNode 转 error ToolMessage → route_after_tool 走 fallback（LLM 知识 + 免责声明）。"""  # noqa: E501
    calc = make_fake_tool("calc", fail_with=RuntimeError("boom"))
    graph = build_graph(
        tool_call_messages([[{"name": "calc", "args": {"expression": "1/0"}, "id": "c1"}]], "占位"),
        tools=[calc],
    )
    response, _chunks = await run_graph(graph, text="帮我计算")

    assert response is not None
    assert response.finished_reason == "fallback"
    assert response.reply.startswith("占位")
    assert _FALLBACK_DISCLAIMER_TEXT in response.reply
    trace = response.tool_trace
    assert trace[-1].status == "error"
    assert trace[-1].result is not None
    assert trace[-1].result.error is not None
    assert trace[-1].result.error.code == TOOL_ERROR_EXECUTION


async def test_unknown_tool_falls_back(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """模型幻觉出未注册工具名 → ToolNode 产 error ToolMessage → fallback，未派发到真实工具。"""
    rec: list[dict] = []
    calc = make_fake_tool("calc", recorder=rec)
    graph = build_graph(
        tool_call_messages([[{"name": "calculator2", "args": {}, "id": "c1"}]], "占位"),
        tools=[calc],
    )
    response, _chunks = await run_graph(graph, text="帮我算一下")

    assert response is not None
    assert response.finished_reason == "fallback"
    assert rec == []  # 未派发
    trace = response.tool_trace
    assert trace[-1].status == "error"
    assert trace[-1].result is not None
    assert trace[-1].result.error is not None
    assert trace[-1].result.error.code == TOOL_ERROR_UNKNOWN_TOOL


async def test_parallel_tool_calls(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """一条 AI 携带多个 tool_calls → ToolNode 并行执行；单轮 = 1 次迭代。"""
    calc_rec: list[dict] = []
    now_rec: list[dict] = []
    calc = make_fake_tool("calc", recorder=calc_rec)
    now = make_fake_tool("datetime", recorder=now_rec)
    graph = build_graph(
        tool_call_messages(
            [
                [
                    {"name": "calc", "args": {"expression": "1+1"}, "id": "c1"},
                    {"name": "datetime", "args": {}, "id": "c2"},
                ]
            ],
            "并行完成",
        ),
        tools=[calc, now],
    )
    response, _chunks = await run_graph(graph, text="并行算一下")

    assert response is not None
    assert response.finished_reason == "completed"
    assert calc_rec == [{"expression": "1+1"}]
    assert now_rec == [{}]
    trace = response.tool_trace
    assert [(r.tool_name, r.status) for r in trace] == [("calc", "ok"), ("datetime", "ok")]
    # 单轮并行 → tool_iterations 仅 1（两工具同一轮被调，无第二轮）。
    state = graph.get_state({"configurable": {"thread_id": "s1"}})
    assert state.values["tool_iterations"] == 1


async def test_parallel_one_failure_falls_back(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """并行多工具中任一失败 → 任一 error ToolMessage 即降级 fallback（确定性）。"""
    calc_rec: list[dict] = []
    calc = make_fake_tool("calc", recorder=calc_rec)
    bad = make_fake_tool("bad", fail_with=ValueError("bad"))
    graph = build_graph(
        tool_call_messages(
            [
                [
                    {"name": "calc", "args": {"expression": "1+1"}, "id": "c1"},
                    {"name": "bad", "args": {}, "id": "c2"},
                ]
            ],
            "占位",
        ),
        tools=[calc, bad],
    )
    response, _chunks = await run_graph(graph, text="并行执行")

    assert response is not None
    assert response.finished_reason == "fallback"
    assert calc_rec == [{"expression": "1+1"}]  # 两工具都被执行
    statuses = {r.tool_name: r.status for r in response.tool_trace}
    assert statuses["calc"] == "ok"
    assert statuses["bad"] == "error"


async def test_model_decides_no_tool_and_answers_directly(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """工具可用但模型未产出 tool_calls → 直接回答（completed），无工具被调、轨迹为空。"""
    rec: list[dict] = []
    calc = make_fake_tool("calc", recorder=rec)
    graph = build_graph(tool_call_messages([], "无需工具，直接回答"), tools=[calc])
    response, _chunks = await run_graph(graph, text="直接回答我")

    assert response is not None
    assert response.finished_reason == "completed"
    assert response.reply == "无需工具，直接回答"
    assert rec == []
    assert response.tool_trace == []


async def test_call_model_llm_error_falls_back(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """call_model 的 bind_tools LLMError（耗竭）→ fallback；兜底 LLM 亦失败 → 固定话术。"""
    rec: list[dict] = []
    calc = make_fake_tool("calc", recorder=rec)
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="t")
            )
        ],
        tools=[calc],
    )
    response, _chunks = await run_graph(graph, text="帮我计算")

    assert response is not None
    assert response.finished_reason == "fallback"
    assert response.reply == _FALLBACK_GENERIC_TEXT
    assert rec == []


async def test_tool_messages_flow_to_generate_and_history(
    make_llm_service: Callable,
    run_graph: Callable,
) -> None:
    """工具结果以 ToolMessage 进 messages/checkpointer；SYSTEM_PROMPT 声明其不可信身份。

    旧实现把工具结果包成「data:」不可信块注入 prompt 且不落 checkpoint；迁移 ToolNode
    后 ToolMessage 作为普通消息合法进入历史，其不可信身份由 system prompt 声明。
    """
    rec: list[dict] = []
    calc = make_fake_tool("calc", content="19", recorder=rec)
    messages = tool_call_messages(
        [[{"name": "calc", "args": {"expression": "12+7"}, "id": "c1"}]], "答案是 19"
    )
    service = make_llm_service(messages, model_cls=RecordingFakeChatModel)
    graph = build_agent_graph(service, tools=[calc])
    response, _chunks = await run_graph(graph, text="帮我计算 12+7")

    assert response is not None
    assert response.finished_reason == "completed"
    # 末次 call_model 的 prompt 含 ToolMessage + 带不可信声明的 SystemMessage。
    last_prompt = service.chat_model.prompts[-1]
    assert any(isinstance(m, ToolMessage) for m in last_prompt)
    sys_msg = next(m for m in last_prompt if isinstance(m, SystemMessage))
    assert "不可信" in sys_msg.content
    # ToolMessage 合法进入 checkpointed messages（LangChain 惯用法）。
    state = graph.get_state({"configurable": {"thread_id": "s1"}})
    assert any(isinstance(m, ToolMessage) for m in state.values["messages"])


async def test_fallback_empty_reply_degrades_to_fixed_text(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """fallback 的 LLM 返回空串 → 降级为工具失败固定话术，不追加免责声明。"""
    calc = make_fake_tool("calc", fail_with=RuntimeError("boom"))
    graph = build_graph(
        tool_call_messages([[{"name": "calc", "args": {}, "id": "c1"}]], ""),
        tools=[calc],
    )
    response, _chunks = await run_graph(graph, text="帮我计算")

    assert response is not None
    assert response.finished_reason == "fallback"
    assert response.reply == _FALLBACK_TOOL_ERROR_TEXT
    assert _FALLBACK_DISCLAIMER_TEXT not in response.reply


async def test_generate_empty_reply_keeps_history_consistent(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """回归：chat 路径 LLM 返回空串 → 节点内拦截为固定话术，历史不留空 AIMessage。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, ""))
    response, _chunks = await run_graph(graph, text="你好")

    assert response is not None
    assert response.reply == _FALLBACK_GENERIC_TEXT
    assert response.finished_reason == "error"

    state = graph.get_state({"configurable": {"thread_id": "s1"}})
    last_msg = state.values["messages"][-1]
    assert last_msg.type == "ai"
    assert str(last_msg.content) == _FALLBACK_GENERIC_TEXT  # 历史与回复一致，非空


async def test_generate_whitespace_reply_degraded_and_consistent(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """全空白回复同样触发兜底，且历史一致。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, "   \t "))
    response, _chunks = await run_graph(graph, text="你好")

    assert response is not None
    assert response.reply == _FALLBACK_GENERIC_TEXT
    state = graph.get_state({"configurable": {"thread_id": "s1"}})
    assert str(state.values["messages"][-1].content) == _FALLBACK_GENERIC_TEXT


async def test_generate_strips_surrounding_whitespace(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """成功路径 strip 首尾空白，回复与历史均为净文本。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, "  你好！  "))
    response, _chunks = await run_graph(graph, text="你好")

    assert response is not None
    assert response.reply == "你好！"
    state = graph.get_state({"configurable": {"thread_id": "s1"}})
    assert str(state.values["messages"][-1].content) == "你好！"


async def test_classify_failure_falls_back_to_rules(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """LLM 结构化输出失败 → 规则兜底 CHAT → 正常走 chat 路径生成回答。"""
    # 第一条是非结构化文本（classify 解析失败→规则兜底），第二条是正常回答。
    graph = build_graph(
        [
            fake_text_message("非法结构化输出"),
            fake_text_message("你好呀，我很好"),
        ]
    )
    response, _chunks = await run_graph(graph, text="你好")

    assert response is not None
    assert response.reply == "你好呀，我很好"
    assert response.finished_reason == "completed"


async def test_checkpointer_multiturn_context(
    build_graph: Callable,
) -> None:
    """同 thread_id 跨轮恢复上下文；新 thread 隔离（thread_id==session_id 契约）。"""
    messages = (
        chat_turn_messages(Intent.CHAT, "回复1")
        + chat_turn_messages(Intent.CHAT, "回复2")
        + chat_turn_messages(Intent.CHAT, "回复3")
    )
    graph = build_graph(messages)
    s1_config = {"configurable": {"thread_id": "s1"}}

    async for _ in graph.astream(
        {"input": "问1", "session_id": "s1"}, config=s1_config, stream_mode="updates"
    ):
        pass
    async for _ in graph.astream(
        {"input": "问2", "session_id": "s1"}, config=s1_config, stream_mode="updates"
    ):
        pass

    s1_state = graph.get_state(s1_config)
    assert len(s1_state.values["messages"]) == 4  # 2 轮 ×（Human + AI）

    s2_config = {"configurable": {"thread_id": "s2"}}
    async for _ in graph.astream(
        {"input": "问3", "session_id": "s2"}, config=s2_config, stream_mode="updates"
    ):
        pass
    s2_state = graph.get_state(s2_config)
    assert len(s2_state.values["messages"]) == 2  # 新线程从空上下文开始


async def test_trim_history_removes_old_messages(
    build_graph: Callable,
) -> None:
    """trim_history 按 Human 轮裁剪：4 轮后早期消息被 RemoveMessage 移除。

    关闭滚动摘要：本用例聚焦「裁剪」本身，避免 summarize_history 额外消费一次 LLM
    调用把 fake 消息序列错位（滚动摘要行为由 test_context_summary 单独覆盖）。
    """
    cfg = AgentFrameworkConfig(
        graph=AgentGraphConfig(trim_keep_recent_rounds=1),
        memory=MemoryBehaviorConfig(summarize=SummarizeConfig(enabled=False)),
    )
    messages = []
    for i in range(4):
        messages += chat_turn_messages(Intent.CHAT, f"回复{i}")
    graph = build_graph(messages, config=cfg)
    config = {"configurable": {"thread_id": "trim"}}

    for i in range(4):
        async for _ in graph.astream(
            {"input": f"第{i + 1}问", "session_id": "trim"}, config=config, stream_mode="updates"
        ):
            pass

    state = graph.get_state(config)
    msgs = state.values["messages"]
    # 无裁剪时 4 轮应有 8 条；裁剪后仅保留最近 1 轮（当前轮 Human + AI 共 2 条）。
    assert len(msgs) < 8
    assert all(getattr(m, "id", None) for m in msgs)  # 显式消息 id 是 RemoveMessage 裁剪的前提
    contents = [str(getattr(m, "content", "")) for m in msgs]
    assert "第1问" not in contents and "第2问" not in contents


async def test_build_graph_with_store_runs_chat(
    build_graph: Callable,
    run_graph: Callable,
) -> None:
    """P4-1 回归：传入 langgraph Store 后图正常编译并跑通一轮 chat（store 注入不影响既有路径）。"""
    graph = build_graph(
        chat_turn_messages(Intent.CHAT, "你好！"),
        store=InMemoryStore(),
    )
    response, _chunks = await run_graph(graph, text="你好")
    assert isinstance(response, AgentResponse)
    assert response.reply == "你好！"
