"""图级集成：v4.0 T2 澄清式追问（Clarification）。

P4-2 验收（离线 fake LLM + 假工具）：
- 触发源①：CHAT 意图置信度低于 `min_confidence` → `clarify`（CLARIFYING → 反问）；
- 触发源②：ReAct 工具参数缺失（`tool_error.missing_argument`）→ `clarify`（替代降级道歉）；
- 不触发场景：高置信 CHAT / 澄清关闭 / 未知工具（unknown_tool 仍降级）/ plan 模式
  （missing_argument 走重规划，§4.1 例外）；
- 防循环：`clarify_asked` 置位后不再二次追问；`load_context` 每轮重置；
- clarify 自身失败（LLM 异常 / 空结果）→ `fallback_chat` 确定性兜底；
- clarify 的 prompt 带不可信声明（安全约束）；追问作为正常 AI 消息入历史。

fake 消息消费契约（与 tests/conftest 一致）：一次 LLM 调用消费一条消息；
structured 消息经 fake_structured_message（tool name = 类名）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.core.config import AgentFrameworkConfig, ClarifyConfig, SubagentConfig
from agent.core.graph import build_agent_graph
from agent.core.models import PlanResult, PlanStep
from agent.intent.models import Intent, IntentResult
from agent.response.models import (
    FINISHED_REASON_COMPLETED,
    FINISHED_REASON_FALLBACK,
    FINISHED_REASON_NEEDS_CLARIFICATION,
    ClarifyResult,
)
from agent.response.status import Status
from agent.tools.models import TOOL_ERROR_MISSING_ARGUMENT, TOOL_ERROR_UNKNOWN_TOOL
from tests.conftest import (
    RecordingFakeChatModel,
    StructuredFakeChatModel,
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
)


class _ClockArgs(BaseModel):
    """带必填参数的假工具 schema：`zone` 缺失 → ToolNode 参数校验失败。"""

    zone: str = Field(description="时区")


def _schema_tool(name: str = "clock") -> StructuredTool:
    """构造带 args_schema 的假工具（真实 MCP 工具形态：参数缺失产生 ToolInvocationError）。"""

    async def _run(zone: str) -> str:
        return f"time in {zone}"

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=f"{name} 假工具（必填 zone）",
        args_schema=_ClockArgs,
    )


def _tool_call_msg(name: str, call_id: str, args: dict[str, object] | None = None) -> AIMessage:
    """构造产出单条 tool_calls 的 AIMessage（args 缺省 = 空 dict，触发参数缺失）。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


def _clarify_turn(intent: Intent, question: str, options: list[str]) -> list[AIMessage]:
    """触发澄清的一轮 LLM 消息序列：意图 + ClarifyResult（clarify 消费，回合终止）。"""
    return [
        fake_structured_message(IntentResult(intent=intent, confidence=0.3, reason="模糊")),
        fake_structured_message(ClarifyResult(question=question, options=options)),
    ]


# —— 触发源①：意图低置信 ——


async def test_clarify_low_confidence_chat_asks(build_graph, run_graph) -> None:
    """CHAT 低置信（0.3 < 0.5）→ clarify：CLARIFYING + 反问 + options + 历史一致。"""
    graph = build_graph(_clarify_turn(Intent.CHAT, "你是想查天气还是新闻？", ["天气", "新闻"]))
    response, _ = await run_graph(graph, text="我想知道一下那个")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
    assert response.clarification is not None
    assert response.clarification.question == "你是想查天气还是新闻？"
    assert response.clarification.options == ["天气", "新闻"]
    # reply 即反问正文（前端口播/文本展示）。
    assert response.reply == "你是想查天气还是新闻？"
    # 状态轨迹含 CLARIFYING。
    assert Status.CLARIFYING in [e.status for e in response.status_trace]

    # 追问作为正常 AI 消息写入历史（与 fallback/generate 一致）。
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    last = state.values["messages"][-1]
    assert last.type == "ai"
    assert str(last.content) == "你是想查天气还是新闻？"
    assert state.values.get("clarify_asked") is True
    assert state.values.get("clarification") is not None


async def test_clarify_high_confidence_chat_not_asked(build_graph, run_graph) -> None:
    """高置信 CHAT → 直接回答（不澄清，零回归）。"""
    graph = build_graph(chat_turn_messages(Intent.CHAT, "好的，有什么可以帮你？"))
    response, _ = await run_graph(graph, text="你好")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert response.clarification is None
    assert Status.CLARIFYING not in [e.status for e in response.status_trace]


async def test_clarify_disabled_low_confidence_direct_answer(build_graph, run_graph) -> None:
    """clarify.enabled=False：低置信 CHAT 直接回答（配置关闭零回归）。"""
    cfg = AgentFrameworkConfig(clarify=ClarifyConfig(enabled=False))
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.CHAT, confidence=0.3, reason="模糊")
            ),
            fake_text_message("直接回答"),
        ],
        config=cfg,
    )
    response, _ = await run_graph(graph, text="我想知道一下那个")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert response.reply == "直接回答"
    assert Status.CLARIFYING not in [e.status for e in response.status_trace]


async def test_clarify_low_confidence_tool_use_asks(build_graph, run_graph) -> None:
    """触发源①扩展：TOOL_USE 低置信（模糊工具请求，如「帮我算一下」）→ 澄清而非硬答。"""
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.3, reason="缺算式")
            ),
            fake_structured_message(
                ClarifyResult(question="你想让我计算什么？请给出算式", options=[])
            ),
        ],
        tools=[_schema_tool()],
    )
    response, _ = await run_graph(graph, text="帮我算一下")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
    assert response.clarification is not None
    assert "算式" in response.clarification.question
    assert Status.CLARIFYING in [e.status for e in response.status_trace]
    # 未进入工具执行（澄清优先于硬调工具）。
    assert response.tool_trace == []


async def test_clarify_low_confidence_plan_asks(build_graph, run_graph) -> None:
    """触发源①扩展：PLAN 低置信（模糊复合任务）→ 澄清而非盲目规划。"""
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.PLAN, confidence=0.35, reason="对象不明")
            ),
            fake_structured_message(
                ClarifyResult(question="你想让我整理什么内容？", options=["文档", "数据", "图片"])
            ),
        ],
        tools=[_schema_tool()],
    )
    response, _ = await run_graph(graph, text="把那个整理一下发给我")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
    assert response.clarification is not None
    assert response.clarification.options == ["文档", "数据", "图片"]
    assert Status.CLARIFYING in [e.status for e in response.status_trace]


# —— 触发源②：工具参数缺失 ——


async def test_clarify_missing_argument_triggers(build_graph, run_graph) -> None:
    """ReAct 工具参数缺失（missing_argument）→ clarify（替代降级道歉）。"""
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="工具")
            ),
            _tool_call_msg("clock", "c1"),  # args={} → zone 缺失 → ToolInvocationError
            fake_structured_message(
                ClarifyResult(
                    question="请问要查询哪个时区的时间？", options=["UTC", "Asia/Shanghai"]
                )
            ),
        ],
        tools=[_schema_tool()],
    )
    response, _ = await run_graph(graph, text="帮我查一下时间")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
    assert response.clarification is not None
    assert "时区" in response.clarification.question
    # 工具轨迹：参数缺失被细分为 missing_argument（§4.4）。
    assert response.tool_trace[0].status == "error"
    assert response.tool_trace[0].result is not None
    assert response.tool_trace[0].result.error is not None
    assert response.tool_trace[0].result.error.code == TOOL_ERROR_MISSING_ARGUMENT
    assert Status.CLARIFYING in [e.status for e in response.status_trace]


async def test_clarify_unknown_tool_still_falls_back(build_graph, run_graph) -> None:
    """未知工具（模型幻觉，unknown_tool）不触发澄清 → 既有 fallback 降级（零回归）。"""
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="工具")
            ),
            _tool_call_msg("ghost_tool", "g1"),
            fake_text_message("兜底"),
        ],
        tools=[make_fake_tool("clock", content="ok")],
    )
    response, _ = await run_graph(graph, text="调用一下那个工具")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_FALLBACK
    assert response.clarification is None
    assert response.tool_trace[0].result is not None
    assert response.tool_trace[0].result.error is not None
    assert response.tool_trace[0].result.error.code == TOOL_ERROR_UNKNOWN_TOOL
    assert Status.CLARIFYING not in [e.status for e in response.status_trace]


# —— plan 模式例外：missing_argument 走重规划，不澄清 ——


async def test_plan_mode_missing_argument_replans_not_clarify(build_graph, run_graph) -> None:
    """plan 模式参数缺失 → 走重规划（§4.1 例外），不触发澄清。"""
    plan = PlanResult(
        summary="两步",
        steps=[PlanStep(goal="查询时区时间", tool="clock"), PlanStep(goal="汇总", tool=None)],
    )
    new_plan = PlanResult(summary="重规划", steps=[PlanStep(goal="查询 UTC 时间", tool="clock")])
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.PLAN, confidence=0.95, reason="复合")
            ),
            fake_structured_message(plan),
            _tool_call_msg("clock", "c1"),  # 第 1 步参数缺失 → 重规划
            fake_structured_message(new_plan),
            _tool_call_msg("clock", "c2", {"zone": "UTC"}),  # 新计划第 1 步成功
            fake_text_message("整合回答"),
        ],
        # 双无依赖步计划 + v4.0 串行语义验证：显式关闭 T7 并行子代理。
        config=AgentFrameworkConfig(subagents=SubagentConfig(enabled=False)),
        tools=[_schema_tool()],
    )
    response, _ = await run_graph(graph, text="查时间并汇总")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert response.clarification is None
    assert Status.CLARIFYING not in [e.status for e in response.status_trace]
    # 首步的缺失参数被细分并进入轨迹（plan 模式仍记录错误码，只是路由走重规划）。
    assert response.tool_trace[0].result is not None
    assert response.tool_trace[0].result.error is not None
    assert response.tool_trace[0].result.error.code == TOOL_ERROR_MISSING_ARGUMENT
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state.values.get("replanned") is True


# —— clarify 自身失败：确定性兜底 ——


class _RaiseAtClarify(StructuredFakeChatModel):
    """第 raise_at 次 LLM 调用抛 RuntimeError（模拟 clarify 的 LLM 失败）。

    计数在 super()._generate 消费消息之后自增再抛，保证被抛的那条消息已从
    迭代器消费、后续调用能取到下一条（fallback_chat 的文本消息）。
    """

    raise_at: int = 1  # 0=classify, 1=clarify
    calls: int = Field(default=0, exclude=True)

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        self.calls += 1
        if self.calls - 1 == self.raise_at:
            raise RuntimeError("injected clarify failure")
        return result


async def test_clarify_llm_failure_falls_back(make_llm_service, run_graph) -> None:
    """clarify 的 LLM 调用失败 → route_after_clarify → fallback_chat 确定性兜底。"""
    service = make_llm_service(
        [
            fake_structured_message(
                IntentResult(intent=Intent.CHAT, confidence=0.3, reason="模糊")
            ),
            fake_structured_message(ClarifyResult(question="q", options=[])),
            fake_text_message("兜底回答"),
        ],
        model_cls=_RaiseAtClarify,
    )
    graph = build_agent_graph(service)
    response, _ = await run_graph(graph, text="我想知道一下那个")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_FALLBACK
    assert response.clarification is None
    assert response.reply.startswith("兜底回答")  # fallback_chat 的 LLM 兜底尝试成功
    assert Status.CLARIFYING in [e.status for e in response.status_trace]


async def test_clarify_empty_result_falls_back(build_graph, run_graph) -> None:
    """clarify 返回空 question（模型未产出有效反问）→ fallback_chat 兜底。"""
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.CHAT, confidence=0.3, reason="模糊")
            ),
            fake_structured_message(ClarifyResult(question="", options=[])),
            fake_text_message("兜底"),
        ]
    )
    response, _ = await run_graph(graph, text="我想知道一下那个")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_FALLBACK
    assert response.clarification is None


# —— 安全约束：prompt 带不可信声明 ——


async def test_clarify_prompt_contains_untrusted_declaration(make_llm_service, run_graph) -> None:
    """clarify 的 prompt 必须声明用户输入为不可信数据（安全约束）。"""
    service = make_llm_service(
        _clarify_turn(Intent.CHAT, "你是想查 A 还是 B？", ["A", "B"]),
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(service)
    _, _ = await run_graph(graph, text="我想知道一下那个")

    # prompts[0] = classify，prompts[1] = clarify。
    clarify_prompt = "".join(str(getattr(m, "content", "")) for m in service.chat_model.prompts[1])
    assert "不可信" in clarify_prompt


# —— 防循环与跨轮重置 ——


async def test_clarify_state_reset_across_turns(build_graph, run_graph) -> None:
    """跨轮重置：澄清轮结束保留 clarify_asked，下一轮 load_context 全部清空。"""
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.CHAT, confidence=0.3, reason="模糊")
            ),
            fake_structured_message(ClarifyResult(question="q1", options=["A"])),
            *chat_turn_messages(Intent.CHAT, "好的"),
        ]
    )
    resp1, _ = await run_graph(graph, text="我想知道一下那个", session_id="s1")
    state1 = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state1.values.get("clarify_asked") is True
    assert state1.values.get("clarification") is not None

    resp2, _ = await run_graph(graph, text="那算了", session_id="s1")
    assert resp1 is not None and resp2 is not None
    state2 = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state2.values.get("clarify_asked") is False
    assert state2.values.get("clarification") is None
    assert resp2.clarification is None
