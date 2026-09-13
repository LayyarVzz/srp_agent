"""图级集成：v4.0 T1 多步任务编排（Plan-and-Solve）。

P4-1 验收（离线 fake LLM + 假工具）：
- 复合任务 → PLANNING → 步骤级 USING_TOOL（「第 i/N 步」）→ 串行工具执行 → 整合回答（completed）；
- 单步失败 → replan_task 重规划继续执行；重规划后再失败且已有 ≥1 步产出 → partial（不丢已得结果）；
- 0 步成功 → fallback_chat 确定性降级；规划非法 / plan 禁用 → 回退 ReAct（既有工具循环零回归）；
- plan 模式工具预算（max_tool_calls_per_plan）独立于 ReAct 全局预算；
- load_context 每轮重置 plan / plan_step / plan_steps_done / replanned。

fake 消息消费契约（与 tests/conftest 一致）：一次 LLM 调用消费一条消息；
ToolNode 执行不消费；structured 消息经 fake_structured_message（tool name = 类名）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from pydantic import Field

from agent.core.config import AgentFrameworkConfig, PlanConfig, SubagentConfig
from agent.core.graph import _PLAN_BLOCK_HEADER, build_agent_graph
from agent.core.models import PlanResult, PlanStep
from agent.intent.models import Intent, IntentResult
from agent.response.models import (
    FINISHED_REASON_COMPLETED,
    FINISHED_REASON_FALLBACK,
    FINISHED_REASON_PARTIAL,
)
from agent.response.status import Status
from tests.conftest import (
    RecordingFakeChatModel,
    StructuredFakeChatModel,
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
)


def _plan_run_messages(plan_msgs: list[AIMessage], final_reply: str) -> list[AIMessage]:
    """PLAN 轮 LLM 消息序列：意图(PLAN) + 规划/执行消息（按真实执行顺序）+ 整合回答。

    `plan_msgs` 由调用方显式按序给出：PlanResult（plan_task 消费）、每步 execute_step
    的 AIMessage（工具步=带 tool_calls / 变换步=纯文本）、重规划 PlanResult
    （replan_task 消费，插在失败步消息之后）……与 fake 逐条消费契约一一对应。
    """
    return [
        fake_structured_message(IntentResult(intent=Intent.PLAN, confidence=0.95, reason="test")),
        *plan_msgs,
        fake_text_message(final_reply),
    ]


def _tool_call_msg(name: str, call_id: str) -> AIMessage:
    """构造产出单条 tool_calls 的 AIMessage（execute_step 的工具步输出）。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": call_id}])


def _serial_cfg() -> AgentFrameworkConfig:
    """子代理关闭的框架配置：本文件验证 v4.0 串行执行语义（T7 默认开启后显式钉住）。"""
    return AgentFrameworkConfig(subagents=SubagentConfig(enabled=False))


def _step_label(status_trace: list[Any], index: int, total: int) -> bool:
    """状态轨迹中是否出现第 index/total 步的 USING_TOOL（tool_name 带进度标签）。"""
    return any(
        e.status == Status.USING_TOOL
        and e.tool_name is not None
        and f"第 {index}/{total} 步" in e.tool_name
        for e in status_trace
    )


async def test_plan_happy_path_serial_execution(build_graph, run_graph) -> None:
    """复合任务串行执行：变换步直接作答 + 工具步 ToolNode 执行 → completed 整合。"""
    plan = PlanResult(
        summary="先总结资料，再翻译成英文",
        steps=[
            PlanStep(goal="总结毕设资料", tool=None),
            PlanStep(goal="把总结翻译成英文", tool="translate"),
        ],
    )
    calc = make_fake_tool("translate", content="English summary")
    graph = build_graph(
        _plan_run_messages(
            [
                fake_structured_message(plan),
                fake_text_message("资料已总结"),  # 第 1 步：变换步直接作答
                _tool_call_msg("translate", "t1"),  # 第 2 步：工具步
            ],
            "整合后的最终回答",
        ),
        config=_serial_cfg(),
        tools=[calc],
    )
    response, _ = await run_graph(graph, text="把毕设资料找出来、总结、翻译成英文")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert response.reply == "整合后的最终回答"
    # 状态轨迹：PLANNING → 步骤级 USING_TOOL（第 1/2 步、第 2/2 步）→ SPEAKING。
    assert Status.PLANNING in [e.status for e in response.status_trace]
    assert _step_label(response.status_trace, 1, 2)
    assert _step_label(response.status_trace, 2, 2)
    # 工具轨迹：翻译步执行成功。
    assert [r.tool_name for r in response.tool_trace] == ["translate"]

    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert (state.values.get("plan_step") or 0) == 2
    assert (state.values.get("plan_steps_done") or 0) == 2
    assert state.values.get("replanned") is False


async def test_plan_execute_step_prompt_contains_plan_block(make_llm_service, run_graph) -> None:
    """execute_step 的 prompt 含计划块（不可信声明 + 步骤进度）。"""
    plan = PlanResult(
        summary="两步",
        steps=[PlanStep(goal="查时间", tool="clock"), PlanStep(goal="翻译", tool=None)],
    )
    service = make_llm_service(
        _plan_run_messages(
            [
                fake_structured_message(plan),
                _tool_call_msg("clock", "c1"),
                fake_text_message("时间已查"),
            ],
            "最终回答",
        ),
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(
        service, _serial_cfg(), tools=[make_fake_tool("clock", content="10:00")]
    )
    _, _ = await run_graph(graph, text="查时间并翻译")

    prompts = service.chat_model.prompts
    # 规划调用（prompts[1]）之后是 execute_step 的调用：含计划块与当前步指令。
    step_text = "".join(str(getattr(m, "content", "")) for m in prompts[2])
    assert _PLAN_BLOCK_HEADER in step_text
    assert "1/2 [执行中/未完成]" in step_text
    assert "2/2 [待执行]" in step_text
    assert "只完成这一步的目标" in step_text


async def test_plan_step_tool_failure_triggers_replan(build_graph, run_graph) -> None:
    """单步工具失败（未重规划过）→ replan_task 重规划 → 新计划继续执行（completed）。"""
    plan = PlanResult(
        summary="两步",
        steps=[PlanStep(goal="查询数据", tool="fetch"), PlanStep(goal="汇总", tool="bad_tool")],
    )
    ok_tool = make_fake_tool("fetch", content="数据")
    bad_tool = make_fake_tool("bad_tool", fail_with=RuntimeError("boom"))
    new_plan = PlanResult(summary="重规划", steps=[PlanStep(goal="查询数据", tool="fetch")])
    graph = build_graph(
        _plan_run_messages(
            [
                fake_structured_message(plan),
                _tool_call_msg("fetch", "f1"),  # 第 1 步成功
                _tool_call_msg("bad_tool", "b1"),  # 第 2 步失败 → 重规划
                fake_structured_message(new_plan),  # replan_task 消费
                _tool_call_msg("fetch", "f2"),  # 新计划第 1 步
            ],
            "整合回答",
        ),
        config=_serial_cfg(),
        tools=[ok_tool, bad_tool],
    )
    response, _ = await run_graph(graph, text="查数据并汇总")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    # 两次 PLANNING（初始规划 + 重规划）。
    assert [e.status for e in response.status_trace].count(Status.PLANNING) == 2
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state.values.get("replanned") is True
    assert (state.values.get("plan_steps_done") or 0) == 2  # 两轮计划各成功 1 步


async def test_replan_failure_with_prior_success_is_partial(build_graph, run_graph) -> None:
    """重规划后再失败且已有 ≥1 步成功产出 → partial（整合已得结果，不丢）。"""
    plan = PlanResult(
        summary="两步",
        steps=[PlanStep(goal="查询数据", tool="fetch"), PlanStep(goal="汇总", tool="bad")],
    )
    ok_tool = make_fake_tool("fetch", content="数据")
    bad_tool = make_fake_tool("bad", fail_with=RuntimeError("boom"))
    new_plan = PlanResult(summary="重规划", steps=[PlanStep(goal="再汇总", tool="bad")])
    graph = build_graph(
        _plan_run_messages(
            [
                fake_structured_message(plan),
                _tool_call_msg("fetch", "f1"),  # 第 1 步成功
                _tool_call_msg("bad", "b1"),  # 第 2 步失败 → 重规划
                fake_structured_message(new_plan),  # replan_task 消费
                _tool_call_msg("bad", "b2"),  # 新计划第 1 步再失败（已重规划过）
            ],
            "部分成功回答",
        ),
        config=_serial_cfg(),
        tools=[ok_tool, bad_tool],
    )
    response, _ = await run_graph(graph, text="查数据并汇总")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_PARTIAL
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert (state.values.get("plan_steps_done") or 0) == 1  # 首轮第 1 步成功产出


async def test_zero_steps_succeeded_falls_back(build_graph, run_graph) -> None:
    """重规划后再失败且 0 步成功 → fallback_chat 确定性降级。"""
    plan = PlanResult(summary="单步", steps=[PlanStep(goal="查数据", tool="bad")])
    bad_tool = make_fake_tool("bad", fail_with=RuntimeError("boom"))
    new_plan = PlanResult(summary="重规划", steps=[PlanStep(goal="再查", tool="bad")])
    graph = build_graph(
        _plan_run_messages(
            [
                fake_structured_message(plan),
                _tool_call_msg("bad", "b1"),  # 第 1 步失败 → 重规划
                fake_structured_message(new_plan),  # replan_task 消费
                _tool_call_msg("bad", "b2"),  # 新计划第 1 步再失败（0 步成功）
            ],
            "兜底话术",
        ),
        tools=[bad_tool],
    )
    response, _ = await run_graph(graph, text="查数据")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_FALLBACK
    # fallback_chat 对 LLM 兜底回答确定性追加免责声明（设计行为）。
    assert response.reply.startswith("兜底话术")
    assert "未经核实" in response.reply
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert (state.values.get("plan_steps_done") or 0) == 0


async def test_invalid_plan_falls_back_to_react(build_graph, run_graph) -> None:
    """规划非法（工具名不在注册表）→ plan=None → 回退 ReAct 工具循环（completed）。"""
    bad_plan = PlanResult(summary="幻觉计划", steps=[PlanStep(goal="查", tool="ghost_tool")])
    calc = make_fake_tool("calc", content="19")
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.PLAN, confidence=0.95, reason="test")
            ),
            fake_structured_message(bad_plan),  # 校验失败 → 回退
            _tool_call_msg("calc", "c1"),
            fake_text_message("ReAct 回答"),
        ],
        tools=[calc],
    )
    response, _ = await run_graph(graph, text="帮我计算 12+7")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert [r.tool_name for r in response.tool_trace] == ["calc"]  # ReAct 执行了工具
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state.values.get("plan") is None  # 规划被拒绝


async def test_plan_disabled_falls_back_to_react(build_graph, run_graph) -> None:
    """plan.enabled=False：PLAN 意图直接回退 ReAct（与 v3.0 行为一致，零回归）。"""
    cfg = AgentFrameworkConfig(plan=PlanConfig(enabled=False))
    calc = make_fake_tool("calc", content="19")
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.PLAN, confidence=0.95, reason="test")
            ),
            _tool_call_msg("calc", "c1"),
            fake_text_message("ReAct 回答"),
        ],
        config=cfg,
        tools=[calc],
    )
    response, _ = await run_graph(graph, text="帮我计算 12+7")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert [r.tool_name for r in response.tool_trace] == ["calc"]
    assert Status.PLANNING not in [e.status for e in response.status_trace]


async def test_plan_tool_budget_leads_to_partial(build_graph, run_graph) -> None:
    """plan 模式达 max_tool_calls_per_plan → generate_answer 整合（partial）。"""
    cfg = AgentFrameworkConfig(plan=PlanConfig(max_tool_calls_per_plan=1))
    plan = PlanResult(summary="单步", steps=[PlanStep(goal="查询", tool="fetch")])
    calc = make_fake_tool("fetch", content="数据")
    graph = build_graph(
        _plan_run_messages(
            [fake_structured_message(plan), _tool_call_msg("fetch", "f1")],
            "已达预算的回答",
        ),
        config=cfg,
        tools=[calc],
    )
    response, _ = await run_graph(graph, text="查询数据")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_PARTIAL
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert (state.values.get("plan_step") or 0) == 0  # 未推进即被预算截断


class _RaiseAtCallModel(StructuredFakeChatModel):
    """第 raise_at 次 LLM 调用抛 RuntimeError（模拟 execute_step 的 LLM 失败）。

    计数在 super()._generate 消费消息之后自增再抛，保证被抛的那条消息已从
    迭代器消费、后续调用能取到下一条（replan 的 PlanResult）。
    """

    raise_at: int = 2  # 0=classify, 1=plan_task, 2=execute_step
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
            raise RuntimeError("injected step failure")
        return result


async def test_execute_step_llm_error_triggers_replan(make_llm_service, run_graph) -> None:
    """execute_step 的 LLM 调用失败（llm_error.*）→ 与工具失败同语义：重规划。"""
    plan = PlanResult(summary="两步", steps=[PlanStep(goal="查询", tool="fetch")])
    new_plan = PlanResult(summary="重规划", steps=[PlanStep(goal="查询", tool="fetch")])
    service = make_llm_service(
        [
            fake_structured_message(
                IntentResult(intent=Intent.PLAN, confidence=0.95, reason="test")
            ),
            fake_structured_message(plan),
            _tool_call_msg("fetch", "f1"),  # 被 execute_step 消费后抛错
            fake_structured_message(new_plan),  # replan_task 消费
            _tool_call_msg("fetch", "f2"),
            fake_text_message("最终回答"),
        ],
        model_cls=_RaiseAtCallModel,
    )
    graph = build_agent_graph(service, tools=[make_fake_tool("fetch", content="数据")])
    response, _ = await run_graph(graph, text="查询数据")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state.values.get("replanned") is True


async def test_plan_state_reset_across_turns(build_graph, run_graph) -> None:
    """跨轮重置：PLAN 轮结束保留 plan 状态，下一轮 CHAT 的 load_context 全部清空。"""
    plan = PlanResult(summary="单步", steps=[PlanStep(goal="总结", tool=None)])
    calc = make_fake_tool("x", content="ok")
    # 两轮消息一次性提供：第 1 轮 PLAN（4 条：意图/规划/变换步/整合），第 2 轮 CHAT（2 条）。
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.PLAN, confidence=0.95, reason="test")
            ),
            fake_structured_message(plan),
            fake_text_message("总结完毕"),
            fake_text_message("整合回答"),
            *chat_turn_messages(Intent.CHAT, "好的"),
        ],
        tools=[calc],
    )
    resp1, _ = await run_graph(graph, text="帮我总结一下", session_id="s1")
    state1 = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state1.values.get("plan") is not None
    assert (state1.values.get("plan_steps_done") or 0) == 1

    resp2, _ = await run_graph(graph, text="你好", session_id="s1")
    assert resp1 is not None and resp2 is not None
    state2 = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state2.values.get("plan") is None
    assert (state2.values.get("plan_step") or 0) == 0
    assert (state2.values.get("plan_steps_done") or 0) == 0
    assert state2.values.get("replanned") is False


async def test_non_plan_turn_plan_stays_none(build_graph, run_graph) -> None:
    """非 PLAN 轮（TOOL_USE）不进入规划：plan 全程 None，ReAct 行为零回归。"""
    calc = make_fake_tool("calc", content="19")
    graph = build_graph(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="test")
            ),
            _tool_call_msg("calc", "c1"),
            fake_text_message("计算回答"),
        ],
        tools=[calc],
    )
    response, _ = await run_graph(graph, text="帮我计算 12+7")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert Status.PLANNING not in [e.status for e in response.status_trace]
    state = await graph.aget_state({"configurable": {"thread_id": "s1"}})
    assert state.values.get("plan") is None
