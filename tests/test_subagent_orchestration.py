"""图级集成：T7 内部并发子代理（dev-version5.0.md §6 / V5-M5 验收）。

覆盖（离线 routed fake LLM + 假工具）：
- 双无依赖步计划 → DELEGATING + Send 并行扇出 + join 回填 → 串行收尾步 → completed；
- 批内失败 → replan（≤1 次）；重规划后再失败且 ≥1 步成功 → partial（不丢已得结果）；
  0 步成功 → fallback_chat 确定性降级；
- SubagentConfig.enabled=False / max_parallel=1 → 与 v4.0 串行路径逐字节一致（零回归）；
- plan 工具预算按「本轮全部工具调用记录数」计入（子代理调用不豁免）；
- load_context 每轮重置子代理状态（subagent_results/dispatch_round/完成索引）。

fake 消费契约：RoutedFakeChatModel 按 prompt 特征文本路由（并行分支各自消费独立脚本，
与分支调度顺序无关）；未命中 marker 走默认迭代器（意图分类 + 最终整合文本）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from pydantic import SecretStr

from agent.core.config import AgentFrameworkConfig, LLMConfig, PlanConfig, SubagentConfig
from agent.core.graph import build_agent_graph
from agent.core.models import PlanResult, PlanStep, SubagentResult
from agent.intent.models import Intent, IntentResult
from agent.llm import LLMService
from agent.response.models import (
    FINISHED_REASON_COMPLETED,
    FINISHED_REASON_FALLBACK,
    FINISHED_REASON_PARTIAL,
)
from agent.response.status import Status
from agent.tools.models import ToolCallRecord
from tests.conftest import (
    RoutedFakeChatModel,
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
)

# 计划/子代理 prompt 的特征 marker（与 graph.py 模板文本强耦合，改模板须同步）。
# 注意：replan 模板同样以「你是任务规划器」开头，不能用该短语区分两者。
_M_PLAN = "分解为有序、可逐步执行的计划"
_M_REPLAN = "重新规划一份完整的新计划"
_M_SUB = "子任务目标："
_M_SERIAL = "执行计划第"


def _plan_intent() -> AIMessage:
    return fake_structured_message(IntentResult(intent=Intent.PLAN, confidence=0.95, reason="复合"))


def _chat_intent() -> AIMessage:
    return fake_structured_message(IntentResult(intent=Intent.CHAT, confidence=0.95, reason="闲聊"))


def _tool_call_msg(name: str, call_id: str, args: dict[str, Any] | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


def _routed_llm(routes: list[tuple[str, list[AIMessage]]], default: list[AIMessage]) -> LLMService:
    model = RoutedFakeChatModel(routes=routes, messages=iter(default))
    return LLMService(config=LLMConfig(api_key=SecretStr("sk-x")), chat_model=model)


def _build_graph(
    routes: list[tuple[str, list[AIMessage]]],
    default: list[AIMessage],
    *,
    cfg: AgentFrameworkConfig | None = None,
    tools: list | None = None,
) -> Any:
    """构建注入 routed fake 的主图（默认配置 = 生产默认，子代理开启）。"""
    return build_agent_graph(_routed_llm(routes, default), cfg, tools=tools)


def _serial_cfg() -> AgentFrameworkConfig:
    """子代理关闭（v4.0 零回归验证用）。"""
    return AgentFrameworkConfig(subagents=SubagentConfig(enabled=False))


async def _finish(graph: Any, text: str) -> tuple[Any, Any]:
    """驱动一轮会话（updates 流），返回 (AgentResponse, 最终图状态)。"""
    response = None
    config = {"configurable": {"thread_id": "s1"}}
    async for chunk in graph.astream(
        {"input": text, "session_id": "s1"}, config=config, stream_mode="updates"
    ):
        for updates in chunk.values():
            if updates and "response" in updates:
                response = updates["response"]
    state = await graph.aget_state(config)
    return response, state.values


async def test_parallel_batch_fans_out_and_joins() -> None:
    """双无依赖步 → DELEGATING 扇出 2 个子代理 → join 回填 → 单步就绪落回串行收尾。"""
    plan = PlanResult(
        summary="并行计算两路结果再汇总",
        steps=[
            PlanStep(goal="计算甲组数值", tool="calc"),
            PlanStep(goal="计算乙组数值", tool="calc"),
            PlanStep(goal="汇总两步结果", tool=None, depends_on=[0, 1]),
        ],
    )
    recorder: list[dict] = []
    calc = make_fake_tool("calc", content="19", recorder=recorder)
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (
                _M_SUB + "计算甲组数值",
                [_tool_call_msg("calc", "c1", {"x": 1}), fake_text_message("甲组结果 19")],
            ),
            (
                _M_SUB + "计算乙组数值",
                [_tool_call_msg("calc", "c2", {"x": 2}), fake_text_message("乙组结果 40")],
            ),
            (_M_SERIAL + " 3/3 步", [fake_text_message("汇总完成")]),
        ],
        [_plan_intent(), fake_text_message("最终整合回答")],
        tools=[calc],
    )
    response, values = await _finish(graph, "并行计算甲乙两组并汇总")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert response.reply == "最终整合回答"
    # 状态轨迹：DELEGATING（并行处理 2 个子任务）→ 串行步 USING_TOOL → SPEAKING。
    delegating = [e for e in response.status_trace if e.status == Status.DELEGATING]
    assert len(delegating) == 1
    assert "正在并行处理 2 个子任务" in (delegating[0].message or "")
    # 工具轨迹：两个子代理各一次 calc（并行分支产出均透出为 tool 事件）。
    assert sorted(r.arguments.get("x") for r in response.tool_trace) == [1, 2]
    assert all(r.status == "ok" for r in response.tool_trace)
    # 两个并行分支真的都执行了工具（而非串行复用）。
    assert sorted(r["x"] for r in recorder) == [1, 2]
    # 状态回填：完成索引 = 两个并行步 + 串行步（推进节点合成时入账）；
    # 批序号 1；串行步推进指针；合成串行结果入账。
    assert values.get("plan_steps_completed") == [0, 1, 2]
    assert values.get("dispatch_round") == 1
    assert values.get("plan_step") == 3
    assert values.get("plan_steps_done") == 3
    results = values.get("subagent_results") or []
    assert {r.step_index for r in results} == {0, 1, 2}
    serial_result = next(r for r in results if r.step_index == 2)
    assert serial_result.ok is True
    assert serial_result.summary == "汇总完成"


async def test_batch_failure_triggers_replan() -> None:
    """批内一步失败 → join 置错 → replan_task 重规划 → 新计划串行继续（completed）。"""
    plan = PlanResult(
        summary="两路查询后汇总",
        steps=[
            PlanStep(goal="查询甲数据", tool="fetch"),
            PlanStep(goal="汇总数据", tool="bad_tool"),
        ],
    )
    new_plan = PlanResult(summary="重规划", steps=[PlanStep(goal="再查询", tool="fetch")])
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (_M_SUB + "查询甲数据", [_tool_call_msg("fetch", "f1"), fake_text_message("甲数据")]),
            (_M_SUB + "汇总数据", [_tool_call_msg("bad_tool", "b1")]),
            (_M_REPLAN, [fake_structured_message(new_plan)]),
            (_M_SERIAL + " 1/1 步", [_tool_call_msg("fetch", "f2")]),
        ],
        [_plan_intent(), fake_text_message("整合回答")],
        tools=[
            make_fake_tool("fetch", content="数据"),
            make_fake_tool("bad_tool", fail_with=RuntimeError("boom")),
        ],
    )
    response, values = await _finish(graph, "查询并汇总")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    # 两次 PLANNING（初始规划 + 重规划）；重规划后批序号清零且未再扇出（单步就绪走串行）。
    assert [e.status for e in response.status_trace].count(Status.PLANNING) == 2
    assert values.get("replanned") is True
    assert values.get("dispatch_round") == 0
    assert values.get("plan_steps_done") == 2  # join 记账 1 步 + 串行推进 1 步
    # 工具轨迹：fetch 成功 ×2 + bad_tool 失败 ×1（失败步进入轨迹，驱动重规划）。
    by_name: dict[str, list[ToolCallRecord]] = {}
    for record in response.tool_trace:
        by_name.setdefault(record.tool_name, []).append(record)
    assert len(by_name["fetch"]) == 2 and all(r.status == "ok" for r in by_name["fetch"])
    assert by_name["bad_tool"][0].status == "error"
    # 失败子代理结果保留在结果列表（ok=False），供整合阶段说明失败原因。
    results = values.get("subagent_results") or []
    assert any(r.step_index == 1 and r.ok is False for r in results)


async def test_replan_failure_with_prior_success_is_partial() -> None:
    """重规划后再失败且已有 ≥1 步成功产出 → partial（整合已得结果，不丢）。"""
    plan = PlanResult(
        summary="两路查询后汇总",
        steps=[
            PlanStep(goal="查询甲数据", tool="fetch"),
            PlanStep(goal="汇总数据", tool="bad"),
        ],
    )
    new_plan = PlanResult(summary="重规划", steps=[PlanStep(goal="再汇总", tool="bad")])
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (_M_SUB + "查询甲数据", [_tool_call_msg("fetch", "f1"), fake_text_message("甲数据")]),
            (_M_SUB + "汇总数据", [_tool_call_msg("bad", "b1")]),
            (_M_REPLAN, [fake_structured_message(new_plan)]),
            (_M_SERIAL + " 1/1 步", [_tool_call_msg("bad", "b2")]),
        ],
        [_plan_intent(), fake_text_message("部分成功回答")],
        tools=[
            make_fake_tool("fetch", content="数据"),
            make_fake_tool("bad", fail_with=RuntimeError("boom")),
        ],
    )
    response, values = await _finish(graph, "查询并汇总")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_PARTIAL
    assert response.reply == "部分成功回答"
    assert values.get("plan_steps_done") == 1  # 首批 join 记账的成功步（跨计划累计）


async def test_zero_success_after_replan_falls_back() -> None:
    """重规划后的批再全部失败且 0 步成功 → fallback_chat 确定性降级。"""
    plan = PlanResult(
        summary="两路查询",
        steps=[PlanStep(goal="故障一步", tool="bad"), PlanStep(goal="故障二步", tool="bad")],
    )
    new_plan = PlanResult(
        summary="重规划",
        steps=[PlanStep(goal="再故障一步", tool="bad"), PlanStep(goal="再故障二步", tool="bad")],
    )
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (_M_SUB + "故障一步", [_tool_call_msg("bad", "b1")]),
            (_M_SUB + "故障二步", [_tool_call_msg("bad", "b2")]),
            (_M_REPLAN, [fake_structured_message(new_plan)]),
            (_M_SUB + "再故障一步", [_tool_call_msg("bad", "b3")]),
            (_M_SUB + "再故障二步", [_tool_call_msg("bad", "b4")]),
        ],
        [_plan_intent(), fake_text_message("兜底回答文本")],
        tools=[make_fake_tool("bad", fail_with=RuntimeError("boom"))],
    )
    response, values = await _finish(graph, "两路查询")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_FALLBACK
    assert response.reply.startswith("兜底回答文本")
    assert "未经核实" in response.reply  # 免责声明由系统统一追加
    assert values.get("plan_steps_done") == 0
    # 重规划后批次重新扇出（dispatch_round 重置后再次推进）；4 个失败结果全部保留。
    assert values.get("dispatch_round") == 1
    assert len(values.get("subagent_results") or []) == 4


async def test_subagent_disabled_serial_zero_regression() -> None:
    """enabled=False → 与 v4.0 串行路径一致：步骤级 USING_TOOL、无 DELEGATING、零状态写入。"""
    plan = PlanResult(
        summary="两步串行",
        steps=[
            PlanStep(goal="查询甲数据", tool="fetch"),
            PlanStep(goal="查询乙数据", tool="fetch"),
        ],
    )
    graph = _build_graph(
        [],
        [
            _plan_intent(),
            fake_structured_message(plan),
            _tool_call_msg("fetch", "f1"),
            _tool_call_msg("fetch", "f2"),
            fake_text_message("整合回答"),
        ],
        cfg=_serial_cfg(),
        tools=[make_fake_tool("fetch", content="数据")],
    )
    response, values = await _finish(graph, "查询甲乙数据")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    statuses = [e.status for e in response.status_trace]
    assert Status.DELEGATING not in statuses
    assert any(
        e.status == Status.USING_TOOL and e.tool_name and "第 1/2 步" in e.tool_name
        for e in response.status_trace
    )
    assert any(
        e.status == Status.USING_TOOL and e.tool_name and "第 2/2 步" in e.tool_name
        for e in response.status_trace
    )
    # 零写入：无扇出、无结果合成（禁用时逐字节回到 v4.0 顺序路径）。
    assert values.get("subagent_results") == []
    assert values.get("dispatch_round") == 0
    assert values.get("plan_steps_completed") == []
    assert values.get("plan_step") == 2


async def test_max_parallel_one_serial_zero_regression() -> None:
    """max_parallel=1 → 与禁用等价（无并行收益即走串行，零回归）。"""
    cfg = AgentFrameworkConfig(subagents=SubagentConfig(enabled=True, max_parallel=1))
    plan = PlanResult(
        summary="两步",
        steps=[
            PlanStep(goal="查询甲数据", tool="fetch"),
            PlanStep(goal="查询乙数据", tool="fetch"),
        ],
    )
    graph = _build_graph(
        [],
        [
            _plan_intent(),
            fake_structured_message(plan),
            _tool_call_msg("fetch", "f1"),
            _tool_call_msg("fetch", "f2"),
            fake_text_message("整合回答"),
        ],
        cfg=cfg,
        tools=[make_fake_tool("fetch", content="数据")],
    )
    response, values = await _finish(graph, "查询甲乙数据")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert Status.DELEGATING not in [e.status for e in response.status_trace]
    assert values.get("subagent_results") == []


async def test_plan_budget_counts_subagent_tool_calls() -> None:
    """plan 预算按全部工具调用记录数计：预算 1 → 扇出截为 1 个子任务 → 整合（partial）。"""
    cfg = AgentFrameworkConfig(plan=PlanConfig(max_tool_calls_per_plan=1))
    plan = PlanResult(
        summary="两路查询",
        steps=[
            PlanStep(goal="查询甲数据", tool="fetch"),
            PlanStep(goal="查询乙数据", tool="fetch"),
        ],
    )
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (_M_SUB + "查询甲数据", [_tool_call_msg("fetch", "f1"), fake_text_message("甲数据")]),
            (_M_SUB + "查询乙数据", [_tool_call_msg("fetch", "f2"), fake_text_message("乙数据")]),
        ],
        [_plan_intent(), fake_text_message("预算内整合回答")],
        cfg=cfg,
        tools=[make_fake_tool("fetch", content="数据")],
    )
    response, values = await _finish(graph, "两路查询")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_PARTIAL
    # 预算余额 1 → 批内只扇出 1 个子任务；join 后预算耗尽 → 整合收尾。
    assert len(response.tool_trace) == 1
    assert len(values.get("subagent_results") or []) == 1
    assert values.get("plan_step") == 1  # 串行指针同步到首个未完成步


async def test_serial_then_parallel_mixed_mode() -> None:
    """串行首步完成后依赖步就绪（≥2）→ 第二轮扇出：混合模式与上游结果传递。"""
    plan = PlanResult(
        summary="先查基准再并行分析",
        steps=[
            PlanStep(goal="查询基准数据", tool="fetch"),
            PlanStep(goal="分析甲维度", tool="calc", depends_on=[0]),
            PlanStep(goal="分析乙维度", tool="calc", depends_on=[0]),
        ],
    )
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (_M_SERIAL + " 1/3 步", [_tool_call_msg("fetch", "f1")]),
            (
                _M_SUB + "分析甲维度",
                [_tool_call_msg("calc", "c1"), fake_text_message("甲分析完成")],
            ),
            (
                _M_SUB + "分析乙维度",
                [_tool_call_msg("calc", "c2"), fake_text_message("乙分析完成")],
            ),
        ],
        [_plan_intent(), fake_text_message("最终整合回答")],
        tools=[
            make_fake_tool("fetch", content="基准数据"),
            make_fake_tool("calc", content="分析ok"),
        ],
    )
    response, values = await _finish(graph, "查基准并分析甲乙")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_COMPLETED
    assert Status.DELEGATING in [e.status for e in response.status_trace]
    # 串行 1 步 + 并行 2 步全部完成；并行批为第二轮（串行步不扇出）。
    assert values.get("dispatch_round") == 1
    assert values.get("plan_steps_done") == 3
    assert values.get("plan_step") == 3
    results = values.get("subagent_results") or []
    assert {r.step_index for r in results} == {0, 1, 2}
    # 串行步合成结果：摘要取本步工具输出（后续并行批的 upstream 来源）。
    serial = next(r for r in results if r.step_index == 0)
    assert serial.ok is True
    assert serial.summary == "基准数据"
    assert serial.tool_summary == "fetch(ok)"


async def test_subagent_state_reset_across_turns() -> None:
    """load_context 每轮重置：并行轮后接普通对话轮，子代理状态清零。"""
    plan = PlanResult(
        summary="两路查询",
        steps=[
            PlanStep(goal="查询甲数据", tool="fetch"),
            PlanStep(goal="查询乙数据", tool="fetch"),
        ],
    )
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (_M_SUB + "查询甲数据", [_tool_call_msg("fetch", "f1"), fake_text_message("甲数据")]),
            (_M_SUB + "查询乙数据", [_tool_call_msg("fetch", "f2"), fake_text_message("乙数据")]),
        ],
        [
            _plan_intent(),
            fake_text_message("第一轮整合回答"),
            _chat_intent(),
            fake_text_message("不客气"),
        ],
        tools=[make_fake_tool("fetch", content="数据")],
    )
    response1, values1 = await _finish(graph, "两路查询")
    assert response1 is not None
    assert len(values1.get("subagent_results") or []) == 2

    response2, values2 = await _finish(graph, "谢谢")
    assert response2 is not None
    assert response2.reply == "不客气"
    # 第二轮（CHAT 意图）子代理记账状态归零；subagent_results 为 operator.add 通道
    # 跨轮累积（与 tool_calls 同口径），计划级隔离由 base 偏移承担（见 plan_task）。
    assert values2.get("dispatch_round") == 0
    assert values2.get("plan_steps_completed") == []
    assert len(values2.get("subagent_results") or []) == 2


async def test_subagent_result_model_in_state() -> None:
    """子代理结果经 operator.add 回填主图状态（模型契约在真实图中成立）。"""
    plan = PlanResult(
        summary="两路查询",
        steps=[
            PlanStep(goal="查询甲数据", tool="fetch"),
            PlanStep(goal="查询乙数据", tool="fetch"),
        ],
    )
    graph = _build_graph(
        [
            (_M_PLAN, [fake_structured_message(plan)]),
            (_M_SUB + "查询甲数据", [_tool_call_msg("fetch", "f1"), fake_text_message("甲数据")]),
            (_M_SUB + "查询乙数据", [_tool_call_msg("fetch", "f2"), fake_text_message("乙数据")]),
        ],
        [_plan_intent(), fake_text_message("整合回答")],
        tools=[make_fake_tool("fetch", content="数据")],
    )
    _, values = await _finish(graph, "两路查询")

    results = values.get("subagent_results") or []
    assert all(isinstance(r, SubagentResult) for r in results)
    assert all(r.ok for r in results)
    assert all(r.summary for r in results)
    assert all(r.error is None for r in results)
