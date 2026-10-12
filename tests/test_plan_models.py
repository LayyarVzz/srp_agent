"""agent/core/models —— PlanResult/PlanStep 模型与 validate_plan_result 校验单测（v4.0 T1）。

规划校验是「确定性护栏」：步骤数上限 / depends_on 拓扑序 / 工具名在注册表，
非法计划由规划节点把 plan 置 None → 回退 ReAct（零回归），故必须纯函数可测。
"""

from __future__ import annotations

from agent.core.graph import validate_plan_result
from agent.core.models import PlanResult, PlanStep


def test_plan_step_defaults() -> None:
    """PlanStep 可选字段默认：tool=None / depends_on=[] / expected_output=None。"""
    step = PlanStep(goal="查资料")
    assert step.tool is None
    assert step.depends_on == []
    assert step.expected_output is None


def test_plan_result_defaults() -> None:
    """PlanResult 默认：summary 空串 / steps 空列表（非法态，由校验拒绝）。"""
    plan = PlanResult()
    assert plan.summary == ""
    assert plan.steps == []


def test_validate_accepts_valid_plan() -> None:
    """合法计划：工具名在注册表、depends_on 只引用更早索引。"""
    plan = PlanResult(
        summary="两步计划",
        steps=[
            PlanStep(goal="检索", tool="search", depends_on=[], expected_output="片段"),
            PlanStep(goal="总结", tool=None, depends_on=[0]),
        ],
    )
    assert validate_plan_result(plan, known_tools=["search"], max_steps=6)


def test_validate_rejects_empty_steps() -> None:
    """空步骤列表 → 非法（1..max_plan_steps 的下界）。"""
    assert not validate_plan_result(PlanResult(), known_tools=[], max_steps=6)


def test_validate_rejects_too_many_steps() -> None:
    """步骤数超 max_steps → 非法（主收敛约束）。"""
    plan = PlanResult(steps=[PlanStep(goal=f"步{i}") for i in range(3)])
    assert not validate_plan_result(plan, known_tools=[], max_steps=2)


def test_validate_rejects_unknown_tool() -> None:
    """步骤期望工具不在注册表（模型幻觉）→ 非法。"""
    plan = PlanResult(steps=[PlanStep(goal="查资料", tool="ghost_tool")])
    assert not validate_plan_result(plan, known_tools=["search"], max_steps=6)


def test_validate_accepts_none_tool() -> None:
    """tool=None 的纯 LLM 变换步合法（总结/翻译无需外部工具）。"""
    plan = PlanResult(steps=[PlanStep(goal="翻译成英文")])
    assert validate_plan_result(plan, known_tools=[], max_steps=6)


def test_validate_rejects_forward_dependency() -> None:
    """depends_on 引用更晚步骤 → 非法（破坏串行拓扑序）。"""
    plan = PlanResult(steps=[PlanStep(goal="a", depends_on=[1]), PlanStep(goal="b")])
    assert not validate_plan_result(plan, known_tools=[], max_steps=6)


def test_validate_rejects_self_dependency() -> None:
    """depends_on 引用自身 → 非法。"""
    plan = PlanResult(steps=[PlanStep(goal="a", depends_on=[0])])
    assert not validate_plan_result(plan, known_tools=[], max_steps=6)


def test_validate_rejects_negative_dependency() -> None:
    """depends_on 负数索引 → 非法。"""
    plan = PlanResult(steps=[PlanStep(goal="a", depends_on=[-1])])
    assert not validate_plan_result(plan, known_tools=[], max_steps=6)
