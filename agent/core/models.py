"""编排内核模型：多步任务规划。

`PlanResult` / `PlanStep` 是 `plan_task` / `replan_task` 的结构化 LLM 输出
（`LLMService.ainvoke_structured`），也是图状态 `plan` 字段的载体。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """计划中的一步：只做一件事的目标 + 期望工具 + 依赖关系。

    `tool` 为 None 表示纯 LLM 变换步（总结/翻译等，无需外部工具）；
    `depends_on` 为 0-based 索引列表，只允许引用更早的步骤（拓扑序由
    `validate_plan_result` 在规划期校验，执行按 `plan_step` 指针严格串行）。
    """

    goal: str  # 步骤目标（自然语言，供执行与进度展示）
    tool: str | None = None  # 期望工具名；None = 纯 LLM 变换步
    depends_on: list[int] = Field(default_factory=list)  # 上游步骤索引（0-based，空 = 无依赖）
    expected_output: str | None = None  # 步骤产出描述（供整合引用）


class PlanResult(BaseModel):
    """一次规划的结构化输出：整体概述 + 有序步骤列表。"""

    summary: str = ""  # 整体计划概述（供 StatusEvent 展示）
    steps: list[PlanStep] = Field(default_factory=list)  # 1..max_plan_steps（校验在规划节点）


class SubagentResult(BaseModel):
    """单个子代理步骤的执行结果（T7 并行子代理，dev-version5.0.md §9）。

    子图状态独立于主图，只把本结果经 `subagent_results`（operator.add）回填主图，
    按 `step_index` 定位所属计划步骤；`summary` 是子代理对本步产出的自然语言摘要
    （属不可信数据，注入整合 prompt 时须带声明头）。
    """

    step_index: int  # 所属计划步骤索引（0-based；重规划后由 plan_steps_completed 区分计划代）
    ok: bool  # 该步是否成功（子代理内工具失败 / LLM 失败 → False，触发重规划语义）
    summary: str = ""  # 本步产出摘要（供最终整合引用）
    error: str | None = None  # 失败原因（ok=False 时非空）
    tool_summary: str = ""  # 本步工具调用摘要（如 "calc(ok)"；无工具调用为空）
