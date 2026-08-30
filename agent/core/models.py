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
    depends_on: list[int] = Field(default_factory=list)  # 本步需要的上游产出所对应的步骤索引（0-based，空 = 无依赖）
    expected_output: str | None = None  # 步骤产出描述（供整合引用）


class PlanResult(BaseModel):
    """一次规划的结构化输出：整体概述 + 有序步骤列表。"""

    summary: str = ""  # 整体计划概述（供 StatusEvent 展示）
    steps: list[PlanStep] = Field(default_factory=list)  # 1..max_plan_steps（校验在规划节点）
