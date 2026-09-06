"""统一输出模型（AgentResponse）与 finished_reason 常量。

所有 Agent 输出必须收敛到 `AgentResponse`。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.response.status import StatusEvent
from agent.share.models import Citation
from agent.tools.models import ToolCallRecord

# —— finished_reason 常量（结束原因，供前端/调用方区分处理路径）——
FINISHED_REASON_COMPLETED = "completed"  # 正常完成
FINISHED_REASON_TOOL_LIMIT = "tool_limit"  # 工具迭代达上限
FINISHED_REASON_FALLBACK = "fallback"  # 走了降级/兜底路径
FINISHED_REASON_ERROR = "error"  # 出错降级
FINISHED_REASON_PARTIAL = "partial"  # 部分成功：多步任务有 ≥1 步产出、其余失败/中断
FINISHED_REASON_NEEDS_CLARIFICATION = "needs_clarification"  # 澄清追问：等待用户补充/选择


class ClarifyResult(BaseModel):
    """clarify 节点的 LLM 结构化输出。

    与 `Clarification` 同构但独立声明：前者是「模型产出契约」，后者是「响应契约」，
    各自可独立演进（如后续为产出加结构化约束，不影响前端渲染契约）。
    """

    question: str  # 反问正文
    options: list[str] = Field(default_factory=list)  # 候选选项（空 = 纯开放反问）


class Clarification(BaseModel):
    """澄清追问响应契约：随 `AgentResponse.clarification` 下发，供前端渲染选项按钮。"""

    question: str
    options: list[str] = Field(default_factory=list)


class AnswerToken(BaseModel):
    """最终回答的 token 级增量（SSE `token` 帧载荷）。

    与 `AgentResponse` 同域：正常路径（模型输出无首尾空白）下各帧 `delta`
    拼接 == `reply`；回答节点对首尾空白的确定性裁剪、以及流式中途失败后的
    降级，均以 `done`（AgentResponse.reply）为权威，token 帧仅作增量预览。
    只在回答节点（generate_answer / fallback_chat）的 LLM 流式生成期间产生，
    属过程事件，不进 AgentResponse / InteractionResult。
    """

    delta: str


class AgentResponse(BaseModel):
    """Agent 统一响应：回答 + 引用 + 状态轨迹 + 工具轨迹 + 结束原因。"""

    session_id: str
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    status_trace: list[StatusEvent] = Field(default_factory=list)
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    finished_reason: str = FINISHED_REASON_COMPLETED
    # 澄清追问：非 None 时 `finished_reason == needs_clarification`，reply 即反问正文。
    clarification: Clarification | None = None
