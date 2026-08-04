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


class AgentResponse(BaseModel):
    """Agent 统一响应：回答 + 引用 + 状态轨迹 + 工具轨迹 + 结束原因。"""

    session_id: str
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    status_trace: list[StatusEvent] = Field(default_factory=list)
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    finished_reason: str = FINISHED_REASON_COMPLETED
