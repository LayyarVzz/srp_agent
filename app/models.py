"""app 层 DTO：HTTP 边界结构体。

`phase` 是本层对 `AgentResponse` 的**契约适配**（api.md §6.2）：把 Agent 内部语义
`finished_reason`（6 值）收敛为前端唯一分支依据 `phase`（3 个互斥值）——一码一行为、
数量最小；`finished_reason` 本身仍随响应下发（日志/调试/可选弱提示用）。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from agent.response.models import (
    FINISHED_REASON_ERROR,
    FINISHED_REASON_NEEDS_CLARIFICATION,
    AgentResponse,
    Clarification,
)
from agent.response.status import StatusEvent
from agent.session.models import SessionContext
from agent.share.models import Citation
from agent.tools.models import ToolCallRecord


class Phase(StrEnum):
    """决策层状态码：前端唯一分支依据（互斥，每码对应唯一 UI 行为）。"""

    ANSWER = "answer"  # 播报 answer
    CLARIFY = "clarify"  # 反问：渲染 question + options，不播报
    ERROR = "error"  # 异常收尾：播报 answer + 错误弱提示


def to_phase(finished_reason: str) -> Phase:
    """finished_reason（Agent 语义）→ phase（前端契约）。

    WHY 收敛而非透传：completed / tool_limit / partial / fallback 在 UI 上都是
    「播报」，差异降级为 answer 分支内的可选提示；needs_clarification 与 error
    各自独立成码（反问不播报、异常弱提示）。新增 finished_reason 时先问
    「UI 行为是否不同」。
    """
    if finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION:
        return Phase.CLARIFY
    if finished_reason == FINISHED_REASON_ERROR:
        return Phase.ERROR
    return Phase.ANSWER


class TextInteractionRequest(BaseModel):
    """文本交互请求。`session_id` 缺省时服务端自动创建（响应带回）。"""

    session_id: str | None = Field(default=None, description="会话 id；缺省自动创建")
    text: str = Field(min_length=1, max_length=8000, description="用户消息（对齐 max_input_chars）")


class InteractionResult(BaseModel):
    """交互响应（HTTP 200 体 / SSE done 载荷）。

    字段 = AgentResponse 全量（reply → answer 字段名映射）+ 前端契约 phase
    + ok / user_id / source / input_text。`from_response` 是本层唯一的契约适配点。
    """

    ok: bool = True
    phase: Phase
    session_id: str
    user_id: str
    source: Literal["text", "voice"]
    input_text: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    status_trace: list[StatusEvent] = Field(default_factory=list)
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    finished_reason: str = "completed"
    clarification: Clarification | None = None

    @classmethod
    def from_response(
        cls,
        resp: AgentResponse,
        *,
        user_id: str,
        source: Literal["text", "voice"],
        input_text: str,
    ) -> InteractionResult:
        """由 AgentResponse 派生：追加 phase / ok / user_id / source / input_text。"""
        return cls(
            phase=to_phase(resp.finished_reason),
            session_id=resp.session_id,
            user_id=user_id,
            source=source,
            input_text=input_text,
            answer=resp.reply,
            citations=resp.citations,
            status_trace=resp.status_trace,
            tool_trace=resp.tool_trace,
            finished_reason=resp.finished_reason,
            clarification=resp.clarification,
        )


class SessionCreated(BaseModel):
    """创建会话响应：SessionContext + ok 信封。"""

    ok: bool = True
    session_id: str
    user_id: str
    created_at: datetime
    meta: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_context(cls, ctx: SessionContext) -> SessionCreated:
        return cls(**ctx.model_dump())


class SessionListResponse(BaseModel):
    """会话列表响应：按 created_at 降序。"""

    ok: bool = True
    sessions: list[SessionContext] = Field(default_factory=list)
