from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TextInteractionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000, description="用户输入文本")
    session_id: str | None = None


class SessionContext(BaseModel):
    ok: bool = True
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    created_at: str
    meta: dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    ok: bool = True
    sessions: list[SessionContext]


class Clarification(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)


class StatusEvent(BaseModel):
    status: str
    tool_name: str | None = None
    message: str | None = None


class InteractionResponse(BaseModel):
    ok: bool = True
    phase: Literal["answer", "clarify", "error"] = "answer"
    session_id: str
    user_id: str
    source: Literal["text", "voice"]
    input_text: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    status_trace: list[StatusEvent] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    finished_reason: str = "completed"
    clarification: Clarification | None = None


class RecentLogsResponse(BaseModel):
    ok: bool = True
    logs: list[dict[str, Any]]
