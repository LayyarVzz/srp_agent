from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TextInteractionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="用户输入文本")
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None


class InteractionResponse(BaseModel):
    ok: bool = True
    session_id: str
    user_id: str | None = None
    source: Literal["text", "voice"]
    input_text: str
    answer: str


class RecentLogsResponse(BaseModel):
    ok: bool = True
    logs: list[dict[str, Any]]
