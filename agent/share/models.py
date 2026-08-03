"""Agent 模块公共数据模型（跨子模块共享）。

WHY 独立成模块：`Citation` 被 tools（``ToolResult.citations``）、memory
（``MemoryRecallResult.sources``）、response（``AgentResponse.citations``）三方引用，
若放任意一个子模块都会与其它子模块形成循环导入。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """来源引用：回答中引用的任何来源必须存在于本次检索结果集内（护栏校验）。"""

    source_id: str
    source_title: str
    snippet: str
    score: float | None = Field(default=None, description="相关性得分（如有）")
    url: str | None = None
