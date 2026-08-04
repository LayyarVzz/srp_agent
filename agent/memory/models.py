"""记忆数据模型（P1 类型级预留，无实现）。

每条 `MemoryItem` 必须携带 kind / session_id / user_id / timestamp / provenance；
召回结果必须带来源。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agent.share.models import Citation


class MemoryItem(BaseModel):
    """一条长期记忆（事实 / 片段 / 偏好…），必须携带会话与用户作用域。"""

    id: str
    kind: str  # fact / episode / preference / ...
    content: str
    session_id: str
    user_id: str
    timestamp: datetime
    provenance: str  # 来源（会话、工具等）


class MemoryRecallResult(BaseModel):
    """记忆召回结果：记忆条目 + 来源引用。"""

    items: list[MemoryItem] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
