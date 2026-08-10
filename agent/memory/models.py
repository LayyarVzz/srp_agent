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
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryRecallResult(BaseModel):
    """记忆召回结果：记忆条目 + 来源引用。"""

    items: list[MemoryItem] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)

class MemoryExtraction(BaseModel):
    """一次结构化抽取结果（来自 LLM，可独立理解）。

    `kind` 为宽松 str（不约束 Literal）：
    强约束时模型偶发输出超纲值会导致整批校验失败被丢弃（降级保险，见 kind 策略）。
    `importance` 由模型给出，作为 v2.0 非语义召回的确定性排序键。
    """
    kind: str  # fact / episode / preference（召回端未知值归入 other 组）
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryExtractionResult(BaseModel):
    """整批抽取结果；没有值得记住的内容时为空列表。"""

    memories: list[MemoryExtraction] = Field(default_factory=list)
