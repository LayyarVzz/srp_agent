"""RAG MCP 返回数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeSource(BaseModel):
    """知识片段来源。"""

    id: str
    title: str
    url: str | None = None


class KnowledgeChunk(BaseModel):
    """一次检索命中的知识片段。"""

    content: str
    source: KnowledgeSource
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float


class SearchKnowledgeResponse(BaseModel):
    """知识检索响应。"""

    query: str
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
