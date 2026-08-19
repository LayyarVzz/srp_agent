"""Internal RAG data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.rag_mcp.schemas import KnowledgeSource


class DocumentChunk(BaseModel):
    """A source-aware knowledge chunk used inside the RAG pipeline."""

    chunk_id: str
    content: str
    source: KnowledgeSource
    metadata: dict[str, object] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its retriever score."""

    chunk: DocumentChunk
    score: float
