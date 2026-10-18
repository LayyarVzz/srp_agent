"""Internal RAG data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.rag_mcp.schemas import KnowledgeSource, RetrievalScoreDetails


class DocumentChunk(BaseModel):
    """A source-aware knowledge chunk used inside the RAG pipeline."""

    chunk_id: str
    content: str
    source: KnowledgeSource
    metadata: dict[str, object] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its retriever-specific primary score.

    For BM25 retrieval, ``score`` is the raw BM25 score. For vector retrieval,
    it is the inner-product similarity of L2-normalized vectors. For hybrid
    retrieval, it is a normalized RRF ranking score; normalized RRF is neither
    a probability nor a calibrated semantic-relevance score.
    """

    chunk: DocumentChunk
    score: float
    score_details: RetrievalScoreDetails | None = None
