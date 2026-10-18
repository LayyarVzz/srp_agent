"""RAG MCP 返回数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field, FiniteFloat, PositiveInt


class KnowledgeSource(BaseModel):
    """知识片段来源。"""

    id: str
    title: str
    url: str | None = None


class RetrievalScoreDetails(BaseModel):
    """Retriever-specific scores and ranks for one knowledge hit.

    ``bm25`` is the raw BM25 score, ``vector`` is the inner-product similarity
    of L2-normalized vectors, and ``rrf`` is the raw reciprocal-rank-fusion
    score. Ranks are one-based. These optional diagnostics do not change the
    meaning of the enclosing result's primary ``score`` field.
    """

    bm25: FiniteFloat | None = None
    vector: FiniteFloat | None = None
    rrf: FiniteFloat | None = None
    bm25_rank: PositiveInt | None = None
    vector_rank: PositiveInt | None = None


class KnowledgeChunk(BaseModel):
    """一次检索命中的知识片段。"""

    content: str
    source: KnowledgeSource
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float
    chunk_id: str | None = None
    score_details: RetrievalScoreDetails | None = None


class SearchKnowledgeResponse(BaseModel):
    """知识检索响应。"""

    query: str
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
