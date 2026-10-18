"""Deterministic tests for the RAG MCP retrieval data contracts."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.schemas import (
    KnowledgeChunk,
    KnowledgeSource,
    RetrievalScoreDetails,
    SearchKnowledgeResponse,
)


@pytest.fixture
def source() -> KnowledgeSource:
    return KnowledgeSource(id="policy", title="Employee policy")


@pytest.fixture
def document_chunk(source: KnowledgeSource) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="policy:0",
        content="Annual leave policy",
        source=source,
        metadata={"chunk_index": 0},
    )


def test_legacy_retrieved_chunk_construction_remains_valid(
    document_chunk: DocumentChunk,
) -> None:
    retrieved = RetrievedChunk(chunk=document_chunk, score=1.25)

    assert retrieved.chunk == document_chunk
    assert retrieved.score == 1.25
    assert retrieved.score_details is None


def test_retrieved_chunk_with_score_details_serializes(
    document_chunk: DocumentChunk,
) -> None:
    retrieved = RetrievedChunk(
        chunk=document_chunk,
        score=0.75,
        score_details=RetrievalScoreDetails(
            bm25=2.5,
            vector=0.8,
            rrf=0.03,
            bm25_rank=1,
            vector_rank=2,
        ),
    )

    payload = retrieved.model_dump(mode="json")
    assert payload["score_details"] == {
        "bm25": 2.5,
        "vector": 0.8,
        "rrf": 0.03,
        "bm25_rank": 1,
        "vector_rank": 2,
    }


def test_legacy_knowledge_chunk_construction_remains_valid(source: KnowledgeSource) -> None:
    chunk = KnowledgeChunk(content="Policy", source=source, score=1.0)

    assert chunk.chunk_id is None
    assert chunk.score_details is None
    assert chunk.metadata == {}


def test_knowledge_chunk_serializes_chunk_id(source: KnowledgeSource) -> None:
    chunk = KnowledgeChunk(
        chunk_id="policy:3",
        content="Policy",
        source=source,
        score=0.9,
    )

    assert chunk.model_dump(mode="json")["chunk_id"] == "policy:3"


def test_knowledge_chunk_serializes_score_details(source: KnowledgeSource) -> None:
    chunk = KnowledgeChunk(
        content="Policy",
        source=source,
        score=0.5,
        score_details=RetrievalScoreDetails(vector=0.72, vector_rank=1),
    )

    details = chunk.model_dump(mode="json")["score_details"]
    assert details == {
        "bm25": None,
        "vector": 0.72,
        "rrf": None,
        "bm25_rank": None,
        "vector_rank": 1,
    }


def test_search_response_top_level_structure_is_unchanged(source: KnowledgeSource) -> None:
    response = SearchKnowledgeResponse(
        query="annual leave",
        chunks=[KnowledgeChunk(content="Policy", source=source, score=0.5)],
    )

    payload = response.model_dump(mode="json")
    assert set(payload) == {"query", "chunks"}
    assert payload["query"] == "annual leave"
    assert isinstance(payload["chunks"], list)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bm25", "not-a-score"),
        ("vector", math.inf),
        ("rrf", math.nan),
        ("bm25_rank", 0),
        ("vector_rank", -1),
    ],
)
def test_invalid_score_detail_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RetrievalScoreDetails(**{field: value})
