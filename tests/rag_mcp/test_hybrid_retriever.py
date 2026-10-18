"""Deterministic tests for BM25/vector hybrid retrieval."""

from __future__ import annotations

import logging
import traceback
from typing import Any

import faiss
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

import services.rag_mcp.rag.retriever as retriever_module
from services.rag_mcp.rag.fusion import reciprocal_rank_fusion
from services.rag_mcp.rag.index import FaissIndexBundle, FaissIndexManifest
from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.rag.retriever import (
    BM25Retriever,
    ChunkConsistencyError,
    HybridRetrievalError,
    HybridRetriever,
    VectorRetriever,
)
from services.rag_mcp.schemas import KnowledgeSource

_SYNTHETIC_SECRET = "SYNTHETIC_PROVIDER_SECRET_DO_NOT_LOG"


class _SpyRetriever:
    def __init__(
        self,
        results: list[RetrievedChunk] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return list(self.results)


class _FakeEmbeddings(Embeddings):
    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("Hybrid retrieval must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.query_vector


def _chunk(
    chunk_id: str,
    *,
    content: str | None = None,
    source: KnowledgeSource | None = None,
    metadata: dict[str, object] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content or f"Content for {chunk_id}",
        source=source or KnowledgeSource(id="policy", title="Employee policy"),
        metadata=metadata if metadata is not None else {"chunk_id": chunk_id},
    )


def _hit(
    chunk_id: str,
    score: float,
    *,
    chunk: DocumentChunk | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk or _chunk(chunk_id), score=score)


def _manifest(*, chunk_count: int, dimensions: int = 2) -> FaissIndexManifest:
    return FaissIndexManifest(
        schema_version=1,
        index_type="IndexFlatIP",
        metric="inner_product",
        normalized=True,
        embedding_provider="openai_compatible",
        embedding_model="fake-embedding",
        embedding_dimensions=dimensions,
        chunk_count=chunk_count,
        index_sha256="0" * 64,
        chunks_sha256="1" * 64,
    )


def _hybrid(
    bm25: _SpyRetriever | None = None,
    vector: _SpyRetriever | None = None,
    **kwargs: Any,
) -> tuple[HybridRetriever, _SpyRetriever, _SpyRetriever]:
    bm25_spy = bm25 or _SpyRetriever()
    vector_spy = vector or _SpyRetriever()
    hybrid = HybridRetriever(
        bm25=bm25_spy,  # type: ignore[arg-type]
        vector=vector_spy,  # type: ignore[arg-type]
        **kwargs,
    )
    return hybrid, bm25_spy, vector_spy


def test_valid_query_calls_both_channels_once_and_fuses_candidates() -> None:
    bm25 = _SpyRetriever([_hit("lexical", 2.0), _hit("shared", 5.0)])
    vector = _SpyRetriever([_hit("vector", 0.7), _hit("shared", 0.9)])
    hybrid, _, _ = _hybrid(bm25, vector, candidate_multiplier=2)

    results = hybrid.search("  leave policy  ", top_k=3)

    assert bm25.calls == [("leave policy", 6)]
    assert vector.calls == [("leave policy", 6)]
    assert [result.chunk.chunk_id for result in results] == ["shared", "lexical", "vector"]
    shared = results[0]
    assert shared.score == pytest.approx(1.0)
    assert shared.score_details.bm25 == 5.0
    assert shared.score_details.vector == 0.9
    assert shared.score_details.bm25_rank == 1
    assert shared.score_details.vector_rank == 1
    assert shared.score_details.rrf == pytest.approx(2.0 / 61.0)


def test_candidate_k_is_capped_by_max_candidates() -> None:
    hybrid, bm25, vector = _hybrid(candidate_multiplier=3, max_candidates=10)

    assert hybrid.search("policy", top_k=4) == []
    assert bm25.calls == [("policy", 10)]
    assert vector.calls == [("policy", 10)]


def test_top_k_equal_to_max_candidates_is_allowed() -> None:
    hybrid, bm25, vector = _hybrid(candidate_multiplier=3, max_candidates=5)

    assert hybrid.search("policy", top_k=5) == []
    assert bm25.calls == [("policy", 5)]
    assert vector.calls == [("policy", 5)]


def test_top_k_above_max_candidates_is_rejected_before_search() -> None:
    hybrid, bm25, vector = _hybrid(max_candidates=5)

    with pytest.raises(ValueError, match="must not exceed max_candidates"):
        hybrid.search("policy", top_k=6)

    assert bm25.calls == []
    assert vector.calls == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_without_search(top_k: int) -> None:
    hybrid, bm25, vector = _hybrid()

    assert hybrid.search("policy", top_k=top_k) == []
    assert bm25.calls == []
    assert vector.calls == []


@pytest.mark.parametrize("top_k", [True, 1.5, "5"])
def test_non_integer_top_k_is_rejected(top_k: object) -> None:
    hybrid, bm25, vector = _hybrid()

    with pytest.raises(ValueError, match="top_k must be an integer"):
        hybrid.search("policy", top_k=top_k)  # type: ignore[arg-type]

    assert bm25.calls == []
    assert vector.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\t\r\n"])
def test_empty_query_does_not_call_either_channel(query: str) -> None:
    hybrid, bm25, vector = _hybrid()

    assert hybrid.search(query) == []
    assert bm25.calls == []
    assert vector.calls == []


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("candidate_multiplier", 0),
        ("candidate_multiplier", True),
        ("max_candidates", -1),
        ("max_candidates", False),
        ("rrf_k", 0),
        ("rrf_k", 1.5),
    ],
)
def test_constructor_rejects_invalid_positive_integer_parameters(
    parameter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        _hybrid(**{parameter: value})


def test_bm25_zero_and_negative_scores_are_filtered_before_rrf() -> None:
    bm25 = _SpyRetriever(
        [
            _hit("positive", 0.5),
            _hit("zero", 0.0),
            _hit("negative", -1.0),
        ]
    )
    hybrid, _, _ = _hybrid(bm25=bm25)

    results = hybrid.search("policy")

    assert [result.chunk.chunk_id for result in results] == ["positive"]
    assert results[0].score_details.bm25 == 0.5
    assert results[0].score_details.bm25_rank == 1
    assert results[0].score_details.vector is None


@pytest.mark.parametrize("active_channel", ["bm25", "vector"])
def test_one_normal_empty_channel_allows_other_channel_results(active_channel: str) -> None:
    hit = _hit("only", 2.0 if active_channel == "bm25" else 0.8)
    bm25 = _SpyRetriever([hit] if active_channel == "bm25" else [])
    vector = _SpyRetriever([hit] if active_channel == "vector" else [])
    hybrid, _, _ = _hybrid(bm25, vector)

    results = hybrid.search("policy")

    assert [result.chunk.chunk_id for result in results] == ["only"]
    assert results[0].score == pytest.approx(0.5)
    assert results[0].score_details.bm25_rank == (1 if active_channel == "bm25" else None)
    assert results[0].score_details.vector_rank == (1 if active_channel == "vector" else None)


def test_two_normal_empty_channels_return_empty_list() -> None:
    hybrid, bm25, vector = _hybrid()

    assert hybrid.search("policy") == []
    assert len(bm25.calls) == 1
    assert len(vector.calls) == 1


@pytest.mark.parametrize("failed_channel", ["bm25", "vector"])
def test_single_channel_failure_logs_and_degrades(
    failed_channel: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError(f"{failed_channel} unavailable: {_SYNTHETIC_SECRET}")
    healthy_hit = _hit("healthy", 2.0 if failed_channel == "vector" else 0.8)
    bm25 = _SpyRetriever(
        [healthy_hit] if failed_channel == "vector" else [],
        error=failure if failed_channel == "bm25" else None,
    )
    vector = _SpyRetriever(
        [healthy_hit] if failed_channel == "bm25" else [],
        error=failure if failed_channel == "vector" else None,
    )
    hybrid, _, _ = _hybrid(bm25, vector)

    with caplog.at_level(logging.WARNING, logger=retriever_module.__name__):
        results = hybrid.search("policy")

    assert [result.chunk.chunk_id for result in results] == ["healthy"]
    details = results[0].score_details
    assert details.bm25_rank == (None if failed_channel == "bm25" else 1)
    assert details.vector_rank == (None if failed_channel == "vector" else 1)
    assert f"channel={failed_channel}" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert _SYNTHETIC_SECRET not in caplog.text
    assert "Traceback" not in caplog.text


def test_both_channel_failures_raise_safe_diagnostic_hybrid_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bm25_error = RuntimeError(f"bm25 unavailable: {_SYNTHETIC_SECRET}")
    vector_error = ConnectionError(f"vector unavailable: {_SYNTHETIC_SECRET}")
    hybrid, bm25, vector = _hybrid(
        _SpyRetriever(error=bm25_error),
        _SpyRetriever(error=vector_error),
    )

    with caplog.at_level(logging.WARNING, logger=retriever_module.__name__):
        with pytest.raises(
            HybridRetrievalError,
            match="Both hybrid retrieval channels failed",
        ) as exc:
            hybrid.search("policy")

    assert exc.value.channel_errors == {"bm25": bm25_error, "vector": vector_error}
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert exc.value.__suppress_context__ is True
    assert "bm25=RuntimeError" in str(exc.value)
    assert "vector=ConnectionError" in str(exc.value)
    assert _SYNTHETIC_SECRET not in str(exc.value)
    assert _SYNTHETIC_SECRET not in "".join(traceback.format_exception(exc.value))
    assert _SYNTHETIC_SECRET not in caplog.text
    assert "Traceback" not in caplog.text
    assert len(bm25.calls) == 1
    assert len(vector.calls) == 1


@pytest.mark.parametrize(
    ("field", "vector_chunk"),
    [
        ("content", _chunk("shared", content="Different content")),
        (
            "source",
            _chunk(
                "shared",
                source=KnowledgeSource(id="other", title="Other source"),
            ),
        ),
        ("metadata", _chunk("shared", metadata={"version": 2})),
    ],
)
def test_shared_chunk_payload_mismatch_is_rejected(
    field: str,
    vector_chunk: DocumentChunk,
) -> None:
    bm25 = _SpyRetriever([_hit("shared", 3.0)])
    vector = _SpyRetriever([_hit("shared", 0.8, chunk=vector_chunk)])
    hybrid, _, _ = _hybrid(bm25, vector)

    with pytest.raises(ChunkConsistencyError, match=field):
        hybrid.search("policy")


def test_real_retriever_collections_are_checked_once_at_initialization() -> None:
    bm25_chunk = _chunk("shared", content="BM25 content")
    vector_chunk = _chunk("shared", content="Vector content")
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    vector = VectorRetriever(
        bundle=FaissIndexBundle(
            index=index,
            chunks=[vector_chunk],
            manifest=_manifest(chunk_count=1),
        ),
        embedder=_FakeEmbeddings([1.0, 0.0]),
    )

    with pytest.raises(ChunkConsistencyError, match="content"):
        HybridRetriever(bm25=BM25Retriever([bm25_chunk]), vector=vector)


def test_real_retriever_collections_require_identical_chunk_ids() -> None:
    bm25_chunk = _chunk("bm25-only")
    vector_chunk = _chunk("vector-only")
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    vector = VectorRetriever(
        bundle=FaissIndexBundle(
            index=index,
            chunks=[vector_chunk],
            manifest=_manifest(chunk_count=1),
        ),
        embedder=_FakeEmbeddings([1.0, 0.0]),
    )

    with pytest.raises(ChunkConsistencyError, match="different chunk_id"):
        HybridRetriever(bm25=BM25Retriever([bm25_chunk]), vector=vector)


def test_hybrid_calls_b2_rrf_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[RetrievedChunk], list[RetrievedChunk], int, int]] = []

    def _rrf_spy(
        bm25_results: list[RetrievedChunk],
        vector_results: list[RetrievedChunk],
        *,
        k: int,
        top_k: int,
    ) -> list[RetrievedChunk]:
        calls.append((bm25_results, vector_results, k, top_k))
        return reciprocal_rank_fusion(bm25_results, vector_results, k=k, top_k=top_k)

    monkeypatch.setattr(retriever_module, "reciprocal_rank_fusion", _rrf_spy)
    hybrid, _, _ = _hybrid(
        _SpyRetriever([_hit("bm25", 2.0)]),
        _SpyRetriever([_hit("vector", 0.8)]),
        rrf_k=37,
    )

    results = hybrid.search("policy", top_k=2)

    assert len(calls) == 1
    assert calls[0][2:] == (37, 2)
    assert {result.chunk.chunk_id for result in results} == {"bm25", "vector"}


def test_real_bm25_and_faiss_index_flat_ip_are_both_used() -> None:
    chunks = [
        _chunk("policy:0", content="Vacation entitlement days"),
        _chunk("policy:1", content="Medical reimbursement rules"),
        _chunk("policy:2", content="Travel expense approval"),
    ]
    index = faiss.IndexFlatIP(2)
    diagonal = 1.0 / np.sqrt(2.0)
    index.add(
        np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [diagonal, diagonal]],
            dtype=np.float32,
        )
    )
    embedder = _FakeEmbeddings([0.0, 1.0])
    vector = VectorRetriever(
        bundle=FaissIndexBundle(index=index, chunks=chunks, manifest=_manifest(chunk_count=3)),
        embedder=embedder,
    )
    hybrid = HybridRetriever(bm25=BM25Retriever(chunks), vector=vector)

    results = hybrid.search("vacation", top_k=2)

    by_id = {result.chunk.chunk_id: result for result in results}
    assert set(by_id) == {"policy:0", "policy:1"}
    assert by_id["policy:0"].score_details.bm25 is not None
    assert by_id["policy:0"].score_details.vector is None
    assert by_id["policy:1"].score_details.bm25 is None
    assert by_id["policy:1"].score_details.vector == pytest.approx(1.0)
    assert embedder.query_calls == ["vacation"]
