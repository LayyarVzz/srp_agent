"""Deterministic VectorRetriever tests backed by a real FAISS IndexFlatIP."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from services.rag_mcp.rag.index import FaissIndexBundle, FaissIndexManifest
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.retriever import (
    FaissSearchError,
    QueryEmbeddingError,
    QueryVectorValidationError,
    VectorRetriever,
)
from services.rag_mcp.schemas import KnowledgeSource

_DIMS = 3


class _FakeEmbeddings(Embeddings):
    def __init__(
        self,
        query_vector: Any,
        *,
        error: Exception | None = None,
    ) -> None:
        self._query_vector = query_vector
        self._error = error
        self.query_calls: list[str] = []
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        raise AssertionError("VectorRetriever must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        if self._error is not None:
            raise self._error
        return self._query_vector


def _chunks() -> list[DocumentChunk]:
    source = KnowledgeSource(id="policy", title="Employee policy")
    return [
        DocumentChunk(
            chunk_id=f"policy:{index}",
            content=content,
            source=source,
            metadata={"chunk_index": index, "label": label},
        )
        for index, (content, label) in enumerate(
            [
                ("Vector X policy", "x"),
                ("Vector Y policy", "y"),
                ("Diagonal policy", "diagonal"),
            ]
        )
    ]


def _manifest() -> FaissIndexManifest:
    return FaissIndexManifest(
        schema_version=1,
        index_type="IndexFlatIP",
        metric="inner_product",
        normalized=True,
        embedding_provider="openai_compatible",
        embedding_model="fake-embedding",
        embedding_dimensions=_DIMS,
        chunk_count=3,
        index_sha256="0" * 64,
        chunks_sha256="1" * 64,
    )


def _bundle() -> FaissIndexBundle:
    index = faiss.IndexFlatIP(_DIMS)
    diagonal = 1.0 / np.sqrt(2.0)
    index.add(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [diagonal, diagonal, 0.0],
            ],
            dtype=np.float32,
        )
    )
    return FaissIndexBundle(index=index, chunks=_chunks(), manifest=_manifest())


def test_vector_search_uses_real_faiss_and_maps_complete_chunks() -> None:
    embedder = _FakeEmbeddings([10.0, 0.0, 0.0])
    retriever = VectorRetriever(bundle=_bundle(), embedder=embedder)

    results = retriever.search("  annual leave  ", top_k=3)

    assert [result.chunk.chunk_id for result in results] == ["policy:0", "policy:2"]
    assert results[0].chunk.metadata == {"chunk_index": 0, "label": "x"}
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-6)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert all(result.score_details is not None for result in results)
    assert all(result.score_details.vector == result.score for result in results)
    assert embedder.query_calls == ["annual leave"]
    assert embedder.document_calls == 0


def test_query_vector_is_l2_normalized_before_real_faiss_search() -> None:
    embedder = _FakeEmbeddings([3.0, 4.0, 0.0])
    retriever = VectorRetriever(bundle=_bundle(), embedder=embedder)

    results = retriever.search("normalized", top_k=3)

    assert results[0].chunk.chunk_id == "policy:2"
    assert results[0].score == pytest.approx(0.7 * np.sqrt(2.0), abs=1e-6)
    x_hit = next(result for result in results if result.chunk.chunk_id == "policy:0")
    y_hit = next(result for result in results if result.chunk.chunk_id == "policy:1")
    assert x_hit.score == pytest.approx(0.6, abs=1e-6)
    assert y_hit.score == pytest.approx(0.8, abs=1e-6)


def test_top_k_is_strictly_applied() -> None:
    retriever = VectorRetriever(
        bundle=_bundle(),
        embedder=_FakeEmbeddings([1.0, 0.9, 0.0]),
    )

    results = retriever.search("policy", top_k=2)

    assert len(results) == 2
    assert [result.chunk.chunk_id for result in results] == ["policy:2", "policy:0"]


def test_vector_min_score_is_strictly_greater_than_threshold() -> None:
    retriever = VectorRetriever(
        bundle=_bundle(),
        embedder=_FakeEmbeddings([1.0, 0.0, 0.0]),
        vector_min_score=1.0 / np.sqrt(2.0),
    )

    results = retriever.search("policy", top_k=3)

    assert [result.chunk.chunk_id for result in results] == ["policy:0"]


@pytest.mark.parametrize("query", ["", "   ", "\t\r\n"])
def test_empty_query_returns_without_embedding(query: str) -> None:
    embedder = _FakeEmbeddings([1.0, 0.0, 0.0])
    retriever = VectorRetriever(bundle=_bundle(), embedder=embedder)

    assert retriever.search(query, top_k=3) == []
    assert embedder.query_calls == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_without_embedding(top_k: int) -> None:
    embedder = _FakeEmbeddings([1.0, 0.0, 0.0])
    retriever = VectorRetriever(bundle=_bundle(), embedder=embedder)

    assert retriever.search("policy", top_k=top_k) == []
    assert embedder.query_calls == []


@pytest.mark.parametrize(
    ("query_vector", "message"),
    [
        ([1.0, 0.0], "dimension"),
        ([[1.0, 0.0, 0.0]], "1D vector"),
        ([], "1D vector"),
    ],
)
def test_invalid_query_shape_or_dimension_is_rejected(
    query_vector: Any,
    message: str,
) -> None:
    embedder = _FakeEmbeddings(query_vector)
    retriever = VectorRetriever(bundle=_bundle(), embedder=embedder)

    with pytest.raises(QueryVectorValidationError, match=message):
        retriever.search("policy")
    assert embedder.query_calls == ["policy"]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_query_vector_is_rejected(invalid_value: float) -> None:
    embedder = _FakeEmbeddings([invalid_value, 0.0, 0.0])
    retriever = VectorRetriever(bundle=_bundle(), embedder=embedder)

    with pytest.raises(QueryVectorValidationError, match="NaN or Inf"):
        retriever.search("policy")


def test_zero_query_vector_is_rejected() -> None:
    retriever = VectorRetriever(
        bundle=_bundle(),
        embedder=_FakeEmbeddings([0.0, 0.0, 0.0]),
    )

    with pytest.raises(QueryVectorValidationError, match="zero vector"):
        retriever.search("policy")


def test_non_numeric_query_vector_is_rejected() -> None:
    retriever = VectorRetriever(
        bundle=_bundle(),
        embedder=_FakeEmbeddings(["invalid", 0.0, 0.0]),
    )

    with pytest.raises(QueryVectorValidationError, match="float32"):
        retriever.search("policy")


def test_embedding_failure_is_wrapped_and_chained() -> None:
    provider_error = ConnectionError("provider unavailable")
    embedder = _FakeEmbeddings(None, error=provider_error)
    retriever = VectorRetriever(bundle=_bundle(), embedder=embedder)

    with pytest.raises(QueryEmbeddingError, match="Failed to embed") as error_info:
        retriever.search("policy")

    assert error_info.value.__cause__ is provider_error
    assert embedder.query_calls == ["policy"]


def test_faiss_search_failure_is_wrapped_and_chained() -> None:
    faiss_error = RuntimeError("simulated FAISS failure")

    class _FailingIndex:
        d = _DIMS
        ntotal = 3

        def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
            raise faiss_error

    bundle = FaissIndexBundle(index=_FailingIndex(), chunks=_chunks(), manifest=_manifest())
    retriever = VectorRetriever(bundle=bundle, embedder=_FakeEmbeddings([1.0, 0.0, 0.0]))

    with pytest.raises(FaissSearchError, match="FAISS vector search failed") as error_info:
        retriever.search("policy")

    assert error_info.value.__cause__ is faiss_error


@pytest.mark.parametrize("minimum_score", [float("nan"), float("inf"), "invalid"])
def test_invalid_vector_min_score_is_rejected(minimum_score: Any) -> None:
    with pytest.raises(ValueError, match="finite number"):
        VectorRetriever(
            bundle=_bundle(),
            embedder=_FakeEmbeddings([1.0, 0.0, 0.0]),
            vector_min_score=minimum_score,
        )
