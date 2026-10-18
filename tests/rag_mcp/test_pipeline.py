"""Deterministic tests for RAGPipeline retriever injection and result mapping."""

from __future__ import annotations

from typing import Any, cast

import faiss
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

import services.rag_mcp.rag.retriever as retriever_module
from services.rag_mcp.rag.fusion import reciprocal_rank_fusion
from services.rag_mcp.rag.index import FaissIndexBundle, FaissIndexManifest
from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.rag.pipeline import PipelineMappingError, RAGPipeline
from services.rag_mcp.rag.retriever import (
    BM25Retriever,
    HybridRetrievalError,
    HybridRetriever,
    VectorRetriever,
)
from services.rag_mcp.schemas import (
    KnowledgeSource,
    RetrievalScoreDetails,
    SearchKnowledgeResponse,
)


class _FakeRetriever:
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
        raise AssertionError("Pipeline search must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.query_vector


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
                ("Vacation entitlement days", "vacation"),
                ("Medical reimbursement rules", "medical"),
                ("Travel expense approval", "travel"),
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
        embedding_dimensions=2,
        chunk_count=3,
        index_sha256="0" * 64,
        chunks_sha256="1" * 64,
    )


def _real_hybrid(
    *,
    max_candidates: int = 50,
) -> tuple[HybridRetriever, _FakeEmbeddings]:
    chunks = _chunks()
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
        bundle=FaissIndexBundle(index=index, chunks=chunks, manifest=_manifest()),
        embedder=embedder,
    )
    return (
        HybridRetriever(
            bm25=BM25Retriever(chunks),
            vector=vector,
            max_candidates=max_candidates,
        ),
        embedder,
    )


def test_legacy_chunks_constructor_remains_compatible() -> None:
    pipeline = RAGPipeline(_chunks())

    results = pipeline.search("vacation", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "policy:0"
    assert results[0].content == "Vacation entitlement days"
    assert results[0].source.id == "policy"
    assert results[0].metadata == {"chunk_index": 0, "label": "vacation"}
    assert results[0].score > 0.0
    assert results[0].score_details is None


def test_injected_retriever_receives_exact_query_and_top_k() -> None:
    fake = _FakeRetriever()
    pipeline = RAGPipeline(retriever=fake)

    assert pipeline.search("  exact query  ", top_k=7) == []
    assert fake.calls == [("  exact query  ", 7)]


def test_all_retrieval_fields_are_mapped_without_rescoring() -> None:
    source = KnowledgeSource(id="handbook", title="Handbook", url="https://example.test")
    details = RetrievalScoreDetails(
        bm25=4.5,
        vector=0.8,
        rrf=0.03,
        bm25_rank=1,
        vector_rank=2,
    )
    retrieved = RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="handbook:7",
            content="Mapped content",
            source=source,
            metadata={"page": 7, "section": "leave"},
        ),
        score=0.4321,
        score_details=details,
    )
    pipeline = RAGPipeline(retriever=_FakeRetriever([retrieved]))

    result = pipeline.search("query", top_k=1)[0]

    assert result.chunk_id == "handbook:7"
    assert result.content == "Mapped content"
    assert result.source == source
    assert result.metadata == {"page": 7, "section": "leave"}
    assert result.score == 0.4321
    assert result.score_details == details
    assert result.model_dump()["score_details"] == details.model_dump()


def test_search_response_top_level_contract_remains_query_and_chunks() -> None:
    pipeline = RAGPipeline(retriever=_FakeRetriever())

    response = SearchKnowledgeResponse(query="empty", chunks=pipeline.search("empty"))

    assert response.model_dump() == {"query": "empty", "chunks": []}


def test_empty_retriever_result_maps_to_empty_chunks() -> None:
    fake = _FakeRetriever()
    pipeline = RAGPipeline(retriever=fake)

    assert pipeline.search("query", top_k=3) == []
    assert fake.calls == [("query", 3)]


def test_retriever_exception_propagates_unchanged() -> None:
    failure = HybridRetrievalError("hybrid unavailable")
    pipeline = RAGPipeline(retriever=_FakeRetriever(error=failure))

    with pytest.raises(HybridRetrievalError) as exc:
        pipeline.search("query")

    assert exc.value is failure


def test_hybrid_top_k_limit_propagates_through_pipeline() -> None:
    hybrid, embedder = _real_hybrid(max_candidates=2)
    pipeline = RAGPipeline(retriever=hybrid)

    with pytest.raises(ValueError, match="must not exceed max_candidates"):
        pipeline.search("vacation", top_k=3)

    assert embedder.query_calls == []


def test_mapping_failure_has_context_and_preserves_cause() -> None:
    invalid_results = cast(list[RetrievedChunk], [object()])
    pipeline = RAGPipeline(retriever=_FakeRetriever(invalid_results))

    with pytest.raises(PipelineMappingError, match="index=0, chunk_id=None") as exc:
        pipeline.search("query")

    assert isinstance(exc.value.__cause__, AttributeError)


def test_non_list_retriever_result_is_rejected() -> None:
    class _InvalidRetriever:
        def search(self, query: str, top_k: int = 5) -> Any:
            return (query, top_k)

    pipeline = RAGPipeline(retriever=_InvalidRetriever())  # type: ignore[arg-type]

    with pytest.raises(PipelineMappingError, match="must be a list"):
        pipeline.search("query")


@pytest.mark.parametrize(
    "build",
    [
        lambda: RAGPipeline(),
        lambda: RAGPipeline(_chunks(), retriever=_FakeRetriever()),
    ],
)
def test_constructor_rejects_missing_or_ambiguous_source(build: Any) -> None:
    with pytest.raises(ValueError):
        build()


def test_pipeline_does_not_repeat_hybrid_rrf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def _rrf_spy(
        bm25_results: list[RetrievedChunk],
        vector_results: list[RetrievedChunk],
        *,
        k: int,
        top_k: int,
    ) -> list[RetrievedChunk]:
        nonlocal call_count
        call_count += 1
        return reciprocal_rank_fusion(bm25_results, vector_results, k=k, top_k=top_k)

    monkeypatch.setattr(retriever_module, "reciprocal_rank_fusion", _rrf_spy)
    hybrid, _ = _real_hybrid()
    pipeline = RAGPipeline(retriever=hybrid)

    pipeline.search("vacation", top_k=2)

    assert call_count == 1


def test_real_bm25_faiss_hybrid_pipeline_integration() -> None:
    hybrid, embedder = _real_hybrid()
    pipeline = RAGPipeline(retriever=hybrid)

    results = pipeline.search("vacation", top_k=2)

    by_id = {result.chunk_id: result for result in results}
    assert set(by_id) == {"policy:0", "policy:1"}
    assert by_id["policy:0"].score_details.bm25 is not None
    assert by_id["policy:0"].score_details.vector is None
    assert by_id["policy:1"].score_details.bm25 is None
    assert by_id["policy:1"].score_details.vector == pytest.approx(1.0)
    assert all(result.score_details.rrf is not None for result in results)
    assert embedder.query_calls == ["vacation"]
