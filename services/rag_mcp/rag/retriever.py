"""Lexical and vector retrievers for source-aware RAG chunks."""

from __future__ import annotations

import logging
import math
import re

import numpy as np
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from services.rag_mcp.rag.fusion import reciprocal_rank_fusion
from services.rag_mcp.rag.index import FaissIndexBundle
from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.schemas import RetrievalScoreDetails

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese text with bigrams and keep alphanumeric runs."""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        if value.isascii():
            tokens.append(value.lower())
            continue

        if len(value) == 1:
            tokens.append(value)
            continue

        tokens.extend(value[index : index + 2] for index in range(len(value) - 1))

    return tokens


def _score_chunk(query_chars: set[str], chunk: DocumentChunk) -> float:
    """Score a chunk by character overlap with the query."""
    chunk_chars = set(chunk.content)
    return len(query_chars & chunk_chars) / len(query_chars)


def retrieve_chunks(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve top chunks by simple character-overlap score."""
    if not query or not chunks or top_k <= 0:
        return []

    query_chars = set(query)
    if not query_chars:
        return []

    scored_chunks = [
        (score, index, chunk)
        for index, chunk in enumerate(chunks)
        if (score := _score_chunk(query_chars, chunk)) > 0
    ]
    if not scored_chunks:
        return []

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    return [
        RetrievedChunk(
            chunk=chunk,
            score=score,
        )
        for score, _, chunk in scored_chunks[:top_k]
    ]


class BM25Retriever:
    """BM25 retriever over source-aware document chunks."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks
        self._tokenized_corpus = [_tokenize(chunk.content) for chunk in chunks]
        self._index = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top chunks by BM25 score."""
        if not query or not self._chunks or top_k <= 0 or self._index is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        scored_chunks = [
            (float(score), index, chunk)
            for index, (score, chunk) in enumerate(zip(scores, self._chunks, strict=True))
        ]
        scored_chunks.sort(key=lambda item: (-item[0], item[1]))

        return [
            RetrievedChunk(
                chunk=chunk,
                score=score,
            )
            for score, _, chunk in scored_chunks[:top_k]
        ]


class VectorRetrievalError(RuntimeError):
    """Base error for a failed vector-retrieval query."""


class QueryEmbeddingError(VectorRetrievalError):
    """The embedding provider failed while embedding a query."""


class QueryVectorValidationError(VectorRetrievalError):
    """The query embedding is not a valid vector for the loaded FAISS index."""


class FaissSearchError(VectorRetrievalError):
    """FAISS failed to search the validated index or returned an invalid shape."""


class VectorRetriever:
    """Semantic retriever over a prevalidated, L2-normalized FAISS bundle."""

    def __init__(
        self,
        *,
        bundle: FaissIndexBundle,
        embedder: Embeddings,
        vector_min_score: float = 0.0,
    ) -> None:
        try:
            minimum_score = float(vector_min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("vector_min_score must be a finite number") from exc
        if not math.isfinite(minimum_score):
            raise ValueError("vector_min_score must be a finite number")

        self._bundle = bundle
        self._embedder = embedder
        self._vector_min_score = minimum_score

    def _embed_query(self, query: str) -> np.ndarray:
        try:
            raw_vector = self._embedder.embed_query(query)
        except Exception as exc:  # Provider SDKs do not expose one stable exception hierarchy.
            raise QueryEmbeddingError("Failed to embed the RAG query") from exc

        try:
            vector = np.asarray(raw_vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise QueryVectorValidationError(
                "Query embedding cannot be converted to float32"
            ) from exc

        if vector.ndim != 1 or vector.size == 0:
            raise QueryVectorValidationError("Query embedding must be a non-empty 1D vector")
        if vector.shape[0] != self._bundle.index.d:
            raise QueryVectorValidationError(
                "Query embedding dimension does not match the FAISS index: "
                f"{vector.shape[0]} != {self._bundle.index.d}"
            )
        if not np.all(np.isfinite(vector)):
            raise QueryVectorValidationError("Query embedding contains NaN or Inf")

        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm):
            raise QueryVectorValidationError("Query embedding norm is not finite")
        if norm == 0.0:
            raise QueryVectorValidationError("Query embedding is a zero vector")

        normalized = np.ascontiguousarray((vector / norm).reshape(1, -1), dtype=np.float32)
        if not np.isclose(np.linalg.norm(normalized[0]), 1.0, rtol=1e-5, atol=1e-6):
            raise QueryVectorValidationError("Query embedding could not be L2-normalized")
        return normalized

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Embed one query and return at most ``top_k`` vector-similar chunks."""
        normalized_query = query.strip()
        if not normalized_query or top_k <= 0:
            return []
        if self._bundle.index.ntotal <= 0:
            return []

        query_matrix = self._embed_query(normalized_query)
        search_k = min(top_k, int(self._bundle.index.ntotal))
        try:
            scores, row_ids = self._bundle.index.search(query_matrix, search_k)
        except Exception as exc:  # FAISS exposes search failures as RuntimeError subclasses.
            raise FaissSearchError("FAISS vector search failed") from exc

        if scores.ndim != 2 or row_ids.ndim != 2 or scores.shape != row_ids.shape:
            raise FaissSearchError("FAISS vector search returned invalid result arrays")
        if scores.shape[0] != 1:
            raise FaissSearchError("FAISS vector search returned an invalid query batch size")

        candidates: list[tuple[float, int]] = []
        for raw_score, raw_row_id in zip(scores[0], row_ids[0], strict=True):
            row_id = int(raw_row_id)
            if row_id < 0 or row_id >= len(self._bundle.chunks):
                continue
            score = float(raw_score)
            if not math.isfinite(score):
                raise FaissSearchError("FAISS vector search returned a non-finite score")
            if score > self._vector_min_score:
                candidates.append((score, row_id))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(
                chunk=self._bundle.chunks[row_id],
                score=score,
                score_details=RetrievalScoreDetails(vector=score),
            )
            for score, row_id in candidates[:top_k]
        ]


class HybridRetrievalError(RuntimeError):
    """Hybrid retrieval cannot safely produce a result."""

    def __init__(
        self,
        message: str,
        *,
        channel_errors: dict[str, Exception] | None = None,
    ) -> None:
        super().__init__(message)
        self.channel_errors = channel_errors or {}


class ChunkConsistencyError(HybridRetrievalError):
    """Two retrieval channels returned different chunks for one chunk id."""


def _require_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prepare_hybrid_candidates(
    results: list[RetrievedChunk],
    *,
    channel: str,
    positive_only: bool,
) -> list[RetrievedChunk]:
    """Validate channel scores, filter if requested, and enforce descending order."""
    candidates: list[RetrievedChunk] = []
    for result in results:
        if not isinstance(result, RetrievedChunk):
            raise TypeError(f"{channel} returned a non-RetrievedChunk result")
        score = float(result.score)
        if not math.isfinite(score):
            raise ValueError(f"{channel} returned a non-finite score")
        if positive_only and score <= 0.0:
            continue
        candidates.append(result)

    return sorted(candidates, key=lambda result: result.score, reverse=True)


def _validate_shared_chunks(
    bm25_results: list[RetrievedChunk],
    vector_results: list[RetrievedChunk],
) -> None:
    """Reject divergent payloads for ids returned by both retrieval channels."""
    bm25_chunks: dict[str, DocumentChunk] = {}
    vector_chunks: dict[str, DocumentChunk] = {}
    for result in bm25_results:
        bm25_chunks.setdefault(result.chunk.chunk_id, result.chunk)
    for result in vector_results:
        vector_chunks.setdefault(result.chunk.chunk_id, result.chunk)

    for chunk_id in bm25_chunks.keys() & vector_chunks.keys():
        bm25_chunk = bm25_chunks[chunk_id]
        vector_chunk = vector_chunks[chunk_id]
        mismatches = [
            field
            for field in ("content", "source", "metadata")
            if getattr(bm25_chunk, field) != getattr(vector_chunk, field)
        ]
        if mismatches:
            fields = ", ".join(mismatches)
            raise ChunkConsistencyError(
                f"Hybrid retrieval chunk mismatch for chunk_id={chunk_id!r}: {fields}"
            )


def _validate_chunk_collections(
    bm25_chunks: list[DocumentChunk],
    vector_chunks: list[DocumentChunk],
) -> None:
    """Validate the complete in-memory chunk collections once at initialization."""
    bm25_by_id = {chunk.chunk_id: chunk for chunk in bm25_chunks}
    vector_by_id = {chunk.chunk_id: chunk for chunk in vector_chunks}
    if len(bm25_by_id) != len(bm25_chunks):
        raise ChunkConsistencyError("BM25 chunk collection contains duplicate chunk_id values")
    if len(vector_by_id) != len(vector_chunks):
        raise ChunkConsistencyError("Vector chunk collection contains duplicate chunk_id values")
    if bm25_by_id.keys() != vector_by_id.keys():
        raise ChunkConsistencyError(
            "BM25 and vector chunk collections contain different chunk_id values"
        )

    _validate_shared_chunks(
        [RetrievedChunk(chunk=chunk, score=0.0) for chunk in bm25_chunks],
        [RetrievedChunk(chunk=chunk, score=0.0) for chunk in vector_chunks],
    )


class HybridRetriever:
    """Combine positive BM25 and vector candidates with RAG-local RRF."""

    def __init__(
        self,
        *,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        candidate_multiplier: int = 3,
        max_candidates: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self._bm25 = bm25
        self._vector = vector
        self._candidate_multiplier = _require_positive_integer(
            candidate_multiplier,
            name="candidate_multiplier",
        )
        self._max_candidates = _require_positive_integer(
            max_candidates,
            name="max_candidates",
        )
        self._rrf_k = _require_positive_integer(rrf_k, name="rrf_k")
        if isinstance(bm25, BM25Retriever) and isinstance(vector, VectorRetriever):
            _validate_chunk_collections(bm25._chunks, vector._bundle.chunks)

    def _search_channel(
        self,
        *,
        channel: str,
        retriever: BM25Retriever | VectorRetriever,
        query: str,
        candidate_k: int,
        positive_only: bool,
    ) -> tuple[list[RetrievedChunk], Exception | None]:
        try:
            results = retriever.search(query, candidate_k)
            return (
                _prepare_hybrid_candidates(
                    results,
                    channel=channel,
                    positive_only=positive_only,
                ),
                None,
            )
        except Exception as exc:
            logger.warning(
                "Hybrid retrieval channel failed: channel=%s error_type=%s",
                channel,
                type(exc).__name__,
            )
            return [], exc

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search both channels and fuse their candidates, degrading one channel on failure."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if top_k <= 0:
            return []
        if top_k > self._max_candidates:
            raise ValueError(
                f"top_k must not exceed max_candidates ({self._max_candidates})"
            )

        normalized_query = query.strip()
        if not normalized_query:
            return []

        candidate_k = min(
            top_k * self._candidate_multiplier,
            self._max_candidates,
        )
        bm25_results, bm25_error = self._search_channel(
            channel="bm25",
            retriever=self._bm25,
            query=normalized_query,
            candidate_k=candidate_k,
            positive_only=True,
        )
        vector_results, vector_error = self._search_channel(
            channel="vector",
            retriever=self._vector,
            query=normalized_query,
            candidate_k=candidate_k,
            positive_only=False,
        )

        if bm25_error is not None and vector_error is not None:
            raise HybridRetrievalError(
                "Both hybrid retrieval channels failed: "
                f"bm25={type(bm25_error).__name__}, "
                f"vector={type(vector_error).__name__}",
                channel_errors={"bm25": bm25_error, "vector": vector_error},
            ) from None

        _validate_shared_chunks(bm25_results, vector_results)
        return reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            k=self._rrf_k,
            top_k=top_k,
        )
