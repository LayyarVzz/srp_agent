"""RAG-local reciprocal rank fusion for lexical and vector retrieval results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.schemas import RetrievalScoreDetails

DEFAULT_RRF_K = 60


class RRFValidationError(ValueError):
    """An RRF parameter or input retrieval result is invalid."""


@dataclass
class _FusedState:
    chunk: DocumentChunk
    bm25_score: float | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None

    @property
    def channel_count(self) -> int:
        return int(self.bm25_rank is not None) + int(self.vector_rank is not None)

    @property
    def best_rank(self) -> int:
        ranks = [rank for rank in (self.bm25_rank, self.vector_rank) if rank is not None]
        return min(ranks)


def _validate_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RRFValidationError(f"{name} must be a positive integer")
    return value


def _unique_ranked_results(
    results: Sequence[RetrievedChunk],
    *,
    channel: Literal["bm25", "vector"],
) -> list[tuple[int, RetrievedChunk, float]]:
    """Validate one channel and assign compact ranks after first-occurrence deduplication."""
    unique: list[tuple[RetrievedChunk, float]] = []
    seen_ids: set[str] = set()

    for result in results:
        if not isinstance(result, RetrievedChunk):
            raise RRFValidationError(f"{channel} results must contain RetrievedChunk values")
        chunk_id = result.chunk.chunk_id
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise RRFValidationError(f"{channel} result contains an empty chunk_id")
        try:
            score = float(result.score)
        except (TypeError, ValueError) as exc:
            raise RRFValidationError(f"{channel} result contains an invalid score") from exc
        if not math.isfinite(score):
            raise RRFValidationError(f"{channel} result contains a non-finite score")

        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique.append((result, score))

    return [
        (rank, result, score)
        for rank, (result, score) in enumerate(unique, start=1)
    ]


def reciprocal_rank_fusion(
    bm25_results: Sequence[RetrievedChunk],
    vector_results: Sequence[RetrievedChunk],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Fuse BM25 and vector rankings by ``sum(1 / (k + 1-based rank))``.

    The public ``score`` is normalized as ``min(1, raw_rrf / (2 / (k + 1)))``.
    It is a ranking/display value for numeric compatibility, not a probability,
    calibrated semantic relevance, or evidence that a query is truly related to
    a chunk. Raw channel scores and the unnormalized RRF value remain available
    in ``score_details``.

    Within each channel, the first occurrence of a ``chunk_id`` wins and later
    duplicates are removed before compact ranks are assigned. Across channels,
    BM25 supplies the representative DocumentChunk when both channels contain
    the same id; otherwise the available channel supplies it.
    """
    validated_k = _validate_positive_integer(k, name="k")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RRFValidationError("top_k must be an integer")
    if top_k <= 0:
        return []

    bm25_ranked = _unique_ranked_results(bm25_results, channel="bm25")
    vector_ranked = _unique_ranked_results(vector_results, channel="vector")
    if not bm25_ranked and not vector_ranked:
        return []

    states: dict[str, _FusedState] = {}
    for rank, result, score in bm25_ranked:
        states[result.chunk.chunk_id] = _FusedState(
            chunk=result.chunk,
            bm25_score=score,
            bm25_rank=rank,
        )

    for rank, result, score in vector_ranked:
        state = states.get(result.chunk.chunk_id)
        if state is None:
            state = _FusedState(chunk=result.chunk)
            states[result.chunk.chunk_id] = state
        state.vector_score = score
        state.vector_rank = rank

    def raw_rrf(state: _FusedState) -> float:
        ranks = (state.bm25_rank, state.vector_rank)
        return sum(1.0 / (validated_k + rank) for rank in ranks if rank is not None)

    ordered_states = sorted(
        states.items(),
        key=lambda item: (
            -raw_rrf(item[1]),
            -item[1].channel_count,
            item[1].best_rank,
            item[0],
        ),
    )
    maximum_rrf = 2.0 / (validated_k + 1)

    fused: list[RetrievedChunk] = []
    for _, state in ordered_states[:top_k]:
        raw_score = raw_rrf(state)
        normalized_score = min(1.0, raw_score / maximum_rrf)
        fused.append(
            RetrievedChunk(
                chunk=state.chunk,
                score=normalized_score,
                score_details=RetrievalScoreDetails(
                    bm25=state.bm25_score,
                    vector=state.vector_score,
                    rrf=raw_score,
                    bm25_rank=state.bm25_rank,
                    vector_rank=state.vector_rank,
                ),
            )
        )
    return fused


__all__ = ["DEFAULT_RRF_K", "RRFValidationError", "reciprocal_rank_fusion"]
