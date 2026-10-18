"""Deterministic tests for RAG-local reciprocal rank fusion."""

from __future__ import annotations

import math
from typing import Literal

import pytest

from services.rag_mcp.rag.fusion import RRFValidationError, reciprocal_rank_fusion
from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.schemas import KnowledgeSource


def _hit(
    chunk_id: str,
    score: float,
    *,
    content: str | None = None,
    channel: Literal["bm25", "vector"] = "bm25",
) -> RetrievedChunk:
    source = KnowledgeSource(id="policy", title="Employee policy")
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            content=content or f"Content for {chunk_id}",
            source=source,
            metadata={"channel": channel},
        ),
        score=score,
    )


def test_single_bm25_channel_uses_one_rrf_contribution() -> None:
    results = reciprocal_rank_fusion(
        [_hit("a", 4.2), _hit("b", 2.1)],
        [],
        top_k=5,
    )

    assert [result.chunk.chunk_id for result in results] == ["a", "b"]
    assert results[0].score_details.rrf == pytest.approx(1.0 / 61.0)
    assert results[1].score_details.rrf == pytest.approx(1.0 / 62.0)
    assert results[0].score == pytest.approx(0.5)
    assert results[0].score_details.bm25 == 4.2
    assert results[0].score_details.vector is None
    assert results[0].score_details.bm25_rank == 1
    assert results[0].score_details.vector_rank is None


def test_single_vector_channel_preserves_vector_details() -> None:
    results = reciprocal_rank_fusion(
        [],
        [_hit("a", 0.87, channel="vector")],
    )

    assert results[0].score == pytest.approx(0.5)
    assert results[0].score_details.bm25 is None
    assert results[0].score_details.vector == 0.87
    assert results[0].score_details.bm25_rank is None
    assert results[0].score_details.vector_rank == 1


def test_two_channel_rrf_accumulates_and_normalizes() -> None:
    bm25 = [_hit("other", 3.0), _hit("shared", 2.5, content="BM25 copy")]
    vector = [_hit("shared", 0.9, content="Vector copy", channel="vector")]

    results = reciprocal_rank_fusion(bm25, vector)
    shared = next(result for result in results if result.chunk.chunk_id == "shared")
    expected_raw = 1.0 / 62.0 + 1.0 / 61.0
    expected_normalized = expected_raw / (2.0 / 61.0)

    assert shared.score_details.rrf == pytest.approx(expected_raw)
    assert shared.score == pytest.approx(expected_normalized)
    assert shared.score_details.bm25_rank == 2
    assert shared.score_details.vector_rank == 1
    assert shared.score_details.bm25 == 2.5
    assert shared.score_details.vector == 0.9


def test_cross_channel_duplicate_returns_one_bm25_representative() -> None:
    bm25_hit = _hit("shared", 5.0, content="BM25 representative")
    vector_hit = _hit("shared", 0.8, content="Vector representative", channel="vector")

    results = reciprocal_rank_fusion([bm25_hit], [vector_hit])

    assert len(results) == 1
    assert results[0].chunk.content == "BM25 representative"
    assert results[0].score == pytest.approx(1.0)


def test_duplicate_within_channel_contributes_once_and_ranks_are_compacted() -> None:
    bm25 = [
        _hit("a", 9.0),
        _hit("a", 8.0, content="duplicate"),
        _hit("b", 7.0),
    ]

    results = reciprocal_rank_fusion(bm25, [])
    by_id = {result.chunk.chunk_id: result for result in results}

    assert list(by_id) == ["a", "b"]
    assert by_id["a"].score_details.rrf == pytest.approx(1.0 / 61.0)
    assert by_id["a"].score_details.bm25 == 9.0
    assert by_id["a"].score_details.bm25_rank == 1
    assert by_id["b"].score_details.rrf == pytest.approx(1.0 / 62.0)
    assert by_id["b"].score_details.bm25_rank == 2


def test_disjoint_channels_retain_all_results_and_original_scores() -> None:
    results = reciprocal_rank_fusion(
        [_hit("bm25", 6.5)],
        [_hit("vector", 0.75, channel="vector")],
    )
    by_id = {result.chunk.chunk_id: result for result in results}

    assert set(by_id) == {"bm25", "vector"}
    assert by_id["bm25"].score_details.bm25 == 6.5
    assert by_id["bm25"].score_details.vector is None
    assert by_id["vector"].score_details.bm25 is None
    assert by_id["vector"].score_details.vector == 0.75


def test_sorting_uses_chunk_id_for_symmetric_rankings() -> None:
    bm25 = [_hit("a", 1.0), _hit("b", 1.0)]
    vector = [
        _hit("b", 0.9, channel="vector"),
        _hit("a", 0.9, channel="vector"),
    ]

    results = reciprocal_rank_fusion(bm25, vector)

    assert results[0].score_details.rrf == pytest.approx(results[1].score_details.rrf)
    assert [result.chunk.chunk_id for result in results] == ["a", "b"]


def test_sorting_prefers_more_channels_when_raw_rrf_ties() -> None:
    bm25 = [_hit("single", 3.0), _hit("filler-b", 2.0), _hit("double", 1.0)]
    vector = [
        _hit("filler-v1", 0.9, channel="vector"),
        _hit("filler-v2", 0.8, channel="vector"),
        _hit("double", 0.7, channel="vector"),
    ]

    results = reciprocal_rank_fusion(bm25, vector, k=1, top_k=10)
    positions = {result.chunk.chunk_id: index for index, result in enumerate(results)}
    by_id = {result.chunk.chunk_id: result for result in results}

    assert by_id["single"].score_details.rrf == pytest.approx(0.5)
    assert by_id["double"].score_details.rrf == pytest.approx(0.5)
    assert positions["double"] < positions["single"]


def test_sorting_prefers_better_channel_rank_after_other_ties() -> None:
    bm25 = [
        _hit("best-rank", 5.0),
        _hit("balanced", 4.0),
        _hit("filler-b3", 3.0),
        _hit("filler-b4", 2.0),
        _hit("filler-b5", 1.0),
    ]
    vector = [
        _hit("filler-v1", 0.9, channel="vector"),
        _hit("balanced", 0.8, channel="vector"),
        _hit("filler-v3", 0.7, channel="vector"),
        _hit("filler-v4", 0.6, channel="vector"),
        _hit("best-rank", 0.5, channel="vector"),
    ]

    results = reciprocal_rank_fusion(bm25, vector, k=1, top_k=10)
    positions = {result.chunk.chunk_id: index for index, result in enumerate(results)}
    by_id = {result.chunk.chunk_id: result for result in results}

    assert by_id["best-rank"].score_details.rrf == pytest.approx(2.0 / 3.0)
    assert by_id["balanced"].score_details.rrf == pytest.approx(2.0 / 3.0)
    assert positions["best-rank"] < positions["balanced"]


def test_top_k_strictly_truncates_after_fusion() -> None:
    bm25 = [_hit("a", 3.0), _hit("b", 2.0), _hit("c", 1.0)]
    vector = [_hit("c", 0.9, channel="vector")]

    results = reciprocal_rank_fusion(bm25, vector, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c"


def test_empty_channels_return_empty_list() -> None:
    assert reciprocal_rank_fusion([], []) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_empty_list(top_k: int) -> None:
    assert reciprocal_rank_fusion([_hit("a", 1.0)], [], top_k=top_k) == []


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(k: object) -> None:
    with pytest.raises(RRFValidationError, match="positive integer"):
        reciprocal_rank_fusion([], [], k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [1.5, True, "5"])
def test_non_integer_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(RRFValidationError, match="top_k must be an integer"):
        reciprocal_rank_fusion([], [], top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("chunk_id", ["", "   "])
def test_empty_chunk_id_is_rejected(chunk_id: str) -> None:
    with pytest.raises(RRFValidationError, match="empty chunk_id"):
        reciprocal_rank_fusion([_hit(chunk_id, 1.0)], [])


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("channel", ["bm25", "vector"])
def test_non_finite_channel_score_is_rejected(
    score: float,
    channel: Literal["bm25", "vector"],
) -> None:
    hit = _hit("a", score, channel=channel)
    bm25 = [hit] if channel == "bm25" else []
    vector = [hit] if channel == "vector" else []

    with pytest.raises(RRFValidationError, match="non-finite score"):
        reciprocal_rank_fusion(bm25, vector)


def test_duplicate_with_non_finite_score_is_still_rejected() -> None:
    results = [_hit("a", 1.0), _hit("a", math.nan)]

    with pytest.raises(RRFValidationError, match="non-finite score"):
        reciprocal_rank_fusion(results, [])
