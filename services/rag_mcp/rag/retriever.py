"""Simple character-overlap retriever."""

from __future__ import annotations


def _score_chunk(query_chars: set[str], chunk: str) -> float:
    """Score a chunk by character overlap with the query."""
    chunk_chars = set(chunk)
    return len(query_chars & chunk_chars) / len(query_chars)


def retrieve_chunks(
    query: str,
    chunks: list[str],
    top_k: int = 5,
) -> list[str]:
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
    return [chunk for _, _, chunk in scored_chunks[:top_k]]
