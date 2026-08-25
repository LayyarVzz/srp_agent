"""Simple character-overlap retriever."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


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
