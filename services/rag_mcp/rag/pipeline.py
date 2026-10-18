"""RAG MCP 内部检索流程入口。"""

from __future__ import annotations

from typing import Protocol

from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.rag.retriever import BM25Retriever
from services.rag_mcp.schemas import KnowledgeChunk


class Retriever(Protocol):
    """Minimal retrieval contract consumed by the RAG pipeline."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return at most ``top_k`` retrieved chunks for one query."""
        ...


class PipelineMappingError(RuntimeError):
    """A retriever result could not be mapped to the public knowledge schema."""


class RAGPipeline:
    """Map an injected retriever's results to public knowledge chunks."""

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        *,
        retriever: Retriever | None = None,
    ) -> None:
        """Use an injected retriever or build the legacy BM25 retriever from chunks."""
        if chunks is not None and retriever is not None:
            raise ValueError("Provide either chunks or retriever, not both")
        if retriever is None:
            if chunks is None:
                raise ValueError("Either chunks or retriever must be provided")
            self._chunks: list[DocumentChunk] | None = chunks
            self._retriever: Retriever = BM25Retriever(chunks)
            return
        if not callable(getattr(retriever, "search", None)):
            raise TypeError("retriever must provide a callable search method")

        self._chunks = None
        self._retriever = retriever

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """Retrieve chunks and map every retrieval field without rescoring."""
        retrieved_chunks = self._retriever.search(query=query, top_k=top_k)
        if not isinstance(retrieved_chunks, list):
            raise PipelineMappingError("Retriever search result must be a list")

        chunks: list[KnowledgeChunk] = []
        for index, retrieved in enumerate(retrieved_chunks):
            try:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=retrieved.chunk.chunk_id,
                        content=retrieved.chunk.content,
                        source=retrieved.chunk.source,
                        metadata=retrieved.chunk.metadata,
                        score=retrieved.score,
                        score_details=retrieved.score_details,
                    )
                )
            except Exception as exc:
                chunk_id = getattr(getattr(retrieved, "chunk", None), "chunk_id", None)
                raise PipelineMappingError(
                    "Failed to map retriever result at "
                    f"index={index}, chunk_id={chunk_id!r}"
                ) from exc
        return chunks
