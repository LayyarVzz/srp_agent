"""Simple text splitter for RAG knowledge."""

from __future__ import annotations

from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.schemas import KnowledgeSource


def split_text(text: str, source: KnowledgeSource) -> list[DocumentChunk]:
    """Split text into non-empty chunks by blank lines."""
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return [
        DocumentChunk(
            chunk_id=f"{source.id}:{chunk_index}",
            content=chunk,
            source=source,
            metadata={"chunk_index": chunk_index},
        )
        for chunk_index, chunk in enumerate(chunks)
    ]
