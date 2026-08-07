"""Knowledge search tool."""

from __future__ import annotations

from services.rag_mcp.rag.pipeline import RAGPipeline
from services.rag_mcp.schemas import SearchKnowledgeResponse


def search_knowledge(query: str, top_k: int = 5) -> SearchKnowledgeResponse:
    """Search knowledge chunks through the RAG pipeline."""
    pipeline = RAGPipeline()
    chunks = pipeline.search(query=query, top_k=top_k)
    return SearchKnowledgeResponse(query=query, chunks=chunks)
