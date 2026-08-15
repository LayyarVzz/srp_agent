"""Knowledge search tool."""

from __future__ import annotations

from services.rag_mcp.rag.pipeline import RAGPipeline
from services.rag_mcp.schemas import SearchKnowledgeResponse

_pipeline = RAGPipeline()


def search_knowledge(query: str, top_k: int = 5) -> SearchKnowledgeResponse:
    """Search knowledge chunks through the RAG pipeline."""
    chunks = _pipeline.search(query=query, top_k=top_k)
    return SearchKnowledgeResponse(query=query, chunks=chunks)
