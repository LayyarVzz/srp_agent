"""Knowledge search tool."""

from __future__ import annotations

from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.pipeline import RAGPipeline
from services.rag_mcp.schemas import KnowledgeSource, SearchKnowledgeResponse

KNOWLEDGE_FILE_PATH = "services/rag_mcp/knowledge/data/employee_policy.txt"
KNOWLEDGE_SOURCE = KnowledgeSource(
    id="employee_policy",
    title="员工制度",
    url=None,
)

_builder = KnowledgeBuilder(KNOWLEDGE_FILE_PATH, KNOWLEDGE_SOURCE)
_chunks = _builder.build()
_pipeline = RAGPipeline(_chunks)


def search_knowledge(query: str, top_k: int = 5) -> SearchKnowledgeResponse:
    """Search knowledge chunks through the RAG pipeline."""
    chunks = _pipeline.search(query=query, top_k=top_k)
    return SearchKnowledgeResponse(query=query, chunks=chunks)
