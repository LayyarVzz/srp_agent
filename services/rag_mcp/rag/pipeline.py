"""RAG MCP 内部检索流程入口。"""

from __future__ import annotations

from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.retriever import BM25Retriever
from services.rag_mcp.schemas import KnowledgeChunk


class RAGPipeline:
    """简单 RAG Pipeline。"""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        """为当前Pipeline实例创建一次BM25检索器。"""
        self._chunks = chunks
        self._retriever = BM25Retriever(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """按 query 检索知识片段。"""
        retrieved_chunks = self._retriever.search(query=query, top_k=top_k)

        return [
            KnowledgeChunk(
                content=retrieved.chunk.content,
                source=retrieved.chunk.source,
                metadata=retrieved.chunk.metadata,
                score=retrieved.score,
            )
            for retrieved in retrieved_chunks
        ]
