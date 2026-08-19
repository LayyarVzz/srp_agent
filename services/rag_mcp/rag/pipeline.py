"""RAG MCP 内部检索流程入口。"""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.retriever import BM25Retriever
from services.rag_mcp.rag.splitter import split_text
from services.rag_mcp.schemas import KnowledgeChunk, KnowledgeSource

KNOWLEDGE_FILE_PATH = "services/rag_mcp/knowledge/data/employee_policy.txt"


class RAGPipeline:
    """简单 RAG Pipeline。"""

    def __init__(self) -> None:
        """为当前Pipeline实例加载并切分一次知识文本。"""
        text = load_text_file(KNOWLEDGE_FILE_PATH)
        source = KnowledgeSource(
            id="employee_policy",
            title="员工制度",
            url=None,
        )
        self._chunks: list[DocumentChunk] = split_text(text, source)
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
