"""RAG MCP 内部检索流程入口。"""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.retriever import retrieve_chunks
from services.rag_mcp.rag.splitter import split_text
from services.rag_mcp.schemas import KnowledgeChunk, KnowledgeSource

KNOWLEDGE_FILE_PATH = "services/rag_mcp/knowledge/data/employee_policy.txt"


class RAGPipeline:
    """简单 RAG Pipeline。"""

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """按 query 检索知识片段。"""
        text = load_text_file(KNOWLEDGE_FILE_PATH)
        chunks = split_text(text)
        retrieved_chunks = retrieve_chunks(query=query, chunks=chunks, top_k=top_k)
        source = KnowledgeSource(
            id="employee_policy",
            title="员工制度",
            url=None,
        )

        return [
            KnowledgeChunk(
                content=chunk,
                source=source,
                metadata={"chunk_index": index},
                # 当前 MVP Retriever 只返回文本片段，尚未返回相关性分数。
                score=0.0,
            )
            for index, chunk in enumerate(retrieved_chunks)
        ]
