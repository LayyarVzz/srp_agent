"""RAG MCP 内部检索流程入口。"""

from __future__ import annotations

from services.rag_mcp.schemas import KnowledgeChunk, KnowledgeSource


class RAGPipeline:
    """简单 RAG Pipeline，当前仅返回固定 Mock 数据。"""

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """按 query 检索知识片段。

        当前实现不调用 LLM、Embedding、向量数据库或 Retriever，仅用于验证数据流。
        """
        chunks = [
            KnowledgeChunk(
                content="员工年假按照工作年限计算。",
                source=KnowledgeSource(
                    id="doc001",
                    title="员工制度.pdf",
                    url=None,
                ),
                metadata={
                    "chunk_id": "chunk001",
                    "page": 5,
                },
                score=0.92,
            )
        ]
        return chunks[:top_k]
