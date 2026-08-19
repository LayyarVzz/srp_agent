"""Retriever回归验证脚本。"""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.models import DocumentChunk, RetrievedChunk
from services.rag_mcp.rag.retriever import BM25Retriever, retrieve_chunks
from services.rag_mcp.rag.splitter import split_text
from services.rag_mcp.schemas import KnowledgeSource


def _build_source() -> KnowledgeSource:
    return KnowledgeSource(
        id="employee_policy",
        title="员工制度",
        url=None,
    )


def _validate_chunk_identity(chunk: DocumentChunk) -> None:
    chunk_index = chunk.metadata.get("chunk_index")
    if not isinstance(chunk_index, int):
        raise RuntimeError("chunk.metadata中的chunk_index无效")
    if chunk.chunk_id != f"{chunk.source.id}:{chunk_index}":
        raise RuntimeError("chunk_id与chunk_index/source.id不匹配")


def _print_results(title: str, results: list[RetrievedChunk]) -> None:
    print(title)
    for rank, retrieved in enumerate(results, start=1):
        chunk = retrieved.chunk
        chunk_index = chunk.metadata["chunk_index"]
        print(f"{rank}. content: {chunk.content}")
        print(f"   chunk_id: {chunk.chunk_id}")
        print(f"   chunk_index: {chunk_index}")
        print(f"   score: {retrieved.score}")
        print(f"   source: {chunk.source.model_dump()}")
        print(f"   metadata: {chunk.metadata}")


def _validate_bm25_results(results: list[RetrievedChunk], top_k: int) -> None:
    if len(results) > top_k:
        raise RuntimeError("BM25Retriever检索结果数量超过top_k")
    if not results:
        raise RuntimeError("BM25Retriever未返回任何结果")
    if not all(isinstance(retrieved, RetrievedChunk) for retrieved in results):
        raise RuntimeError("BM25Retriever返回结果不是RetrievedChunk")
    if not all(isinstance(retrieved.chunk, DocumentChunk) for retrieved in results):
        raise RuntimeError("BM25Retriever返回的chunk不是DocumentChunk")
    if not all(isinstance(retrieved.score, float) for retrieved in results):
        raise RuntimeError("BM25Retriever返回的score不是float")

    for retrieved in results:
        _validate_chunk_identity(retrieved.chunk)

    if not any(
        "10 年" in retrieved.chunk.content and "10 天" in retrieved.chunk.content
        for retrieved in results
    ):
        raise RuntimeError("BM25Retriever检索结果中未包含10年年假相关内容")


def main() -> None:
    """读取、切分、检索并打印匹配的DocumentChunk。"""
    text = load_text_file("services/rag_mcp/knowledge/data/employee_policy.txt")
    source = _build_source()
    chunks = split_text(text, source)
    query = "员工工作10年有多少天年假？"
    results = retrieve_chunks(
        query=query,
        chunks=chunks,
        top_k=3,
    )

    if not chunks:
        raise RuntimeError("Splitter未返回任何DocumentChunk")
    if len(results) > 3:
        raise RuntimeError("top_k=3时检索结果数量超过3")
    if not results:
        raise RuntimeError("Retriever未返回任何结果")
    if not all(isinstance(retrieved, RetrievedChunk) for retrieved in results):
        raise RuntimeError("Retriever返回结果不是RetrievedChunk")
    if not all(isinstance(retrieved.chunk, DocumentChunk) for retrieved in results):
        raise RuntimeError("RetrievedChunk.chunk不是DocumentChunk")
    if not all(isinstance(retrieved.score, float) for retrieved in results):
        raise RuntimeError("RetrievedChunk.score不是float")
    if not all(0.0 < retrieved.score <= 1.0 for retrieved in results):
        raise RuntimeError("RetrievedChunk.score超出预期范围")
    if not any(
        "10 年" in retrieved.chunk.content and "10 天" in retrieved.chunk.content
        for retrieved in results
    ):
        raise RuntimeError("检索结果中未包含10年年假相关内容")

    print("原始文本类型:")
    print(type(text))
    print("chunk数量:")
    print(len(chunks))
    print("query:")
    print(query)
    print("检索结果数量:")
    print(len(results))
    print("字符Retriever结果:")
    identity_preserved_after_sort = False
    for rank, retrieved in enumerate(results, start=1):
        chunk = retrieved.chunk
        _validate_chunk_identity(chunk)
        chunk_index = chunk.metadata["chunk_index"]
        if chunk_index != rank - 1:
            identity_preserved_after_sort = True

        print(f"{rank}. content: {chunk.content}")
        print(f"   chunk_id: {chunk.chunk_id}")
        print(f"   chunk_index: {chunk_index}")
        print(f"   score: {retrieved.score}")
        print(f"   source: {chunk.source.model_dump()}")
        print(f"   metadata: {chunk.metadata}")

    if not identity_preserved_after_sort:
        raise RuntimeError("本次结果未体现排序后chunk_index保持原始值")

    print("Retriever DocumentChunk验证通过")

    bm25_retriever = BM25Retriever(chunks)
    bm25_results = bm25_retriever.search(query=query, top_k=3)
    _validate_bm25_results(bm25_results, top_k=3)
    print("BM25Retriever结果数量:")
    print(len(bm25_results))
    _print_results("BM25Retriever结果:", bm25_results)
    print("BM25Retriever验证通过")


if __name__ == "__main__":
    main()
