"""Splitter回归验证脚本。"""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.splitter import split_text
from services.rag_mcp.schemas import KnowledgeSource


def _build_source() -> KnowledgeSource:
    return KnowledgeSource(
        id="employee_policy",
        title="员工制度",
        url=None,
    )


def main() -> None:
    """读取员工制度文本并打印DocumentChunk切分结果。"""
    text = load_text_file("services/rag_mcp/knowledge/data/employee_policy.txt")
    source = _build_source()
    chunks = split_text(text, source)
    expected_contents = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    if not chunks:
        raise RuntimeError("Splitter未返回任何chunk")
    if len(chunks) != len(expected_contents):
        raise RuntimeError("DocumentChunk数量与原始空行切分结果不一致")

    print("原始文本类型:")
    print(type(text))
    print("切分结果类型:")
    print(type(chunks))
    print("chunk数量:")
    print(len(chunks))
    print("chunk内容:")
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, DocumentChunk):
            raise RuntimeError("chunk不是DocumentChunk")
        if chunk.content != expected_contents[chunk_index]:
            raise RuntimeError("chunk.content与原始切分文本不一致")
        if chunk.source != source:
            raise RuntimeError("chunk.source与传入KnowledgeSource不一致")
        if chunk.metadata.get("chunk_index") != chunk_index:
            raise RuntimeError("chunk_index未按原始切分顺序递增")
        if chunk.chunk_id != f"{source.id}:{chunk_index}":
            raise RuntimeError("chunk_id不符合当前MVP规则")

        print(f"{chunk_index + 1}. content: {chunk.content}")
        print(f"   chunk_id: {chunk.chunk_id}")
        print(f"   source: {chunk.source.model_dump()}")
        print(f"   metadata: {chunk.metadata}")

    print("Splitter DocumentChunk验证通过")


if __name__ == "__main__":
    main()
