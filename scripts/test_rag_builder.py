"""KnowledgeBuilder回归验证脚本。"""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.splitter import split_text
from services.rag_mcp.schemas import KnowledgeSource

KNOWLEDGE_FILE_PATH = "services/rag_mcp/knowledge/data/employee_policy.txt"


def _build_source() -> KnowledgeSource:
    return KnowledgeSource(
        id="employee_policy",
        title="员工制度",
        url=None,
    )


def _validate_chunks(chunks: list[DocumentChunk], expected_chunks: list[DocumentChunk]) -> None:
    if not chunks:
        raise RuntimeError("KnowledgeBuilder.build()未返回任何DocumentChunk")
    if len(chunks) != len(expected_chunks):
        raise RuntimeError("KnowledgeBuilder.build()返回数量与Splitter结果不一致")

    for chunk_index, chunk in enumerate(chunks):
        expected_chunk = expected_chunks[chunk_index]
        if not isinstance(chunk, DocumentChunk):
            raise RuntimeError("KnowledgeBuilder.build()返回元素不是DocumentChunk")
        if chunk.source.id != "employee_policy":
            raise RuntimeError("chunk.source.id不符合预期")
        if chunk.source.title != "员工制度":
            raise RuntimeError("chunk.source.title不符合预期")
        if chunk.chunk_id != f"{chunk.source.id}:{chunk_index}":
            raise RuntimeError("chunk_id未保持当前规则")
        if chunk.metadata.get("chunk_index") != chunk_index:
            raise RuntimeError("metadata.chunk_index未保持当前规则")
        if chunk != expected_chunk:
            raise RuntimeError("KnowledgeBuilder.build()结果与Splitter结果不一致")


def main() -> None:
    """验证KnowledgeBuilder只负责构建DocumentChunk。"""
    source = _build_source()
    builder = KnowledgeBuilder(KNOWLEDGE_FILE_PATH, source)
    text = load_text_file(KNOWLEDGE_FILE_PATH)
    expected_chunks = split_text(text, source)

    chunks = builder.build()
    repeated_chunks = builder.build()

    _validate_chunks(chunks, expected_chunks)
    if repeated_chunks != chunks:
        raise RuntimeError("重复build结果结构不稳定")

    print("KnowledgeBuilder验证通过")
    print(f"chunk数量: {len(chunks)}")
    for chunk in chunks:
        print(f"{chunk.chunk_id}: {chunk.content}")


if __name__ == "__main__":
    main()
