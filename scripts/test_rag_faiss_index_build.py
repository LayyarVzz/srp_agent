"""Offline FAISS index build smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.build_rag_faiss_index import build_faiss_index, load_chunks
from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.schemas import KnowledgeSource


def main() -> None:
    """Build index into a temp dir and verify row/chunk alignment."""
    import faiss

    input_path = Path("services/rag_mcp/knowledge/data/employee_policy.txt")
    source = KnowledgeSource(id="employee_policy", title="员工制度")
    expected_chunks = KnowledgeBuilder(input_path, source).build()

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path, chunks_path, chunk_count, dimension = build_faiss_index(
            input_path=input_path,
            source=source,
            output_dir=temp_dir,
        )

        if not index_path.exists():
            raise RuntimeError("index.faiss未生成")
        if not chunks_path.exists():
            raise RuntimeError("chunks.json未生成")

        index = faiss.read_index(str(index_path))
        chunks = load_chunks(chunks_path)

        if index.ntotal != chunk_count:
            raise RuntimeError("index.ntotal与构建返回chunk数量不一致")
        if index.ntotal != len(chunks):
            raise RuntimeError("index.ntotal与chunks.json数量不一致")
        if index.ntotal != len(expected_chunks):
            raise RuntimeError("index.ntotal与KnowledgeBuilder输出数量不一致")
        if index.d != dimension:
            raise RuntimeError("index维度与Embedding实际维度不一致")

        for row, (actual, expected) in enumerate(zip(chunks, expected_chunks, strict=True)):
            if actual != expected:
                raise RuntimeError(f"chunks.json第{row}行未能按原顺序还原DocumentChunk")

        print("RAG FAISS离线构建验证通过")
        print(f"chunk数量: {index.ntotal}")
        print(f"向量维度: {index.d}")


if __name__ == "__main__":
    main()
