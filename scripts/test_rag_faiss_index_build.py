"""Offline FAISS index build smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings

from scripts.build_rag_faiss_index import build_faiss_index, load_chunks
from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.index import MANIFEST_FILE_NAME, load_faiss_index_bundle
from services.rag_mcp.schemas import KnowledgeSource
from shared import EmbeddingConfig


class _DeterministicEmbeddings(Embeddings):
    """Small offline embedding double for the build smoke test."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((row + column) % self._dimensions + 1) for column in range(self._dimensions)]
            for row, _ in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [1.0] * self._dimensions


def main() -> None:
    """Build index into a temp dir and verify row/chunk alignment."""
    import faiss

    input_path = Path("services/rag_mcp/knowledge/data/employee_policy.txt")
    source = KnowledgeSource(id="employee_policy", title="员工制度")
    expected_chunks = KnowledgeBuilder(input_path, source).build()
    config = EmbeddingConfig(
        enabled=True,
        provider="openai_compatible",
        model="offline-smoke-embedding",
        dims=4,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path, chunks_path, chunk_count, dimension = build_faiss_index(
            input_path=input_path,
            source=source,
            output_dir=temp_dir,
            embedding_config=config,
            embedder=_DeterministicEmbeddings(config.dims),
        )

        if not index_path.exists():
            raise RuntimeError("index.faiss未生成")
        if not chunks_path.exists():
            raise RuntimeError("chunks.json未生成")
        manifest_path = Path(temp_dir) / MANIFEST_FILE_NAME
        if not manifest_path.exists():
            raise RuntimeError("manifest.json未生成")

        bundle = load_faiss_index_bundle(temp_dir, config)
        index = bundle.index
        chunks = load_chunks(chunks_path)

        if index.ntotal != chunk_count:
            raise RuntimeError("index.ntotal与构建返回chunk数量不一致")
        if index.ntotal != len(chunks):
            raise RuntimeError("index.ntotal与chunks.json数量不一致")
        if index.ntotal != len(expected_chunks):
            raise RuntimeError("index.ntotal与KnowledgeBuilder输出数量不一致")
        if index.d != dimension:
            raise RuntimeError("index维度与Embedding实际维度不一致")
        if not isinstance(index, faiss.IndexFlatIP):
            raise RuntimeError("index类型不是IndexFlatIP")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise RuntimeError("index metric不是inner product")

        for row, (actual, expected) in enumerate(zip(chunks, expected_chunks, strict=True)):
            if actual != expected:
                raise RuntimeError(f"chunks.json第{row}行未能按原顺序还原DocumentChunk")

        print("RAG FAISS离线构建验证通过")
        print(f"chunk数量: {index.ntotal}")
        print(f"向量维度: {index.d}")


if __name__ == "__main__":
    main()
