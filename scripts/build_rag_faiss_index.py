"""离线构建 RAG FAISS 索引。

本脚本是文档 Embedding 的唯一执行边界：读取单个 TXT/Markdown 知识源，经
KnowledgeBuilder 生成 DocumentChunk，再用 shared Embedding 底座生成向量并写入
预构建索引。RAG MCP 在线启动阶段只应加载产物，不应重新 embed 全量文档。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.schemas import KnowledgeSource
from settings import get_settings
from shared import EmbeddingConfig, EmbeddingsFactory

DEFAULT_OUTPUT_DIR = Path("services/rag_mcp/knowledge/index")
INDEX_FILE_NAME = "index.faiss"
CHUNKS_FILE_NAME = "chunks.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline FAISS index for RAG knowledge.")
    parser.add_argument("--input", required=True, help="Single .txt/.md/.markdown knowledge file.")
    parser.add_argument("--source-id", required=True, help="Knowledge source id.")
    parser.add_argument("--source-title", required=True, help="Knowledge source title.")
    parser.add_argument("--source-url", default=None, help="Optional source URL.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for index.faiss and chunks.json.",
    )
    return parser.parse_args()


def _build_embedder():
    """按运行环境构造 shared Embeddings。

    WHY 配置读取留在离线脚本：公共 Embedding 底座不读环境，在线 RAG 也不在启动时
    embed 文档；只有本脚本把 RuntimeSettings 转成 EmbeddingConfig。
    """
    settings = get_settings()
    config = EmbeddingConfig.from_runtime(
        enabled=True,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )
    return EmbeddingsFactory(config).build()


def _normalize_vectors(vectors: list[list[float]]) -> np.ndarray:
    """转为 float32 并 L2 normalize，配合 IndexFlatIP 表示 cosine similarity。"""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Embedding 结果必须是非空二维向量")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding 结果包含零向量，无法 L2 normalize")
    return matrix / norms


def build_faiss_index(
    *,
    input_path: str | Path,
    source: KnowledgeSource,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, int, int]:
    """构建并写出 index.faiss + chunks.json，返回产物路径、chunk 数和向量维度。"""
    import faiss

    chunks = KnowledgeBuilder(input_path, source).build()
    if not chunks:
        raise ValueError("知识源未生成任何 DocumentChunk")

    embedder = _build_embedder()
    vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    matrix = _normalize_vectors(vectors)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / INDEX_FILE_NAME
    chunks_path = output / CHUNKS_FILE_NAME

    # FAISS row 顺序与 chunks.json 数组顺序严格一致，在线加载时按 row id 取同位置 chunk。
    faiss.write_index(index, str(index_path))
    chunks_payload = [chunk.model_dump(mode="json") for chunk in chunks]
    chunks_path.write_text(
        json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_path, chunks_path, len(chunks), matrix.shape[1]


def load_chunks(chunks_path: str | Path) -> list[DocumentChunk]:
    """从 chunks.json 还原 DocumentChunk，供离线测试校验产物结构。"""
    payload = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    return [DocumentChunk.model_validate(item) for item in payload]


def main() -> None:
    args = _parse_args()
    source = KnowledgeSource(id=args.source_id, title=args.source_title, url=args.source_url)
    index_path, chunks_path, chunk_count, dimension = build_faiss_index(
        input_path=args.input,
        source=source,
        output_dir=args.output_dir,
    )
    print("RAG FAISS索引构建完成")
    print(f"index: {index_path}")
    print(f"chunks: {chunks_path}")
    print(f"chunk数量: {chunk_count}")
    print(f"向量维度: {dimension}")


if __name__ == "__main__":
    main()
