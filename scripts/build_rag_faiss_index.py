"""离线构建 RAG FAISS 索引。

本脚本是文档 Embedding 的唯一执行边界：读取单个 TXT/Markdown 知识源，经
KnowledgeBuilder 生成 DocumentChunk，再用 shared Embedding 底座生成向量并写入
预构建索引。RAG MCP 在线启动阶段只应加载产物，不应重新 embed 全量文档。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.index import (
    CHUNKS_FILE_NAME,
    CURRENT_SCHEMA_VERSION,
    INDEX_FILE_NAME,
    MANIFEST_FILE_NAME,
    FaissIndexManifest,
    load_faiss_index_bundle,
)
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.schemas import KnowledgeSource
from settings import get_settings
from shared import EmbeddingConfig

DEFAULT_OUTPUT_DIR = Path("services/rag_mcp/knowledge/index")
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_EMBEDDING_TIMEOUT = 20.0
DEFAULT_EMBEDDING_MAX_RETRIES = 0
_ARTIFACT_FILE_NAMES = (INDEX_FILE_NAME, CHUNKS_FILE_NAME, MANIFEST_FILE_NAME)


class IndexPublishError(RuntimeError):
    """A staged index bundle could not be published or safely verified."""


class EmbeddingBatchError(RuntimeError):
    """A bounded document-embedding batch failed or returned an invalid count."""


def _diagnostic(message: str) -> None:
    """Emit one immediately visible, non-sensitive offline-build diagnostic event."""
    print(f"[rag-faiss-build] {message}", file=sys.stderr, flush=True)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline FAISS index for RAG knowledge.")
    parser.add_argument("--input", required=True, help="Single .txt/.md/.markdown knowledge file.")
    parser.add_argument("--source-id", required=True, help="Knowledge source id.")
    parser.add_argument("--source-title", required=True, help="Knowledge source title.")
    parser.add_argument("--source-url", default=None, help="Optional source URL.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for index.faiss, chunks.json, and manifest.json.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=_positive_int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Maximum texts per embedding request (default: 10).",
    )
    parser.add_argument(
        "--embedding-timeout",
        type=_positive_finite_float,
        default=DEFAULT_EMBEDDING_TIMEOUT,
        help="Embedding request timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--embedding-max-retries",
        type=_non_negative_int,
        default=DEFAULT_EMBEDDING_MAX_RETRIES,
        help="Maximum embedding request retries (default: 0).",
    )
    return parser.parse_args()


def _build_embedding_config() -> EmbeddingConfig:
    """按运行环境构造 shared EmbeddingConfig。

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
    return config


def _validate_embedding_options(
    *,
    batch_size: int,
    timeout: float,
    max_retries: int,
) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("embedding_batch_size must be a positive integer")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("embedding_timeout must be a positive finite number")
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError("embedding_max_retries must be a non-negative integer")


def _build_offline_embedder(
    config: EmbeddingConfig,
    *,
    batch_size: int,
    timeout: float,
    max_retries: int,
) -> Embeddings:
    """Build a bounded offline client using only public OpenAIEmbeddings options.

    The shared factory intentionally keeps common runtime defaults and does not expose
    request timeout/retry overrides. Reconstructing the same public configuration here
    avoids mutating private SDK clients while keeping the shared configuration contract.
    """
    _validate_embedding_options(
        batch_size=batch_size,
        timeout=timeout,
        max_retries=max_retries,
    )
    kwargs: dict[str, object] = {
        "model": config.model,
        "check_embedding_ctx_length": config.check_embedding_ctx_length,
        "chunk_size": batch_size,
        "timeout": float(timeout),
        "max_retries": max_retries,
    }
    if config.pass_dimensions:
        kwargs["dimensions"] = config.dims
    if config.base_url:
        kwargs["base_url"] = config.base_url
    api_key = config.api_key.get_secret_value()
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


def _embed_documents_in_batches(
    embedder: Embeddings,
    texts: list[str],
    *,
    batch_size: int,
) -> list[list[float]]:
    """Embed texts serially in bounded batches while preserving input order."""
    vectors: list[list[float]] = []
    total_batches = math.ceil(len(texts) / batch_size)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_number = start // batch_size + 1
        _diagnostic(
            f"embedding_batch_start batch={batch_number}/{total_batches} size={len(batch)}"
        )
        started_at = time.perf_counter()
        try:
            batch_vectors = embedder.embed_documents(batch)
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            _diagnostic(
                "embedding_batch_failed "
                f"batch={batch_number}/{total_batches} size={len(batch)} "
                f"elapsed_seconds={elapsed:.3f} exception_type={type(exc).__name__}"
            )
            raise EmbeddingBatchError(
                f"Embedding batch {batch_number} failed (size={len(batch)})"
            ) from exc
        if len(batch_vectors) != len(batch):
            elapsed = time.perf_counter() - started_at
            _diagnostic(
                "embedding_batch_failed "
                f"batch={batch_number}/{total_batches} size={len(batch)} "
                f"elapsed_seconds={elapsed:.3f} exception_type=EmbeddingBatchError"
            )
            raise EmbeddingBatchError(
                "Embedding batch returned an unexpected vector count: "
                f"batch={batch_number}, expected={len(batch)}, actual={len(batch_vectors)}"
            )
        vectors.extend(batch_vectors)
        elapsed = time.perf_counter() - started_at
        _diagnostic(
            "embedding_batch_success "
            f"batch={batch_number}/{total_batches} size={len(batch)} "
            f"elapsed_seconds={elapsed:.3f} returned={len(batch_vectors)} "
            f"cumulative={len(vectors)}"
        )
    return vectors


def _normalize_vectors(
    vectors: list[list[float]],
    *,
    expected_count: int,
    expected_dimension: int,
) -> np.ndarray:
    """Validate, convert, and L2-normalize document embeddings for IndexFlatIP."""
    if len(vectors) != expected_count:
        raise ValueError(
            f"Embedding 向量数量与 chunk 数量不一致: {len(vectors)} != {expected_count}"
        )
    if not vectors:
        raise ValueError("Embedding 结果必须是非空二维向量")

    try:
        row_dimensions = [len(vector) for vector in vectors]
    except TypeError as exc:
        raise ValueError("Embedding 结果必须是二维向量") from exc
    if not row_dimensions or row_dimensions[0] <= 0:
        raise ValueError("Embedding 向量维度必须大于零")
    if any(dimension != row_dimensions[0] for dimension in row_dimensions):
        raise ValueError("Embedding 结果包含维度不一致的向量")
    if row_dimensions[0] != expected_dimension:
        raise ValueError(
            "Embedding 实际维度与配置维度不一致: "
            f"{row_dimensions[0]} != {expected_dimension}"
        )

    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Embedding 结果无法转换为 float32") from exc
    if matrix.ndim != 2 or matrix.shape != (expected_count, expected_dimension):
        raise ValueError("Embedding 结果必须是形状正确的二维向量")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embedding 结果包含 NaN 或 Inf")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding 结果包含零向量，无法 L2 normalize")
    normalized = np.ascontiguousarray(matrix / norms, dtype=np.float32)
    normalized_norms = np.linalg.norm(normalized, axis=1)
    if not np.allclose(normalized_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Embedding 向量 L2 normalize 后未达到单位长度")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while block := artifact.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_staged_bundle(
    *,
    staging_dir: Path,
    index: object,
    chunks: list[DocumentChunk],
    embedding_config: EmbeddingConfig,
) -> None:
    import faiss

    index_path = staging_dir / INDEX_FILE_NAME
    chunks_path = staging_dir / CHUNKS_FILE_NAME
    manifest_path = staging_dir / MANIFEST_FILE_NAME

    faiss.write_index(index, str(index_path))
    chunks_payload = [chunk.model_dump(mode="json") for chunk in chunks]
    chunks_path.write_bytes(
        (json.dumps(chunks_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )

    manifest = FaissIndexManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        index_type="IndexFlatIP",
        metric="inner_product",
        normalized=True,
        embedding_provider=embedding_config.provider,
        embedding_model=embedding_config.model,
        embedding_dimensions=embedding_config.dims,
        chunk_count=len(chunks),
        index_sha256=_sha256(index_path),
        chunks_sha256=_sha256(chunks_path),
    )
    manifest_path.write_bytes(
        (json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
    )


def _replace_artifact(source: Path, target: Path) -> None:
    """Atomically replace one file when the OS and filesystem support it."""
    os.replace(source, target)


def _publish_bundle(
    *,
    staging_dir: Path,
    output_dir: Path,
    embedding_config: EmbeddingConfig,
) -> None:
    """Publish three validated files with best-effort rollback on ordinary failures.

    Individual ``os.replace`` calls are atomic, but the three-file publication is
    not a filesystem transaction. Publishing the manifest last prevents a mixed
    generation from passing A2 validation. Process or machine interruption can
    still leave a temporarily invalid target bundle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(
        tempfile.mkdtemp(prefix=".rag-faiss-backup-", dir=str(output_dir.parent))
    )
    backed_up: set[str] = set()
    published: list[str] = []
    keep_backup = False

    try:
        for name in _ARTIFACT_FILE_NAMES:
            target = output_dir / name
            if not target.exists():
                continue
            if not target.is_file():
                raise IndexPublishError(f"目标产物路径不是普通文件: {target}")
            shutil.copy2(target, backup_dir / name)
            backed_up.add(name)

        try:
            # manifest 最后替换；旧 manifest 不会认可只发布了一部分的新文件。
            for name in _ARTIFACT_FILE_NAMES:
                _replace_artifact(staging_dir / name, output_dir / name)
                published.append(name)
            load_faiss_index_bundle(output_dir, embedding_config)
        except Exception as exc:
            rollback_errors: list[str] = []
            for name in reversed(published):
                target = output_dir / name
                try:
                    if name in backed_up:
                        os.replace(backup_dir / name, target)
                    elif target.exists():
                        target.unlink()
                except OSError:
                    rollback_errors.append(name)

            if rollback_errors:
                keep_backup = True
                names = ", ".join(rollback_errors)
                raise IndexPublishError(
                    "FAISS 索引发布失败且回滚不完整；"
                    f"未恢复: {names}；备份保留在: {backup_dir}"
                ) from exc
            raise IndexPublishError("FAISS 索引发布失败，原有产物已恢复") from exc
    finally:
        if not keep_backup:
            shutil.rmtree(backup_dir, ignore_errors=True)


def build_faiss_index(
    *,
    input_path: str | Path,
    source: KnowledgeSource,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    embedding_config: EmbeddingConfig | None = None,
    embedder: Embeddings | None = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    embedding_timeout: float = DEFAULT_EMBEDDING_TIMEOUT,
    embedding_max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES,
) -> tuple[Path, Path, int, int]:
    """Build, validate, and safely publish a schema-v1 FAISS artifact bundle."""
    import faiss

    _diagnostic("build_start")
    _validate_embedding_options(
        batch_size=embedding_batch_size,
        timeout=embedding_timeout,
        max_retries=embedding_max_retries,
    )

    chunks = KnowledgeBuilder(input_path, source).build()
    _diagnostic(f"document_load_complete chunks={len(chunks)}")
    if not chunks:
        raise ValueError("知识源未生成任何 DocumentChunk")

    config = embedding_config or _build_embedding_config()
    client_mode = "injected" if embedder is not None else "runtime"
    _diagnostic(f"embedding_client_init_start mode={client_mode}")
    selected_embedder = embedder or _build_offline_embedder(
        config,
        batch_size=embedding_batch_size,
        timeout=embedding_timeout,
        max_retries=embedding_max_retries,
    )
    _diagnostic(f"embedding_client_init_complete mode={client_mode}")
    expected_batches = math.ceil(len(chunks) / embedding_batch_size)
    _diagnostic(
        f"embedding_plan chunks={len(chunks)} batches={expected_batches} "
        f"batch_size={embedding_batch_size}"
    )
    vectors = _embed_documents_in_batches(
        selected_embedder,
        [chunk.content for chunk in chunks],
        batch_size=embedding_batch_size,
    )
    _diagnostic(f"embedding_complete vectors={len(vectors)}")
    matrix = _normalize_vectors(
        vectors,
        expected_count=len(chunks),
        expected_dimension=config.dims,
    )

    _diagnostic("faiss_build_start")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    if not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("FAISS 索引类型不是 IndexFlatIP")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS 索引 metric 不是 inner product")
    _diagnostic(f"faiss_build_complete rows={index.ntotal} dimensions={index.d}")

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = output / INDEX_FILE_NAME
    chunks_path = output / CHUNKS_FILE_NAME

    with tempfile.TemporaryDirectory(
        prefix=".rag-faiss-build-",
        dir=str(output.parent),
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        _write_staged_bundle(
            staging_dir=staging_dir,
            index=index,
            chunks=chunks,
            embedding_config=config,
        )
        # A2 is the single compatibility authority for both staged and published artifacts.
        load_faiss_index_bundle(staging_dir, config)
        _diagnostic("staging_validation_complete")
        _diagnostic("publish_start")
        _publish_bundle(
            staging_dir=staging_dir,
            output_dir=output,
            embedding_config=config,
        )
        _diagnostic("publish_complete artifacts=3")

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
        embedding_batch_size=args.embedding_batch_size,
        embedding_timeout=args.embedding_timeout,
        embedding_max_retries=args.embedding_max_retries,
    )
    print("RAG FAISS索引构建完成")
    print(f"index: {index_path}")
    print(f"chunks: {chunks_path}")
    print(f"manifest: {Path(args.output_dir) / MANIFEST_FILE_NAME}")
    print(f"chunk数量: {chunk_count}")
    print(f"向量维度: {dimension}")


if __name__ == "__main__":
    main()
