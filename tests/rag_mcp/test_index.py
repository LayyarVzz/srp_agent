"""Deterministic tests for loading and validating RAG FAISS artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pytest

from services.rag_mcp.rag.index import (
    CHUNKS_FILE_NAME,
    INDEX_FILE_NAME,
    MANIFEST_FILE_NAME,
    DuplicateChunkIdError,
    EmbeddingConfigMismatchError,
    FaissIndexReadError,
    FaissIndexValidationError,
    IndexArtifactMissingError,
    IndexChecksumMismatchError,
    IndexChunkValidationError,
    IndexDimensionMismatchError,
    IndexJsonDecodeError,
    IndexManifestValidationError,
    load_faiss_index_bundle,
)
from shared import EmbeddingConfig

_DIMS = 3
_MODEL = "test-embedding-model"
_PROVIDER = "openai_compatible"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunks_payload() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "policy:0",
            "content": "Annual leave policy",
            "source": {"id": "policy", "title": "Employee policy", "url": None},
            "metadata": {"chunk_index": 0},
        },
        {
            "chunk_id": "policy:1",
            "content": "Sick leave policy",
            "source": {"id": "policy", "title": "Employee policy", "url": None},
            "metadata": {"chunk_index": 1},
        },
    ]


def _embedding_config(
    *,
    provider: str = _PROVIDER,
    model: str = _MODEL,
    dims: int = _DIMS,
) -> EmbeddingConfig:
    return EmbeddingConfig(enabled=True, provider=provider, model=model, dims=dims)


def _read_manifest(index_dir: Path) -> dict[str, Any]:
    return json.loads((index_dir / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, payload: dict[str, Any]) -> None:
    (index_dir / MANIFEST_FILE_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _refresh_hash(index_dir: Path, artifact_name: str) -> None:
    manifest = _read_manifest(index_dir)
    manifest[f"{artifact_name.split('.')[0]}_sha256"] = _sha256(index_dir / artifact_name)
    _write_manifest(index_dir, manifest)


def _write_index(index_dir: Path, index: Any) -> None:
    faiss.write_index(index, str(index_dir / INDEX_FILE_NAME))


def _create_bundle(index_dir: Path) -> EmbeddingConfig:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = index_dir / CHUNKS_FILE_NAME
    chunks_path.write_text(
        json.dumps(_chunks_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index = faiss.IndexFlatIP(_DIMS)
    index.add(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    _write_index(index_dir, index)

    _write_manifest(
        index_dir,
        {
            "schema_version": 1,
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
            "normalized": True,
            "embedding_provider": _PROVIDER,
            "embedding_model": _MODEL,
            "embedding_dimensions": _DIMS,
            "chunk_count": 2,
            "index_sha256": _sha256(index_dir / INDEX_FILE_NAME),
            "chunks_sha256": _sha256(chunks_path),
        },
    )
    return _embedding_config()


def test_load_valid_faiss_index_bundle(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)

    bundle = load_faiss_index_bundle(tmp_path, config)

    assert isinstance(bundle.index, faiss.IndexFlatIP)
    assert bundle.index.ntotal == 2
    assert bundle.index.d == _DIMS
    assert [chunk.chunk_id for chunk in bundle.chunks] == ["policy:0", "policy:1"]
    assert bundle.manifest.embedding_model == _MODEL
    assert bundle.manifest.normalized is True


@pytest.mark.parametrize("artifact_name", [INDEX_FILE_NAME, CHUNKS_FILE_NAME, MANIFEST_FILE_NAME])
def test_missing_artifact_is_rejected(tmp_path: Path, artifact_name: str) -> None:
    config = _create_bundle(tmp_path)
    (tmp_path / artifact_name).unlink()

    with pytest.raises(IndexArtifactMissingError, match=artifact_name):
        load_faiss_index_bundle(tmp_path, config)


def test_non_regular_artifact_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    manifest_path = tmp_path / MANIFEST_FILE_NAME
    manifest_path.unlink()
    manifest_path.mkdir()

    with pytest.raises(IndexArtifactMissingError, match=MANIFEST_FILE_NAME):
        load_faiss_index_bundle(tmp_path, config)


def test_invalid_manifest_json_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    (tmp_path / MANIFEST_FILE_NAME).write_text("{invalid", encoding="utf-8")

    with pytest.raises(IndexJsonDecodeError, match=MANIFEST_FILE_NAME):
        load_faiss_index_bundle(tmp_path, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("index_type", "IndexFlatL2"),
        ("metric", "cosine"),
        ("normalized", False),
        ("embedding_dimensions", 0),
        ("index_sha256", "not-a-sha256"),
    ],
)
def test_invalid_manifest_schema_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    config = _create_bundle(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest[field] = value
    _write_manifest(tmp_path, manifest)

    with pytest.raises(IndexManifestValidationError):
        load_faiss_index_bundle(tmp_path, config)


def test_unknown_manifest_field_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["future_field"] = "unsupported"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(IndexManifestValidationError):
        load_faiss_index_bundle(tmp_path, config)


def test_invalid_chunks_json_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    (tmp_path / CHUNKS_FILE_NAME).write_text("[invalid", encoding="utf-8")
    _refresh_hash(tmp_path, CHUNKS_FILE_NAME)

    with pytest.raises(IndexJsonDecodeError, match=CHUNKS_FILE_NAME):
        load_faiss_index_bundle(tmp_path, config)


def test_non_array_chunks_payload_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    (tmp_path / CHUNKS_FILE_NAME).write_text("{}", encoding="utf-8")
    _refresh_hash(tmp_path, CHUNKS_FILE_NAME)

    with pytest.raises(IndexChunkValidationError, match="JSON array"):
        load_faiss_index_bundle(tmp_path, config)


def test_invalid_document_chunk_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    chunks = _chunks_payload()
    chunks[0]["source"] = "not-a-source"
    (tmp_path / CHUNKS_FILE_NAME).write_text(json.dumps(chunks), encoding="utf-8")
    _refresh_hash(tmp_path, CHUNKS_FILE_NAME)

    with pytest.raises(IndexChunkValidationError, match="DocumentChunk"):
        load_faiss_index_bundle(tmp_path, config)


@pytest.mark.parametrize("chunk_id", ["", "   "])
def test_empty_chunk_id_is_rejected(tmp_path: Path, chunk_id: str) -> None:
    config = _create_bundle(tmp_path)
    chunks = _chunks_payload()
    chunks[0]["chunk_id"] = chunk_id
    (tmp_path / CHUNKS_FILE_NAME).write_text(json.dumps(chunks), encoding="utf-8")
    _refresh_hash(tmp_path, CHUNKS_FILE_NAME)

    with pytest.raises(DuplicateChunkIdError, match="empty chunk_id"):
        load_faiss_index_bundle(tmp_path, config)


def test_duplicate_chunk_id_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    chunks = _chunks_payload()
    chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
    (tmp_path / CHUNKS_FILE_NAME).write_text(json.dumps(chunks), encoding="utf-8")
    _refresh_hash(tmp_path, CHUNKS_FILE_NAME)

    with pytest.raises(DuplicateChunkIdError, match="duplicate chunk_id"):
        load_faiss_index_bundle(tmp_path, config)


@pytest.mark.parametrize("artifact_name", [INDEX_FILE_NAME, CHUNKS_FILE_NAME])
def test_checksum_mismatch_is_rejected(tmp_path: Path, artifact_name: str) -> None:
    config = _create_bundle(tmp_path)
    with (tmp_path / artifact_name).open("ab") as artifact:
        artifact.write(b"tampered")

    with pytest.raises(IndexChecksumMismatchError, match=artifact_name):
        load_faiss_index_bundle(tmp_path, config)


def test_corrupt_faiss_index_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    (tmp_path / INDEX_FILE_NAME).write_bytes(b"not a FAISS index")
    _refresh_hash(tmp_path, INDEX_FILE_NAME)

    with pytest.raises(FaissIndexReadError, match=INDEX_FILE_NAME):
        load_faiss_index_bundle(tmp_path, config)


def test_wrong_faiss_metric_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    index = faiss.IndexFlatL2(_DIMS)
    index.add(np.zeros((2, _DIMS), dtype=np.float32))
    _write_index(tmp_path, index)
    _refresh_hash(tmp_path, INDEX_FILE_NAME)

    with pytest.raises(FaissIndexValidationError, match="metric"):
        load_faiss_index_bundle(tmp_path, config)


def test_wrong_faiss_type_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(_DIMS))
    index.add_with_ids(
        np.zeros((2, _DIMS), dtype=np.float32),
        np.asarray([10, 11], dtype=np.int64),
    )
    _write_index(tmp_path, index)
    _refresh_hash(tmp_path, INDEX_FILE_NAME)

    with pytest.raises(FaissIndexValidationError, match="type"):
        load_faiss_index_bundle(tmp_path, config)


@pytest.mark.parametrize("manifest_count", [1, 3])
def test_manifest_chunk_count_mismatch_is_rejected(
    tmp_path: Path,
    manifest_count: int,
) -> None:
    config = _create_bundle(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["chunk_count"] = manifest_count
    _write_manifest(tmp_path, manifest)

    with pytest.raises(FaissIndexValidationError, match="row count"):
        load_faiss_index_bundle(tmp_path, config)


def test_faiss_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    index = faiss.IndexFlatIP(_DIMS)
    index.add(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    _write_index(tmp_path, index)
    _refresh_hash(tmp_path, INDEX_FILE_NAME)

    with pytest.raises(FaissIndexValidationError, match="row count"):
        load_faiss_index_bundle(tmp_path, config)


def test_faiss_and_manifest_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _create_bundle(tmp_path)
    index = faiss.IndexFlatIP(4)
    index.add(np.zeros((2, 4), dtype=np.float32))
    _write_index(tmp_path, index)
    _refresh_hash(tmp_path, INDEX_FILE_NAME)

    with pytest.raises(IndexDimensionMismatchError):
        load_faiss_index_bundle(tmp_path, config)


def test_runtime_embedding_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    _create_bundle(tmp_path)

    with pytest.raises(IndexDimensionMismatchError):
        load_faiss_index_bundle(tmp_path, _embedding_config(dims=4))


@pytest.mark.parametrize(
    "config",
    [
        _embedding_config(provider="openai"),
        _embedding_config(model="different-model"),
    ],
)
def test_runtime_embedding_identity_mismatch_is_rejected(
    tmp_path: Path,
    config: EmbeddingConfig,
) -> None:
    _create_bundle(tmp_path)

    with pytest.raises(EmbeddingConfigMismatchError):
        load_faiss_index_bundle(tmp_path, config)
