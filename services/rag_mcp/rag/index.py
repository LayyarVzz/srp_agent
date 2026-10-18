"""Load and validate persisted FAISS artifacts for the RAG MCP service."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError

from services.rag_mcp.rag.models import DocumentChunk
from shared import EmbeddingConfig

INDEX_FILE_NAME = "index.faiss"
CHUNKS_FILE_NAME = "chunks.json"
MANIFEST_FILE_NAME = "manifest.json"
CURRENT_SCHEMA_VERSION = 1

_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_DOCUMENT_CHUNKS = TypeAdapter(list[DocumentChunk])


class RAGIndexError(RuntimeError):
    """Base error for an invalid or unreadable persisted RAG index."""


class IndexArtifactMissingError(RAGIndexError):
    """A required index artifact is missing or is not a regular file."""


class IndexArtifactReadError(RAGIndexError):
    """A required index artifact cannot be read from disk."""


class IndexJsonDecodeError(RAGIndexError):
    """A JSON index artifact is not valid UTF-8 JSON."""


class IndexManifestValidationError(RAGIndexError):
    """The index manifest does not satisfy the supported schema."""


class IndexChunkValidationError(RAGIndexError):
    """The persisted chunk payload cannot be restored as DocumentChunk objects."""


class DuplicateChunkIdError(IndexChunkValidationError):
    """Persisted chunks contain an empty or duplicate chunk identifier."""


class IndexChecksumMismatchError(RAGIndexError):
    """An artifact digest differs from the digest recorded in the manifest."""


class FaissIndexReadError(RAGIndexError):
    """The FAISS index cannot be deserialized."""


class FaissIndexValidationError(RAGIndexError):
    """The loaded FAISS index is not the supported IndexFlatIP layout."""


class IndexDimensionMismatchError(RAGIndexError):
    """FAISS, manifest, and runtime embedding dimensions do not agree."""


class EmbeddingConfigMismatchError(RAGIndexError):
    """The runtime embedding provider or model differs from the build manifest."""


class FaissIndexManifest(BaseModel):
    """Versioned contract shared by the offline builder and online loader.

    Schema version 1 supports only an L2-normalized ``IndexFlatIP`` index. The
    two SHA-256 values cover the exact bytes of the separately stored FAISS and
    chunk artifacts. Unknown fields are rejected so a newer manifest cannot be
    silently interpreted with older rules.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    index_type: Literal["IndexFlatIP"]
    metric: Literal["inner_product"]
    normalized: Literal[True]
    embedding_provider: Literal["openai", "openai_compatible"]
    embedding_model: _NonEmptyString
    embedding_dimensions: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    index_sha256: _Sha256Hex
    chunks_sha256: _Sha256Hex


@dataclass(frozen=True, slots=True)
class FaissIndexBundle:
    """A validated FAISS index and its row-aligned document chunks."""

    index: Any
    chunks: list[DocumentChunk]
    manifest: FaissIndexManifest


def _require_regular_file(path: Path) -> None:
    if not path.is_file():
        raise IndexArtifactMissingError(
            f"Required RAG index artifact is missing or not a regular file: {path.name}"
        )


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IndexArtifactReadError(f"Unable to read RAG index artifact: {path.name}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IndexJsonDecodeError(f"Invalid JSON in RAG index artifact: {path.name}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            while block := artifact.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise IndexArtifactReadError(f"Unable to read RAG index artifact: {path.name}") from exc
    return digest.hexdigest()


def _load_manifest(path: Path) -> FaissIndexManifest:
    payload = _read_json(path)
    try:
        return FaissIndexManifest.model_validate(payload)
    except ValidationError as exc:
        raise IndexManifestValidationError(
            f"RAG index manifest does not match schema version {CURRENT_SCHEMA_VERSION}"
        ) from exc


def _validate_checksum(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise IndexChecksumMismatchError(f"SHA-256 mismatch for RAG index artifact: {path.name}")


def _load_chunks(path: Path) -> list[DocumentChunk]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise IndexChunkValidationError("chunks.json must contain a JSON array")

    try:
        chunks = _DOCUMENT_CHUNKS.validate_python(payload)
    except ValidationError as exc:
        raise IndexChunkValidationError(
            "chunks.json contains an invalid DocumentChunk entry"
        ) from exc

    seen_ids: set[str] = set()
    for chunk in chunks:
        if not chunk.chunk_id.strip():
            raise DuplicateChunkIdError("chunks.json contains an empty chunk_id")
        if chunk.chunk_id in seen_ids:
            raise DuplicateChunkIdError(
                f"chunks.json contains a duplicate chunk_id: {chunk.chunk_id}"
            )
        seen_ids.add(chunk.chunk_id)
    return chunks


def _load_faiss_index(path: Path) -> tuple[Any, Any]:
    import faiss

    try:
        index = faiss.read_index(str(path))
    except Exception as exc:  # FAISS exposes deserialization failures as RuntimeError subclasses.
        raise FaissIndexReadError(f"Unable to read FAISS index artifact: {path.name}") from exc
    return faiss, index


def _validate_embedding_config(
    manifest: FaissIndexManifest,
    embedding_config: EmbeddingConfig,
) -> None:
    if manifest.embedding_provider != embedding_config.provider:
        raise EmbeddingConfigMismatchError(
            "Embedding provider differs between manifest and runtime configuration"
        )
    if manifest.embedding_model != embedding_config.model:
        raise EmbeddingConfigMismatchError(
            "Embedding model differs between manifest and runtime configuration"
        )


def _validate_faiss_layout(
    *,
    faiss: Any,
    index: Any,
    chunks: list[DocumentChunk],
    manifest: FaissIndexManifest,
    embedding_config: EmbeddingConfig,
) -> None:
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise FaissIndexValidationError("FAISS index metric must be inner product")
    if not isinstance(index, faiss.IndexFlatIP):
        raise FaissIndexValidationError("FAISS index type must be IndexFlatIP")

    if index.ntotal != len(chunks) or index.ntotal != manifest.chunk_count:
        raise FaissIndexValidationError(
            "FAISS row count, chunks.json length, and manifest chunk_count must match"
        )

    if index.d != manifest.embedding_dimensions or index.d != embedding_config.dims:
        raise IndexDimensionMismatchError(
            "FAISS, manifest, and runtime embedding dimensions must match"
        )


def load_faiss_index_bundle(
    index_dir: str | Path,
    embedding_config: EmbeddingConfig,
) -> FaissIndexBundle:
    """Load a complete, validated FAISS artifact bundle without embedding data.

    The loader performs no document splitting and makes no embedding API calls.
    Loading fails closed: invalid artifacts never trigger a silent BM25 fallback.
    """

    directory = Path(index_dir)
    index_path = directory / INDEX_FILE_NAME
    chunks_path = directory / CHUNKS_FILE_NAME
    manifest_path = directory / MANIFEST_FILE_NAME

    for path in (index_path, chunks_path, manifest_path):
        _require_regular_file(path)

    manifest = _load_manifest(manifest_path)
    _validate_embedding_config(manifest, embedding_config)
    _validate_checksum(index_path, manifest.index_sha256)
    _validate_checksum(chunks_path, manifest.chunks_sha256)

    chunks = _load_chunks(chunks_path)
    faiss, index = _load_faiss_index(index_path)
    _validate_faiss_layout(
        faiss=faiss,
        index=index,
        chunks=chunks,
        manifest=manifest,
        embedding_config=embedding_config,
    )
    return FaissIndexBundle(index=index, chunks=chunks, manifest=manifest)


__all__ = [
    "CHUNKS_FILE_NAME",
    "CURRENT_SCHEMA_VERSION",
    "INDEX_FILE_NAME",
    "MANIFEST_FILE_NAME",
    "DuplicateChunkIdError",
    "EmbeddingConfigMismatchError",
    "FaissIndexBundle",
    "FaissIndexManifest",
    "FaissIndexReadError",
    "FaissIndexValidationError",
    "IndexArtifactMissingError",
    "IndexArtifactReadError",
    "IndexChecksumMismatchError",
    "IndexChunkValidationError",
    "IndexDimensionMismatchError",
    "IndexJsonDecodeError",
    "IndexManifestValidationError",
    "RAGIndexError",
    "load_faiss_index_bundle",
]
