"""Deterministic tests for the offline RAG FAISS index builder."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

import scripts.build_rag_faiss_index as builder
from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.index import (
    CHUNKS_FILE_NAME,
    INDEX_FILE_NAME,
    MANIFEST_FILE_NAME,
    load_faiss_index_bundle,
)
from services.rag_mcp.schemas import KnowledgeSource
from shared import EmbeddingConfig

_DIMS = 4
_MODEL = "deterministic-test-embedding"


class _FakeEmbeddings(Embeddings):
    def __init__(
        self,
        vectors: list[list[float]],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self._vectors = vectors
        self._position = 0
        self._fail_on_call = fail_on_call
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        if self._fail_on_call == len(self.document_calls):
            raise RuntimeError("simulated embedding batch failure")
        start = self._position
        self._position += len(texts)
        return self._vectors[start : self._position]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("The offline builder must not call embed_query")


def _source() -> KnowledgeSource:
    return KnowledgeSource(id="policy", title="Employee policy")


def _config(*, dims: int = _DIMS, api_key: str = "") -> EmbeddingConfig:
    return EmbeddingConfig(
        enabled=True,
        provider="openai_compatible",
        model=_MODEL,
        dims=dims,
        api_key=api_key,
    )


def _write_knowledge_file(path: Path, marker: str = "A") -> Path:
    knowledge_path = path / f"knowledge-{marker}.txt"
    knowledge_path.write_text(
        f"{marker}" * 80 + "\n\n" + f"{marker.lower()}" * 80,
        encoding="utf-8",
    )
    return knowledge_path


def _write_many_chunk_knowledge_file(path: Path, count: int = 74) -> Path:
    knowledge_path = path / "many-chunks.txt"
    paragraphs = [f"{index:02d}-" + chr(65 + index % 26) * 77 for index in range(count)]
    knowledge_path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return knowledge_path


def _vectors(count: int, dims: int = _DIMS) -> list[list[float]]:
    vectors: list[list[float]] = []
    for row in range(count):
        vector = [0.1] * dims
        vector[row % dims] = 1.0
        vectors.append(vector)
    return vectors


def _chunk_count(input_path: Path) -> int:
    return len(KnowledgeBuilder(input_path, _source()).build())


def _build(
    *,
    input_path: Path,
    output_dir: Path,
    vectors: list[list[float]],
    config: EmbeddingConfig | None = None,
) -> tuple[Path, Path, int, int]:
    return builder.build_faiss_index(
        input_path=input_path,
        source=_source(),
        output_dir=output_dir,
        embedding_config=config or _config(),
        embedder=_FakeEmbeddings(vectors),
    )


def _artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        name: (output_dir / name).read_bytes()
        for name in (INDEX_FILE_NAME, CHUNKS_FILE_NAME, MANIFEST_FILE_NAME)
    }


def test_build_generates_a2_compatible_three_file_bundle(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    expected_chunks = KnowledgeBuilder(input_path, _source()).build()
    vectors = _vectors(len(expected_chunks))
    output_dir = tmp_path / "index"

    index_path, chunks_path, chunk_count, dimension = _build(
        input_path=input_path,
        output_dir=output_dir,
        vectors=vectors,
    )
    bundle = load_faiss_index_bundle(output_dir, _config())

    assert index_path == output_dir / INDEX_FILE_NAME
    assert chunks_path == output_dir / CHUNKS_FILE_NAME
    assert (output_dir / MANIFEST_FILE_NAME).is_file()
    assert chunk_count == len(expected_chunks) == bundle.index.ntotal
    assert dimension == _DIMS == bundle.index.d
    assert bundle.chunks == expected_chunks
    assert isinstance(bundle.index, faiss.IndexFlatIP)
    assert bundle.index.metric_type == faiss.METRIC_INNER_PRODUCT

    for row, vector in enumerate(vectors):
        expected = np.asarray(vector, dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(bundle.index.reconstruct(row), expected, rtol=1e-6, atol=1e-6)
        assert np.linalg.norm(bundle.index.reconstruct(row)) == pytest.approx(1.0, abs=1e-6)


def test_manifest_matches_embedding_and_file_bytes(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    count = _chunk_count(input_path)
    output_dir = tmp_path / "index"
    _build(input_path=input_path, output_dir=output_dir, vectors=_vectors(count))

    manifest = json.loads((output_dir / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "normalized": True,
        "embedding_provider": "openai_compatible",
        "embedding_model": _MODEL,
        "embedding_dimensions": _DIMS,
        "chunk_count": count,
        "index_sha256": hashlib.sha256((output_dir / INDEX_FILE_NAME).read_bytes()).hexdigest(),
        "chunks_sha256": hashlib.sha256((output_dir / CHUNKS_FILE_NAME).read_bytes()).hexdigest(),
    }


def test_builder_calls_embed_documents_once_in_chunk_order(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    chunks = KnowledgeBuilder(input_path, _source()).build()
    embedder = _FakeEmbeddings(_vectors(len(chunks)))

    builder.build_faiss_index(
        input_path=input_path,
        source=_source(),
        output_dir=tmp_path / "index",
        embedding_config=_config(),
        embedder=embedder,
    )

    assert embedder.document_calls == [[chunk.content for chunk in chunks]]


def test_builder_embeds_74_chunks_in_order_across_eight_bounded_batches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = _write_many_chunk_knowledge_file(tmp_path)
    chunks = KnowledgeBuilder(input_path, _source()).build()
    assert len(chunks) == 74
    embedder = _FakeEmbeddings(_vectors(len(chunks)))

    builder.build_faiss_index(
        input_path=input_path,
        source=_source(),
        output_dir=tmp_path / "index",
        embedding_config=_config(api_key="diagnostic-secret"),
        embedder=embedder,
    )

    assert [len(batch) for batch in embedder.document_calls] == [10, 10, 10, 10, 10, 10, 10, 4]
    assert [text for batch in embedder.document_calls for text in batch] == [
        chunk.content for chunk in chunks
    ]
    stderr = capsys.readouterr().err
    lines = stderr.splitlines()
    assert "[rag-faiss-build] build_start" in lines
    assert "[rag-faiss-build] document_load_complete chunks=74" in lines
    assert "[rag-faiss-build] embedding_client_init_start mode=injected" in lines
    assert "[rag-faiss-build] embedding_client_init_complete mode=injected" in lines
    assert "[rag-faiss-build] embedding_plan chunks=74 batches=8 batch_size=10" in lines
    previous_position = -1
    cumulative = 0
    for batch_number, size in enumerate([10, 10, 10, 10, 10, 10, 10, 4], start=1):
        cumulative += size
        start_marker = f"embedding_batch_start batch={batch_number}/8 size={size}"
        success_marker = f"embedding_batch_success batch={batch_number}/8 size={size}"
        start_position = next(index for index, line in enumerate(lines) if start_marker in line)
        success_position = next(index for index, line in enumerate(lines) if success_marker in line)
        assert previous_position < start_position < success_position
        assert f"returned={size}" in lines[success_position]
        assert f"cumulative={cumulative}" in lines[success_position]
        previous_position = success_position
    assert "[rag-faiss-build] embedding_complete vectors=74" in lines
    assert "[rag-faiss-build] faiss_build_complete rows=74 dimensions=4" in lines
    assert "[rag-faiss-build] staging_validation_complete" in lines
    assert "[rag-faiss-build] publish_complete artifacts=3" in lines
    assert "diagnostic-secret" not in stderr
    assert chunks[0].content not in stderr


def test_openai_embeddings_http_client_receives_bounded_batches() -> None:
    class FakeEmbeddingEndpoint:
        def __init__(self) -> None:
            self.inputs: list[list[str]] = []

        def create(self, *, input: list[str], **_: object) -> dict[str, object]:
            self.inputs.append(input)
            return {
                "data": [
                    {"embedding": vector}
                    for vector in _vectors(len(input))
                ]
            }

    endpoint = FakeEmbeddingEndpoint()
    embedder = builder.OpenAIEmbeddings(
        model=_MODEL,
        dimensions=_DIMS,
        check_embedding_ctx_length=False,
        chunk_size=10,
        timeout=20.0,
        max_retries=0,
        api_key="test-key",
        client=endpoint,
        async_client=object(),
    )
    texts = [f"chunk-{index}" for index in range(74)]

    vectors = builder._embed_documents_in_batches(embedder, texts, batch_size=10)

    assert [len(batch) for batch in endpoint.inputs] == [10, 10, 10, 10, 10, 10, 10, 4]
    assert [text for batch in endpoint.inputs for text in batch] == texts
    assert len(vectors) == len(texts)


def test_embedding_count_mismatch_is_rejected(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    count = _chunk_count(input_path)

    with pytest.raises(builder.EmbeddingBatchError, match="unexpected vector count"):
        _build(
            input_path=input_path,
            output_dir=tmp_path / "index",
            vectors=_vectors(count - 1),
        )


def test_embedding_rows_with_different_dimensions_are_rejected(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    count = _chunk_count(input_path)
    vectors = _vectors(count)
    vectors[-1] = [1.0] * (_DIMS - 1)

    with pytest.raises(ValueError, match="维度不一致"):
        _build(input_path=input_path, output_dir=tmp_path / "index", vectors=vectors)


def test_embedding_dimension_must_match_config(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    count = _chunk_count(input_path)

    with pytest.raises(ValueError, match="实际维度与配置维度不一致"):
        _build(
            input_path=input_path,
            output_dir=tmp_path / "index",
            vectors=_vectors(count),
            config=_config(dims=_DIMS + 1),
        )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_embedding_values_are_rejected(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    input_path = _write_knowledge_file(tmp_path)
    vectors = _vectors(_chunk_count(input_path))
    vectors[0][0] = invalid_value

    with pytest.raises(ValueError, match="NaN 或 Inf"):
        _build(input_path=input_path, output_dir=tmp_path / "index", vectors=vectors)


def test_zero_embedding_vector_is_rejected(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    vectors = _vectors(_chunk_count(input_path))
    vectors[0] = [0.0] * _DIMS

    with pytest.raises(ValueError, match="零向量"):
        _build(input_path=input_path, output_dir=tmp_path / "index", vectors=vectors)


def test_non_numeric_embedding_value_is_rejected(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path)
    vectors = _vectors(_chunk_count(input_path))
    vectors[0][0] = "not-a-number"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="无法转换为 float32"):
        _build(input_path=input_path, output_dir=tmp_path / "index", vectors=vectors)


def test_build_failure_preserves_existing_bundle(tmp_path: Path) -> None:
    input_path = _write_knowledge_file(tmp_path, "A")
    output_dir = tmp_path / "index"
    count = _chunk_count(input_path)
    _build(input_path=input_path, output_dir=output_dir, vectors=_vectors(count))
    original = _artifact_bytes(output_dir)

    invalid_vectors = _vectors(count)
    invalid_vectors[0] = [0.0] * _DIMS
    with pytest.raises(ValueError, match="零向量"):
        _build(
            input_path=input_path,
            output_dir=output_dir,
            vectors=invalid_vectors,
        )

    assert _artifact_bytes(output_dir) == original
    load_faiss_index_bundle(output_dir, _config())
    assert not list(tmp_path.glob(".rag-faiss-build-*"))


def test_embedding_batch_failure_preserves_existing_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old_input = _write_knowledge_file(tmp_path, "A")
    output_dir = tmp_path / "index"
    old_count = _chunk_count(old_input)
    _build(input_path=old_input, output_dir=output_dir, vectors=_vectors(old_count))
    original = _artifact_bytes(output_dir)
    capsys.readouterr()

    input_path = _write_many_chunk_knowledge_file(tmp_path)
    chunks = KnowledgeBuilder(input_path, _source()).build()
    embedder = _FakeEmbeddings(_vectors(len(chunks)), fail_on_call=3)

    with pytest.raises(builder.EmbeddingBatchError, match=r"batch 3 failed \(size=10\)"):
        builder.build_faiss_index(
            input_path=input_path,
            source=_source(),
            output_dir=output_dir,
            embedding_config=_config(api_key="diagnostic-secret"),
            embedder=embedder,
        )

    stderr = capsys.readouterr().err
    assert [len(batch) for batch in embedder.document_calls] == [10, 10, 10]
    assert "embedding_batch_success batch=1/8 size=10" in stderr
    assert "embedding_batch_success batch=2/8 size=10" in stderr
    assert "embedding_batch_start batch=3/8 size=10" in stderr
    assert "embedding_batch_failed batch=3/8 size=10" in stderr
    assert "exception_type=RuntimeError" in stderr
    assert "embedding_batch_start batch=4/8" not in stderr
    assert "simulated embedding batch failure" not in stderr
    assert "diagnostic-secret" not in stderr
    assert chunks[0].content not in stderr
    assert _artifact_bytes(output_dir) == original
    load_faiss_index_bundle(output_dir, _config())
    assert not list(tmp_path.glob(".rag-faiss-build-*"))


def test_publish_failure_restores_existing_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_input = _write_knowledge_file(tmp_path, "A")
    output_dir = tmp_path / "index"
    old_count = _chunk_count(old_input)
    _build(input_path=old_input, output_dir=output_dir, vectors=_vectors(old_count))
    original = _artifact_bytes(output_dir)

    real_replace = builder._replace_artifact
    call_count = 0

    def fail_second_replace(source: Path, target: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated publish failure")
        real_replace(source, target)

    monkeypatch.setattr(builder, "_replace_artifact", fail_second_replace)
    new_input = _write_knowledge_file(tmp_path, "B")
    new_count = _chunk_count(new_input)

    with pytest.raises(builder.IndexPublishError, match="原有产物已恢复"):
        _build(input_path=new_input, output_dir=output_dir, vectors=_vectors(new_count))

    assert _artifact_bytes(output_dir) == original
    load_faiss_index_bundle(output_dir, _config())
    assert not list(tmp_path.glob(".rag-faiss-build-*"))
    assert not list(tmp_path.glob(".rag-faiss-backup-*"))


def test_existing_cli_arguments_remain_compatible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "knowledge.txt"
    output_dir = tmp_path / "index"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_rag_faiss_index.py",
            "--input",
            str(input_path),
            "--source-id",
            "policy",
            "--source-title",
            "Employee policy",
            "--source-url",
            "https://example.test/policy",
            "--output-dir",
            str(output_dir),
        ],
    )

    args = builder._parse_args()

    assert args.input == str(input_path)
    assert args.source_id == "policy"
    assert args.source_title == "Employee policy"
    assert args.source_url == "https://example.test/policy"
    assert args.output_dir == str(output_dir)
    assert args.embedding_batch_size == 10
    assert args.embedding_timeout == 20.0
    assert args.embedding_max_retries == 0


def test_embedding_cli_arguments_accept_explicit_valid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_rag_faiss_index.py",
            "--input",
            "knowledge.txt",
            "--source-id",
            "policy",
            "--source-title",
            "Employee policy",
            "--embedding-batch-size",
            "7",
            "--embedding-timeout",
            "12.5",
            "--embedding-max-retries",
            "1",
        ],
    )

    args = builder._parse_args()

    assert args.embedding_batch_size == 7
    assert args.embedding_timeout == 12.5
    assert args.embedding_max_retries == 1


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--embedding-batch-size", "0"),
        ("--embedding-batch-size", "-1"),
        ("--embedding-batch-size", "true"),
        ("--embedding-batch-size", "1.5"),
        ("--embedding-timeout", "0"),
        ("--embedding-timeout", "-1"),
        ("--embedding-timeout", "nan"),
        ("--embedding-timeout", "inf"),
        ("--embedding-timeout", "false"),
        ("--embedding-max-retries", "-1"),
        ("--embedding-max-retries", "true"),
        ("--embedding-max-retries", "1.5"),
    ],
)
def test_embedding_cli_arguments_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_rag_faiss_index.py",
            "--input",
            "knowledge.txt",
            "--source-id",
            "policy",
            "--source-title",
            "Employee policy",
            flag,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        builder._parse_args()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"batch_size": True, "timeout": 20.0, "max_retries": 0}, "batch_size"),
        ({"batch_size": 10, "timeout": False, "max_retries": 0}, "timeout"),
        ({"batch_size": 10, "timeout": 20.0, "max_retries": True}, "max_retries"),
    ],
)
def test_programmatic_embedding_options_reject_booleans(
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        builder._validate_embedding_options(**options)  # type: ignore[arg-type]


def test_offline_client_receives_public_timeout_retry_and_batch_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = _FakeEmbeddings([])

    def capture_client(**kwargs: object) -> Embeddings:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(builder, "OpenAIEmbeddings", capture_client)

    client = builder._build_offline_embedder(
        _config(),
        batch_size=7,
        timeout=12.5,
        max_retries=1,
    )

    assert client is sentinel
    assert captured["model"] == _MODEL
    assert captured["dimensions"] == _DIMS
    assert captured["check_embedding_ctx_length"] is False
    assert captured["chunk_size"] == 7
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 1
