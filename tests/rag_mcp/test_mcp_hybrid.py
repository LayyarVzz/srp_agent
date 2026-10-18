"""Offline production-assembly tests for the Hybrid RAG MCP tool."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import faiss
import numpy as np
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

import services.rag_mcp.rag.index as index_module
from services.rag_mcp.config import DEFAULT_RAG_INDEX_DIR, RAGSettings, get_rag_settings
from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.index import (
    CHUNKS_FILE_NAME,
    INDEX_FILE_NAME,
    MANIFEST_FILE_NAME,
    FaissIndexBundle,
    FaissIndexManifest,
    IndexArtifactMissingError,
)
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.pipeline import RAGPipeline
from services.rag_mcp.rag.retriever import (
    HybridRetrievalError,
    HybridRetriever,
    QueryEmbeddingError,
)
from services.rag_mcp.schemas import KnowledgeSource
from settings import get_settings
from shared import EmbeddingConfig

_MODEL = "fake-embedding"
_DIMS = 2
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYNTHETIC_SECRET = "SYNTHETIC_PROVIDER_SECRET_DO_NOT_LOG"

_FAILING_STDIO_SERVER = """
import importlib
import os

import langchain_openai
from langchain_core.embeddings import Embeddings

_SECRET = os.environ["RAG_TEST_PROVIDER_SECRET"]


class _FailingEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise AssertionError("online RAG must not embed documents")

    def embed_query(self, text):
        raise RuntimeError(_SECRET)


def _build_failing_query_embedder(**kwargs):
    return _FailingEmbeddings()


langchain_openai.OpenAIEmbeddings = _build_failing_query_embedder
search_module = importlib.import_module("services.rag_mcp.tools.search")


def _fail_bm25(query, top_k=5):
    raise RuntimeError(_SECRET)


search_module._pipeline._retriever._bm25.search = _fail_bm25

from services.rag_mcp.server import mcp

mcp.run()
"""


class _FakeEmbeddings(Embeddings):
    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector
        self.query_calls: list[str] = []
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        raise AssertionError("Online RAG search must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.query_vector


def _chunks() -> list[DocumentChunk]:
    source = KnowledgeSource(id="policy", title="Employee policy")
    return [
        DocumentChunk(
            chunk_id=f"policy:{index}",
            content=content,
            source=source,
            metadata={"chunk_index": index, "label": label},
        )
        for index, (content, label) in enumerate(
            [
                ("Vacation entitlement policy", "vacation"),
                ("Medical reimbursement rules", "medical"),
                ("Travel expense approval", "travel"),
            ]
        )
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(index_dir: Path) -> None:
    index_dir.mkdir(parents=True)
    index = faiss.IndexFlatIP(_DIMS)
    diagonal = 1.0 / np.sqrt(2.0)
    index.add(
        np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [diagonal, diagonal]],
            dtype=np.float32,
        )
    )
    index_path = index_dir / INDEX_FILE_NAME
    chunks_path = index_dir / CHUNKS_FILE_NAME
    faiss.write_index(index, str(index_path))
    chunks_path.write_text(
        json.dumps(
            [chunk.model_dump(mode="json") for chunk in _chunks()],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = FaissIndexManifest(
        schema_version=1,
        index_type="IndexFlatIP",
        metric="inner_product",
        normalized=True,
        embedding_provider="openai_compatible",
        embedding_model=_MODEL,
        embedding_dimensions=_DIMS,
        chunk_count=3,
        index_sha256=_sha256(index_path),
        chunks_sha256=_sha256(chunks_path),
    )
    (index_dir / MANIFEST_FILE_NAME).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _stdio_failure_environment(index_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "RAG_INDEX_DIR": str(index_dir),
            "RAG_CANDIDATE_MULTIPLIER": "2",
            "RAG_MAX_CANDIDATES": "5",
            "RAG_RRF_K": "60",
            "RAG_VECTOR_MIN_SCORE": "0",
            "RAG_QUERY_EMBEDDING_TIMEOUT": "10",
            "RAG_QUERY_EMBEDDING_MAX_RETRIES": "0",
            "EMBEDDING_ENABLED": "true",
            "EMBEDDING_MODEL": _MODEL,
            "EMBEDDING_DIMS": str(_DIMS),
            "EMBEDDING_BASE_URL": "http://unused.test/v1",
            "EMBEDDING_API_KEY": "fake-test-key",
            "FASTMCP_CHECK_FOR_UPDATES": "off",
            "FASTMCP_SHOW_SERVER_BANNER": "false",
            "RAG_TEST_PROVIDER_SECRET": _SYNTHETIC_SECRET,
        }
    )
    return environment


def _remove_rag_tool_modules() -> None:
    for name in (
        "services.rag_mcp.server",
        "services.rag_mcp.tools.search",
        "services.rag_mcp.tools",
    ):
        sys.modules.pop(name, None)


@pytest.fixture
def production_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    index_dir = tmp_path / "index"
    _write_bundle(index_dir)
    monkeypatch.setenv("RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("RAG_CANDIDATE_MULTIPLIER", "2")
    monkeypatch.setenv("RAG_MAX_CANDIDATES", "5")
    monkeypatch.setenv("RAG_RRF_K", "60")
    monkeypatch.setenv("RAG_VECTOR_MIN_SCORE", "0")
    monkeypatch.setenv("EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_MODEL", _MODEL)
    monkeypatch.setenv("EMBEDDING_DIMS", str(_DIMS))
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://unused.test/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "fake-test-key")

    get_settings.cache_clear()
    get_rag_settings.cache_clear()
    fake_embedder = _FakeEmbeddings([1.0, 0.0])
    client_init_calls: list[dict[str, object]] = []

    def _build_fake_client(**kwargs: object) -> Embeddings:
        client_init_calls.append(kwargs)
        return fake_embedder

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", _build_fake_client)

    load_calls: list[Path] = []
    loaded_bundles: list[FaissIndexBundle] = []
    real_loader = index_module.load_faiss_index_bundle

    def _counting_loader(
        directory: str | Path,
        embedding_config: EmbeddingConfig,
    ) -> FaissIndexBundle:
        load_calls.append(Path(directory))
        bundle = real_loader(directory, embedding_config)
        loaded_bundles.append(bundle)
        return bundle

    monkeypatch.setattr(index_module, "load_faiss_index_bundle", _counting_loader)

    def _unexpected_source_build(self: KnowledgeBuilder) -> list[DocumentChunk]:
        raise AssertionError("Production RAG must not load or split source documents")

    monkeypatch.setattr(KnowledgeBuilder, "build", _unexpected_source_build)
    _remove_rag_tool_modules()
    search_module = importlib.import_module("services.rag_mcp.tools.search")

    yield SimpleNamespace(
        search=search_module,
        embedder=fake_embedder,
        load_calls=load_calls,
        bundle=loaded_bundles[0],
        index_dir=index_dir,
        client_init_calls=client_init_calls,
    )

    _remove_rag_tool_modules()
    get_settings.cache_clear()
    get_rag_settings.cache_clear()


def test_rag_settings_defaults() -> None:
    settings = RAGSettings(_env_file=None)

    assert DEFAULT_RAG_INDEX_DIR == Path(
        "services/rag_mcp/knowledge/index/uat_company_policy"
    )
    assert settings.rag_index_dir == DEFAULT_RAG_INDEX_DIR
    assert settings.rag_candidate_multiplier == 3
    assert settings.rag_max_candidates == 50
    assert settings.rag_rrf_k == 60
    assert settings.rag_vector_min_score == 0.0
    assert settings.rag_query_embedding_timeout == 10.0
    assert settings.rag_query_embedding_max_retries == 0


def test_default_query_embedding_client_options_are_forwarded(production_env: Any) -> None:
    assert len(production_env.client_init_calls) == 1
    options = production_env.client_init_calls[0]

    assert options["timeout"] == 10.0
    assert options["max_retries"] == 0
    assert options["model"] == _MODEL
    assert options["dimensions"] == _DIMS
    assert options["check_embedding_ctx_length"] is False


def test_custom_query_embedding_client_options_are_forwarded(
    production_env: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture_client(**kwargs: object) -> Embeddings:
        captured.update(kwargs)
        return production_env.embedder

    monkeypatch.setattr(production_env.search, "OpenAIEmbeddings", _capture_client)
    production_env.search.build_rag_pipeline(
        rag_settings=RAGSettings(
            rag_index_dir=production_env.index_dir,
            rag_query_embedding_timeout=2.5,
            rag_query_embedding_max_retries=1,
        ),
        embedding_config=EmbeddingConfig(
            enabled=True,
            model=_MODEL,
            dims=_DIMS,
            base_url="http://unused.test/v1",
            api_key=SecretStr("fake-test-key"),
        ),
    )

    assert captured["timeout"] == 2.5
    assert captured["max_retries"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rag_query_embedding_timeout", True),
        ("rag_query_embedding_timeout", 0),
        ("rag_query_embedding_timeout", -1),
        ("rag_query_embedding_timeout", 60.1),
        ("rag_query_embedding_timeout", float("nan")),
        ("rag_query_embedding_timeout", float("inf")),
        ("rag_query_embedding_max_retries", True),
        ("rag_query_embedding_max_retries", -1),
        ("rag_query_embedding_max_retries", 4),
        ("rag_query_embedding_max_retries", 1.5),
        ("rag_query_embedding_max_retries", "1.5"),
    ],
)
def test_invalid_query_embedding_settings_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        RAGSettings(_env_file=None, **{field: value})


def test_query_embedding_settings_accept_environment_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_QUERY_EMBEDDING_TIMEOUT", "7.5")
    monkeypatch.setenv("RAG_QUERY_EMBEDDING_MAX_RETRIES", "2")

    settings = RAGSettings(_env_file=None)

    assert settings.rag_query_embedding_timeout == 7.5
    assert settings.rag_query_embedding_max_retries == 2


def test_production_pipeline_is_hybrid_and_uses_one_chunk_collection(
    production_env: Any,
) -> None:
    pipeline = production_env.search._pipeline

    assert isinstance(pipeline, RAGPipeline)
    assert isinstance(pipeline._retriever, HybridRetriever)
    hybrid = pipeline._retriever
    assert hybrid._bm25._chunks is hybrid._vector._bundle.chunks
    assert hybrid._vector._bundle.chunks is production_env.bundle.chunks
    assert isinstance(hybrid._vector._bundle.index, faiss.IndexFlatIP)


def test_production_query_executes_both_channels_and_returns_score_details(
    production_env: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hybrid = production_env.search._pipeline._retriever
    bm25_calls: list[tuple[str, int]] = []
    vector_calls: list[tuple[str, int]] = []
    real_bm25_search = hybrid._bm25.search
    real_vector_search = hybrid._vector.search

    def _bm25_spy(query: str, top_k: int = 5) -> Any:
        bm25_calls.append((query, top_k))
        return real_bm25_search(query, top_k)

    def _vector_spy(query: str, top_k: int = 5) -> Any:
        vector_calls.append((query, top_k))
        return real_vector_search(query, top_k)

    monkeypatch.setattr(hybrid._bm25, "search", _bm25_spy)
    monkeypatch.setattr(hybrid._vector, "search", _vector_spy)

    response = production_env.search.search_knowledge("vacation", top_k=2)

    assert bm25_calls == [("vacation", 4)]
    assert vector_calls == [("vacation", 4)]
    assert response.query == "vacation"
    assert response.chunks[0].chunk_id == "policy:0"
    details = response.chunks[0].score_details
    assert details.bm25 is not None
    assert details.vector == pytest.approx(1.0)
    assert details.bm25_rank == 1
    assert details.vector_rank == 1
    assert details.rrf == pytest.approx(2.0 / 61.0)
    assert response.chunks[0].score == pytest.approx(1.0)
    assert production_env.embedder.query_calls == ["vacation"]
    assert production_env.embedder.document_calls == 0


def test_index_is_loaded_once_and_never_reloaded_per_query(production_env: Any) -> None:
    assert production_env.load_calls == [production_env.index_dir]

    production_env.search.search_knowledge("vacation", top_k=1)
    production_env.search.search_knowledge("medical", top_k=1)

    assert production_env.load_calls == [production_env.index_dir]
    assert production_env.embedder.document_calls == 0


def test_query_embedding_timeout_degrades_to_bm25_without_reloading_index(
    production_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    load_count = len(production_env.load_calls)

    def _raise_timeout(text: str) -> list[float]:
        calls.append(text)
        raise TimeoutError("simulated bounded query timeout")

    monkeypatch.setattr(production_env.embedder, "embed_query", _raise_timeout)
    with caplog.at_level(logging.WARNING, logger="services.rag_mcp.rag.retriever"):
        response = production_env.search.search_knowledge("vacation", top_k=2)

    assert calls == ["vacation"]
    assert len(production_env.load_calls) == load_count
    assert response.chunks
    assert [chunk.chunk_id for chunk in response.chunks] == ["policy:0"]
    details = response.chunks[0].score_details
    assert details.bm25 is not None
    assert details.bm25_rank == 1
    assert details.vector is None
    assert details.vector_rank is None
    assert details.rrf == pytest.approx(1.0 / 61.0)
    assert response.chunks[0].score == pytest.approx(0.5)
    assert "channel=vector error_type=QueryEmbeddingError" in caplog.text


def test_other_vector_error_keeps_existing_bm25_degradation(
    production_env: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_provider_error(text: str) -> list[float]:
        raise RuntimeError(f"simulated provider failure for {text}")

    monkeypatch.setattr(production_env.embedder, "embed_query", _raise_provider_error)

    response = production_env.search.search_knowledge("medical", top_k=2)

    assert response.chunks
    for chunk in response.chunks:
        assert chunk.score_details.bm25 is not None
        assert chunk.score_details.vector is None
        assert chunk.score_details.vector_rank is None


def test_both_channels_failing_remains_an_explicit_error(
    production_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hybrid = production_env.search._pipeline._retriever

    def _raise_bm25_error(query: str, top_k: int = 5) -> Any:
        raise RuntimeError(f"{_SYNTHETIC_SECRET}: {query}/{top_k}")

    def _raise_timeout(text: str) -> list[float]:
        raise TimeoutError(f"{_SYNTHETIC_SECRET}: {text}")

    monkeypatch.setattr(hybrid._bm25, "search", _raise_bm25_error)
    monkeypatch.setattr(production_env.embedder, "embed_query", _raise_timeout)

    with caplog.at_level(logging.WARNING, logger="services.rag_mcp.rag.retriever"):
        with pytest.raises(HybridRetrievalError) as exc:
            production_env.search.search_knowledge("vacation", top_k=1)

    assert isinstance(exc.value.channel_errors["bm25"], RuntimeError)
    vector_error = exc.value.channel_errors["vector"]
    assert isinstance(vector_error, QueryEmbeddingError)
    assert isinstance(vector_error.__cause__, TimeoutError)
    assert _SYNTHETIC_SECRET in str(vector_error.__cause__)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert exc.value.__suppress_context__ is True
    assert _SYNTHETIC_SECRET not in str(exc.value)
    assert _SYNTHETIC_SECRET not in "".join(traceback.format_exception(exc.value))
    assert _SYNTHETIC_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_stdio_double_failure_does_not_expose_provider_secret(tmp_path: Path) -> None:
    index_dir = tmp_path / "stdio-index"
    stderr_path = tmp_path / "stdio-server.stderr.log"
    _write_bundle(index_dir)
    transport = StdioTransport(
        command=sys.executable,
        args=["-c", _FAILING_STDIO_SERVER],
        env=_stdio_failure_environment(index_dir),
        cwd=str(_REPO_ROOT),
        keep_alive=False,
        log_file=stderr_path,
    )

    async with Client(transport) as client:
        result = await client.call_tool(
            "search_knowledge",
            {"query": "vacation", "top_k": 1},
            raise_on_error=False,
        )

    assert result.is_error is True
    tool_error = repr(result)
    server_stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    assert "Both hybrid retrieval channels failed" in tool_error
    assert "bm25=RuntimeError" in tool_error
    assert "vector=QueryEmbeddingError" in tool_error
    assert _SYNTHETIC_SECRET not in tool_error
    assert _SYNTHETIC_SECRET not in server_stderr


def test_tool_name_signature_and_direct_boundaries(production_env: Any) -> None:
    tool = production_env.search.search_knowledge
    signature = inspect.signature(tool)

    assert tool.__name__ == "search_knowledge"
    assert list(signature.parameters) == ["query", "top_k"]
    assert signature.parameters["top_k"].default == 5
    with pytest.raises(TypeError, match="query must be a string"):
        tool(123, top_k=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="top_k must be an integer"):
        tool("vacation", top_k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        tool("vacation", top_k=0)
    with pytest.raises(ValueError, match="RAG_MAX_CANDIDATES"):
        tool("vacation", top_k=6)


async def test_in_process_fastmcp_call_is_hybrid_and_strict(production_env: Any) -> None:
    server: ModuleType = importlib.import_module("services.rag_mcp.server")

    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == ["search_knowledge"]

        result = await client.call_tool(
            "search_knowledge",
            {"query": "vacation", "top_k": 1},
        )
        assert result.is_error is False
        payload = result.structured_content
        assert payload["query"] == "vacation"
        assert payload["chunks"][0]["chunk_id"] == "policy:0"
        assert payload["chunks"][0]["score_details"]["bm25"] is not None
        assert payload["chunks"][0]["score_details"]["vector"] == pytest.approx(1.0)

        bool_result = await client.call_tool(
            "search_knowledge",
            {"query": "vacation", "top_k": True},
            raise_on_error=False,
        )
        assert bool_result.is_error is True


async def test_in_process_mcp_timeout_returns_honest_bm25_details(
    production_env: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_timeout(text: str) -> list[float]:
        raise TimeoutError(f"simulated timeout for {text}")

    monkeypatch.setattr(production_env.embedder, "embed_query", _raise_timeout)
    server: ModuleType = importlib.import_module("services.rag_mcp.server")

    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "search_knowledge",
            {"query": "vacation", "top_k": 1},
        )

    assert result.is_error is False
    chunk = result.structured_content["chunks"][0]
    assert chunk["chunk_id"] == "policy:0"
    assert chunk["score_details"]["bm25"] is not None
    assert chunk["score_details"]["bm25_rank"] == 1
    assert chunk["score_details"]["vector"] is None
    assert chunk["score_details"]["vector_rank"] is None


def test_missing_embedding_configuration_fails_before_index_load(production_env: Any) -> None:
    load_count = len(production_env.load_calls)

    with pytest.raises(
        production_env.search.RAGServiceInitializationError,
        match="embedding configuration is unavailable",
    ):
        production_env.search.build_rag_pipeline(
            rag_settings=RAGSettings(rag_index_dir=production_env.index_dir),
            embedding_config=EmbeddingConfig(),
            embedder=production_env.embedder,
        )

    assert len(production_env.load_calls) == load_count


def test_missing_index_fails_without_bm25_fallback(
    production_env: Any,
    tmp_path: Path,
) -> None:
    config = EmbeddingConfig(
        enabled=True,
        model=_MODEL,
        dims=_DIMS,
        api_key=SecretStr("fake-test-key"),
    )

    with pytest.raises(IndexArtifactMissingError, match=r"index\.faiss"):
        production_env.search.build_rag_pipeline(
            rag_settings=RAGSettings(rag_index_dir=tmp_path / "missing"),
            embedding_config=config,
            embedder=production_env.embedder,
        )


def test_embedding_client_failure_does_not_expose_secret(
    production_env: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "super-secret-api-key"
    config = EmbeddingConfig(
        enabled=True,
        model=_MODEL,
        dims=_DIMS,
        api_key=SecretStr(secret),
    )

    def _fail_build(**kwargs: object) -> Embeddings:
        raise RuntimeError(f"provider rejected api_key={secret}")

    monkeypatch.setattr(production_env.search, "OpenAIEmbeddings", _fail_build)
    with pytest.raises(
        production_env.search.RAGServiceInitializationError,
        match="RuntimeError",
    ) as exc:
        production_env.search.build_rag_pipeline(
            rag_settings=RAGSettings(rag_index_dir=production_env.index_dir),
            embedding_config=config,
        )

    assert secret not in str(exc.value)
    assert exc.value.__cause__ is None
