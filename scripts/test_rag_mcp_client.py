"""Temporary client script for validating the RAG MCP tool chain."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from services.rag_mcp.rag.index import (
    CHUNKS_FILE_NAME,
    INDEX_FILE_NAME,
    MANIFEST_FILE_NAME,
    FaissIndexManifest,
)
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.schemas import KnowledgeSource

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_NAME = "search_knowledge"
_MODEL = "fake-embedding"
_DIMS = 2

_SERVER_BOOTSTRAP = """
import langchain_openai
from langchain_core.embeddings import Embeddings

class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise AssertionError("online RAG must not embed documents")

    def embed_query(self, text):
        return [1.0, 0.0]

def _build_fake_query_embedder(**kwargs):
    if kwargs.get("dimensions") != 2:
        raise AssertionError("query embedding dimensions were not forwarded")
    if kwargs.get("timeout") != 10.0:
        raise AssertionError("query embedding timeout was not forwarded")
    if kwargs.get("max_retries") != 0:
        raise AssertionError("query embedding retries were not forwarded")
    return _FakeEmbeddings()

langchain_openai.OpenAIEmbeddings = _build_fake_query_embedder

from services.rag_mcp.server import mcp
mcp.run()
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_test_bundle(index_dir: Path) -> None:
    index_dir.mkdir(parents=True)
    chunks = [
        DocumentChunk(
            chunk_id=f"policy:{index}",
            content=content,
            source=KnowledgeSource(id="policy", title="Employee policy"),
            metadata={"chunk_index": index},
        )
        for index, content in enumerate(
            [
                "Vacation entitlement policy",
                "Medical reimbursement rules",
                "Travel expense approval",
            ]
        )
    ]
    diagonal = 1.0 / np.sqrt(2.0)
    index = faiss.IndexFlatIP(_DIMS)
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
            [chunk.model_dump(mode="json") for chunk in chunks],
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
        chunk_count=len(chunks),
        index_sha256=_sha256(index_path),
        chunks_sha256=_sha256(chunks_path),
    )
    (index_dir / MANIFEST_FILE_NAME).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _server_environment(index_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "RAG_INDEX_DIR": str(index_dir),
            "RAG_CANDIDATE_MULTIPLIER": "2",
            "RAG_MAX_CANDIDATES": "5",
            "RAG_RRF_K": "60",
            "RAG_VECTOR_MIN_SCORE": "0",
            "EMBEDDING_ENABLED": "true",
            "EMBEDDING_MODEL": _MODEL,
            "EMBEDDING_DIMS": str(_DIMS),
            "EMBEDDING_BASE_URL": "http://unused.test/v1",
            "EMBEDDING_API_KEY": "fake-test-key",
        }
    )
    return environment


async def main() -> None:
    """Call the real stdio MCP transport with temporary offline Hybrid artifacts."""
    with tempfile.TemporaryDirectory(prefix="rag-mcp-stdio-") as temporary_directory:
        index_dir = Path(temporary_directory) / "index"
        _write_test_bundle(index_dir)
        transport = StdioTransport(
            command=sys.executable,
            args=["-c", _SERVER_BOOTSTRAP],
            env=_server_environment(index_dir),
            cwd=str(REPO_ROOT),
            keep_alive=False,
        )
        client = Client(transport)

        async with client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]

            print("Tool列表:")
            print(json.dumps(tool_names, ensure_ascii=False, indent=2))
            if tool_names != [TOOL_NAME]:
                raise RuntimeError(f"未找到唯一的RAG Tool: {TOOL_NAME}")

            result = await client.call_tool(
                TOOL_NAME,
                {
                    "query": "vacation",
                    "top_k": 2,
                },
            )
            response = result.structured_content
            if (
                not isinstance(response, dict)
                or "query" not in response
                or "chunks" not in response
            ):
                raise RuntimeError("返回结果不符合 SearchKnowledgeResponse 结构")
            if response["query"] != "vacation":
                raise RuntimeError("Hybrid MCP响应未保留原始query")
            if not response["chunks"]:
                raise RuntimeError("Hybrid MCP调用未返回知识片段")
            first = response["chunks"][0]
            details = first.get("score_details") or {}
            if first.get("chunk_id") != "policy:0":
                raise RuntimeError("Hybrid MCP首条结果未命中预期Chunk")
            try:
                public_score = float(first["score"])
                bm25_score = float(details["bm25"])
                vector_score = float(details["vector"])
                rrf_score = float(details["rrf"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("Hybrid MCP首条结果缺少完整分数明细") from exc
            if not all(
                math.isfinite(score)
                for score in (public_score, bm25_score, vector_score, rrf_score)
            ):
                raise RuntimeError("Hybrid MCP首条结果包含非有限分数")
            if details.get("bm25_rank") != 1 or details.get("vector_rank") != 1:
                raise RuntimeError("Hybrid MCP首条结果未同时获得BM25和Vector第一名")
            if not math.isclose(vector_score, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError("Hybrid MCP首条结果未使用预期FAISS向量分数")
            if not math.isclose(rrf_score, 2.0 / 61.0, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError("Hybrid MCP首条结果的RRF分数不正确")
            if not math.isclose(public_score, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError("Hybrid MCP首条结果的归一化score不正确")

            invalid = await client.call_tool(
                TOOL_NAME,
                {"query": "vacation", "top_k": True},
                raise_on_error=False,
            )
            if not invalid.is_error:
                raise RuntimeError("top_k布尔值未被严格拒绝")

            print("Hybrid stdio调用返回结果:")
            print(json.dumps(response, ensure_ascii=False, indent=2))
            print("Hybrid stdio客户端验证通过")


if __name__ == "__main__":
    asyncio.run(main())
