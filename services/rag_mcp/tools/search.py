"""Knowledge search tool."""

from __future__ import annotations

from typing import Annotated

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import Field

from services.rag_mcp.config import RAGSettings, get_rag_settings
from services.rag_mcp.rag.index import load_faiss_index_bundle
from services.rag_mcp.rag.pipeline import RAGPipeline
from services.rag_mcp.rag.retriever import BM25Retriever, HybridRetriever, VectorRetriever
from services.rag_mcp.schemas import SearchKnowledgeResponse
from settings import get_settings
from shared import EmbeddingConfig, EmbeddingsFactory

StrictQuery = Annotated[str, Field(strict=True)]
StrictTopK = Annotated[int, Field(strict=True)]


class RAGServiceInitializationError(RuntimeError):
    """The RAG MCP production retrieval chain cannot be initialized safely."""


def _build_embedding_config() -> EmbeddingConfig:
    """Project the shared runtime settings into the shared embedding contract."""
    settings = get_settings()
    return EmbeddingConfig.from_runtime(
        enabled=settings.embedding_enabled,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )


def _build_query_embedder(
    embedding_config: EmbeddingConfig,
    rag_settings: RAGSettings,
) -> Embeddings:
    """Build the RAG query client with bounded public SDK options.

    The shared factory intentionally owns project-wide defaults and does not expose
    per-request timeout/retry overrides. The online RAG service therefore reconstructs
    the same public OpenAI-compatible client configuration without touching private
    SDK state or changing the shared contract.
    """
    kwargs: dict[str, object] = {
        "model": embedding_config.model,
        "check_embedding_ctx_length": embedding_config.check_embedding_ctx_length,
        "timeout": float(rag_settings.rag_query_embedding_timeout),
        "max_retries": rag_settings.rag_query_embedding_max_retries,
    }
    if embedding_config.pass_dimensions:
        kwargs["dimensions"] = embedding_config.dims
    if embedding_config.base_url:
        kwargs["base_url"] = embedding_config.base_url
    api_key = embedding_config.api_key.get_secret_value()
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


def build_rag_pipeline(
    *,
    rag_settings: RAGSettings,
    embedding_config: EmbeddingConfig,
    embedder: Embeddings | None = None,
) -> RAGPipeline:
    """Build one validated Hybrid RAG pipeline without reading source documents."""
    embedding_factory = EmbeddingsFactory(embedding_config, embedder=embedder)
    if not embedding_factory.is_available():
        raise RAGServiceInitializationError(
            "RAG embedding configuration is unavailable; "
            "enable embeddings and configure a model with positive dimensions"
        )

    bundle = load_faiss_index_bundle(rag_settings.rag_index_dir, embedding_config)
    try:
        selected_embedder = (
            embedding_factory.build()
            if embedder is not None
            else _build_query_embedder(embedding_config, rag_settings)
        )
    except Exception as exc:
        raise RAGServiceInitializationError(
            "Failed to initialize the RAG embedding client: "
            f"{type(exc).__name__}"
        ) from None

    bm25 = BM25Retriever(bundle.chunks)
    vector = VectorRetriever(
        bundle=bundle,
        embedder=selected_embedder,
        vector_min_score=rag_settings.rag_vector_min_score,
    )
    hybrid = HybridRetriever(
        bm25=bm25,
        vector=vector,
        candidate_multiplier=rag_settings.rag_candidate_multiplier,
        max_candidates=rag_settings.rag_max_candidates,
        rrf_k=rag_settings.rag_rrf_k,
    )
    return RAGPipeline(retriever=hybrid)


_rag_settings = get_rag_settings()
_embedding_config = _build_embedding_config()
_pipeline = build_rag_pipeline(
    rag_settings=_rag_settings,
    embedding_config=_embedding_config,
)


def search_knowledge(query: StrictQuery, top_k: StrictTopK = 5) -> SearchKnowledgeResponse:
    """Search the prebuilt knowledge index through BM25/vector Hybrid RRF."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if top_k > _rag_settings.rag_max_candidates:
        raise ValueError(
            f"top_k must not exceed RAG_MAX_CANDIDATES "
            f"({_rag_settings.rag_max_candidates})"
        )

    chunks = _pipeline.search(query=query, top_k=top_k)
    return SearchKnowledgeResponse(query=query, chunks=chunks)


__all__ = [
    "RAGServiceInitializationError",
    "build_rag_pipeline",
    "search_knowledge",
]
