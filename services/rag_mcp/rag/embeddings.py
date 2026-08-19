"""OpenAI-compatible embedding client for RAG."""

from __future__ import annotations

from openai import OpenAI

from settings import RuntimeSettings, get_settings


class EmbeddingClient:
    """Generate text embeddings through an OpenAI-compatible API."""

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        runtime_settings = settings or get_settings()
        api_key = runtime_settings.embedding_api_key.get_secret_value()
        base_url = runtime_settings.embedding_base_url
        model = runtime_settings.embedding_model

        if not api_key:
            raise ValueError("EMBEDDING_API_KEY未配置")
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL未配置")
        if not model:
            raise ValueError("EMBEDDING_MODEL未配置")

        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts with the configured model."""
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query text with the configured model."""
        return self.embed_documents([text])[0]
