"""Knowledge build helpers for RAG."""

from __future__ import annotations

from pathlib import Path

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.splitter import split_markdown, split_text
from services.rag_mcp.schemas import KnowledgeSource


class KnowledgeBuilder:
    """Build source-aware document chunks from a local text knowledge file."""

    def __init__(self, file_path: str | Path, source: KnowledgeSource) -> None:
        self._file_path = file_path
        self._source = source

    def build(self) -> list[DocumentChunk]:
        """Load and split the configured text knowledge file."""
        text = load_text_file(self._file_path)
        suffix = Path(self._file_path).suffix.lower()
        if suffix in {".md", ".markdown"}:
            return split_markdown(text, self._source)

        return split_text(text, self._source)
