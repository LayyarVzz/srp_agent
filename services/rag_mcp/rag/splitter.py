"""Simple text splitter for RAG knowledge."""

from __future__ import annotations


def split_text(text: str) -> list[str]:
    """Split text into non-empty chunks by blank lines."""
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
