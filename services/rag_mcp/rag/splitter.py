"""Text splitter for RAG knowledge."""

from __future__ import annotations

import re

from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.schemas import KnowledgeSource

_CHUNK_SIZE = 80
_PARAGRAPH_SEPARATOR = "\n\n"
_SENTENCE_END_PATTERN = re.compile(r"[^。！？；]+[。！？；]?")


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    return [
        sentence.group(0).strip()
        for sentence in _SENTENCE_END_PATTERN.finditer(paragraph)
        if sentence.group(0).strip()
    ]


def _split_long_sentence(sentence: str) -> list[str]:
    return [
        sentence[index : index + _CHUNK_SIZE]
        for index in range(0, len(sentence), _CHUNK_SIZE)
    ]


def _pack_texts(texts: list[str], separator: str) -> list[str]:
    chunks: list[str] = []
    current = ""

    for text in texts:
        candidate = text if not current else f"{current}{separator}{text}"
        if current and len(candidate) > _CHUNK_SIZE:
            chunks.append(current)
            current = text
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _split_long_paragraph(paragraph: str) -> list[str]:
    sentence_units: list[str] = []
    for sentence in _split_sentences(paragraph):
        if len(sentence) <= _CHUNK_SIZE:
            sentence_units.append(sentence)
            continue

        sentence_units.extend(_split_long_sentence(sentence))

    return _pack_texts(sentence_units, "")


def split_text(text: str, source: KnowledgeSource) -> list[DocumentChunk]:
    """Split text into source-aware chunks."""
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) > _CHUNK_SIZE:
            if current_paragraphs:
                chunks.append(_PARAGRAPH_SEPARATOR.join(current_paragraphs))
                current_paragraphs = []
            chunks.extend(_split_long_paragraph(paragraph))
            continue

        candidate_paragraphs = [*current_paragraphs, paragraph]
        candidate = _PARAGRAPH_SEPARATOR.join(candidate_paragraphs)
        if current_paragraphs and len(candidate) > _CHUNK_SIZE:
            chunks.append(_PARAGRAPH_SEPARATOR.join(current_paragraphs))
            current_paragraphs = [paragraph]
        else:
            current_paragraphs = candidate_paragraphs

    if current_paragraphs:
        chunks.append(_PARAGRAPH_SEPARATOR.join(current_paragraphs))

    return [
        DocumentChunk(
            chunk_id=f"{source.id}:{chunk_index}",
            content=chunk,
            source=source,
            metadata={"chunk_index": chunk_index},
        )
        for chunk_index, chunk in enumerate(chunks)
    ]
