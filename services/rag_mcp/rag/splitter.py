"""Text splitter for RAG knowledge."""

from __future__ import annotations

import re

from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.schemas import KnowledgeSource

_CHUNK_SIZE = 80
_PARAGRAPH_SEPARATOR = "\n\n"
_SENTENCE_END_PATTERN = re.compile(r"[^。！？；]+[。！？；]?")
_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


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


def _split_body_text(text: str) -> list[str]:
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

    return chunks


def _build_document_chunks(
    chunks: list[str],
    source: KnowledgeSource,
    heading_paths: list[list[str]] | None = None,
) -> list[DocumentChunk]:
    document_chunks: list[DocumentChunk] = []
    for chunk_index, chunk in enumerate(chunks):
        metadata: dict[str, object] = {"chunk_index": chunk_index}
        if heading_paths is not None:
            metadata["heading_path"] = heading_paths[chunk_index]

        document_chunks.append(
            DocumentChunk(
                chunk_id=f"{source.id}:{chunk_index}",
                content=chunk,
                source=source,
                metadata=metadata,
            )
        )

    return document_chunks


def split_text(text: str, source: KnowledgeSource) -> list[DocumentChunk]:
    """Split text into source-aware chunks."""
    chunks = _split_body_text(text)
    return _build_document_chunks(chunks, source)


def split_markdown(text: str, source: KnowledgeSource) -> list[DocumentChunk]:
    """Split Markdown text into source-aware chunks with heading metadata."""
    chunks: list[str] = []
    heading_paths: list[list[str]] = []
    heading_path: list[str] = []
    section_lines: list[str] = []

    def flush_section() -> None:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            return

        section_chunks = _split_body_text(section_text)
        chunks.extend(section_chunks)
        heading_paths.extend([heading_path.copy() for _ in section_chunks])
        section_lines.clear()

    for line in text.splitlines():
        heading_match = _MARKDOWN_HEADING_PATTERN.match(line.strip())
        if heading_match is None:
            section_lines.append(line)
            continue

        flush_section()
        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()
        heading_path = [*heading_path[: level - 1], title]

    flush_section()
    return _build_document_chunks(chunks, source, heading_paths)
