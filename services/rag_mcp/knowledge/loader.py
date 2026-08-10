"""Local text knowledge loader."""

from __future__ import annotations

from pathlib import Path


def load_text_file(file_path: str | Path) -> str:
    """Load a local UTF-8 txt knowledge file."""
    path = Path(file_path)
    if path.suffix != ".txt":
        raise ValueError("仅支持读取 .txt 知识文件")
    return path.read_text(encoding="utf-8")
