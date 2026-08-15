"""Temporary script for validating the text splitter."""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.splitter import split_text


def main() -> None:
    """Load employee policy text and print split chunks."""
    text = load_text_file("services/rag_mcp/knowledge/data/employee_policy.txt")
    chunks = split_text(text)

    print("原始文本类型:")
    print(type(text))
    print("切分结果类型:")
    print(type(chunks))
    print("chunk数量:")
    print(len(chunks))
    print("chunk内容:")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")


if __name__ == "__main__":
    main()
