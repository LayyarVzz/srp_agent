"""Temporary script for validating the local knowledge loader."""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file


def main() -> None:
    """Load the employee policy knowledge file and print basic diagnostics."""
    text = load_text_file("services/rag_mcp/knowledge/data/employee_policy.txt")

    print("返回对象类型:")
    print(type(text))
    print("文本长度:")
    print(len(text))
    print("文本前100个字符:")
    print(text[:100])


if __name__ == "__main__":
    main()
