"""Temporary script for validating the simple retriever."""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.retriever import retrieve_chunks
from services.rag_mcp.rag.splitter import split_text


def main() -> None:
    """Load, split, retrieve, and print top matching chunks."""
    text = load_text_file("services/rag_mcp/knowledge/data/employee_policy.txt")
    chunks = split_text(text)
    query = "员工工作10年有多少天年假？"
    results = retrieve_chunks(
        query=query,
        chunks=chunks,
        top_k=3,
    )

    print("原始文本类型:")
    print(type(text))
    print("chunk数量:")
    print(len(chunks))
    print("query:")
    print(query)
    print("检索结果数量:")
    print(len(results))
    print("检索结果:")
    for index, chunk in enumerate(results, start=1):
        print(f"{index}. {chunk}")


if __name__ == "__main__":
    main()
