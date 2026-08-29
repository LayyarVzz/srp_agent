"""FastMCP server for RAG knowledge search."""

from __future__ import annotations

from fastmcp import FastMCP

from services.rag_mcp.tools.search import search_knowledge

mcp = FastMCP("rag-mcp")
mcp.tool(search_knowledge)


if __name__ == "__main__":
    mcp.run()
