"""MCP tool loading factory for Agent-side LangChain tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection

RAG_MCP_SERVER_NAME = "rag"
RAG_MCP_SERVER_MODULE = "services.rag_mcp.server"


def _repo_root() -> Path:
    """Return the project root used as MCP server working directory."""
    return Path(__file__).resolve().parents[2]


def build_rag_mcp_stdio_connection() -> StdioConnection:
    """Build the stdio connection config for the local RAG MCP server."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", RAG_MCP_SERVER_MODULE],
        "cwd": str(_repo_root()),
        "env": {
            **os.environ,
            "FASTMCP_CHECK_FOR_UPDATES": "off",
        },
    }


async def build_tools_from_mcp() -> list[BaseTool]:
    """Load MCP tools and convert them to LangChain tools for the Agent."""
    client = MultiServerMCPClient(
        {
            RAG_MCP_SERVER_NAME: build_rag_mcp_stdio_connection(),
        }
    )
    return await client.get_tools(server_name=RAG_MCP_SERVER_NAME)
