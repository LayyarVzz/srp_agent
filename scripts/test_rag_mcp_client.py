"""Temporary client script for validating the RAG MCP tool chain."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_NAME = "search_knowledge"


async def main() -> None:
    """List RAG MCP tools and call search_knowledge."""
    transport = StdioTransport(
        command="python",
        args=["-m", "services.rag_mcp.server"],
        cwd=str(REPO_ROOT),
    )
    client = Client(transport)

    async with client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        print("Tool列表:")
        print(json.dumps(tool_names, ensure_ascii=False, indent=2))

        if TOOL_NAME not in tool_names:
            raise RuntimeError(f"未找到Tool: {TOOL_NAME}")

        result = await client.call_tool(
            TOOL_NAME,
            {
                "query": "员工年假如何计算",
                "top_k": 5,
            },
        )

        print("调用返回结果:")
        response = result.structured_content
        if "query" not in response or "chunks" not in response:
            raise RuntimeError("返回结果不符合 SearchKnowledgeResponse 结构")
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
