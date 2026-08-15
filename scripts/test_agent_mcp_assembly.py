"""Temporary script for validating Agent graph assembly with MCP tools."""

from __future__ import annotations

import asyncio

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.llm import LLMService
from agent.tools import build_tools_from_mcp
from settings import get_settings


async def main() -> None:
    """Assemble the Agent graph with MCP tools without invoking the graph."""
    settings = get_settings()
    framework_config = AgentFrameworkConfig.get_default()
    llm_config = LLMConfig.from_runtime(
        provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        behavior=framework_config.llm_behavior,
    )
    llm = LLMService(config=llm_config)

    tools = await build_tools_from_mcp()
    tool_names = [tool.name for tool in tools]
    print("MCP tools:")
    print(tool_names)

    if "search_knowledge" not in tool_names:
        raise RuntimeError("未找到 MCP Tool: search_knowledge")

    graph = build_agent_graph(
        llm=llm,
        config=framework_config,
        tools=tools,
    )
    print("Agent graph装配成功:")
    print(type(graph))


if __name__ == "__main__":
    asyncio.run(main())
