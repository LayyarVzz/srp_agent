"""可复用的Agent Runtime装配模块。"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.llm import LLMService
from agent.tools import build_tools_from_mcp
from settings import get_settings


@dataclass(frozen=True)
class AgentRuntime:
    """应用级已编译Agent Graph的持有者。"""

    graph: CompiledStateGraph

    @classmethod
    async def create(cls) -> "AgentRuntime":
        """装配LLM、MCP Tools和Agent Graph，供后续复用。"""
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
        graph = build_agent_graph(
            llm=llm,
            config=framework_config,
            tools=tools,
        )
        return cls(graph=graph)
