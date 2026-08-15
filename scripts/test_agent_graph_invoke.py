"""Temporary script for validating one Agent graph ainvoke run."""

from __future__ import annotations

import asyncio

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.llm import LLMService
from agent.response.models import AgentResponse
from agent.tools import build_tools_from_mcp
from settings import get_settings


DEMO_INPUT = "你好，请简单介绍一下你自己。"
THREAD_ID = "agent-graph-invoke-demo"


async def main() -> None:
    """Assemble the graph and invoke it once with a normal chat input."""
    settings = get_settings()
    if not settings.llm_api_key.get_secret_value():
        raise RuntimeError("未配置 LLM_API_KEY，无法执行需要真实 LLM 的 graph.ainvoke 验证")

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

    graph = build_agent_graph(
        llm=llm,
        config=framework_config,
        tools=tools,
    )

    result = await graph.ainvoke(
        {
            "input": DEMO_INPUT,
        },
        config={
            "configurable": {
                "thread_id": THREAD_ID,
            }
        },
    )

    response = result.get("response")
    final_answer = result.get("final_answer")
    if not isinstance(response, AgentResponse):
        raise RuntimeError("graph.ainvoke 返回结果中未找到 AgentResponse")
    if not final_answer:
        raise RuntimeError("graph.ainvoke 返回结果中未找到 final_answer")

    print("Agent graph运行成功")
    print("AgentResponse:")
    print(response.model_dump())
    print("Final answer:")
    print(final_answer)
    print("Tool calls:")
    print(len(response.tool_trace))


if __name__ == "__main__":
    asyncio.run(main())
