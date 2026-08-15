"""Temporary script for validating Agent invocation of the RAG MCP tool."""

from __future__ import annotations

import asyncio
from typing import Any

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.llm import LLMService
from agent.response.models import AgentResponse
from agent.tools import build_tools_from_mcp
from settings import get_settings


DEMO_INPUT = "员工工作10年有多少天年假？"
THREAD_ID = "agent-rag-mcp-demo"
TARGET_TOOL_NAME = "search_knowledge"


def _dump_value(value: Any) -> Any:
    """Convert simple model-like values for readable printing."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _print_tool_trace(response: AgentResponse) -> None:
    """Print tool call records from AgentResponse."""
    print("Tool calls数量:")
    print(len(response.tool_trace))
    for index, record in enumerate(response.tool_trace, start=1):
        print(f"Tool call {index}:")
        print(f"  tool_name: {record.tool_name}")
        print(f"  arguments: {record.arguments}")
        print(f"  status: {record.status}")
        if record.result and record.result.data:
            content = str(record.result.data.get("content", ""))
            print("  result.content:")
            print(content[:1000])
        if record.result and record.result.error:
            print("  error:")
            print(record.result.error.model_dump())


async def main() -> None:
    """Assemble the Agent graph and invoke it with a RAG-related question."""
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
    if not isinstance(response, AgentResponse):
        raise RuntimeError("graph.ainvoke 返回结果中未找到 AgentResponse")

    print("Final answer:")
    print(result.get("final_answer"))
    print("finished_reason:")
    print(response.finished_reason)
    _print_tool_trace(response)

    successful_rag_calls = [
        record
        for record in response.tool_trace
        if record.tool_name == TARGET_TOOL_NAME and record.status == "ok"
    ]
    if successful_rag_calls:
        print("search_knowledge调用验证:")
        print("ok")
        return

    print("search_knowledge调用验证:")
    print("not_called_or_failed")
    print("intent:")
    print(_dump_value(result.get("intent")))
    print("status_trace:")
    print([event.model_dump() for event in response.status_trace])
    print("tool_trace:")
    print([record.model_dump() for record in response.tool_trace])


if __name__ == "__main__":
    asyncio.run(main())
