"""用于验证AgentRuntime初始化的临时测试脚本。"""

from __future__ import annotations

import asyncio

from langgraph.graph.state import CompiledStateGraph

from agent.runtime import AgentRuntime


async def main() -> None:
    """创建AgentRuntime，但不调用Agent Graph。"""
    runtime = await AgentRuntime.create()

    print("Agent Runtime初始化成功")
    print("Graph类型：")
    print(type(runtime.graph))
    print("是否为CompiledStateGraph：")
    print(isinstance(runtime.graph, CompiledStateGraph))

    if not isinstance(runtime.graph, CompiledStateGraph):
        raise RuntimeError("Agent Runtime未持有CompiledStateGraph")


if __name__ == "__main__":
    asyncio.run(main())
