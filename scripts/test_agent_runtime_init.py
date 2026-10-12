"""用于验证AgentRuntime初始化的临时测试脚本。"""

from __future__ import annotations

import asyncio

from langgraph.graph.state import CompiledStateGraph

from agent.runtime import AgentRuntime
from agent.share.eventloop import ensure_selector_event_loop


async def main() -> None:
    """创建AgentRuntime（不调用Agent Graph），验证装配产物并正确释放生命周期资源。"""
    runtime = await AgentRuntime.create()

    print("Agent Runtime初始化成功")
    print("Graph类型：")
    print(type(runtime.graph))
    print("是否为CompiledStateGraph：")
    print(isinstance(runtime.graph, CompiledStateGraph))

    if not isinstance(runtime.graph, CompiledStateGraph):
        raise RuntimeError("Agent Runtime未持有CompiledStateGraph")

    # create() 已打开 MCP 子进程与会话/记忆后端，验证后必须按逆序关闭。
    await runtime.aclose()


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前设置
    # （与 scripts/demo_chat.py 同策略；配了 DATABASE_URL 时必走此路径）。
    ensure_selector_event_loop()
    asyncio.run(main())
