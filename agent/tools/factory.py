"""MCP 工具装配工厂：连接 MCP 服务 → LangChain BaseTool 列表。

工具一律经 MCP 客户端接入。
本工厂承载「构造客户端 → get_tools → yield → 结束」生命周期，返回的 `list[BaseTool]`
供图内 `ToolNode` 统一执行。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool

from agent.core.config import AgentFrameworkConfig

logger = logging.getLogger(__name__)

# transport 类型里 TypedDict 提供 timeout 字段的（stdio 无该字段，见 sessions.py）。
_HTTP_TRANSPORTS = ("streamable_http", "sse")


@asynccontextmanager
async def build_tools_from_mcp(
    config: AgentFrameworkConfig,
    *,
    servers: dict[str, dict],
) -> AsyncIterator[list[BaseTool]]:
    """从 MCP 服务装配 LangChain 工具列表。

    超时：仅 streamable_http/sse 连接的 TypedDict 有 `timeout` 字段，注入
    `mcp_timeout_s`；stdio 连接无 timeout

    重试：adapters 无原生重试参数，`mcp_max_retries` 仅用于 `get_tools`
    （连接/列表）阶段的有限重试；单次工具执行重试由 ToolNode/模型自纠错承担
    """
    tools_cfg = config.tools
    connections: dict[str, dict] = {}
    for name, conn in servers.items():
        conn = dict(conn)
        if conn.get("transport") in _HTTP_TRANSPORTS and "timeout" not in conn:
            conn["timeout"] = tools_cfg.mcp_timeout_s
        connections[name] = conn

    from langchain_mcp_adapters.client import MultiServerMCPClient

    # 构造 MCP 多服务客户端
    client = MultiServerMCPClient(connections=connections, handle_tool_errors=True)
    last_exc: Exception | None = None
    for attempt in range(tools_cfg.mcp_max_retries + 1):
        try:
            tools = await client.get_tools()
            break
        except Exception as exc:
            # 连接/列表阶段失败统一记日志后按上限重试。
            last_exc = exc
            if attempt < tools_cfg.mcp_max_retries:
                logger.warning(
                    "MCP 工具加载失败（第 %d/%d 次）：%s",
                    attempt + 1,
                    tools_cfg.mcp_max_retries,
                    exc,
                )
    else:
        if last_exc is None:  # pragma: no cover  # 防御：逻辑上不可达
            raise RuntimeError("MCP 工具加载失败：未捕获到异常信息")
        logger.error("MCP 工具加载失败，重试耗尽")
        raise last_exc

    logger.info("已从 MCP 服务加载 %d 个工具", len(tools))
    yield tools
