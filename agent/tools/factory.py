"""MCP 工具装配工厂：连接 MCP 服务 → LangChain BaseTool 列表。

工具一律经 MCP 客户端接入。
本工厂承载「构造客户端 → get_tools → yield → 结束」生命周期，返回的 `list[BaseTool]`
供图内 `ToolNode` 统一执行。

v5.1：客户端挂载 `LarkScopeInterceptor`（把图状态 `user_id` 注入 lark 工具实参，
见 agent/tools/lark_scope.py）；`get_tools()` 后统一剥离 `_lark_scope` 使 LLM 不可见。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool

from agent.core.config import AgentFrameworkConfig
from agent.tools.lark_scope import LarkScopeInterceptor
from agent.tools.retrieval_args import RetrievalQueryInterceptor

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

    返回的工具对象同时承担两个职责：`bind_tools`（模型可见声明）与 `ToolNode`
    （执行实体）。飞书工具的 `_lark_scope` 需「执行侧可见、模型侧不可见」，
    故绑定处一律经 `LarkScopeInterceptor` + 图内 `visible_tools()` 分流
    （见 agent/tools/lark_scope.py），本工厂不做 schema 改写。

    超时：仅 streamable_http/sse 连接的 TypedDict 有 `timeout` 字段，注入
    `mcp_timeout_s`；stdio 连接无 timeout。

    重试：adapters 无原生重试参数，`mcp_max_retries` 仅用于 `get_tools`
    （连接/列表）阶段的有限重试；单次工具执行重试由 ToolNode/模型自纠错承担。
    重试耗尽后**降级为空工具集而非抛错**：工具为空时图内 call_model 无工具可选，
    Agent 退化为纯文本回答（意图识别与降级路径仍可用），请求不失败。
    """
    tools_cfg = config.tools
    connections: dict[str, dict] = {}
    for name, conn in servers.items():
        conn = dict(conn)
        if conn.get("transport") in _HTTP_TRANSPORTS and "timeout" not in conn:
            conn["timeout"] = tools_cfg.mcp_timeout_s
        connections[name] = conn

    from langchain_mcp_adapters.client import MultiServerMCPClient

    # 构造 MCP 多服务客户端；拦截器各按 server_name 自行放行非目标服务（零影响）：
    # - LarkScopeInterceptor：把图状态 user_id 注入 lark 工具实参（v5.1）；
    # - RetrievalQueryInterceptor：归一 rag 检索实参 + 召回不足时惰性补检（v6.0 T2）。
    client = MultiServerMCPClient(
        connections=connections,
        handle_tool_errors=True,
        tool_interceptors=[
            LarkScopeInterceptor(),
            RetrievalQueryInterceptor(config=config.retrieval),
        ],
    )
    last_exc: Exception | None = None
    tools: list[BaseTool] = []
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
        # 重试耗尽：降级为空工具集而非整体崩溃（MCP 服务故障不应拖垮 Agent 进程）。
        # WHY 可观测降级：空工具集下 Agent 仍可跑意图识别/纯文本回答/降级路径，
        # 日志明确记录故障根因；MCP 服务恢复后重启进程（或热重载）即恢复工具能力。
        logger.error("MCP 工具加载失败，重试耗尽，降级为空工具集：%s", last_exc)

    logger.info("已从 MCP 服务加载 %d 个工具", len(tools))
    yield tools
