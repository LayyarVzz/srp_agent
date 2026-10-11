"""agent/tools/factory.py —— build_tools_from_mcp 降级行为测试。

WHY 只测降级路径：MCP 服务不可达且重试耗尽时必须返回空工具集而非抛错
（工具服务故障不应拖垮 Agent 进程），这是工厂的容错契约。
"""

from __future__ import annotations

from agent.core.config import AgentFrameworkConfig, MCPToolsConfig
from agent.tools import build_tools_from_mcp

# 端口 1 的 localhost 连接必然被拒（无服务监听），用于确定性触发 get_tools 失败。
_UNREACHABLE_URL = "http://127.0.0.1:1/mcp"


async def test_mcp_unavailable_yields_empty_tools() -> None:
    """MCP 不可达且重试耗尽（mcp_max_retries=0 → 仅 1 次尝试）→ 空工具集，不抛错。"""
    cfg = AgentFrameworkConfig(tools=MCPToolsConfig(mcp_max_retries=0, mcp_timeout_s=2))
    servers = {"broken": {"transport": "streamable_http", "url": _UNREACHABLE_URL}}

    async with build_tools_from_mcp(cfg, servers=servers) as tools:
        assert tools == []


async def test_mcp_unavailable_with_retries_yields_empty_tools() -> None:
    """带重试（mcp_max_retries=2）同样降级为空工具集（重试本身也失败）。"""
    cfg = AgentFrameworkConfig(tools=MCPToolsConfig(mcp_max_retries=2, mcp_timeout_s=2))
    servers = {"broken": {"transport": "streamable_http", "url": _UNREACHABLE_URL}}

    async with build_tools_from_mcp(cfg, servers=servers) as tools:
        assert tools == []
