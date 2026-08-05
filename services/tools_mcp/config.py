"""tools_mcp 服务运行参数构造。

将根 `settings.py` 的 MCP 运行环境配置映射为 `mcp.run(**kwargs)` 参数。
本模块不运行时 import 根 settings，保持独立、可纯函数单测；若未来要求
零 agent 依赖独立部署，只需替换 `__main__` 的配置来源为本地 settings。
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 运行时避免引入根 settings（连带 agent 框架）
    from settings import RuntimeSettings


class MCPTransport(StrEnum):
    """tools_mcp 运行传输方式（值须与 fastmcp `run(transport=...)` 对齐）。

    注意：fastmcp 服务端传输值用连字符 `streamable-http`；langchain-mcp-adapters
    客户端连接配置里的键是下划线 `streamable_http`。
    """

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"


def build_run_params(settings: RuntimeSettings) -> dict[str, Any]:
    """把根运行环境配置映射为 `mcp.run(**kwargs)` 参数。
    
    fastmcp 在 stdio 下会把多余 kwargs 透传给 `run_stdio_async`（其签名仅
    show_banner/log_level/stateless），传 host/port/path 会 TypeError
    因此 http 系列参数只能出现在非 stdio 分支。
    """
    params: dict[str, Any] = {"transport": settings.mcp_transport.value}
    if settings.mcp_transport is not MCPTransport.STDIO:
        params.update(
            host=settings.mcp_host,
            port=settings.mcp_port,
            path=settings.mcp_streamable_http_path,
            stateless_http=settings.mcp_stateless_http,
        )
    return params
