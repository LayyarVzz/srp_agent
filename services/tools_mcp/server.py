"""tools_mcp FastMCP 服务器：注册 calculator / current_datetime 工具。

启动：`uv run python -m services.tools_mcp`（默认 stdio；配置
MCP_TRANSPORT=streamable-http 即运行于端口，见 README）。
Agent 侧经 langchain-mcp-adapters MultiServerMCPClient 接入。
"""

from __future__ import annotations

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.tools_mcp.datetime_tool import current_datetime

mcp = FastMCP("tools-mcp")

# 工具以模块级函数注册
mcp.tool(name="current_datetime")(current_datetime)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """存活探针：供云上负载均衡 / 容器编排检查，不影响 MCP 协议端点。"""
    return JSONResponse({"status": "ok"})


__all__ = ["mcp"]
