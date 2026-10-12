"""services/tools_mcp FastMCP 服务测试（进程内 Client，零网络、零子进程）。

成功结果读取 `res.content[0].text`。
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_async

from services.tools_mcp.server import mcp


@pytest.fixture
async def tool_client() -> Client:
    """同进程 fastmcp 客户端（直连本服务 FastMCP 实例）。"""
    async with Client(mcp) as client:
        yield client


async def test_datetime_returns_iso_and_weekday(tool_client: Client) -> None:
    res = await tool_client.call_tool("current_datetime", {})
    assert res.is_error is False
    iso, weekday = res.content[0].text.split(" | ")
    dt = datetime.fromisoformat(iso)
    assert dt.tzinfo is not None  # 必须为 aware（本地时区）
    assert weekday == dt.strftime("%A")


async def test_calculate_returns_result(tool_client: Client) -> None:
    """calculate 正常求值：整数结果不带小数点、支持括号/幂。"""
    res = await tool_client.call_tool("calculate", {"expression": "1+2*3"})
    assert res.is_error is False
    assert res.content[0].text == "7"
    res2 = await tool_client.call_tool("calculate", {"expression": "(2+3)**2"})
    assert res2.is_error is False
    assert res2.content[0].text == "25"


async def test_calculate_rejects_unsafe_expression(tool_client: Client) -> None:
    """calculate 拒绝白名单外语法（防注入）：函数调用 / 属性访问必须报错。"""
    res = await tool_client.call_tool(
        "calculate", {"expression": "__import__('os').system('echo hi')"}, raise_on_error=False
    )
    assert res.is_error is True
    res2 = await tool_client.call_tool("calculate", {"expression": ""}, raise_on_error=False)
    assert res2.is_error is True


# —— streamable-http 传输：验证远程部署形态（进程内 uvicorn + 动态空闲端口，无端口冲突）——


async def test_streamable_http_e2e() -> None:
    """以 streamable-http 起真实 HTTP 端点，经网络协议连客户端并调用工具。"""
    async with run_server_async(mcp, transport="streamable-http") as url:
        async with Client(StreamableHttpTransport(url)) as client:
            tools = await client.list_tools()
            assert {t.name for t in tools} == {"current_datetime", "calculate"}
            res = await client.call_tool("current_datetime", {})
            assert res.is_error is False
            assert " | " in res.content[0].text
