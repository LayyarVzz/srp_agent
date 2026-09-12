"""services/lark_mcp FastMCP 服务测试（进程内 Client + fake runner，零 CLI、零网络）。

工具层契约：域级命令组组装正确、as_user 身份位透传、写工具收件人校验、
CLI 失败归一为工具错误、超长输出截断。argv 追加（--as/--format）属于
LarkCliRunner 的职责，在 tests/test_lark_cli.py 用真实桩进程验证。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastmcp import Client

import services.lark_mcp.server as server_module
from services.lark_mcp.cli import LarkCliError
from services.lark_mcp.server import mcp


class FakeLarkCliRunner:
    """记录 (域级命令组, as_user) 并返回预设载荷的 fake 执行器。"""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else {"ok": True}
        self.error: Exception | None = None
        self.max_output_chars = 10_000
        self.calls: list[tuple[list[str], bool]] = []

    async def run(self, args: list[str], *, as_user: bool = False) -> dict[str, Any]:
        self.calls.append((args, as_user))
        if self.error is not None:
            raise self.error
        return self.payload


@pytest.fixture
async def lark_env(monkeypatch) -> tuple[Client, FakeLarkCliRunner]:
    """同进程客户端 + 注入 fake runner（工具在调用时按模块全局名查找 runner）。"""
    fake = FakeLarkCliRunner()
    monkeypatch.setattr(server_module, "runner", fake)
    async with Client(mcp) as client:
        yield client, fake


async def test_im_send_message_defaults_to_bot(lark_env) -> None:
    """默认 bot 身份（as_user=False），域级命令组与收件人标志正确组装。"""
    client, fake = lark_env
    res = await client.call_tool("lark_im_send_message", {"text": "你好", "chat_id": "oc_x"})
    assert res.is_error is False
    assert fake.calls[-1] == (
        ["im", "+messages-send", "--text", "你好", "--chat-id", "oc_x"],
        False,
    )
    assert json.loads(res.content[0].text) == {"ok": True}


async def test_im_send_message_as_user_and_open_id(lark_env) -> None:
    """user_id 收件人 + as_user=true（用户身份）透传。"""
    client, fake = lark_env
    res = await client.call_tool(
        "lark_im_send_message", {"text": "hi", "user_id": "ou_x", "as_user": True}
    )
    assert res.is_error is False
    assert fake.calls[-1] == (["im", "+messages-send", "--text", "hi", "--user-id", "ou_x"], True)


async def test_im_send_message_rejects_missing_receiver(lark_env) -> None:
    """收件人缺失 → 工具错误（execution 语义；schema 必填缺失由 Agent 侧归一 missing_argument）。"""
    client, _ = lark_env
    res = await client.call_tool(
        "lark_im_send_message", {"text": "hi"}, raise_on_error=False
    )
    assert res.is_error is True


async def test_im_send_message_rejects_both_receivers(lark_env) -> None:
    """chat_id 与 user_id 互斥（CLI 契约），同时给出必须报错。"""
    client, _ = lark_env
    res = await client.call_tool(
        "lark_im_send_message",
        {"text": "hi", "chat_id": "oc_x", "user_id": "ou_x"},
        raise_on_error=False,
    )
    assert res.is_error is True


async def test_calendar_agenda_default_today(lark_env) -> None:
    """无日期参数 → 默认今天（CLI 语义），不追加 --start/--end；「我的日程」走 user 身份。"""
    client, fake = lark_env
    res = await client.call_tool("lark_calendar_get_agenda", {"as_user": True})
    assert res.is_error is False
    assert fake.calls[-1] == (["calendar", "+agenda"], True)


async def test_calendar_agenda_with_range(lark_env) -> None:
    client, fake = lark_env
    res = await client.call_tool(
        "lark_calendar_get_agenda", {"start": "2026-09-12", "end": "2026-09-13"}
    )
    assert res.is_error is False
    assert fake.calls[-1] == (
        ["calendar", "+agenda", "--start", "2026-09-12", "--end", "2026-09-13"],
        False,
    )


async def test_task_create_minimal(lark_env) -> None:
    client, fake = lark_env
    res = await client.call_tool("lark_task_create", {"summary": "写周报"})
    assert res.is_error is False
    assert fake.calls[-1] == (["task", "+create", "--summary", "写周报"], False)


async def test_task_create_with_due_and_description(lark_env) -> None:
    client, fake = lark_env
    res = await client.call_tool(
        "lark_task_create",
        {"summary": "交作业", "due": "date:2026-09-20", "description": "第五章"},
    )
    assert res.is_error is False
    assert fake.calls[-1] == (
        [
            "task",
            "+create",
            "--summary",
            "交作业",
            "--due",
            "date:2026-09-20",
            "--description",
            "第五章",
        ],
        False,
    )


async def test_docs_read_uses_markdown_format(lark_env) -> None:
    client, fake = lark_env
    res = await client.call_tool("lark_docs_read", {"doc": "https://xxx.feishu.cn/docs/abc"})
    assert res.is_error is False
    assert fake.calls[-1] == (
        [
            "docs",
            "+fetch",
            "--doc",
            "https://xxx.feishu.cn/docs/abc",
            "--doc-format",
            "markdown",
        ],
        False,
    )


async def test_contact_resolve_forces_user_identity(lark_env) -> None:
    """contact 搜索仅支持 user 身份（CLI 限制），服务层固定 as_user=True。"""
    client, fake = lark_env
    res = await client.call_tool("lark_contact_resolve_name", {"query": "张三"})
    assert res.is_error is False
    assert fake.calls[-1] == (["contact", "+search-user", "--query", "张三"], True)


async def test_cli_failure_maps_to_tool_error(monkeypatch) -> None:
    """lark-cli 调用失败（LarkCliError）→ 工具错误结果（Agent 侧归一 execution）。"""
    fake = FakeLarkCliRunner()
    fake.error = LarkCliError("lark-cli im +messages-send 失败：boom")
    monkeypatch.setattr(server_module, "runner", fake)
    async with Client(mcp) as client:
        res = await client.call_tool(
            "lark_im_send_message", {"text": "hi", "chat_id": "oc_x"}, raise_on_error=False
        )
    assert res.is_error is True


async def test_output_truncation(monkeypatch) -> None:
    """超长输出按 runner.max_output_chars 截断并带截断标记（服务侧护栏兜底）。"""
    fake = FakeLarkCliRunner(payload={"ok": True, "data": "x" * 500})
    fake.max_output_chars = 50
    monkeypatch.setattr(server_module, "runner", fake)
    async with Client(mcp) as client:
        res = await client.call_tool("lark_im_send_message", {"text": "hi", "chat_id": "oc_x"})
    text = res.content[0].text
    assert len(text) == 50 + len("…[输出已截断]")
    assert text.endswith("…[输出已截断]")


async def test_write_tools_declare_high_risk_in_description() -> None:
    """写工具（im/task）description 必须含高风险标注；读工具描述非空（dev-version5.0 §3.2）。"""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    descriptions = {tool.name: tool.description or "" for tool in tools}
    for write_tool in ("lark_im_send_message", "lark_task_create"):
        assert "高风险写操作" in descriptions[write_tool], write_tool
    for read_tool in ("lark_calendar_get_agenda", "lark_docs_read", "lark_contact_resolve_name"):
        assert descriptions[read_tool], read_tool
