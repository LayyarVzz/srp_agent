"""services/lark_mcp FastMCP 服务测试（进程内 Client + fake runner，零 CLI、零网络）。

工具层契约：域级命令组组装正确、作用域 → 凭据解析后透传 runner、写工具收件人校验、
CLI 失败归一为工具错误、超长输出截断。

v5.1 变更（dev-version5.1.md §5.4/§6.4）：工具签名新增 `_lark_scope`（由 agent 侧
拦截器注入，LLM 不可见）、**删除全部 `as_user` 参数**（身份恒为绑定用户本人）；
未绑定 → `LarkUnboundError`（`tool_error.lark_unbound` 语义）。argv 追加
（--as user / --format json）属于 LarkCliRunner 的职责，在 tests/test_lark_cli.py 验证。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastmcp import Client

import services.lark_mcp.server as server_module
from services.lark_mcp.models import LarkCliCredentials
from services.lark_mcp.server import mcp
from shared.lark.errors import LARK_UNBOUND_PREFIX, LarkCliError, LarkUnboundError

SCOPE = "user-a"
# 测试用假凭据（非真实密钥；S105/S106 对常量名/关键字名的通用启发式告警不适用）。
USER_A_UAT = "uat-user-a"
APP_SECRET = "sec"


def _creds(scope: str) -> LarkCliCredentials:
    """按作用域返回不同 UAT 的凭据（用于断言身份隔离）。"""
    return LarkCliCredentials(app_id="app", app_secret=APP_SECRET, user_access_token=f"uat-{scope}")


class FakeLarkCliRunner:
    """记录 (域级命令组, 该次调用所用凭据) 并返回预设载荷的 fake 执行器。"""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else {"ok": True}
        self.error: Exception | None = None
        self.max_output_chars = 10_000
        self.calls: list[tuple[list[str], LarkCliCredentials]] = []

    async def run(self, args: list[str], *, credentials: LarkCliCredentials) -> dict[str, Any]:
        self.calls.append((args, credentials))
        if self.error is not None:
            raise self.error
        return self.payload


class FakeCredentialProvider:
    """作用域 → 凭据的 fake 解析器（未注册的作用域视为未绑定）。"""

    def __init__(self, *, bound: set[str] | None = None, error: Exception | None = None) -> None:
        self.bound = bound if bound is not None else {SCOPE}
        self.error = error
        self.scopes: list[str] = []

    async def resolve(self, scope: str) -> LarkCliCredentials:
        self.scopes.append(scope)
        if self.error is not None:
            raise self.error
        if scope not in self.bound:
            raise LarkUnboundError(f"用户 {scope} 尚未绑定飞书")
        return _creds(scope)


@pytest.fixture
async def lark_env(monkeypatch) -> tuple[Client, FakeLarkCliRunner, FakeCredentialProvider]:
    """同进程客户端 + 注入 fake runner/凭据解析器（工具按模块全局名查找）。"""
    fake = FakeLarkCliRunner()
    provider = FakeCredentialProvider()
    monkeypatch.setattr(server_module, "runner", fake)
    monkeypatch.setattr(server_module, "credential_provider", provider)
    async with Client(mcp) as client:
        yield client, fake, provider


@pytest.fixture
async def list_tools() -> list[Any]:
    async with Client(mcp) as client:
        return await client.list_tools()


async def test_im_send_message_by_chat_id(lark_env) -> None:
    """域级命令组与收件人标志正确组装；身份来自作用域（此处为绑定的 user-a）。"""
    client, fake, provider = lark_env
    res = await client.call_tool(
        "lark_im_send_message", {"text": "你好", "chat_id": "oc_x", "_lark_scope": SCOPE}
    )
    assert res.is_error is False
    args, creds = fake.calls[-1]
    assert args == ["im", "+messages-send", "--text", "你好", "--chat-id", "oc_x"]
    assert creds.user_access_token == USER_A_UAT
    assert provider.scopes == [SCOPE]
    assert json.loads(res.content[0].text) == {"ok": True}


async def test_im_send_message_by_open_id(lark_env) -> None:
    """user_id 收件人分支。"""
    client, fake, _ = lark_env
    res = await client.call_tool(
        "lark_im_send_message", {"text": "hi", "user_id": "ou_x", "_lark_scope": SCOPE}
    )
    assert res.is_error is False
    assert fake.calls[-1][0] == ["im", "+messages-send", "--text", "hi", "--user-id", "ou_x"]


async def test_identity_isolation_between_scopes(lark_env) -> None:
    """V51-M1 核心：不同作用域 → 各自的 UAT，互不串号（A 的调用绝不用 B 的凭据）。"""
    client, fake, provider = lark_env
    provider.bound = {"user-a", "user-b"}
    await client.call_tool("lark_calendar_get_agenda", {"_lark_scope": "user-a"})
    await client.call_tool("lark_calendar_get_agenda", {"_lark_scope": "user-b"})
    assert [c.user_access_token for _, c in fake.calls] == [USER_A_UAT, "uat-user-b"]


async def test_unbound_user_gets_lark_unbound_error(monkeypatch) -> None:
    """未绑定用户 → 工具错误且消息带 `tool_error.lark_unbound` 前缀（图侧走引导，§6.3）。"""
    fake = FakeLarkCliRunner()
    monkeypatch.setattr(server_module, "runner", fake)
    monkeypatch.setattr(server_module, "credential_provider", FakeCredentialProvider(bound=set()))
    async with Client(mcp) as client:
        res = await client.call_tool(
            "lark_calendar_get_agenda", {"_lark_scope": SCOPE}, raise_on_error=False
        )
    assert res.is_error is True
    assert LARK_UNBOUND_PREFIX in res.content[0].text
    assert fake.calls == []  # 未绑定不得启动子进程（无 bot 兜底）


async def test_missing_scope_is_rejected(lark_env) -> None:
    """作用域缺失 → 确定性拒绝（注入链路断裂不猜身份）。"""
    client, fake, _ = lark_env
    res = await client.call_tool("lark_calendar_get_agenda", {}, raise_on_error=False)
    assert res.is_error is True
    assert fake.calls == []


async def test_im_send_message_rejects_missing_receiver(lark_env) -> None:
    """收件人缺失 → 工具错误（execution 语义；schema 必填缺失由 Agent 侧归一 missing_argument）。"""
    client, _, _ = lark_env
    res = await client.call_tool(
        "lark_im_send_message", {"text": "hi", "_lark_scope": SCOPE}, raise_on_error=False
    )
    assert res.is_error is True


async def test_im_send_message_rejects_both_receivers(lark_env) -> None:
    """chat_id 与 user_id 互斥（CLI 契约），同时给出必须报错。"""
    client, _, _ = lark_env
    res = await client.call_tool(
        "lark_im_send_message",
        {"text": "hi", "chat_id": "oc_x", "user_id": "ou_x", "_lark_scope": SCOPE},
        raise_on_error=False,
    )
    assert res.is_error is True


async def test_calendar_agenda_default_today(lark_env) -> None:
    """无日期参数 → 默认今天（CLI 语义），不追加 --start/--end；查的是绑定用户自己的日程。"""
    client, fake, _ = lark_env
    res = await client.call_tool("lark_calendar_get_agenda", {"_lark_scope": SCOPE})
    assert res.is_error is False
    assert fake.calls[-1][0] == ["calendar", "+agenda"]


async def test_calendar_agenda_with_range(lark_env) -> None:
    client, fake, _ = lark_env
    res = await client.call_tool(
        "lark_calendar_get_agenda",
        {"start": "2026-09-12", "end": "2026-09-13", "_lark_scope": SCOPE},
    )
    assert res.is_error is False
    assert fake.calls[-1][0] == [
        "calendar",
        "+agenda",
        "--start",
        "2026-09-12",
        "--end",
        "2026-09-13",
    ]


async def test_task_create_minimal(lark_env) -> None:
    client, fake, _ = lark_env
    res = await client.call_tool("lark_task_create", {"summary": "写周报", "_lark_scope": SCOPE})
    assert res.is_error is False
    assert fake.calls[-1][0] == ["task", "+create", "--summary", "写周报"]


async def test_task_create_with_due_and_description(lark_env) -> None:
    client, fake, _ = lark_env
    res = await client.call_tool(
        "lark_task_create",
        {
            "summary": "交作业",
            "due": "date:2026-09-20",
            "description": "第五章",
            "_lark_scope": SCOPE,
        },
    )
    assert res.is_error is False
    assert fake.calls[-1][0] == [
        "task",
        "+create",
        "--summary",
        "交作业",
        "--due",
        "date:2026-09-20",
        "--description",
        "第五章",
    ]


async def test_docs_read_uses_markdown_format(lark_env) -> None:
    client, fake, _ = lark_env
    res = await client.call_tool(
        "lark_docs_read", {"doc": "https://xxx.feishu.cn/docs/abc", "_lark_scope": SCOPE}
    )
    assert res.is_error is False
    assert fake.calls[-1][0] == [
        "docs",
        "+fetch",
        "--doc",
        "https://xxx.feishu.cn/docs/abc",
        "--doc-format",
        "markdown",
    ]


async def test_contact_resolve_uses_scoped_identity(lark_env) -> None:
    """contact 搜索按绑定用户身份执行（无 bot 身份，身份即作用域）。"""
    client, fake, _ = lark_env
    res = await client.call_tool(
        "lark_contact_resolve_name", {"query": "张三", "_lark_scope": SCOPE}
    )
    assert res.is_error is False
    args, creds = fake.calls[-1]
    assert args == ["contact", "+search-user", "--query", "张三"]
    assert creds.user_access_token == USER_A_UAT


async def test_cli_failure_maps_to_tool_error(monkeypatch) -> None:
    """lark-cli 调用失败（LarkCliError）→ 工具错误结果（Agent 侧归一 execution）。"""
    fake = FakeLarkCliRunner()
    fake.error = LarkCliError("lark-cli im +messages-send 失败：boom")
    monkeypatch.setattr(server_module, "runner", fake)
    monkeypatch.setattr(server_module, "credential_provider", FakeCredentialProvider())
    async with Client(mcp) as client:
        res = await client.call_tool(
            "lark_im_send_message",
            {"text": "hi", "chat_id": "oc_x", "_lark_scope": SCOPE},
            raise_on_error=False,
        )
    assert res.is_error is True


async def test_output_truncation(monkeypatch) -> None:
    """超长输出按 runner.max_output_chars 截断并带截断标记（服务侧护栏兜底）。"""
    fake = FakeLarkCliRunner(payload={"ok": True, "data": "x" * 500})
    fake.max_output_chars = 50
    monkeypatch.setattr(server_module, "runner", fake)
    monkeypatch.setattr(server_module, "credential_provider", FakeCredentialProvider())
    async with Client(mcp) as client:
        res = await client.call_tool(
            "lark_im_send_message", {"text": "hi", "chat_id": "oc_x", "_lark_scope": SCOPE}
        )
    text = res.content[0].text
    assert len(text) == 50 + len("…[输出已截断]")
    assert text.endswith("…[输出已截断]")


async def test_write_tools_declare_high_risk_in_description(list_tools) -> None:
    """写工具（im/task）description 必须含高风险标注；读工具描述非空（dev-version5.0 §3.2）。"""
    descriptions = {tool.name: tool.description or "" for tool in list_tools}
    for write_tool in ("lark_im_send_message", "lark_task_create"):
        assert "高风险写操作" in descriptions[write_tool], write_tool
    for read_tool in ("lark_calendar_get_agenda", "lark_docs_read", "lark_contact_resolve_name"):
        assert descriptions[read_tool], read_tool


async def test_as_user_switch_removed_from_schema(list_tools) -> None:
    """V51-M4：`as_user` 开关从工具面整体删除（身份由作用域决定，无可被猜错的选择项）。"""
    for tool in list_tools:
        props = set((tool.inputSchema or {}).get("properties", {}))
        assert "as_user" not in props, tool.name
