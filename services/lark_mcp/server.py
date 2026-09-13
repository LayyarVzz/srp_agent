"""lark_mcp FastMCP 服务：飞书最简工具集（dev-version5.0 §3.2，5 个工具）。

工具面（读/写与身份约定见各工具 docstring，即 MCP description）：
- lark_im_send_message      im 消息发送（写，高风险标注）
- lark_calendar_get_agenda  calendar 日程查询（读）
- lark_task_create          task 任务创建（写，高风险标注）
- lark_docs_read            docs 文档读取（读）
- lark_contact_resolve_name contact 通讯录解析（读，仅 user 身份）

错误契约：工具内参数校验失败 raise ValueError（FastMCP 转错误结果）；
lark-cli 调用失败 raise LarkCliError → Agent 侧统一归一 `tool_error.execution`。
工具输出经 `lark_output_max_chars` 截断，一律视为不可信数据（ToolMessage 语义）。
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.lark_mcp.cli import LarkCliRunner
from services.lark_mcp.config import LarkMCPRuntimeSettings

_settings = LarkMCPRuntimeSettings()
# 模块级单例：工具函数共享同一 runner（命令解析/超时/截断策略一致）；
# 测试经 monkeypatch 替换为 fake runner（见 tests/test_lark_mcp.py）。
runner = LarkCliRunner(
    command=_settings.lark_cli_command,
    timeout_s=_settings.lark_cli_timeout_s,
    max_output_chars=_settings.lark_output_max_chars,
)

mcp = FastMCP("lark-mcp")


def _serialize(payload: dict[str, Any]) -> str:
    """把 CLI JSON 载荷收敛为工具输出字符串（超长截断，护栏在服务侧兜底）。"""
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > runner.max_output_chars:
        text = text[: runner.max_output_chars] + "…[输出已截断]"
    return text


async def lark_im_send_message(
    text: str,
    chat_id: str | None = None,
    user_id: str | None = None,
    as_user: bool = False,
) -> str:
    """发送飞书消息（高风险写操作：仅在用户明确要求发送时调用）。

    收件人二选一：chat_id（会话 ID，oc_ 开头）或 user_id（用户 open_id，ou_ 开头，
    不知道 ID 时先用 lark_contact_resolve_name 按姓名解析）。
    默认以 bot 身份发送；as_user=true 表示以当前登录用户身份发送。
    """
    if bool(chat_id) == bool(user_id):
        raise ValueError("收件人必须二选一：chat_id 或 user_id")
    args = ["im", "+messages-send", "--text", text]
    args += ["--chat-id", chat_id] if chat_id else ["--user-id", user_id or ""]
    return _serialize(await runner.run(args, as_user=as_user))


async def lark_calendar_get_agenda(
    start: str | None = None,
    end: str | None = None,
    as_user: bool = False,
) -> str:
    """查询飞书日程列表（默认今天；start/end 为 ISO 8601 日期或时间，可省略）。

    查询「我的日程」必须传 as_user=true：bot 身份只能看到 bot 自己的日历，
    看不到登录用户的日程。
    """
    args = ["calendar", "+agenda"]
    if start:
        args += ["--start", start]
    if end:
        args += ["--end", end]
    return _serialize(await runner.run(args, as_user=as_user))


async def lark_task_create(
    summary: str,
    due: str | None = None,
    description: str | None = None,
    as_user: bool = False,
) -> str:
    """创建飞书任务（高风险写操作：仅在用户明确要求创建时调用）。

    summary 为任务标题（必填）；due 支持 ISO 8601、date:YYYY-MM-DD 或相对日期
    （如 +2d 表示两天后）；description 为任务详情（可选）。
    默认以 bot 身份创建；as_user=true 时以当前登录用户身份创建。
    """
    args = ["task", "+create", "--summary", summary]
    if due:
        args += ["--due", due]
    if description:
        args += ["--description", description]
    return _serialize(await runner.run(args, as_user=as_user))


async def lark_docs_read(doc: str, as_user: bool = False) -> str:
    """读取飞书文档正文（doc 传文档 URL 或文档 token，返回 markdown 文本）。

    默认以 bot 身份读取（要求 bot 对该文档有阅读权限）；as_user=true 时以
    当前登录用户身份读取（可读用户可见的文档）。输出超长会被截断。
    """
    args = ["docs", "+fetch", "--doc", doc, "--doc-format", "markdown"]
    return _serialize(await runner.run(args, as_user=as_user))


async def lark_contact_resolve_name(query: str) -> str:
    """按姓名/邮箱等关键词搜索飞书用户，返回 open_id 与 p2p_chat_id。

    结果中的 user open_id（ou_ 开头）或 p2p_chat_id 可直接作为发消息收件人。
    飞书通讯录搜索接口仅支持用户身份，本工具固定以登录用户身份执行
    （需已完成 lark-cli 用户授权 `auth login`）。
    """
    args = ["contact", "+search-user", "--query", query]
    return _serialize(await runner.run(args, as_user=True))


# 工具以模块级函数显式命名注册（镜像 tools_mcp/server.py 惯例）。
mcp.tool(name="lark_im_send_message")(lark_im_send_message)
mcp.tool(name="lark_calendar_get_agenda")(lark_calendar_get_agenda)
mcp.tool(name="lark_task_create")(lark_task_create)
mcp.tool(name="lark_docs_read")(lark_docs_read)
mcp.tool(name="lark_contact_resolve_name")(lark_contact_resolve_name)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """存活探针（不影响 MCP 协议端点，镜像 tools_mcp）。"""
    return JSONResponse({"status": "ok"})
