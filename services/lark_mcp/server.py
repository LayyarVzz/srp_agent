"""lark_mcp FastMCP 服务：飞书工具面（dev-version5.0 §3.2 的 5 个工具 + v5.1 绑定工具）。

工具面（读/写与身份约定见各工具 docstring，即 MCP description）：
- lark_im_send_message      im 消息发送（写，高风险标注）
- lark_calendar_get_agenda  calendar 日程查询（读）
- lark_task_create          task 任务创建（写，高风险标注）
- lark_docs_read            docs 文档读取（读）
- lark_contact_resolve_name contact 通讯录解析（读）

v5.1 身份模型（dev-version5.1.md §4/§5/§6）：
- **无 bot 身份**（D5.1.4）：argv 恒 `--as user`，原 `as_user` 开关已删除
  —— 身份不再是「LLM 可猜的开关」，而是「谁在调用」；
- 每个工具带 `_lark_scope: str | None`：由 agent 侧 `LarkScopeInterceptor` 从图状态
  `user_id` **无条件注入**，并已从 `tool_call_schema` 剥离 → **LLM 看不到也填不了**；
- 作用域 → 各自绑定的 UAT（加密落库，必要时刷新），**互不可见、互不串号**。

错误契约：参数校验失败 raise ValueError；未绑定 / 刷新失败 raise
`LarkUnboundError`（消息带 `tool_error.lark_unbound` 前缀 → 图侧走绑定引导）；
lark-cli 调用失败 raise `LarkCliError` → Agent 侧归一 `tool_error.execution`。
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
from services.lark_mcp.credentials import ConfigCredentialProvider, LarkCredentialProvider
from services.lark_mcp.models import LarkCliCredentials
from shared.lark.errors import LarkUnboundError

_settings = LarkMCPRuntimeSettings()
# 模块级单例：工具函数共享同一 runner（命令解析/超时/截断策略一致）；
# 测试经 monkeypatch 替换为 fake runner（见 tests/test_lark_mcp.py）。
runner = LarkCliRunner(
    command=_settings.lark_cli_command,
    timeout_s=_settings.lark_cli_timeout_s,
    max_output_chars=_settings.lark_output_max_chars,
)
# 作用域 → 凭据解析器（默认配置态单用户；启用绑定时由装配层替换为绑定查库实现）。
credential_provider: LarkCredentialProvider = ConfigCredentialProvider(_settings)

mcp = FastMCP("lark-mcp")


def _serialize(payload: dict[str, Any]) -> str:
    """把 CLI JSON 载荷收敛为工具输出字符串（超长截断，护栏在服务侧兜底）。"""
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > runner.max_output_chars:
        text = text[: runner.max_output_chars] + "…[输出已截断]"
    return text


def _require_scope(scope: str | None) -> str:
    """校验作用域非空（缺失说明装配/注入链路断裂，属确定性拒绝而非猜身份）。"""
    if not scope:
        raise ValueError("缺少调用作用域 _lark_scope（应由 agent 侧拦截器注入）")
    return scope


async def _run(args: list[str], *, scope: str | None) -> str:
    """统一执行入口：作用域 → 该用户凭据 → lark-cli 子进程 → JSON 收敛输出。

    WHY 统一入口：五个工具的执行纪律完全一致（解析凭据 → 跑 CLI → 截断），
    收敛在此避免逐工具重复「未绑定判定」逻辑（漏一处即产生身份歧义）。
    """
    credentials: LarkCliCredentials = await credential_provider.resolve(_require_scope(scope))
    return _serialize(await runner.run(args, credentials=credentials))


async def lark_im_send_message(
    text: str,
    chat_id: str | None = None,
    user_id: str | None = None,
    _lark_scope: str | None = None,
) -> str:
    """发送飞书消息（高风险写操作：仅在用户明确要求发送时调用）。

    收件人二选一：chat_id（会话 ID，oc_ 开头）或 user_id（用户 open_id，ou_ 开头，
    不知道 ID 时先用 lark_contact_resolve_name 按姓名解析）。
    消息以**当前绑定用户本人**的身份发送（无机器人身份）。
    """
    if bool(chat_id) == bool(user_id):
        raise ValueError("收件人必须二选一：chat_id 或 user_id")
    args = ["im", "+messages-send", "--text", text]
    args += ["--chat-id", chat_id] if chat_id else ["--user-id", user_id or ""]
    return await _run(args, scope=_lark_scope)


async def lark_calendar_get_agenda(
    start: str | None = None,
    end: str | None = None,
    _lark_scope: str | None = None,
) -> str:
    """查询飞书日程列表（默认今天；start/end 为 ISO 8601 日期或时间，可省略）。

    查询的是**当前绑定用户自己的**日程（身份恒为本人，无 bot 视角）。
    """
    args = ["calendar", "+agenda"]
    if start:
        args += ["--start", start]
    if end:
        args += ["--end", end]
    return await _run(args, scope=_lark_scope)


async def lark_task_create(
    summary: str,
    due: str | None = None,
    description: str | None = None,
    _lark_scope: str | None = None,
) -> str:
    """创建飞书任务（高风险写操作：仅在用户明确要求创建时调用）。

    summary 为任务标题（必填）；due 支持 ISO 8601、date:YYYY-MM-DD 或相对日期
    （如 +2d 表示两天后）；description 为任务详情（可选）。
    任务创建在**当前绑定用户自己的**任务清单下。
    """
    args = ["task", "+create", "--summary", summary]
    if due:
        args += ["--due", due]
    if description:
        args += ["--description", description]
    return await _run(args, scope=_lark_scope)


async def lark_docs_read(doc: str, _lark_scope: str | None = None) -> str:
    """读取飞书文档正文（doc 传文档 URL 或文档 token，返回 markdown 文本）。

    以**当前绑定用户**的身份读取（可读该用户有权访问的文档）。输出超长会被截断。
    """
    args = ["docs", "+fetch", "--doc", doc, "--doc-format", "markdown"]
    return await _run(args, scope=_lark_scope)


async def lark_contact_resolve_name(query: str, _lark_scope: str | None = None) -> str:
    """按姓名/邮箱等关键词搜索飞书用户，返回 open_id 与 p2p_chat_id。

    结果中的 user open_id（ou_ 开头）或 p2p_chat_id 可直接作为发消息收件人。
    搜索以**当前绑定用户**的身份执行（通讯录可见范围即该用户本人的可见范围）。
    """
    args = ["contact", "+search-user", "--query", query]
    return await _run(args, scope=_lark_scope)


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


__all__ = [
    "LarkUnboundError",
    "credential_provider",
    "lark_calendar_get_agenda",
    "lark_contact_resolve_name",
    "lark_docs_read",
    "lark_im_send_message",
    "lark_task_create",
    "mcp",
    "runner",
]
