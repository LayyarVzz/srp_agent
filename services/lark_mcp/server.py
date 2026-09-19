"""lark_mcp FastMCP 服务：飞书工具面（dev-version5.0 §3.2 的 5 个工具 + v5.1 绑定工具）。

工具面（读/写与身份约定见各工具 docstring，即 MCP description）：
- lark_im_send_message      im 消息发送（写，高风险标注）
- lark_calendar_get_agenda  calendar 日程查询（读）
- lark_task_create          task 任务创建（写，高风险标注）
- lark_docs_read            docs 文档读取（读）
- lark_contact_resolve_name contact 通讯录解析（读）
- lark_bind_start / lark_bind_complete / lark_bind_status / lark_unbind（v5.1 绑定闭环）

v5.1 身份模型（dev-version5.1.md §4/§5/§6）：
- **无 bot 身份**（D5.1.4）：argv 恒 `--as user`，原 `as_user` 开关已删除
  —— 身份不再是「LLM 可猜的开关」，而是「谁在调用」；
- 每个工具带 `_lark_scope: str | None`：由 agent 侧 `LarkScopeInterceptor` 从图状态
  `user_id` **无条件注入**；agent 侧以 `visible_tools()` 给模型绑定**剥离该参数**的
  副本 → **LLM 看不到也填不了**（服务端 `inputSchema` 仍保留它，故注入的实参可落地）；
- 作用域 → 各自绑定的 UAT（加密落库，必要时刷新），**互不可见、互不串号**。

装配（§8.2）：进程启动时按「绑定功能是否可用」组装凭据提供者与绑定服务 ——
有库 + 有 `lark_token_key` + 有应用凭据 → `BindingCredentialProvider`（多用户隔离）；
否则退回配置态单用户凭据（本地 demo / 既有 v5.0 用法，零回归）。
绑定域的建表与连接池生命周期由本模块的 `_binding_lifespan` 承担（启动 `setup()` /
退出 `aclose()`），DSN 取 `LARK_DATABASE_URL`，缺省回退 `DATABASE_URL`
（见 config.py 的 `binding_database_url`）。

错误契约：参数校验失败 raise ValueError；未绑定 / 刷新失败 raise
`LarkUnboundError`（消息带 `tool_error.lark_unbound` 前缀 → 图侧走绑定引导）；
lark-cli 调用失败 raise `LarkCliError` → Agent 侧归一 `tool_error.execution`。
工具输出经 `lark_output_max_chars` 截断，一律视为不可信数据（ToolMessage 语义）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.lark_mcp.binding import LarkBindingService
from services.lark_mcp.cli import LarkCliRunner
from services.lark_mcp.config import LarkMCPRuntimeSettings
from services.lark_mcp.credentials import (
    ConfigCredentialProvider,
    LarkCredentialProvider,
    build_credential_provider,
)
from services.lark_mcp.models import LarkCliCredentials
from shared.lark.errors import LarkBoundError, LarkUnboundError
from shared.lark.oauth import LarkOAuthClient
from shared.lark.repository import build_lark_binding_repository
from shared.lark.token_cipher import build_token_cipher

logger = logging.getLogger(__name__)

_settings = LarkMCPRuntimeSettings()
# 模块级单例：工具函数共享同一 runner（命令解析/超时/截断策略一致）；
# 测试经 monkeypatch 替换为 fake runner（见 tests/test_lark_mcp.py）。
runner = LarkCliRunner(
    command=_settings.lark_cli_command,
    timeout_s=_settings.lark_cli_timeout_s,
    max_output_chars=_settings.lark_output_max_chars,
)


def _build_binding_pieces(
    settings: LarkMCPRuntimeSettings,
) -> tuple[LarkOAuthClient | None, Any]:
    """按配置组装 OAuth 客户端与绑定仓库；绑定不可用时返回 (None, None)。

    WHY 在建服务前判定：绑定链路的三个前提（应用凭据 / 加密密钥 / 库）任一缺失，
    都应明确表现为「绑定不可用」而非运行期才崩 —— fail-closed（§7/§13）。
    """
    app_id = settings.lark_app_id or ""
    app_secret = settings.lark_app_secret.get_secret_value()
    if not (settings.lark_binding_enabled and app_id and app_secret):
        logger.info("未配置飞书应用凭据或绑定被关闭：绑定工具将返回不可用说明")
        return None, None
    cipher = build_token_cipher(settings.lark_token_key.get_secret_value())
    repository = build_lark_binding_repository(
        cipher,
        settings.binding_database_url,
        refresh_skew_s=settings.lark_token_refresh_skew_s,
    )
    oauth = LarkOAuthClient(
        app_id=app_id,
        app_secret=app_secret,
        accounts_base_url=settings.lark_accounts_base_url,
        open_base_url=settings.lark_open_base_url,
        timeout_s=settings.lark_oauth_timeout_s,
    )
    return oauth, repository


_oauth_client, _binding_repository = _build_binding_pieces(_settings)

# 作用域 → 凭据解析器：绑定可用时按用户隔离；否则退回配置态单用户（零回归）。
credential_provider: LarkCredentialProvider = build_credential_provider(
    _settings,
    repository=_binding_repository,
    oauth=_oauth_client,
)
# 绑定服务（未启用时保持 None，工具据此返回确定性不可用说明）。
binding_service: LarkBindingService | None = (
    LarkBindingService(
        repository=_binding_repository,
        oauth=_oauth_client,
        app_id=_settings.lark_app_id or "",
    )
    if _binding_repository is not None and _oauth_client is not None
    else None
)


@asynccontextmanager
async def _binding_lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """服务生命周期：启动建表、退出关连接池（绑定域存储的**唯一**装配点）。

    WHY 必须由本服务承担，且必须在启动期完成：
    - `lark_bindings` / `lark_device_flows` 两张表的消费方是**本进程**（agent 经 MCP
      调用过来，不直接持库），故建表只能在此发生 —— 放到 api 侧建是错的：
      绑定域在无 DSN 时走 `sqlite+aiosqlite:///:memory:`，api 建的库与本子进程的库
      是两个互不相干的内存库，看着成功、实际无效；
    - 绑定状态要跨**多次工具调用**共享（`lark_bind_start` 写待定态 → 用户授权 →
      `lark_bind_complete` 读回），而每次工具调用都是新子进程 → 每进程都必须先建表；
    - 建表是幂等的（`create_all(checkfirst=True)`），多副本同时启动安全。
    fail-fast 取舍：建表失败在**进程启动**即暴露（MCP 连接失败 → api 侧既有降级），
    而非等用户第一次说「帮我绑定飞书」才失败 —— 后者会退化成一次误导性的道歉回答。
    """
    if _binding_repository is not None:
        await _binding_repository.setup()
        logger.info("飞书绑定域建表完成（幂等）：lark_bindings / lark_device_flows")
    else:
        logger.info("飞书绑定域未启用（缺应用凭据或 lark_token_key），跳过建表")
    try:
        yield
    finally:
        # 关闭连接池：退出时不关会残留 aiosqlite 工作线程 / psycopg 连接。
        if _binding_repository is not None:
            await _binding_repository.aclose()


mcp = FastMCP("lark-mcp", lifespan=_binding_lifespan)


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


async def _require_binding_service(scope: str | None) -> tuple[LarkBindingService, str]:
    """取绑定服务与作用域；绑定未启用 → `LarkBoundError`（确定性说明，非静默失败）。"""
    service = binding_service
    if service is None:
        raise LarkBoundError(
            "飞书绑定功能未启用（服务端未配置 lark_app_id/app_secret 或 lark_token_key）"
        )
    return service, _require_scope(scope)


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


# —— v5.1 绑定工具（§7）——
# 作用域形参 `_lark_scope` 同样由 agent 侧拦截器注入：**谁在绑定**不由模型决定，
# 故 A 用户无法代 B 用户完成绑定（与工具面同一套隔离机制）。


async def lark_bind_start(_lark_scope: str | None = None) -> str:
    """发起飞书账号绑定：生成授权链接（10 分钟内有效，需用飞书扫码/登录确认）。

    典型流程：用户说「帮我绑定飞书」→ 调用本工具 → 把返回的链接给用户 →
    用户完成授权后调用 lark_bind_complete。
    """
    service, scope = await _require_binding_service(_lark_scope)
    return await service.start(scope)


async def lark_bind_complete(_lark_scope: str | None = None) -> str:
    """完成绑定：检查用户是否已在浏览器完成授权，完成则保存授权信息。

    用户说「我已完成授权」时调用。若用户还没完成授权，返回提示（稍后再试一次即可）。
    """
    service, scope = await _require_binding_service(_lark_scope)
    return await service.complete(scope)


async def lark_bind_status(_lark_scope: str | None = None) -> str:
    """查询当前用户是否已绑定飞书账号、绑定的是哪个账号、授权是否有效。"""
    service, scope = await _require_binding_service(_lark_scope)
    return await service.status(scope)


async def lark_unbind(_lark_scope: str | None = None) -> str:
    """解除飞书绑定（高风险写操作：仅在用户明确要求解绑时调用）。

    会尽力撤销服务端授权并删除本地保存的授权信息；解绑后飞书相关工具需重新绑定。
    """
    service, scope = await _require_binding_service(_lark_scope)
    return await service.unbind(scope)


# 工具以模块级函数显式命名注册（镜像 tools_mcp/server.py 惯例）。
mcp.tool(name="lark_im_send_message")(lark_im_send_message)
mcp.tool(name="lark_calendar_get_agenda")(lark_calendar_get_agenda)
mcp.tool(name="lark_task_create")(lark_task_create)
mcp.tool(name="lark_docs_read")(lark_docs_read)
mcp.tool(name="lark_contact_resolve_name")(lark_contact_resolve_name)
mcp.tool(name="lark_bind_start")(lark_bind_start)
mcp.tool(name="lark_bind_complete")(lark_bind_complete)
mcp.tool(name="lark_bind_status")(lark_bind_status)
mcp.tool(name="lark_unbind")(lark_unbind)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """存活探针（不影响 MCP 协议端点，镜像 tools_mcp）。"""
    return JSONResponse({"status": "ok"})


__all__ = [
    "ConfigCredentialProvider",
    "LarkUnboundError",
    "binding_service",
    "credential_provider",
    "lark_bind_complete",
    "lark_bind_start",
    "lark_bind_status",
    "lark_calendar_get_agenda",
    "lark_contact_resolve_name",
    "lark_docs_read",
    "lark_im_send_message",
    "lark_task_create",
    "lark_unbind",
    "mcp",
    "runner",
]
