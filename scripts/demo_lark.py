"""V5-M1 验收脚本：飞书工具面真实链路演示（dev-version5.0 §3.5）。

经生产装配 `AgentRuntime.create()` 拉起全部 MCP 服务（含 lark_mcp 门控登记），
跑一轮「用户指令 → 选择飞书工具 → 执行 → 整合回答」，打印状态轨迹与 tool_trace。

前置条件：
    1. `.env` 配置 LLM_API_KEY（未配置则跳过）；
    2. lark-cli 已安装且 `LARK_CLI_COMMAND` 可探测（否则 lark_mcp 不登记，零回归降级，
       飞书类问题将走直答/澄清路径）；
    3. 已完成一次性鉴权（`config init` + `auth login`，见 services/lark_mcp/README.md）。

用法（在仓库根目录执行）：
    uv run python -m scripts.demo_lark                      # 默认演示问题集
    uv run python -m scripts.demo_lark "我明天有什么日程"     # 自定义问题（可多条）
    uv run python -m scripts.demo_lark --doc <文档URL>       # 文档总结演示
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agent.runtime import AgentRuntime
from agent.share.eventloop import ensure_selector_event_loop
from services.lark_mcp.cli import resolve_lark_cli_command
from settings import configure_logging, get_settings

logger = logging.getLogger(__name__)

DEMO_USER_ID = "demo-lark-user"

# 文档 §3.5 验收口令（发消息/查日程/建任务；读文档经 --doc 传入）。
DEFAULT_QUESTIONS = [
    "我今天有什么日程？",
    "帮我在飞书创建一个任务：标题「整理周报」，明天截止。",
]


def _build_questions(argv: list[str]) -> list[str]:
    """命令行问题优先：--doc <url> 转「总结这篇飞书文档」，其余位置参数按序作问题。"""
    questions: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--doc" and i + 1 < len(argv):
            questions.append(f"请总结这篇飞书文档的主要内容：{argv[i + 1]}")
            i += 2
            continue
        questions.append(argv[i])
        i += 1
    return questions or DEFAULT_QUESTIONS


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    if not settings.llm_api_key.get_secret_value():
        logger.warning("未配置 LLM_API_KEY，跳过真实调用（可先运行 pytest 验证离线路径）")
        return
    if not resolve_lark_cli_command(settings.lark_cli_command):
        logger.warning(
            "lark-cli 未探测到（LARK_CLI_COMMAND=%r），lark_mcp 将不登记（零回归降级）"
            "——安装与鉴权步骤见 services/lark_mcp/README.md",
            settings.lark_cli_command,
        )

    # 生产装配：lark_mcp 经 _build_mcp_servers 门控登记（启用 + 探测成功）。
    runtime = await AgentRuntime.create(settings=settings)
    try:
        for text in _build_questions(sys.argv[1:]):
            ctx = await runtime.sessions.create(user_id=DEMO_USER_ID)
            logger.info("用户(session=%s): %s", ctx.session_id, text)
            async for event, payload in runtime.chat_stream(
                user_id=DEMO_USER_ID, session_id=ctx.session_id, text=text
            ):
                if event == "status":
                    logger.info(
                        "  [status] %s%s%s",
                        payload.status,
                        f" tool={payload.tool_name}" if payload.tool_name else "",
                        f" {payload.message}" if payload.message else "",
                    )
                elif event == "tool":
                    result = payload.result
                    logger.info(
                        "  [tool] %s → %s（%sms）",
                        payload.tool_name,
                        "ok" if result is not None and result.ok else "error",
                        result.duration_ms if result is not None else 0,
                    )
                elif event == "done":
                    resp = payload
                    logger.info("回答(%s): %s", resp.finished_reason, resp.reply)
                    for record in resp.tool_trace:
                        ok = record.result.ok if record.result is not None else None
                        logger.info(
                            "  [trace] %s → %s", record.tool_name, "ok" if ok else "error"
                        )
    finally:
        await runtime.aclose()


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
