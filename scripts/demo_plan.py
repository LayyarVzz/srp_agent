"""v4.0 T1 验收脚本：真实 LLM 跑通多步任务编排（Plan-and-Solve）。

复合任务 → `Intent.PLAN` → plan_task（PLANNING）→ 步骤级执行（第 i/N 步，
USING_TOOL）→ 整合回答（finished_reason）。日志可见：计划概述 / 每步进度 /
重规划（若单步失败触发）/ 最终 finished_reason。

注意：意图由真实 LLM 分类，非确定性——若某问被判成 TOOL_USE/CHAT（未走规划），
脚本会如实打印该意图并继续，属正常（ReAct 兜底仍可用，见 dev-version4.0.md §2.1）。

工具经 MCP 接入：默认以 stdio 子进程自动拉起 `services.tools_mcp`（自包含演示）；
配置了 `MCP_TRANSPORT=streamable-http` 时改为 streamable_http 连接对应端口。

用法（在仓库根目录执行）：
    uv run python -m scripts.demo_plan                  # 未配置 LLM_API_KEY 时跳过
    LLM_API_KEY=sk-xxx uv run python -m scripts.demo_plan  # DeepSeek
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.core.models import PlanResult
from agent.core.state import NODE_EXECUTE_STEP, NODE_PLAN_TASK, NODE_REPLAN_TASK
from agent.intent.models import Intent
from agent.llm import LLMService
from agent.share.eventloop import ensure_selector_event_loop
from agent.tools import build_tools_from_mcp
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings, configure_logging, get_settings

logger = logging.getLogger(__name__)

# 演示问题：均为「先工具查询 + 再变换输出」的复合任务，引导真实 LLM 判成 PLAN。
DEMO_QUESTIONS = [
    "先查询当前的日期和时间，再把结果翻译成英文，最后整理成一句话告诉我",
    "查询一下今天是几号，然后用英文说出今天的日期",
]


def _tools_mcp_servers(settings: RuntimeSettings) -> dict[str, dict]:
    """组装 tools_mcp 的 langchain-mcp-adapters connection 配置（与 demo_chat 同策略）。"""
    if settings.mcp_transport is MCPTransport.STREAMABLE_HTTP:
        url = f"http://{settings.mcp_host}:{settings.mcp_port}{settings.mcp_streamable_http_path}"
        return {"tools_mcp": {"transport": "streamable_http", "url": url}}
    # stdio 子进程强制 MCP_TRANSPORT=stdio：子进程会继承 `.env` 里的
    # MCP_TRANSPORT=streamable-http，误启 HTTP 并抢端口导致 Connection closed。
    env = dict(os.environ)
    env["MCP_TRANSPORT"] = "stdio"
    return {
        "tools_mcp": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "services.tools_mcp"],
            "env": env,
        }
    }


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    api_key = settings.llm_api_key.get_secret_value()
    if not api_key:
        logger.warning("未配置 LLM_API_KEY，跳过真实调用（可先运行 pytest 验证离线路径）")
        return
    llm_config = LLMConfig.from_runtime(
        provider=settings.llm_provider,
        api_key=api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    service = LLMService(config=llm_config)
    cfg = AgentFrameworkConfig.get_default()
    logger.info(
        "plan 配置：max_plan_steps=%d, max_tool_calls_per_plan=%d",
        cfg.plan.max_plan_steps,
        cfg.plan.max_tool_calls_per_plan,
    )

    async with build_tools_from_mcp(cfg, servers=_tools_mcp_servers(settings)) as tools:
        graph = build_agent_graph(service, cfg, tools=tools)
        logger.info("已加载 %d 个 MCP 工具", len(tools))
        for q_idx, text in enumerate(DEMO_QUESTIONS):
            thread_id = f"plan-demo-{q_idx}"
            config = {"configurable": {"thread_id": thread_id}}
            logger.info("用户(session=%s): %s", thread_id, text)
            async for chunk in graph.astream(
                {"input": text, "session_id": thread_id, "user_id": "demo-user"},
                config=config,
                stream_mode="updates",
            ):
                for node, updates in chunk.items():
                    if not updates:
                        continue
                    for event in updates.get("status_events", []):
                        logger.info(
                            "  [%s] 状态=%s 工具=%s 消息=%s",
                            node,
                            event.status,
                            event.tool_name,
                            event.message,
                        )
                    # 规划节点成功产出 → 打印计划概述与步骤数。
                    if node in (NODE_PLAN_TASK, NODE_REPLAN_TASK):
                        plan = updates.get("plan")
                        if isinstance(plan, PlanResult):
                            logger.info(
                                "  [%s] 计划(%d 步): %s",
                                node,
                                len(plan.steps),
                                plan.summary,
                            )
                    if node == NODE_EXECUTE_STEP and updates.get("messages"):
                        logger.info("  [execute_step] 本步模型输出已就绪（工具调用或直接作答）")
                    if "response" in updates:
                        resp = updates["response"]
                        logger.info(
                            "回答(finished_reason=%s): %s", resp.finished_reason, resp.reply
                        )
            # 读最终状态：展示计划执行进度（步骤指针 / 累计成功步数）。
            final_state = await graph.aget_state(config)
            values = final_state.values
            intent = values.get("intent")
            plan = values.get("plan")
            if intent == Intent.PLAN and plan is not None:
                logger.info(
                    "本轮规划进度：%d/%d 步已推进（累计成功 %d 步）",
                    values.get("plan_step") or 0,
                    len(plan.steps),
                    values.get("plan_steps_done") or 0,
                )
            else:
                logger.info(
                    "本轮意图=%s（未进入规划路径，属正常：真实 LLM 分类或 plan 已回退）",
                    intent.value if intent else "None",
                )


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
