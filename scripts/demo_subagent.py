"""v5.0 T7 验收脚本：真实 LLM 跑通并行子代理（Send 扇出 / join 回填 / 整合）。

复合任务（含 2+ 无依赖步骤）→ `Intent.PLAN` → plan_task（PLANNING）→ 就绪批 ≥2 →
dispatch_subagents（DELEGATING「正在并行处理 N 个子任务」）→ 子代理子图并行执行
（并行 tool 事件）→ join 回填 → 依赖步串行收尾 → 整合回答（finished_reason）。

注意（非确定性，与 demo_plan 同）：意图与计划由真实 LLM 产生——若复合任务被判成
TOOL_USE/CHAT、或计划把所有步骤串成依赖链（就绪批 <2），本轮不会出现 DELEGATING，
脚本会如实打印实际意图与计划结构（属正常，不构成失败）。

工具经 MCP 接入：默认以 stdio 子进程自动拉起 `services.tools_mcp`（自包含演示）；
配置了 `MCP_TRANSPORT=streamable-http` 时改为 streamable_http 连接对应端口。

用法（在仓库根目录执行）：
    uv run python -m scripts.demo_subagent                  # 未配置 LLM_API_KEY 时跳过
    LLM_API_KEY=sk-xxx uv run python -m scripts.demo_subagent  # DeepSeek
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.core.models import PlanResult, SubagentResult
from agent.core.state import (
    NODE_DISPATCH_SUBAGENTS,
    NODE_JOIN_SUBAGENTS,
    NODE_PLAN_TASK,
    NODE_REPLAN_TASK,
    NODE_RUN_SUBAGENT,
)
from agent.intent.models import Intent
from agent.llm import LLMService
from agent.share.eventloop import ensure_selector_event_loop
from agent.tools import build_tools_from_mcp
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings, configure_logging, get_settings

logger = logging.getLogger(__name__)

# 演示问题：均含 2+ 可并行步骤（两路独立计算/查询）+ 依赖汇总步，
# 引导真实 LLM 产出带 depends_on 的并行计划。
DEMO_QUESTIONS = [
    "分别计算 12*7 和 345+678，然后把两个结果相加，最后用一句话告诉我",
    "先查询当前日期，然后分别计算 2 的 10 次方和 100 以内最大的质数，最后汇总成一段话",
]


def _tools_mcp_servers(settings: RuntimeSettings) -> dict[str, dict]:
    """组装 tools_mcp 的 langchain-mcp-adapters connection 配置（与 demo_plan 同策略）。"""
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
        "subagent 配置：enabled=%s, max_parallel=%d, per_subagent_max_tool_calls=%d"
        "（plan 预算 max_tool_calls_per_plan=%d）",
        cfg.subagents.enabled,
        cfg.subagents.max_parallel,
        cfg.subagents.per_subagent_max_tool_calls,
        cfg.plan.max_tool_calls_per_plan,
    )

    async with build_tools_from_mcp(cfg, servers=_tools_mcp_servers(settings)) as tools:
        graph = build_agent_graph(service, cfg, tools=tools)
        logger.info("已加载 %d 个 MCP 工具", len(tools))
        for q_idx, text in enumerate(DEMO_QUESTIONS):
            thread_id = f"subagent-demo-{q_idx}"
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
                    # 规划节点成功产出 → 打印计划结构与依赖关系（并行批的来源）。
                    if node in (NODE_PLAN_TASK, NODE_REPLAN_TASK):
                        plan = updates.get("plan")
                        if isinstance(plan, PlanResult):
                            deps = ", ".join(
                                f"{i}:{s.depends_on or '[]'}" for i, s in enumerate(plan.steps)
                            )
                            logger.info(
                                "  [%s] 计划(%d 步) depends_on: %s", node, len(plan.steps), deps
                            )
                    # 扇出节点 → 打印批序号（join 后再次扇出即第二轮批）。
                    if node == NODE_DISPATCH_SUBAGENTS:
                        logger.info("  [%s] 批序号=%s", node, updates.get("dispatch_round"))
                    # 子代理分支产出 → 逐条打印（并行分支各一条更新）。
                    if node == NODE_RUN_SUBAGENT:
                        for result in updates.get("subagent_results", []):
                            if isinstance(result, SubagentResult):
                                logger.info(
                                    "  [%s] 子任务%d ok=%s 摘要=%s 工具=%s",
                                    node,
                                    result.step_index + 1,
                                    result.ok,
                                    result.summary or "-",
                                    result.tool_summary or "无",
                                )
                    # join 节点 → 打印完成索引与串行指针同步。
                    if node == NODE_JOIN_SUBAGENTS:
                        logger.info(
                            "  [%s] 已完成步骤=%s, plan_step=%s, 预算消耗=%s",
                            node,
                            updates.get("plan_steps_completed"),
                            updates.get("plan_step"),
                            updates.get("tool_iterations"),
                        )
                    if "response" in updates:
                        resp = updates["response"]
                        logger.info(
                            "回答(finished_reason=%s): %s", resp.finished_reason, resp.reply
                        )
            # 读最终状态：展示子代理记账结果（批序号 / 完成索引 / 结果清单）。
            final_state = await graph.aget_state(config)
            values = final_state.values
            intent = values.get("intent")
            plan = values.get("plan")
            if intent == Intent.PLAN and plan is not None:
                logger.info(
                    "本轮子代理进度：批序号=%s, 完成索引=%s, 结果数=%d, 累计成功 %d 步",
                    values.get("dispatch_round"),
                    values.get("plan_steps_completed"),
                    len(values.get("subagent_results") or []),
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
