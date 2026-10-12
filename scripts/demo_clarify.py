"""v4.0 T2 验收脚本：真实 LLM 跑通澄清式追问（Clarification）。

触发源①（意图低置信）：模糊请求 → CHAT 低置信 → `clarify` 节点（CLARIFYING →
`finished_reason=needs_clarification` + options，前端可渲染选项按钮）；
触发源②（工具参数缺失）：请求调用带必填参数 `expression` 的 `calculate` 工具但未提供
参数 → `tool_error.missing_argument` → `clarify`（替代降级道歉）。

注意：意图/澄清内容由真实 LLM 生成，非确定性——若某问被判成高置信 CHAT（未走澄清）、
或模型直接补全了工具参数（未触发缺失），脚本会如实打印实际 finished_reason 并继续
（属正常：ReAct 兜底仍可用，见 dev-version4.0.md §4.1）。
RAG 空结果本版不触发澄清（统一如实答「未检索到依据」），本脚本未接 RAG，仅在此说明。

工具经 MCP 接入：默认以 stdio 子进程自动拉起 `services.tools_mcp`（自包含演示）；
配置了 `MCP_TRANSPORT=streamable-http` 时改为 streamable_http 连接对应端口。

用法（在仓库根目录执行）：
    uv run python -m scripts.demo_clarify                  # 未配置 LLM_API_KEY 时跳过
    LLM_API_KEY=sk-xxx uv run python -m scripts.demo_clarify  # DeepSeek
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.core.state import NODE_CLARIFY
from agent.llm import LLMService
from agent.response.models import FINISHED_REASON_NEEDS_CLARIFICATION
from agent.share.eventloop import ensure_selector_event_loop
from agent.tools import build_tools_from_mcp
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings, configure_logging, get_settings

logger = logging.getLogger(__name__)

# 演示问题：① 模糊请求（意图低置信 → 澄清）；② 缺参数的工具请求（missing_argument → 澄清）。
DEMO_QUESTIONS = [
    "就是那个，你懂的，帮我看看呗",
    "帮我算一下",
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
        "澄清配置：min_confidence=%.2f, max_asks_per_turn=%d, enabled=%s",
        cfg.clarify.min_confidence,
        cfg.clarify.max_asks_per_turn,
        cfg.clarify.enabled,
    )

    async with build_tools_from_mcp(cfg, servers=_tools_mcp_servers(settings)) as tools:
        graph = build_agent_graph(service, cfg, tools=tools)
        logger.info("已加载 %d 个 MCP 工具", len(tools))
        for q_idx, text in enumerate(DEMO_QUESTIONS):
            thread_id = f"clarify-demo-{q_idx}"
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
                    if node == NODE_CLARIFY and updates.get("clarification") is not None:
                        clar = updates["clarification"]
                        logger.info(
                            "  [clarify] 反问：%s（选项：%s）",
                            clar.question,
                            clar.options or "无（开放反问）",
                        )
                    if "response" in updates:
                        resp = updates["response"]
                        if resp.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION:
                            logger.info(
                                "回答(finished_reason=%s, clarification=%s): %s",
                                resp.finished_reason,
                                resp.clarification is not None,
                                resp.reply,
                            )
                        else:
                            logger.info(
                                "回答(finished_reason=%s): %s", resp.finished_reason, resp.reply
                            )
            # 读最终状态：展示澄清状态位（本轮是否已追问）。
            final_state = await graph.aget_state(config)
            values = final_state.values
            logger.info(
                "本轮澄清状态：clarify_asked=%s, clarification=%s",
                values.get("clarify_asked") or False,
                "有" if values.get("clarification") is not None else "无",
            )


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
