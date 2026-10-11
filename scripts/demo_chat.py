"""P2/P4 验收脚本：真实 LLM 跑通 Agent 状态机 chat 路径、工具循环与长期记忆召回。

P4-3 演示：固定 `DEMO_USER_ID`，每个问题用独立 session（跨会话共享长期记忆）；
`load_context` 预加载 preference、`recall_memory` 作为共同上游对闲聊/工具两分支召回
fact/episode，日志可见「预加载 / 按需召回 / 引用」；回答后带外保存（P4-2）落盘新增记忆。

工具经 MCP 接入：默认以 stdio 子进程自动拉起 `services.tools_mcp`（自包含演示）；
配置了 `MCP_TRANSPORT=streamable-http` 时改为 streamable_http 连接对应端口。

用法（在仓库根目录执行）：
    uv run python -m scripts.demo_chat                  # 未配置 LLM_API_KEY 时跳过
    LLM_API_KEY=sk-xxx uv run python -m scripts.demo_chat  # DeepSeek
    LLM_PROVIDER=qwen LLM_API_KEY=sk-xxx uv run python -m scripts.demo_chat  # Qwen
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.core.state import NODE_LOAD_CONTEXT, NODE_RECALL_MEMORY
from agent.llm import LLMService
from agent.memory import (
    KIND_EPISODE,
    KIND_FACT,
    MemoryExtractor,
    MemoryRelationJudge,
    MemoryStore,
    build_memory_backends,
    submit_memory_save,
    wait_pending_saves,
)
from agent.session import build_session_backend
from agent.share.eventloop import ensure_selector_event_loop
from agent.tools import build_tools_from_mcp
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings, configure_logging, get_settings
from shared.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

# 演示用户：预加载 / 召回 / 带外保存共用同一命名空间，跨会话生效。
DEMO_USER_ID = "demo-user"

# 演示会话：工具型请求（datetime）+ 跨会话追问（同一 user 不同 session）+ 纯闲聊记忆问。
# 断词器命中「时间/日期」→ TOOL_USE（datetime），命中「计算」→ TOOL_USE（计算器）；
# 第三条无工具触发词 → CHAT 分支，演示「闲聊也召回 fact」（recall_memory 共同上游）。
DEMO_QUESTIONS = [
    "我是一名中学生，晚上九点时需要睡觉，现在几点了",
    "我中午吃了宫保鸡丁，会不会有问题",
    "我今天生日，打算买个蛋糕庆祝，推荐什么口味",
    "帮我网购一袋花生",
]


# async def _seed_demo_memories(memory_store: MemoryStore, user_id: str) -> None:
#     """幂等种入演示长期记忆，保证预加载/召回路径恒可观察（在真实 LLM 抽取落盘前）。

#     WHY 种子而非纯依赖抽取器：抽取器是真实 LLM、非确定，演示需要确定可见的
#     预加载 preference + 召回 fact，故显式种入两条 seed 记忆。
#     """
#     seeds = (
#         MemoryItem(
#             id="seed-pref-1",
#             kind=KIND_PREFERENCE,
#             content="用户偏好简洁回答，重要信息需二次确认",
#             session_id="seed",
#             user_id=user_id,
#             timestamp=datetime.now(UTC),
#             provenance="seed",
#             importance=0.9,
#         ),
#         MemoryItem(
#             id="seed-fact-1",
#             kind=KIND_FACT,
#             content="用户是中学生，通常晚上九点睡觉",
#             session_id="seed",
#             user_id=user_id,
#             timestamp=datetime.now(UTC),
#             provenance="seed",
#             importance=0.8,
#         ),
#     )
#     for item in seeds:
#         await memory_store.save(item)
#     logger.info("已种入演示长期记忆（%d 条：preference / fact）", len(seeds))


def _tools_mcp_servers(settings: RuntimeSettings) -> dict[str, dict]:
    """组装 tools_mcp 的 langchain-mcp-adapters connection 配置。

    服务器配置了 streamable-http 时，从 `mcp_*` 派生客户端连接地址；
    否则以 stdio 子进程自动拉起本服务。
    """
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
    # 语义召回底座：装配层把 settings 的 EMBEDDING_* 注入 cfg.memory.embedding（P3-1）；
    # 未配置时保持默认关闭 → build_memory_backends 传 index_config=None → 零回归。
    cfg.memory.embedding = EmbeddingConfig.from_runtime(
        enabled=settings.embedding_enabled,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
    )
    # 记忆后端：有 DATABASE_URL 建 AsyncPostgresStore + AsyncPostgresSaver（prod）；
    # 无则统一入口降级 InMemoryStore（短期上下文走 graph 默认 MemorySaver）。
    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    backends = await build_memory_backends(cfg, database_url=database_url)
    store = backends.store
    checkpointer = backends.checkpointer
    # 长期记忆：store 只建一次，图与带外保存端共享同一实例（否则各自后端互不可见）。
    extractor = MemoryExtractor(service, max_input_chars=cfg.graph.max_input_chars)
    # D4-2 合并判定器：与 extractor 共用同一 LLMService，按事实三分类决定去重合并。
    judge = MemoryRelationJudge(service)
    # 会话元数据：session 模块自装（独立 sessions 表），不随记忆后端。
    session_backend = await build_session_backend(database_url, config=cfg.session)
    sessions = session_backend.manager

    # MCP 工具生命周期：context 内连接 → 取工具 → 使用（stdio 子进程随 context 退出回收）。
    # try/finally：无论成败都关闭 Postgres 连接池（须在 wait_pending_saves 之后）。
    try:
        async with build_tools_from_mcp(cfg, servers=_tools_mcp_servers(settings)) as tools:
            # checkpointer= 注入 prod 的 AsyncPostgresSaver（会话/短期）；
            # dev 传 None → MemorySaver。
            graph = build_agent_graph(
                service, cfg, tools=tools, store=store, checkpointer=checkpointer
            )
            # 同一实例包装：保存端与图共享命名空间；混合重排参数与图侧 cfg.memory.recall 一致。
            memory_store = MemoryStore(store, recall_config=cfg.memory.recall)
            # await _seed_demo_memories(memory_store, DEMO_USER_ID)
            logger.info("已加载 %d 个 MCP 工具", len(tools))
            # P4-3 跨会话演示：每问经 SessionManager 发号（thread_id==session_id），同一 user。
            for text in DEMO_QUESTIONS:
                ctx = await sessions.create(user_id=DEMO_USER_ID)
                config = {"configurable": {"thread_id": ctx.session_id}}
                logger.info("用户(session=%s): %s", ctx.session_id, text)
                # S1 归属校验演示：owner 解析成功；非 owner 解析会抛 session_error.not_found。
                resolved = await sessions.resolve(user_id=DEMO_USER_ID, session_id=ctx.session_id)
                logger.info(
                    "会话归属校验通过（user=%s, session=%s）", resolved.user_id, resolved.session_id
                )
                async for chunk in graph.astream(
                    {"input": text, "session_id": ctx.session_id, "user_id": DEMO_USER_ID},
                    config=config,
                    stream_mode="updates",
                ):
                    for node, updates in chunk.items():
                        # 节点返回 {}（无状态变更）时 updates 为 None，跳过。
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
                        # P4-3 预加载：load_context 节点更新里的 memory_context。
                        if node == NODE_LOAD_CONTEXT:
                            for m in updates.get("memory_context") or []:
                                logger.info(
                                    "  [load_context] 预加载长期记忆 [%s] %s", m.kind, m.content
                                )
                        # P4-3 按需召回：memory_context 是合并后列表，只记 fact/episode 防重复。
                        if node == NODE_RECALL_MEMORY:
                            for m in updates.get("memory_context") or []:
                                if m.kind in (KIND_FACT, KIND_EPISODE):
                                    logger.info(
                                        "  [recall_memory] 按需召回 [%s] %s", m.kind, m.content
                                    )
                        if "response" in updates:
                            resp = updates["response"]
                            logger.info("回答(%s): %s", resp.finished_reason, resp.reply)
                            for c in resp.citations:
                                logger.info("  [citation] %s: %s", c.source_title, c.snippet)
                # —— P4-2 挂接点：回答已 format_response 下发（END），取完整消息带外保存 ——
                # stream_mode="updates" 下拿不到完整 messages，需在结束后读最终 state。
                final_state = await graph.aget_state(config)
                values = final_state.values
                saved_session_id = values.get("session_id") or ctx.session_id
                saved_user_id = values.get("user_id") or DEMO_USER_ID  # §4.2 兜底
                submit_memory_save(
                    values.get("messages") or [],
                    session_id=saved_session_id,
                    user_id=saved_user_id,
                    extractor=extractor,
                    store=memory_store,
                    dedup=cfg.memory.dedup,  # V3-M3：重复事实带外保存被 upsert 合并
                    judge=judge,  # D4-2：判定器按事实三分类决定合并/新写
                )
                logger.info(
                    "已提交长期记忆带外保存（session=%s, user=%s）", saved_session_id, saved_user_id
                )
            # S1 会话列表演示：按 user 列出全部会话（前端「历史会话」能力的基础）。
            ctxs = await sessions.list(user_id=DEMO_USER_ID)
            logger.info(
                "用户 %s 现有 %d 个会话: %s",
                DEMO_USER_ID,
                len(ctxs),
                [c.session_id[:8] for c in ctxs],
            )
            # 收尾：等待后台保存落盘，避免 asyncio.run 关闭时取消 pending 任务（验收可观测）。
            await wait_pending_saves()
    finally:
        await session_backend.aclose()
        await backends.aclose()  # InMemory 路径为安全 no-op


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
