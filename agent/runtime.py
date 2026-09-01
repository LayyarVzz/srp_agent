"""可复用的 Agent Runtime 装配模块（组合根）。

WHY 组合根：Agent 的多个模块（LLM / 记忆 / 会话 / MCP 工具 / 图）必须**一次性对齐装配**，
且带外保存端与图共享同一 store / LLMService 实例。
本模块是唯一装配点，FastAPI 等入口只需 `AgentRuntime.create()` / `aclose()`；
`chat` / `chat_stream` 承载一轮对话的编排（run 图 + 带外记忆保存），入口层零业务逻辑。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, cast

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.errors import AGENT_ERROR_INTERNAL, AgentError
from agent.llm import LLMService
from agent.memory import (
    MemoryBackends,
    MemoryExtractor,
    MemoryRelationJudge,
    MemoryStore,
    build_memory_backends,
    submit_memory_save,
    wait_pending_saves,
)
from agent.response.models import AgentResponse
from agent.response.status import StatusEvent
from agent.session import SessionBackend, SessionManager, build_session_backend
from agent.tools import build_tools_from_mcp
from agent.tools.models import ToolCallRecord
from services.rag_mcp.client_config import (
    RAG_MCP_SERVER_NAME,
    build_rag_mcp_stdio_connection,
)
from services.tools_mcp.client_config import (
    TOOLS_MCP_SERVER_NAME,
    build_tools_mcp_http_connection,
    build_tools_mcp_stdio_connection,
)
from services.tools_mcp.config import MCPTransport
from settings import RuntimeSettings, get_settings
from shared.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

# 会话编排事件：chat_stream 产出的领域事件（app 层只做 SSE 编码，不做业务判断）。
ChatStreamEvent = (
    tuple[Literal["status"], StatusEvent]
    | tuple[Literal["tool"], ToolCallRecord]
    | tuple[Literal["done"], AgentResponse]
)


@dataclass
class AgentRuntime:
    """应用级组合根：持有全部装配产物，承载一轮对话的编排（run 图 + 带外记忆保存）。

    `create()` 是生产装配入口；测试可跳过 create 直接构造本 dataclass（注入 fake 组件）。
    生命周期句柄（_tools_cm / _memory_backends / _session_backend）由 create 打开、
    `aclose()` 按逆序关闭；未走 create 构造时句柄为 None，aclose 天然 no-op。
    """

    graph: CompiledStateGraph
    sessions: SessionManager
    memory_store: MemoryStore
    cfg: AgentFrameworkConfig
    extractor: MemoryExtractor | None = None
    judge: MemoryRelationJudge | None = None
    # 生命周期句柄：create() 打开、aclose() 关闭（与 build_memory_backends 的句柄模式一致）。
    _tools_cm: AbstractAsyncContextManager[list[BaseTool]] | None = None
    _memory_backends: MemoryBackends | None = None
    _session_backend: SessionBackend | None = None

    # —— 装配（生产入口）——

    @classmethod
    async def create(cls, *, settings: RuntimeSettings | None = None) -> AgentRuntime:
        """装配 LLM / 记忆 / 会话 / MCP 工具 / 图，一次性对齐三个共享实例。

        装配顺序 = 资源依赖顺序（memory → session → tools → graph）；
        任一步失败按逆序清理已获取资源（acquired 表驱动），不留泄漏。
        """
        settings = settings or get_settings()
        cfg = AgentFrameworkConfig.get_default()
        # 语义召回底座：settings 的 EMBEDDING_* 注入 cfg.memory.embedding
        # （未配置时保持默认关闭 → build_memory_backends 传 index_config=None → 零回归）。
        cfg.memory.embedding = EmbeddingConfig.from_runtime(
            enabled=settings.embedding_enabled,
            model=settings.embedding_model,
            dims=settings.embedding_dims,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key.get_secret_value(),
        )
        llm_config = LLMConfig.from_runtime(
            provider=settings.llm_provider,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            behavior=cfg.llm_behavior,
        )
        llm = LLMService(config=llm_config)

        database_url = settings.database_url.get_secret_value() if settings.database_url else None
        # acquired 记录已获取资源的关闭函数；失败时逆序执行（先拿到的后关）。
        acquired: list[Callable[[], Awaitable[None]]] = []
        try:
            memory_backends = await build_memory_backends(cfg, database_url=database_url)
            acquired.append(memory_backends.aclose)
            session_backend = await build_session_backend(database_url, config=cfg.session)
            acquired.append(session_backend.aclose)
            # 带外保存端与图共享同一 store / LLMService（三个共享实例规则）。
            memory_store = MemoryStore(memory_backends.store, recall_config=cfg.memory.recall)
            extractor = MemoryExtractor(llm, max_input_chars=cfg.graph.max_input_chars)
            judge = MemoryRelationJudge(llm)
            # 工具生命周期：context 句柄持有到 aclose()（修复「async with 提前关闭」问题）。
            # servers 统一登记全部 MCP 服务：rag_mcp（stdio，原有支持）+ tools_mcp（按配置传输）。
            servers = _build_mcp_servers(settings)
            logger.info("注册 MCP 服务：%s", ", ".join(sorted(servers)))
            tools_cm = build_tools_from_mcp(cfg, servers=servers)
            tools = await tools_cm.__aenter__()

            async def _close_tools() -> None:
                await tools_cm.__aexit__(None, None, None)

            acquired.append(_close_tools)
            graph = build_agent_graph(
                llm,
                cfg,
                tools=tools,
                store=memory_backends.store,
                checkpointer=memory_backends.checkpointer,
            )
        except BaseException:
            for closer in reversed(acquired):
                try:
                    await closer()
                except BaseException:
                    logger.exception("装配失败，清理已获取资源异常")
            raise
        logger.info(
            "AgentRuntime 装配完成：%d 个 MCP 工具，store=%s",
            len(tools),
            type(memory_backends.store).__name__,
        )
        return cls(
            graph=graph,
            sessions=session_backend.manager,
            memory_store=memory_store,
            cfg=cfg,
            extractor=extractor,
            judge=judge,
            _tools_cm=tools_cm,
            _memory_backends=memory_backends,
            _session_backend=session_backend,
        )

    async def aclose(self) -> None:
        """按逆序关闭生命周期资源：先排干带外保存，再关 MCP 工具、会话、记忆后端。

        WHY 顺序：带外保存任务写 memory store、读 LLMService，必须先
        `wait_pending_saves()` 排干再关后端，否则保存静默失败（langgraph#6367 同因）；
        其余按装配逆序（tools → session → memory）关闭，句柄置 None 保证幂等。
        """
        await wait_pending_saves()
        if self._tools_cm is not None:
            await self._tools_cm.__aexit__(None, None, None)
            self._tools_cm = None
        if self._session_backend is not None:
            await self._session_backend.aclose()
            self._session_backend = None
        if self._memory_backends is not None:
            await self._memory_backends.aclose()
            self._memory_backends = None

    # —— 会话编排（app 层零业务逻辑的保证）——

    async def chat_stream(
        self, *, user_id: str, session_id: str, text: str
    ) -> AsyncIterator[ChatStreamEvent]:
        """流式跑一轮对话：status/tool 实时下发，done 最后下发完整 AgentResponse。

        内部负责：run 图（stream_mode="updates"）+ 结束后读最终 state 触发带外记忆保存。
        会话归属校验由入口层先行完成；本方法假定 session 已合法。
        """
        config = {"configurable": {"thread_id": session_id}}  # thread_id == session_id 契约
        response: AgentResponse | None = None
        try:
            # stream_mode="updates" 产出 {节点名: 状态增量}（本版本 langgraph 的产出为
            # dict 而非二元组）；空更新节点（trim_history 等）产出 {node: None}/{node: {}}，
            # updates falsy 时跳过。
            async for chunk in self.graph.astream(
                {"input": text, "session_id": session_id, "user_id": user_id},
                config=config,
                stream_mode="updates",
            ):
                for node, updates in chunk.items():
                    if not updates:  # 空更新节点（trim_history 等）产出 None/空 dict
                        continue
                    for event in updates.get("status_events", []):
                        # 节点级状态日志：放在 yield 之前，打印顺序即下发顺序（驱动动画的轨迹）。
                        logger.info(
                            "  [%s] 状态=%s 工具=%s 消息=%s",
                            node,
                            event.status,
                            event.tool_name,
                            event.message,
                        )
                        yield ("status", event)
                    for record in updates.get("tool_calls", []):
                        yield ("tool", record)
                    if "response" in updates:
                        response = updates["response"]
        finally:
            # 带外记忆保存（尽力而为，绝不阻塞/中断主流程；客户端中途断开也会走到这里）。
            # stream_mode="updates" 拿不到完整 messages，必须在图结束后读最终 state。
            if self.extractor is not None:
                try:
                    final = await self.graph.aget_state(config)
                    submit_memory_save(
                        final.values.get("messages") or [],
                        session_id=session_id,
                        user_id=user_id,
                        extractor=self.extractor,
                        store=self.memory_store,
                        dedup=self.cfg.memory.dedup,
                        judge=self.judge,
                    )
                except BaseException as exc:
                    # 含 CancelledError（客户端断开取消生成器）：带外路径失败仅记日志。
                    logger.warning("带外记忆保存触发失败：%s", exc)
        if response is None:
            raise AgentError(AGENT_ERROR_INTERNAL, "Agent 图未产出 AgentResponse")
        yield ("done", response)

    async def chat(self, *, user_id: str, session_id: str, text: str) -> AgentResponse:
        """非流式跑一轮对话，直接返回 AgentResponse（消费 chat_stream 的 done 事件）。"""
        async for event, payload in self.chat_stream(
            user_id=user_id, session_id=session_id, text=text
        ):
            if event == "done":
                return cast(AgentResponse, payload)
        # 防御：chat_stream 必有 done；走到这里说明状态机异常。
        raise AgentError(AGENT_ERROR_INTERNAL, "Agent 图未产出 AgentResponse")


def _build_mcp_servers(settings: RuntimeSettings) -> dict[str, dict]:
    """统一登记全部 MCP 服务（rag_mcp + tools_mcp），连接配置由各服务侧导出。

    rag_mcp 仅 stdio（见 services/rag_mcp/client_config.py）；tools_mcp 双传输：
    streamable-http 走远端地址，stdio 以子进程自动拉起（见 services/tools_mcp/client_config.py）。
    """
    servers: dict[str, dict] = {RAG_MCP_SERVER_NAME: build_rag_mcp_stdio_connection()}
    if settings.mcp_transport is MCPTransport.STREAMABLE_HTTP:
        servers[TOOLS_MCP_SERVER_NAME] = build_tools_mcp_http_connection(
            host=settings.mcp_host,
            port=settings.mcp_port,
            path=settings.mcp_streamable_http_path,
        )
    else:
        servers[TOOLS_MCP_SERVER_NAME] = build_tools_mcp_stdio_connection()
    return servers


__all__ = ["AgentRuntime", "ChatStreamEvent"]
