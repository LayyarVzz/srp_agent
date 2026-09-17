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
from dataclasses import dataclass, field
from typing import Literal, cast

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent.a2a.mapper import peer_user_id, resolve_peer
from agent.a2a.models import A2APeer
from agent.a2a.protocol import A2A_ERROR_INVALID_REQUEST, JSONRPC_INVALID_REQUEST, A2AProtocolError
from agent.a2a.registry import A2ATaskRegistry
from agent.core.config import A2AConfig, AgentFrameworkConfig, LLMConfig
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
from agent.response.models import AgentResponse, AnswerToken
from agent.response.status import StatusEvent
from agent.session import SessionBackend, SessionManager, build_session_backend
from agent.tools import build_tools_from_mcp
from agent.tools.models import ToolCallRecord
from services.a2a_mcp.client_config import (
    A2A_MCP_SERVER_NAME,
    build_a2a_mcp_http_connection,
    build_a2a_mcp_stdio_connection,
)
from services.lark_mcp.cli import resolve_lark_cli_command
from services.lark_mcp.client_config import build_lark_mcp_stdio_connection
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
from shared.lark import LARK_MCP_SERVER_NAME

logger = logging.getLogger(__name__)

# 会话编排事件：chat_stream 产出的领域事件（app 层只做 SSE 编码，不做业务判断）。
# token 事件 = 回答节点（generate_answer / fallback_chat）LLM 流式生成期间的
# 增量预览，正常路径拼接 == done.answer；done 仍是唯一权威终态。
ChatStreamEvent = (
    tuple[Literal["status"], StatusEvent]
    | tuple[Literal["tool"], ToolCallRecord]
    | tuple[Literal["token"], AnswerToken]
    | tuple[Literal["done"], AgentResponse]
)

# A2A 入站任务事件：首事件 ("started", session_id) 供调用方建立 task↔session
# 映射（task_id == session_id），后续事件与 chat_stream 完全同构。
A2AStreamEvent = tuple[Literal["started"], str] | ChatStreamEvent


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
    # A2A 入站任务注册表（task↔session 一一对应；随进程生命周期，无持久化）。
    a2a_tasks: A2ATaskRegistry = field(default_factory=A2ATaskRegistry)

    # —— 装配（生产入口）——

    @classmethod
    async def create(cls, *, settings: RuntimeSettings | None = None) -> AgentRuntime:
        """装配 LLM / 记忆 / 会话 / MCP 工具 / 图，一次性对齐三个共享实例。

        装配顺序 = 资源依赖顺序（memory → session → tools → graph）；
        任一步失败按逆序清理已获取资源（acquired 表驱动），不留泄漏。
        """
        settings = settings or get_settings()
        cfg = AgentFrameworkConfig.get_default()
        # A2A 入站 peer 注册表：settings 的 A2A_* 注入 cfg.a2a（未配置 → 空 peers，
        # 所有远端 peer 被拒 → 零回归；AgentCard 发现不受影响）。
        cfg.a2a = A2AConfig.from_runtime(
            enabled=settings.a2a_enabled, peer_map=settings.a2a_peer_map
        )
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
            # servers 统一登记全部 MCP 服务：rag_mcp（stdio）+ tools_mcp（按配置传输）
            # + lark_mcp（T4，配置 + 命令探测门控，未登记零回归）。
            servers = _build_mcp_servers(settings, cfg)
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
        """流式跑一轮对话：status/tool/token 实时下发，done 最后下发完整 AgentResponse。

        内部负责：run 图（stream_mode=["updates","custom"]）+ 结束后读最终 state
        触发带外记忆保存。updates 通道承载节点级状态/工具/响应轨迹；custom 通道由
        回答节点运行中实时外发「speaking + 回答 token 增量」，其中与 updates 同值的
        speaking 状态帧在此去重，保证事件序 speaking → token* → done。
        会话归属校验由入口层先行完成；本方法假定 session 已合法。
        """
        config = {"configurable": {"thread_id": session_id}}  # thread_id == session_id 契约
        response: AgentResponse | None = None
        # 回答节点 live 外发过的状态帧（custom 通道），用于 updates 同值帧去重。
        live_statuses: list[StatusEvent] = []
        try:
            # 多 stream_mode 时本版本 langgraph 的产出为 (mode, data) 二元组：
            # - ("updates", {节点名: 状态增量})：节点完成后的状态轨迹（同单 mode 语义）；
            # - ("custom", StatusEvent | AnswerToken)：回答节点运行中的实时事件。
            async for mode, data in self.graph.astream(
                {"input": text, "session_id": session_id, "user_id": user_id},
                config=config,
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    if isinstance(data, StatusEvent):
                        live_statuses.append(data)
                        yield ("status", data)
                    elif isinstance(data, AnswerToken):
                        yield ("token", data)
                    else:
                        logger.warning("忽略未知 custom 事件: %r", type(data).__name__)
                    continue
                if mode != "updates":
                    continue
                for node, updates in data.items():
                    if not updates:  # 空更新节点（trim_history 等）产出 None/空 dict
                        continue
                    for event in updates.get("status_events", []):
                        if any(event == live for live in live_statuses):
                            # 已被回答节点 live 外发（speaking 先于 token），防重复下发。
                            continue
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

    # —— A2A 入站编排（T5：peer 作为虚拟用户复用 chat_stream 内核，图零改动）——

    def resolve_a2a_peer(self, peer_id: str) -> A2APeer:
        """校验入站 peer：A2A 关闭 / 未登记 / 已禁用 → A2AProtocolError（a2a.invalid_request）。"""
        if not self.cfg.a2a.enabled:
            raise A2AProtocolError(
                jsonrpc_code=JSONRPC_INVALID_REQUEST,
                a2a_code=A2A_ERROR_INVALID_REQUEST,
                message="A2A 入站未启用",
            )
        return resolve_peer(self.cfg.a2a.peers, peer_id)

    async def run_a2a_task_stream(
        self, *, peer_id: str, text: str
    ) -> AsyncIterator[A2AStreamEvent]:
        """流式执行一个 A2A 入站任务：peer 发号会话 → 复用 chat_stream 内核。

        首事件 ("started", session_id) 承载 task↔session 映射（task_id ==
        session_id）；peer 身份校验在首事件前完成（fail-fast，流开始后不再有 4xx
        等价物）。会话归属 peer 命名空间（`sessions.create` 发号，与人类用户隔离）。
        """
        peer = self.resolve_a2a_peer(peer_id)
        user_id = peer_user_id(peer)
        ctx = await self.sessions.create(user_id=user_id)
        session_id = ctx.session_id
        yield ("started", session_id)
        async for event, payload in self.chat_stream(
            user_id=user_id, session_id=session_id, text=text
        ):
            yield (event, payload)

    async def run_a2a_task(self, *, peer_id: str, text: str) -> AgentResponse:
        """非流式执行 A2A 入站任务，返回最终 AgentResponse（session_id 即 task_id）。"""
        async for event, payload in self.run_a2a_task_stream(peer_id=peer_id, text=text):
            if event == "done":
                return cast(AgentResponse, payload)
        # 防御：run_a2a_task_stream 必有 done；走到这里说明状态机异常。
        raise AgentError(AGENT_ERROR_INTERNAL, "A2A 任务未产出 AgentResponse")


def _build_mcp_servers(settings: RuntimeSettings, cfg: AgentFrameworkConfig) -> dict[str, dict]:
    """统一登记全部 MCP 服务（rag_mcp + tools_mcp + lark_mcp），连接配置由各服务侧导出。

    rag_mcp 仅 stdio（见 services/rag_mcp/client_config.py）；tools_mcp 双传输：
    streamable-http 走远端地址，stdio 以子进程自动拉起（见 services/tools_mcp/client_config.py）。

    lark_mcp（T4）门控装配（dev-version5.0 §3.1）：仅当「框架配置启用（cfg.lark.enabled）
    且环境启用（LARK_CLI_ENABLED）且 lark-cli 命令探测成功」才登记；任一不满足则跳过并记
    日志——无 CLI 环境下 Agent 以既有工具集照常服务（零回归，与 MCP 连接失败降级同语义）。
    
    a2a_mcp（T6 远端智能体协作工具）与 tools_mcp 同传输方式，但**配置+注册表双门控**：
    仅当 a2a_mcp_enabled 且注册表非空才登记——无远端配置时工具集与既有完全一致（零回归）。
    """
    servers: dict[str, dict] = {RAG_MCP_SERVER_NAME: build_rag_mcp_stdio_connection()}
    register_a2a = settings.a2a_mcp_enabled and bool(settings.a2a_mcp_agents)
    if settings.mcp_transport is MCPTransport.STREAMABLE_HTTP:
        servers[TOOLS_MCP_SERVER_NAME] = build_tools_mcp_http_connection(
            host=settings.mcp_host,
            port=settings.mcp_port,
            path=settings.mcp_streamable_http_path,
        )
        if register_a2a:
            servers[A2A_MCP_SERVER_NAME] = build_a2a_mcp_http_connection(
                host=settings.a2a_mcp_host,
                port=settings.a2a_mcp_port,
                path=settings.mcp_streamable_http_path,
            )
    else:
        servers[TOOLS_MCP_SERVER_NAME] = build_tools_mcp_stdio_connection()
        if register_a2a:
            servers[A2A_MCP_SERVER_NAME] = build_a2a_mcp_stdio_connection()
    if cfg.lark.enabled and settings.lark_cli_enabled:
        if resolve_lark_cli_command(settings.lark_cli_command):
            servers[LARK_MCP_SERVER_NAME] = build_lark_mcp_stdio_connection()
        else:
            logger.warning(
                "lark-cli 命令探测失败，跳过 lark_mcp 登记"
                "（无 CLI 环境零回归；可配置 LARK_CLI_COMMAND 指向可执行文件）"
            )
    return servers


__all__ = ["A2AStreamEvent", "AgentRuntime", "ChatStreamEvent"]
