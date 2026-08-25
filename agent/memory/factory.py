"""记忆存储工厂：按配置构造 langgraph BaseStore / checkpointer（dev/prod 切换点）。

对外只暴露统一入口 `build_memory_backends`：传入 `database_url` 则构造
AsyncPostgresStore + AsyncPostgresSaver（prod，各自连接池），无则降级 InMemoryStore
（checkpointer=None，由 graph 编译默认 MemorySaver）。会话元数据由 `agent/session`
模块自装（`build_session_backend`，独立 `sessions` 表），本模块不再装配。
构造细节全部私有，后续换库/新增后端只需改本模块，调用方不变。
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from agent.core.config import AgentFrameworkConfig
from shared.embeddings import EmbeddingsFactory


@dataclass
class MemoryBackends:
    """记忆后端集合：store（长期记忆）+ checkpointer（短期上下文/会话消息）。

    `from_conn_string` 返回 async context manager，此处持有其句柄，`aclose()` 时对称
    `__aexit__` 关闭连接池。InMemory 分支 store/checkpointer 句柄为 None，`aclose()`
    天然 no-op。store 由 MemoryStore 与图共享；checkpointer 供 compile(checkpointer=)
    （thread_id == session_id 承载短期上下文）。会话元数据由 `agent/session` 自装，
    不在本集合内。
    """

    store: BaseStore
    checkpointer: BaseCheckpointSaver | None = None
    _store_cm: AbstractAsyncContextManager[BaseStore] | None = None
    _saver_cm: AbstractAsyncContextManager[BaseCheckpointSaver] | None = None

    async def aclose(self) -> None:
        """关闭连接池。调用方须先 `wait_pending_saves()` 排干带外保存，再关闭。

        WHY 顺序：langgraph#6367 AsyncPostgresStore 关闭会残留后台批量任务，
        先等带外写入落盘、再关池可规避 "Task was destroyed" 告警。
        """
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(None, None, None)
        if self._store_cm is not None:
            await self._store_cm.__aexit__(None, None, None)


async def build_memory_backends(
    cfg: AgentFrameworkConfig,
    *,
    database_url: str | None = None,
    embedder: Embeddings | None = None,
) -> MemoryBackends:
    """统一记忆后端入口：有 DSN 建 Postgres 后端；无 DSN 降级 InMemory。

    显式 store_type=postgres 但缺 DSN → ValueError 快速失败（防生产误配静默
    退回内存）；未显式指定而 DSN 已配置 → 自动启用 postgres（dev 零配置仍 in_memory）。
    只装配 store/checkpointer；会话元数据由 `agent/session` 的 `build_session_backend` 自装。

    WHY 守卫放工厂而非 config：config 约定不读环境变量，DSN 属运行环境（settings.py），
    由装配层把两者汇合后在此裁决。`if database_url is not None` 同时完成类型收窄。
    """
    if cfg.memory.store_type == "postgres" and not database_url:
        raise ValueError("store_type=postgres 必须配置 DATABASE_URL（settings.py / .env）")
    # 语义召回底座：把 embedding 转译为 langgraph Store 的 index 配置。
    factory = EmbeddingsFactory(cfg.memory.embedding, embedder=embedder)
    index_config = factory.build_index_config()
    if database_url is not None:
        return await _build_postgres_backends(database_url=database_url, index=index_config)
    return MemoryBackends(store=InMemoryStore(index=index_config), checkpointer=None)


async def _build_postgres_backends(
    *,
    database_url: str,
    index: dict[str, Any] | None = None,
) -> MemoryBackends:
    """prod 后端：AsyncPostgresStore + AsyncPostgresSaver（长期记忆 + 短期上下文）。

    checkpointer 与 Store 是 compile() 的两个独立参数、各有独立表结构
    （checkpoints/checkpoint_writes/checkpoint_blobs + store）。共用同一 Postgres
    实例，`.setup()` 幂等建表，首次运行自动完成。`index` 非 None 时语义检索底座生效：
    `.setup()` 会建 pgvector 扩展 + `store_vectors` 表 + HNSW 索引（见 P1.md §2.1）。
    会话元数据由 `agent/session` 自装，落在独立 `sessions` 表（不经本工厂）。
    """
    store_cm = AsyncPostgresStore.from_conn_string(database_url, index=index)
    store = await store_cm.__aenter__()
    try:
        saver_cm = AsyncPostgresSaver.from_conn_string(database_url)
        saver = await saver_cm.__aenter__()
    except BaseException:
        await store_cm.__aexit__(None, None, None)
        raise
    try:
        await store.setup()  # 建 store 表（幂等）
        await saver.setup()  # 建 checkpoints / checkpoint_writes / checkpoint_blobs（幂等）
    except BaseException:
        await saver_cm.__aexit__(None, None, None)
        await store_cm.__aexit__(None, None, None)
        raise
    return MemoryBackends(
        store=store,
        checkpointer=saver,
        _store_cm=store_cm,
        _saver_cm=saver_cm,
    )
