"""记忆存储工厂：按配置构造 langgraph BaseStore / checkpointer（dev/prod 切换点）。

对外只暴露统一入口 `build_memory_backends`：传入 `database_url` 则构造
AsyncPostgresStore + AsyncPostgresSaver（prod，各自连接池），无则降级 InMemoryStore
（checkpointer=None，由 graph 编译默认 MemorySaver）。构造细节全部私有，
后续换库/新增后端只需改本模块，调用方不变。
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from agent.core.config import AgentFrameworkConfig


@dataclass
class MemoryBackends:
    """记忆后端对：store（长期记忆 + 会话元数据）+ checkpointer（会话/短期）。

    `from_conn_string` 返回 async context manager，此处持有其句柄，`aclose()` 时对称
    `__aexit__` 关闭连接池。InMemory 分支两句柄为 None，`aclose()` 天然 no-op。
    store 由 MemoryStore 与 SessionManager 共享同一实例；checkpointer 供
    compile(checkpointer=)（thread_id == session_id 承载短期上下文）。
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
    cfg: AgentFrameworkConfig, *, database_url: str | None = None
) -> MemoryBackends:
    """统一记忆后端入口：有 DSN 建 Postgres 后端；无 DSN 降级 InMemory。

    显式 store_type=postgres 但缺 DSN → ValueError 快速失败（防生产误配静默
    退回内存）；未显式指定而 DSN 已配置 → 自动启用 postgres（dev 零配置仍 in_memory）。
    """
    if _resolve_store_type(cfg, database_url) == "postgres":
        return await _build_postgres_backends(database_url=database_url, cfg=cfg)
    return MemoryBackends(store=InMemoryStore(), checkpointer=None)


def _resolve_store_type(
    cfg: AgentFrameworkConfig, database_url: str | None
) -> Literal["in_memory", "postgres"]:
    """裁决单一事实：显式 store_type=postgres 但无 DSN → 快速失败；有 DSN → postgres。

    WHY 放工厂而非 config：config 约定不读环境变量，DSN 属运行环境（settings.py），
    由装配层把两者汇合后在此裁决。守卫后 store_type 已不再参与返回分支（有 DSN
    必建 PG），故仅以 DSN 有无决定。
    """
    if cfg.memory.store_type == "postgres" and not database_url:
        raise ValueError("store_type=postgres 必须配置 DATABASE_URL（settings.py / .env）")
    return "postgres" if database_url else "in_memory"


async def _build_postgres_backends(
    *,
    database_url: str,
    cfg: AgentFrameworkConfig,
) -> MemoryBackends:
    """prod 后端：AsyncPostgresStore + AsyncPostgresSaver，同一 DSN、各自连接池/独立表。

    checkpointer 与 Store 是 compile() 的两个独立参数、各有独立表结构
    （checkpoints/checkpoint_writes/checkpoint_blobs + store），共用同一 Postgres 实例
    `.setup()` 幂等建表，首次运行自动完成。
    """
    store_cm = AsyncPostgresStore.from_conn_string(database_url)
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
    if cfg.session.ttl_minutes is not None:
        # PostgresStore.supports_ttl=True，但过期清理需显式启动（避免默认空转后台任务）。
        store.start_ttl_sweeper()
    return MemoryBackends(
        store=store,
        checkpointer=saver,
        _store_cm=store_cm,
        _saver_cm=saver_cm,
    )
