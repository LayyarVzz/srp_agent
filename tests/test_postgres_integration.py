"""Postgres 记忆后端集成测试（需 DATABASE_URL 进程环境变量，无则整体跳过）。

与 InMemory 单测（test_memory_adapter / test_session）镜像对应，验证三层记忆在
AsyncPostgresStore / AsyncPostgresSaver 上的行为一致性：
  1. 长期记忆：MemoryStore save/recall round-trip（命名空间 (user_id, "long_term")）
  2. 会话元数据：SessionManager create/resolve/list/delete（TTL 生效：supports_ttl=True）
  3. 短期上下文：PostgresSaver 跨图实例持久化（同 thread_id 恢复 messages）
  4. 命名空间隔离：用户 A 的长期记忆/会话不泄漏给用户 B

运行方式（cmd）：
    set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/srp_agent
    uv run pytest tests/test_postgres_integration.py -v
注意：数据库跨运行持久化，故用唯一 user/thread/memory id（不做跨运行总数断言）。
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from agent.core.config import AgentFrameworkConfig, DedupConfig
from agent.core.graph import build_agent_graph
from agent.errors import SessionError
from agent.intent.models import Intent
from agent.memory import (
    KIND_FACT,
    KIND_PREFERENCE,
    MemoryItem,
    MemoryStore,
    build_memory_backends,
    normalize_content_hash,
)
from agent.session import SQLAlchemySessionRepository, build_session_backend
from tests.conftest import chat_turn_messages

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="未设置 DATABASE_URL（进程环境变量），跳过 Postgres 集成测试",
)


def _async_dsn() -> str:
    """进程环境变量 DATABASE_URL → SQLAlchemy postgresql+psycopg 方言（复用 psycopg 驱动）。"""
    dsn = os.environ["DATABASE_URL"]
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+psycopg://", 1)
    return dsn


def _make_item(
    *,
    id: str,
    kind: str = KIND_FACT,
    content: str = "事实",
    session_id: str = "s1",
    user_id: str = "u1",
    importance: float = 0.5,
    content_hash: str | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=id,
        kind=kind,
        content=content,
        session_id=session_id,
        user_id=user_id,
        timestamp=datetime.now(UTC),
        provenance="test-pg",
        importance=importance,
        content_hash=normalize_content_hash(content) if content_hash is None else content_hash,
    )


@pytest.fixture
async def pg_backends():
    """经生产工厂构建记忆后端（测真实装配路径），teardown 关闭连接池。"""
    backends = await build_memory_backends(
        AgentFrameworkConfig.get_default(),
        database_url=os.environ["DATABASE_URL"],
    )
    yield backends
    await backends.aclose()


@pytest.fixture
async def session_backend():
    """会话模块自装入构建会话后端（独立于记忆后端），teardown 关闭连接池。"""
    backend = await build_session_backend(os.environ["DATABASE_URL"])
    yield backend
    await backend.aclose()


async def test_long_term_roundtrip_postgres(pg_backends) -> None:
    """长期记忆：save 后 recall 返回同一 MemoryItem（镜像 test_save_recall_roundtrip）。"""
    store = MemoryStore(pg_backends.store)
    uid = f"u-{uuid.uuid4().hex[:8]}"
    item = _make_item(
        id=f"m-{uuid.uuid4().hex[:8]}",
        kind=KIND_PREFERENCE,
        content="用户偏好简洁回答",
        user_id=uid,
    )
    await store.save(item)

    result = await store.recall(user_id=uid)
    assert [m.id for m in result.items] == [item.id]
    got = result.items[0]
    assert got.kind == KIND_PREFERENCE
    assert got.content == "用户偏好简洁回答"
    assert got.timestamp == item.timestamp
    assert got.importance == 0.5
    assert result.sources[0].source_id == item.id


async def test_session_crud_postgres(session_backend) -> None:
    """会话元数据：create/resolve/list/delete 往返；独立 sessions 表承载（分表达成）。"""
    assert isinstance(session_backend.repository, SQLAlchemySessionRepository)
    sessions = session_backend.manager
    uid = f"u-{uuid.uuid4().hex[:8]}"

    ctx = await sessions.create(user_id=uid)
    resolved = await sessions.resolve(user_id=uid, session_id=ctx.session_id)
    assert resolved.session_id == ctx.session_id
    assert resolved.user_id == uid

    listed = await sessions.list(user_id=uid)
    assert ctx.session_id in {c.session_id for c in listed}

    # 分表断言：会话行落在独立 sessions 表；store 表无 {user}.sessions 前缀（旧布局已迁走）。
    engine = create_async_engine(_async_dsn())
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT session_id FROM sessions WHERE user_id = :uid"), {"uid": uid}
                )
            ).first()
            assert row is not None and row[0] == ctx.session_id
            leaked = (
                await conn.execute(
                    text("SELECT 1 FROM store WHERE prefix LIKE '%.sessions' LIMIT 1")
                )
            ).first()
            assert leaked is None  # 会话不再经 langgraph store 命名空间
    finally:
        await engine.dispose()

    await sessions.delete(user_id=uid, session_id=ctx.session_id)
    with pytest.raises(SessionError):
        await sessions.resolve(user_id=uid, session_id=ctx.session_id)


async def test_short_term_persists_across_graph_instances(pg_backends, make_llm_service) -> None:
    """短期上下文：PostgresSaver 跨图实例持久化（同 thread_id 恢复两轮 messages）。"""
    store = pg_backends.store
    saver = pg_backends.checkpointer
    assert saver is not None
    cfg = AgentFrameworkConfig.get_default()
    thread_id = f"t-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # 图实例 #1：写一轮对话（2 条消息）到 checkpointer。
    graph1 = build_agent_graph(
        make_llm_service(chat_turn_messages(Intent.CHAT, "第一轮回答")),
        cfg,
        store=store,
        checkpointer=saver,
    )
    async for _ in graph1.astream(
        {"input": "你好", "session_id": thread_id, "user_id": "u-1"},
        config=config,
        stream_mode="updates",
    ):
        pass

    # 图实例 #2：换一个 graph 对象但复用同一 saver，应能恢复图 #1 的上下文。
    graph2 = build_agent_graph(
        make_llm_service(chat_turn_messages(Intent.CHAT, "第二轮回答")),
        cfg,
        store=store,
        checkpointer=saver,
    )
    state = await graph2.aget_state(config)
    messages = state.values.get("messages") or []
    assert len(messages) == 2  # 图 #1 的 user + ai 两轮前上下文已持久化


async def test_namespace_isolation_postgres(pg_backends, session_backend) -> None:
    """命名空间隔离：A 用户的长期记忆与会话不出现在 B 用户（镜像 test_recall_user_isolation）。"""
    store = MemoryStore(pg_backends.store)
    sessions = session_backend.manager
    uid_a = f"a-{uuid.uuid4().hex[:8]}"
    uid_b = f"b-{uuid.uuid4().hex[:8]}"

    item_a = f"m-{uuid.uuid4().hex[:8]}"
    item_b = f"m-{uuid.uuid4().hex[:8]}"
    await store.save(_make_item(id=item_a, user_id=uid_a))
    await store.save(_make_item(id=item_b, user_id=uid_b))
    ctx_b = await sessions.create(user_id=uid_b)

    # A 召回只见自己的记忆（B 的记忆不跨命名空间泄漏）；B 的会话列表只见自己的会话。
    result_a = await store.recall(user_id=uid_a)
    assert [m.id for m in result_a.items] == [item_a]
    listed_b = await sessions.list(user_id=uid_b)
    assert {c.session_id for c in listed_b} == {ctx_b.session_id}


# —— V3-M3 保存去重（L1 content-hash 精确层 + L3 新写；fixture 未配 embeddings → 语义层跳过）——

_THRESHOLD = DedupConfig().semantic_threshold


async def test_dedup_exact_merge_postgres(pg_backends) -> None:
    """L1 精确层：完全重复第二次 upsert 合并（原 id 复用、条数不涨），镜像 InMemory 版。"""
    store = MemoryStore(pg_backends.store)
    uid = f"u-{uuid.uuid4().hex[:8]}"
    base_id = f"m-{uuid.uuid4().hex[:8]}"
    await store.save(_make_item(id=base_id, user_id=uid, content="用户喜欢喝美式咖啡"))
    outcome = await store.upsert(
        _make_item(id=f"m-{uuid.uuid4().hex[:8]}", user_id=uid, content="用户喜欢喝美式咖啡"),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "merged"
    assert outcome.item.id == base_id
    result = await store.recall(user_id=uid, top_k=10)
    assert len(result.items) == 1


async def test_dedup_normalized_variant_merge_postgres(pg_backends) -> None:
    """L1 精确层归一化变体：仅标点差异也合并（Postgres filter JSONB 等值命中）。"""
    store = MemoryStore(pg_backends.store)
    uid = f"u-{uuid.uuid4().hex[:8]}"
    base_id = f"m-{uuid.uuid4().hex[:8]}"
    await store.save(_make_item(id=base_id, user_id=uid, content="用户喜欢喝美式咖啡"))
    outcome = await store.upsert(
        _make_item(id=f"m-{uuid.uuid4().hex[:8]}", user_id=uid, content="用户喜欢喝美式咖啡。"),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "merged"
    assert outcome.item.id == base_id
    result = await store.recall(user_id=uid, top_k=10)
    assert len(result.items) == 1


async def test_dedup_distinct_insert_postgres(pg_backends) -> None:
    """L3 新写：不同事实 upsert → inserted，条数 +1。"""
    store = MemoryStore(pg_backends.store)
    uid = f"u-{uuid.uuid4().hex[:8]}"
    await store.save(
        _make_item(id=f"m-{uuid.uuid4().hex[:8]}", user_id=uid, content="用户喜欢喝美式咖啡")
    )
    outcome = await store.upsert(
        _make_item(id=f"m-{uuid.uuid4().hex[:8]}", user_id=uid, content="用户是程序员"),
        semantic_threshold=_THRESHOLD,
    )
    assert outcome.action == "inserted"
    result = await store.recall(user_id=uid, top_k=10)
    assert len(result.items) == 2
