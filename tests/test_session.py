"""agent/session/ —— 会话元数据仓库 + 管理单测（离线，SQLite memory）。

覆盖：create/resolve 往返、归属强校验（未命中 / 跨用户 → not_found）、按 user 列会话、
delete 归属校验、TTL 折算 expires_at 与过期不可见、非法 id 校验、与图的 thread_id
契约集成。存储经 `SQLAlchemySessionRepository`（sqlite+aiosqlite:///:memory:）真实
跑 SQLAlchemy 路径，不依赖 langgraph store。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import SecretStr

from agent.core.config import LLMConfig, SessionBehaviorConfig
from agent.core.graph import build_agent_graph
from agent.errors import (
    SESSION_ERROR_INVALID_ID,
    SESSION_ERROR_NOT_FOUND,
    SessionError,
)
from agent.intent.models import Intent
from agent.llm import LLMService
from agent.session import (
    SessionManager,
    SQLAlchemySessionRepository,
    build_session_backend,
    build_session_repository,
)
from agent.session.repository import SessionRow
from tests.conftest import StructuredFakeChatModel, chat_turn_messages


@pytest.fixture
async def session_repo():
    """每个用例独立的 SQLite memory 会话仓库（真实 SQLAlchemy 路径，自动建表）。"""
    repo = build_session_repository(None)
    await repo.setup()
    yield repo
    await repo.aclose()


def _make_manager(repo, *, ttl_minutes: int | None = None) -> SessionManager:
    return SessionManager(repo, config=SessionBehaviorConfig(ttl_minutes=ttl_minutes))


async def test_build_session_backend_in_memory() -> None:
    """会话模块自装入：无 DSN → SQLite memory 仓库 + SessionManager；create 可用、aclose no-op。

    验证 `agent/session` 与记忆后端解耦：本入口独立于 `build_memory_backends` 构造。
    """
    backend = await build_session_backend(None, config=SessionBehaviorConfig())
    try:
        assert isinstance(backend.repository, SQLAlchemySessionRepository)
        assert isinstance(backend.manager, SessionManager)
        ctx = await backend.manager.create(user_id="u1")
        resolved = await backend.manager.resolve(user_id="u1", session_id=ctx.session_id)
        assert resolved.session_id == ctx.session_id
    finally:
        await backend.aclose()


async def test_create_resolve_roundtrip(session_repo) -> None:
    """create 后 resolve 返回同一 SessionContext；存储事实落在独立 sessions 表。"""
    mgr = _make_manager(session_repo)
    ctx = await mgr.create(user_id="u1")

    assert ctx.session_id
    assert ctx.user_id == "u1"
    assert ctx.created_at is not None

    got = await mgr.resolve(user_id="u1", session_id=ctx.session_id)
    assert got.session_id == ctx.session_id
    assert got.user_id == "u1"
    assert got.created_at == ctx.created_at

    # 存储事实：经仓库可查（独立 sessions 表行，不再依赖 langgraph store 命名空间）。
    stored = await session_repo.get(user_id="u1", session_id=ctx.session_id)
    assert stored is not None
    assert stored.session_id == ctx.session_id
    assert stored.created_at == ctx.created_at


async def test_resolve_unknown_raises(session_repo) -> None:
    """未创建的 session_id → SessionError(session_error.not_found)。"""
    mgr = _make_manager(session_repo)
    with pytest.raises(SessionError) as ei:
        await mgr.resolve(user_id="u1", session_id="nope")
    assert ei.value.code == SESSION_ERROR_NOT_FOUND


async def test_resolve_cross_user_is_not_found(session_repo) -> None:
    """跨用户解析 → not_found（仓库按 user_id 过滤，不泄露存在性，防枚举）。"""
    mgr = _make_manager(session_repo)
    ctx = await mgr.create(user_id="u1")
    with pytest.raises(SessionError) as ei:
        await mgr.resolve(user_id="u2", session_id=ctx.session_id)
    assert ei.value.code == SESSION_ERROR_NOT_FOUND


async def test_list_user_isolation_and_order(session_repo) -> None:
    """list 仅含本 user 的会话，且按 created_at 降序。"""
    mgr = _make_manager(session_repo)
    c1 = await mgr.create(user_id="u1")
    c2 = await mgr.create(user_id="u1")
    await mgr.create(user_id="u2")

    ctxs = await mgr.list(user_id="u1")
    assert {c.session_id for c in ctxs} == {c1.session_id, c2.session_id}
    created = [c.created_at for c in ctxs]
    assert created == sorted(created, reverse=True)
    assert all(c.user_id == "u1" for c in ctxs)


async def test_delete_removes_metadata(session_repo) -> None:
    """delete 后 resolve → not_found。"""
    mgr = _make_manager(session_repo)
    ctx = await mgr.create(user_id="u1")
    await mgr.delete(user_id="u1", session_id=ctx.session_id)
    with pytest.raises(SessionError) as ei:
        await mgr.resolve(user_id="u1", session_id=ctx.session_id)
    assert ei.value.code == SESSION_ERROR_NOT_FOUND


async def test_delete_cross_user_raises(session_repo) -> None:
    """跨用户 delete → not_found（仓库按 user_id 过滤），且不影响 owner 侧记录。"""
    mgr = _make_manager(session_repo)
    ctx = await mgr.create(user_id="u1")
    with pytest.raises(SessionError) as ei:
        await mgr.delete(user_id="u2", session_id=ctx.session_id)
    assert ei.value.code == SESSION_ERROR_NOT_FOUND
    assert (await mgr.resolve(user_id="u1", session_id=ctx.session_id)).session_id == ctx.session_id


async def test_create_ttl_sets_expires_at(session_repo) -> None:
    """TTL 折算为 expires_at 落库（SQLite 亦生效，不再依赖 store 的 supports_ttl）。"""
    mgr = _make_manager(session_repo, ttl_minutes=30)
    ctx = await mgr.create(user_id="u1")

    async with session_repo._session_factory() as session:
        row = await session.get(SessionRow, ctx.session_id)
    assert row is not None
    assert row.expires_at is not None
    # SQLite 读回可能丢 tz（naive），统一视为 UTC 再与生成端（aware UTC）比较。
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    expected = ctx.created_at + timedelta(minutes=30)
    assert abs((expires - expected).total_seconds()) < 1


async def test_expired_session_is_not_found(session_repo) -> None:
    """过期视同不存在：resolve → not_found、list 剔除（读路径过滤 expires_at）。"""
    mgr = _make_manager(session_repo, ttl_minutes=30)
    ctx = await mgr.create(user_id="u1")

    # 模拟时间流逝：把 expires_at 改到过去（仓库对外只暴露读语义，测试直改行）。
    async with session_repo._session_factory() as session:
        row = await session.get(SessionRow, ctx.session_id)
        row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
        await session.commit()

    with pytest.raises(SessionError) as ei:
        await mgr.resolve(user_id="u1", session_id=ctx.session_id)
    assert ei.value.code == SESSION_ERROR_NOT_FOUND
    assert await mgr.list(user_id="u1") == []
    # 过期同未命中：delete 也报 not_found（过期行不可见，删除被视为无效操作）。
    with pytest.raises(SessionError) as ei:
        await mgr.delete(user_id="u1", session_id=ctx.session_id)
    assert ei.value.code == SESSION_ERROR_NOT_FOUND


async def test_invalid_user_id_raises(session_repo) -> None:
    """label 校验：含 "." 的 user_id → SessionError(session_error.invalid_id)。"""
    mgr = _make_manager(session_repo)
    with pytest.raises(SessionError) as ei:
        await mgr.create(user_id="bad.user")
    assert ei.value.code == SESSION_ERROR_INVALID_ID


def _make_llm_service(messages: list[Any]) -> LLMService:
    return LLMService(
        config=LLMConfig(api_key=SecretStr("sk-x")),
        chat_model=StructuredFakeChatModel(messages=iter(messages)),
    )


async def test_graph_thread_contract_with_manager(session_repo) -> None:
    """集成：SessionManager 发号 → graph 以 thread_id=session_id 驱动 → 同线程跨轮恢复；
    且 u2 绕过 resolve 直接以同一 thread_id 驱动能续跑 u1 的线程——证明 checkpointer
    不按 user 隔离，归属校验必须在入口层经 SessionManager.resolve 完成（行为文档化）。

    会话仓库（SQLite）与图上下文（InMemoryStore）解耦：两者互不依赖，契约仅靠
    session_id 字符串桥接。"""
    checkpointer = MemorySaver()  # 两图共享同一 checkpointer 才能复现「跨 user 续跑线程」
    mgr = _make_manager(session_repo)
    ctx = await mgr.create(user_id="u1")
    store = InMemoryStore()  # 图独立使用 InMemoryStore（load_context/记忆召回），与会话仓库无关

    # u1 第一轮：thread 写入 ctx.session_id 下。
    graph = build_agent_graph(
        _make_llm_service(chat_turn_messages(Intent.CHAT, "回复1")),
        tools=None,
        store=store,
        checkpointer=checkpointer,
    )
    async for _ in graph.astream(
        {"input": "你好", "session_id": ctx.session_id, "user_id": ctx.user_id},
        config={"configurable": {"thread_id": ctx.session_id}},
        stream_mode="updates",
    ):
        pass
    state = await graph.aget_state({"configurable": {"thread_id": ctx.session_id}})
    assert len(state.values["messages"]) == 2  # Human + AI

    # u2 绕过 SessionManager.resolve，直接用同一 thread_id 驱动（共享 checkpointer）。
    graph2 = build_agent_graph(
        _make_llm_service(chat_turn_messages(Intent.CHAT, "回复2")),
        tools=None,
        store=store,
        checkpointer=checkpointer,
    )
    async for _ in graph2.astream(
        {"input": "我是用户2", "session_id": ctx.session_id, "user_id": "u2"},
        config={"configurable": {"thread_id": ctx.session_id}},
        stream_mode="updates",
    ):
        pass
    # 两轮消息都留在同一线程（u2 触达了 u1 的历史）→ 反衬入口层 resolve 的必要性。
    state2 = await graph2.aget_state({"configurable": {"thread_id": ctx.session_id}})
    assert len(state2.values["messages"]) == 4

    # 而经 SessionManager 的归属校验，u2 解析不到 u1 的会话。
    with pytest.raises(SessionError) as ei:
        await mgr.resolve(user_id="u2", session_id=ctx.session_id)
    assert ei.value.code == SESSION_ERROR_NOT_FOUND
