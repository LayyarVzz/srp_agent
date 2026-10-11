"""agent/memory/adapter.py —— langgraph Store 适配层单测（离线）。

覆盖：save/recall 往返、user 隔离、importance 确定性排序、kind 过滤、top_k、
来源引用、脏数据跳过、图内节点经注入的 store 读写（P4-1 验收「图内节点可读写 Store」）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from agent.memory import (
    KIND_EPISODE,
    KIND_FACT,
    KIND_OTHER,
    KIND_PREFERENCE,
    LONG_TERM_NAMESPACE,
    MemoryItem,
    MemoryStore,
)


def _make_item(
    *,
    id: str,
    kind: str = KIND_FACT,
    content: str = "事实",
    session_id: str = "s1",
    user_id: str = "u1",
    timestamp: datetime | None = None,
    importance: float = 0.5,
) -> MemoryItem:
    return MemoryItem(
        id=id,
        kind=kind,
        content=content,
        session_id=session_id,
        user_id=user_id,
        timestamp=timestamp or datetime.now(UTC),
        provenance="test",
        importance=importance,
    )


async def test_save_recall_roundtrip() -> None:
    """save 后 recall 返回同一 MemoryItem（timestamp 经 JSON 序列化后还原为 datetime）。"""
    store = MemoryStore(InMemoryStore())
    item = _make_item(id="m1", kind=KIND_PREFERENCE, content="用户偏好简洁回答")
    await store.save(item)

    result = await store.recall(user_id="u1")
    assert len(result.items) == 1
    got = result.items[0]
    assert got.id == "m1"
    assert got.kind == KIND_PREFERENCE
    assert got.content == "用户偏好简洁回答"
    assert got.session_id == "s1"
    assert got.timestamp == item.timestamp
    assert got.importance == 0.5
    # 来源引用：source_id = memory id，snippet = content。
    assert len(result.sources) == 1
    assert result.sources[0].source_id == "m1"
    assert result.sources[0].snippet == "用户偏好简洁回答"


async def test_recall_user_isolation() -> None:
    """命名空间按 user_id 隔离：A 用户的记忆不出现在 B 用户召回。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="m1", user_id="u1"))
    await store.save(_make_item(id="m2", user_id="u2"))

    result = await store.recall(user_id="u1")
    assert [m.id for m in result.items] == ["m1"]


async def test_recall_sorts_by_importance_desc_then_time() -> None:
    """非语义召回：importance 降序；同 importance 按时间倒序兜底。"""
    store = MemoryStore(InMemoryStore())
    t = datetime(2026, 1, 1, tzinfo=UTC)
    await store.save(_make_item(id="low", importance=0.2, timestamp=t))
    await store.save(_make_item(id="high", importance=0.9, timestamp=t))
    await store.save(_make_item(id="mid_old", importance=0.5, timestamp=t))
    await store.save(_make_item(id="mid_new", importance=0.5, timestamp=t + timedelta(hours=1)))

    result = await store.recall(user_id="u1")
    assert [m.id for m in result.items] == ["high", "mid_new", "mid_old", "low"]


async def test_recall_kinds_filter_and_top_k() -> None:
    """kinds 过滤只返回指定 kind；top_k 截断为 importance 最高的前 N 条。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="f1", kind=KIND_FACT, importance=0.4))
    await store.save(_make_item(id="p1", kind=KIND_PREFERENCE, importance=0.9))
    await store.save(_make_item(id="p2", kind=KIND_PREFERENCE, importance=0.7))

    only_pref = await store.recall(user_id="u1", kinds=[KIND_PREFERENCE])
    assert [m.id for m in only_pref.items] == ["p1", "p2"]

    top1 = await store.recall(user_id="u1", top_k=1)
    assert [m.id for m in top1.items] == ["p1"]


async def test_recall_unknown_kind_groups_as_other() -> None:
    """unknown kind（模型漂移）归入 other 组：不污染定向召回，含 KIND_OTHER 时可召回。"""
    store = MemoryStore(InMemoryStore())
    await store.save(_make_item(id="f1", kind=KIND_FACT, content="事实"))
    await store.save(_make_item(id="p1", kind=KIND_PREFERENCE, content="偏好"))
    await store.save(_make_item(id="drift", kind="profile", content="漂移数据"))
    await store.save(_make_item(id="odd", kind="weird", content="另一漂移"))

    # 定向召回不受 unknown kind 污染。
    only_pref = await store.recall(user_id="u1", kinds=[KIND_PREFERENCE])
    assert [m.id for m in only_pref.items] == ["p1"]
    only_fact = await store.recall(user_id="u1", kinds=[KIND_FACT, KIND_EPISODE])
    assert [m.id for m in only_fact.items] == ["f1"]

    # kinds 显式含 KIND_OTHER 时，unknown kind 条目按 other 组被召回。
    with_other = await store.recall(user_id="u1", kinds=[KIND_FACT, KIND_OTHER])
    assert {m.id for m in with_other.items} == {"f1", "drift", "odd"}

    # 全量召回（无 kinds）返回全部条目（unknown kind 原样返回）。
    all_items = await store.recall(user_id="u1")
    assert {m.id for m in all_items.items} == {"f1", "p1", "drift", "odd"}


async def test_recall_skips_dirty_data() -> None:
    """损坏的 value（非法结构）被跳过，不中断召回。"""
    raw = InMemoryStore()
    # 直接写一条缺少必填字段的脏数据（模拟旧版本/外部写入的坏记录）。
    await raw.aput(("u1", LONG_TERM_NAMESPACE), "bad", {"not": "a valid memory item"})
    store = MemoryStore(raw)
    await store.save(_make_item(id="good"))

    result = await store.recall(user_id="u1")
    assert [m.id for m in result.items] == ["good"]


class _StoreNodeState(TypedDict, total=False):
    recalled_count: int


async def _store_node(state: _StoreNodeState, *, store: BaseStore) -> dict[str, int]:
    """图节点：经 LangGraph 注入的 store + MemoryStore 适配层完成写与读。"""
    ms = MemoryStore(store)
    await ms.save(_make_item(id="in-graph", user_id="ug"))
    result = await ms.recall(user_id="ug")
    return {"recalled_count": len(result.items)}


async def test_graph_node_can_read_write_store() -> None:
    """P4-1 验收：compile(store=...) 后图内节点可经 MemoryStore 读写 Store（同一实例）。"""
    injected = InMemoryStore()
    builder = StateGraph(_StoreNodeState)
    builder.add_node("store_node", _store_node)
    builder.add_edge(START, "store_node")
    builder.add_edge("store_node", END)
    graph = builder.compile(store=injected)

    out = await graph.ainvoke({})
    assert out["recalled_count"] == 1
    # 节点读写的是传入的同一 store 实例（数据真实落在注入实例上）。
    assert await injected.asearch(("ug", LONG_TERM_NAMESPACE))
