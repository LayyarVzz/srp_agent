"""agent/memory/persist.py —— 带外保存单测（离线）。

覆盖：save_conversation_memory 全字段/importance 透传/空抽取/单条失败续行/未知 kind 保留，
submit_memory_save fire-and-forget / 无事件循环跳过 / 空消息跳过。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from agent.memory import (
    KIND_FACT,
    KIND_PREFERENCE,
    PROVENANCE_CONVERSATION,
    MemoryItem,
    MemoryStore,
    save_conversation_memory,
    submit_memory_save,
    wait_pending_saves,
)
from agent.memory.models import MemoryExtraction
from agent.memory.persist import _background_tasks


class _StubExtractor:
    """固定返回给定抽取列表的 stub 抽取器（可空）；记录收到的 config 供透传断言。"""

    def __init__(self, extractions: Sequence[MemoryExtraction]) -> None:
        self._extractions = list(extractions)
        self.configs: list[dict | None] = []

    async def extract(
        self, messages: Sequence[BaseMessage], *, config: dict | None = None
    ) -> list[MemoryExtraction]:
        self.configs.append(config)
        return list(self._extractions)


class _FlakyStore:
    """对指定 content 的条目抛异常，其余委托真 MemoryStore（验证单条失败不中断）。"""

    def __init__(self, inner: MemoryStore, *, fail_on: str) -> None:
        self._inner = inner
        self._fail_on = fail_on

    async def save(self, item: MemoryItem) -> None:
        if item.content == self._fail_on:
            raise RuntimeError("boom")
        await self._inner.save(item)


async def test_save_conversation_memory_saves_all_fields() -> None:
    """保存的 MemoryItem 全字段正确（kind/content/session/user/provenance/timestamp）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor(
        [MemoryExtraction(kind=KIND_FACT, content="用户叫小明", importance=0.8)]
    )
    await save_conversation_memory(
        [HumanMessage(content="我叫小明")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )

    result = await store.recall(user_id="u1")
    assert len(result.items) == 1
    item = result.items[0]
    assert item.kind == KIND_FACT
    assert item.content == "用户叫小明"
    assert item.session_id == "s1"
    assert item.user_id == "u1"
    assert item.provenance == PROVENANCE_CONVERSATION
    assert item.timestamp is not None


async def test_save_conversation_memory_importance_passthrough() -> None:
    """抽取 importance 透传到落盘（非默认 0.5），保证 recall 排序信号不退化。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor(
        [MemoryExtraction(kind=KIND_FACT, content="重要事实", importance=0.9)]
    )
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )

    result = await store.recall(user_id="u1")
    assert result.items[0].importance == 0.9


async def test_save_conversation_memory_empty_extraction() -> None:
    """抽取结果为空 → 无落盘、不抛。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor([])
    await save_conversation_memory(
        [HumanMessage(content="现在几点了？")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )

    assert (await store.recall(user_id="u1")).items == []


async def test_save_conversation_memory_single_failure_continues() -> None:
    """单条保存失败仅记日志、继续保存其余条目，绝不抛出。"""
    raw = MemoryStore(InMemoryStore())
    store = _FlakyStore(raw, fail_on="失败条目")
    extractor = _StubExtractor(
        [
            MemoryExtraction(kind=KIND_FACT, content="失败条目", importance=0.8),
            MemoryExtraction(kind=KIND_PREFERENCE, content="成功条目", importance=0.9),
        ]
    )
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )

    result = await raw.recall(user_id="u1")
    assert [m.content for m in result.items] == ["成功条目"]


async def test_save_conversation_memory_unknown_kind_kept() -> None:
    """未知 kind（模型漂移）条目原样保存、不抛（召回端归入 other 组仍可达）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor(
        [MemoryExtraction(kind="profile", content="漂移数据", importance=0.5)]
    )
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )

    result = await store.recall(user_id="u1")
    assert len(result.items) == 1
    assert result.items[0].kind == "profile"


async def test_submit_memory_save_fire_and_forget() -> None:
    """submit 同步返回不阻塞；wait_pending_saves 后后台任务已落盘。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor(
        [MemoryExtraction(kind=KIND_FACT, content="用户叫小明", importance=0.8)]
    )
    submit_memory_save(
        [HumanMessage(content="我叫小明")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )
    await wait_pending_saves()

    result = await store.recall(user_id="u1")
    assert len(result.items) == 1
    assert result.items[0].content == "用户叫小明"


def test_submit_memory_save_no_loop_skips() -> None:
    """无事件循环（同步调用）→ 不调度任务、不抛（尽力而为跳过）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor([MemoryExtraction(kind=KIND_FACT, content="x", importance=0.5)])
    submit_memory_save(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )
    assert _background_tasks == set()


async def test_submit_memory_save_empty_messages_skips() -> None:
    """空 messages → 不调度任务。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor([MemoryExtraction(kind=KIND_FACT, content="x")])
    submit_memory_save(
        [],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )
    assert _background_tasks == set()


# —— 观测接线（O3）：callbacks + metadata 经 config 抵达带外 LLM 调用 ——


async def test_save_conversation_memory_forwards_observation_config() -> None:
    """callbacks+metadata 组装为 config 透传抽取器（抽取是带外链路首个 LLM 调用）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor([MemoryExtraction(kind=KIND_FACT, content="用户叫小明")])
    config = {"callbacks": [object()], "metadata": {"langfuse_session_id": "s1"}}
    await save_conversation_memory(
        [HumanMessage(content="我叫小明")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
        callbacks=config["callbacks"],
        metadata=config["metadata"],
    )
    assert extractor.configs == [config]


async def test_save_conversation_memory_without_observation_config_is_none() -> None:
    """未接线（都缺省）→ 抽取器收到 config=None，与接线前逐字同路（零回归判据）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor([])
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
    )
    assert extractor.configs == [None]


async def test_submit_memory_save_forwards_callbacks_and_metadata() -> None:
    """submit 的 callbacks/metadata 经后台任务抵达抽取器（runtime 接线契约）。"""
    store = MemoryStore(InMemoryStore())
    extractor = _StubExtractor([MemoryExtraction(kind=KIND_FACT, content="用户叫小明")])
    handler = object()
    submit_memory_save(
        [HumanMessage(content="我叫小明")],
        session_id="s1",
        user_id="u1",
        extractor=extractor,
        store=store,
        callbacks=[handler],
        metadata={"langfuse_session_id": "s1", "langfuse_user_id": "u1"},
    )
    await wait_pending_saves()

    assert len(extractor.configs) == 1
    assert extractor.configs[0] is not None
    assert extractor.configs[0]["callbacks"] == [handler]
    assert extractor.configs[0]["metadata"] == {
        "langfuse_session_id": "s1",
        "langfuse_user_id": "u1",
    }
