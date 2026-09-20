"""agent/memory/persist.py 值得性接线单测（v6.0 T1）。

覆盖 `save_conversation_memory` / `submit_memory_save` 与值得性判定的接线：
不值得的内容不落库、值得的照常落库（含显式「记住」（explicit）零误伤，V6-M2）、
每轮条数上限按 worth_score 截断、丢弃明细可观测（INFO 日志）、
以及 `worth` 缺省/关闭时与 v5.1 完全一致（零回归）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from agent.core.config import MemoryWorthConfig
from agent.memory import (
    KEEP_EXPLICIT,
    KIND_FACT,
    LONG_TERM_NAMESPACE,
    WORTH_COMMONSENSE,
    MemoryExtraction,
    MemoryItem,
    MemoryStore,
    save_conversation_memory,
    submit_memory_save,
    wait_pending_saves,
)
from agent.memory.persist import REJECT_CATEGORIES, should_keep


class _StubExtractor:
    """固定返回给定抽取列表的 stub 抽取器（顺序即模型输出顺序）。"""

    def __init__(self, extractions: Sequence[MemoryExtraction]) -> None:
        self._extractions = list(extractions)

    async def extract(self, messages: Sequence[BaseMessage]) -> list[MemoryExtraction]:
        return list(self._extractions)


def _extraction(
    content: str, *, category: str = "identity", score: float = 1.0, kind: str = KIND_FACT
) -> MemoryExtraction:
    """构造一条抽取结果（默认「保留类别 + 高分」= 会落库）。"""
    return MemoryExtraction(
        kind=kind, content=content, category=category, worth_score=score, worth_reason="测试"
    )


async def _store_contents(store: MemoryStore, user_id: str = "u1") -> list[str]:
    """当前用户**全部**已落库内容（排序后返回，断言规模用）。

    WHY 不走 `recall()`：「条目是否落库」是存储事实，而 recall 是「按 importance/timestamp
    取 top_k」的召回语义 —— 用它数条数会把「落库几条」与「召回排序取了几条」混为一谈
    （同 importance 时按时间倒序，条数会被 top_k 截断而误导断言）。
    故测试直接读 store 命名空间（`agent/memory/adapter.py::LONG_TERM_NAMESPACE`）。
    """
    hits = await store._store.asearch((user_id, LONG_TERM_NAMESPACE), limit=100)
    return sorted(MemoryItem.model_validate(hit.value).content for hit in hits)


async def _save(
    store: MemoryStore, extractions: Sequence[MemoryExtraction], **kwargs: object
) -> None:
    """驱动一次带外保存（messages 与抽取结果无关，stub 抽取器决定产出）。"""
    await save_conversation_memory(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=_StubExtractor(extractions),
        store=store,
        **kwargs,  # type: ignore[arg-type]
    )


# —— V6-M1：常识/寒暄被拒收，个人事实保留 ——


async def test_commonsense_dropped_identity_kept() -> None:
    """V6-M1：同轮「公共常识 + 用户身份事实」→ 只落身份事实，常识被丢弃。"""
    store = MemoryStore(InMemoryStore())
    await _save(
        store,
        [
            _extraction("用户叫小明，职业是医生", category="identity", score=0.85),
            _extraction(
                "Python 是解释型语言", category=WORTH_COMMONSENSE, score=0.2, kind="episode"
            ),
        ],
    )

    assert await _store_contents(store) == ["用户叫小明，职业是医生"]


@pytest.mark.parametrize("category", sorted(REJECT_CATEGORIES))
async def test_all_reject_categories_dropped(category: str) -> None:
    """四类拒收（commonsense/derivable/self_generated/small_talk）低分时均不落库。"""
    store = MemoryStore(InMemoryStore())
    await _save(store, [_extraction("不值得记的内容", category=category, score=0.25)])
    assert await _store_contents(store) == []


# —— V6-M2：显式「记住 X」零误伤 ——


async def test_explicit_remember_not_dropped_even_if_commonsense() -> None:
    """V6-M2：用户明确要求记住的常识内容 → 落库（explicit 类别不参与拒收）。"""
    store = MemoryStore(InMemoryStore())
    await _save(
        store,
        [
            _extraction(
                "用户要求记住：TCP 三次握手的过程",
                category=KEEP_EXPLICIT,
                score=0.9,
            )
        ],
    )

    assert await _store_contents(store) == ["用户要求记住：TCP 三次握手的过程"]


async def test_explicit_remember_not_dropped_by_turn_cap() -> None:
    """上限截断按 worth_score 排序：explicit 高分不会被低分项挤掉（V6-M2 的边界情形）。

    阈值设 0.8 → 「较低价值事实」被判为不值得（不落库、也不占名额）；
    上限设 2 → 保留 explicit(0.95) 与 0.85 两条，explicit 居首。
    """
    store = MemoryStore(InMemoryStore())
    await _save(
        store,
        [
            _extraction("较低价值事实", score=0.51),
            _extraction("用户显式要求记住的事实", category=KEEP_EXPLICIT, score=0.95),
            _extraction("次高价值事实", score=0.85),
        ],
        worth=MemoryWorthConfig(memories_max_per_turn=2, min_worth_score=0.8),
    )

    assert set(await _store_contents(store)) == {"用户显式要求记住的事实", "次高价值事实"}


# —— 判定关闭 / worth 缺省：v5.1 零回归 ——


async def test_worth_disabled_saves_everything() -> None:
    """worth.enabled=False → 全部落库（含常识），回到 v5.1 行为。"""
    store = MemoryStore(InMemoryStore())
    await _save(
        store,
        [
            _extraction("Python 是解释型语言", category=WORTH_COMMONSENSE, score=0.1),
            _extraction("用户叫小明"),
        ],
        worth=MemoryWorthConfig(enabled=False),
    )

    assert await _store_contents(store) == ["Python 是解释型语言", "用户叫小明"]


async def test_worth_none_defaults_to_judging() -> None:
    """worth 缺省（不传）→ 用默认配置（判定开启）：常识仍被丢弃。"""
    store = MemoryStore(InMemoryStore())
    await _save(store, [_extraction("TCP 三次握手是……", category=WORTH_COMMONSENSE, score=0.2)])
    assert await _store_contents(store) == []


async def test_legacy_extraction_without_worth_fields_saved() -> None:
    """旧抽取器产出（无值得性字段）→ 默认保守保留，落库（向后兼容）。"""
    store = MemoryStore(InMemoryStore())
    legacy = MemoryExtraction(kind=KIND_FACT, content="用户叫小明，职业是医生", importance=0.8)
    await _save(store, [legacy])
    assert await _store_contents(store) == ["用户叫小明，职业是医生"]


# —— 每轮上限 ——


async def test_turn_cap_keeps_highest_worth() -> None:
    """超上限时按 worth_score 降序保留前 N（不被模型输出顺序左右）。"""
    store = MemoryStore(InMemoryStore())
    await _save(
        store,
        [
            _extraction("中价值", score=0.6),
            _extraction("最高价值", score=0.99),
            _extraction("较低价值", score=0.55),
        ],
        worth=MemoryWorthConfig(memories_max_per_turn=2),
    )

    assert await _store_contents(store) == ["中价值", "最高价值"]


async def test_turn_cap_counts_only_kept_items() -> None:
    """被值得性丢弃的条目不占用每轮名额（先判后截）。"""
    store = MemoryStore(InMemoryStore())
    await _save(
        store,
        [
            _extraction("被丢弃常识 1", category=WORTH_COMMONSENSE, score=0.1),
            _extraction("被丢弃常识 2", category=WORTH_COMMONSENSE, score=0.1),
            _extraction("保留事实 1", score=0.8),
            _extraction("保留事实 2", score=0.7),
        ],
        worth=MemoryWorthConfig(memories_max_per_turn=2),
    )

    assert await _store_contents(store) == ["保留事实 1", "保留事实 2"]


# —— 可观测性（V6-M1 验收：日志可见拒收类别与分数）——


async def test_drop_is_logged_with_category_and_score(caplog: pytest.LogCaptureFixture) -> None:
    """丢弃走 INFO 日志，且带 category / score / reason / 内容（可审计、可复盘误丢）。"""
    store = MemoryStore(InMemoryStore())
    with caplog.at_level(logging.INFO, logger="agent.memory.persist"):
        await _save(
            store,
            [
                _extraction("Python 是解释型语言", category=WORTH_COMMONSENSE, score=0.2),
                _extraction("用户叫小明"),
            ],
        )

    text = caplog.text
    assert "记忆丢弃（不值得）" in text
    assert "category=commonsense" in text
    assert "score=0.20" in text
    assert "Python 是解释型语言" in text
    assert "落库 1 条" in text
    assert "不值得 1" in text


async def test_cap_truncation_not_reported_as_worthless(caplog: pytest.LogCaptureFixture) -> None:
    """超上限截断**不得**被打成「不值得记」—— 两种丢弃原因语义不同，混报会误导复盘。

    回归实现中曾出现的缺陷：按「取过 should_keep 的差集」反推丢弃原因时，
    被名额挤掉的「值得记」条目会被误报成类别/分数判定失败。
    """
    store = MemoryStore(InMemoryStore())
    with caplog.at_level(logging.INFO, logger="agent.memory.persist"):
        await _save(
            store,
            [_extraction("最高价值事实", score=0.9), _extraction("次高价值事实", score=0.8)],
            worth=MemoryWorthConfig(memories_max_per_turn=1),
        )

    assert "记忆条数超上限" in caplog.text
    assert "记忆丢弃（不值得）" not in caplog.text  # 关键：不混淆两种原因
    assert "超上限 1" in caplog.text


async def test_no_drop_log_when_everything_kept(caplog: pytest.LogCaptureFixture) -> None:
    """全部保留时不产生「丢弃/判定汇总」噪声日志（只在真有丢弃时打）。"""
    store = MemoryStore(InMemoryStore())
    with caplog.at_level(logging.INFO, logger="agent.memory.persist"):
        await _save(store, [_extraction("用户叫小明")])

    assert "记忆丢弃（不值得）" not in caplog.text
    assert "记忆值得性判定" not in caplog.text


async def test_turn_cap_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """超上限时记日志并列出被截断的条目（可观测）。"""
    store = MemoryStore(InMemoryStore())
    with caplog.at_level(logging.INFO, logger="agent.memory.persist"):
        await _save(
            store,
            [_extraction("事实 A", score=0.9), _extraction("事实 B", score=0.4)],
            worth=MemoryWorthConfig(memories_max_per_turn=1, min_worth_score=0.0),
        )

    assert "记忆条数超上限" in caplog.text
    assert "事实 B" in caplog.text


# —— 带外入口（submit_memory_save）透传 worth ——


async def test_submit_memory_save_passes_worth() -> None:
    """submit_memory_save 经后台任务把 worth 传给 save_conversation_memory（装配层接线）。"""
    store = MemoryStore(InMemoryStore())
    submit_memory_save(
        [HumanMessage(content="x")],
        session_id="s1",
        user_id="u1",
        extractor=_StubExtractor(
            [
                _extraction("Python 是解释型语言", category=WORTH_COMMONSENSE, score=0.1),
                _extraction("用户叫小明"),
            ]
        ),
        store=store,
        worth=MemoryWorthConfig(),
    )
    await wait_pending_saves()

    assert await _store_contents(store) == ["用户叫小明"]


# —— should_keep 仍是唯一决策点（接线不得复制判据）——


def test_should_keep_is_single_decision_point() -> None:
    """persist 导出的 `should_keep` 即判定入口（模块级单点，便于评审与替换）。"""
    cfg = MemoryWorthConfig()
    dropped = _extraction("常识", category=WORTH_COMMONSENSE, score=0.1)
    assert should_keep(dropped, cfg=cfg) is False
    assert should_keep(_extraction("事实"), cfg=cfg) is True


def test_memory_item_carries_no_worth_fields() -> None:
    """值得性字段是写入侧元数据：`MemoryItem` 不得携带（不污染召回与存储）。"""
    assert "worth_score" not in MemoryItem.model_fields
    assert "category" not in MemoryItem.model_fields
    assert "worth_reason" not in MemoryItem.model_fields
