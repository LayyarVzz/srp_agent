"""agent/memory 值得性判定（v6.0 T1）单测。

覆盖 `should_keep` 判定表的全部分支（dev-version6.0.md §4.2）：
① 判定关闭 ② 空内容 ③ 拒收类别 + 低分 → 丢弃 ④ 拒收类别但高分 → 保守保留 ⑤ 保留类别；
外加配置边界校验与「旧数据/缺字段零回归」的向后兼容断言。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.core.config import MemoryBehaviorConfig, MemoryWorthConfig
from agent.memory import (
    KEEP_EXPLICIT,
    REJECT_CATEGORIES,
    WORTH_COMMONSENSE,
    WORTH_DERIVABLE,
    WORTH_SELF_GENERATED,
    WORTH_SMALL_TALK,
    MemoryExtraction,
    should_keep,
)

# —— 类别常量：拒收集合必须恰为四类「可判定硬负面」（多一类就多一份误丢风险）——


def test_reject_categories_exactly_four() -> None:
    """拒收集合 = {commonsense, derivable, self_generated, small_talk}（不多不少）。"""
    assert REJECT_CATEGORIES == frozenset(
        {WORTH_COMMONSENSE, WORTH_DERIVABLE, WORTH_SELF_GENERATED, WORTH_SMALL_TALK}
    )


def test_keep_categories_not_rejected() -> None:
    """保留类别（含 explicit）绝不落在拒收集合内。"""
    for category in ("identity", "preference", "goal", "plan", "relation", "constraint"):
        assert category not in REJECT_CATEGORIES
    assert KEEP_EXPLICIT not in REJECT_CATEGORIES


# —— should_keep 判定表逐行覆盖 ——


def test_should_keep_disabled_always_true() -> None:
    """① enabled=False → 恒 True（v5.1「全部落库」零回归），连空内容也不拦。"""
    cfg = MemoryWorthConfig(enabled=False)
    assert should_keep(
        MemoryExtraction(
            kind="fact", content="Python 是解释型语言", category=WORTH_COMMONSENSE, worth_score=0.1
        ),
        cfg=cfg,
    )
    assert should_keep(MemoryExtraction(kind="fact", content="   ", worth_score=0.1), cfg=cfg)


def test_should_keep_blank_content_false() -> None:
    """② 内容空/纯空白 → False（脏数据防御，与类别无关）。"""
    cfg = MemoryWorthConfig()
    assert should_keep(MemoryExtraction(kind="fact", content="", worth_score=1.0), cfg=cfg) is False
    assert (
        should_keep(
            MemoryExtraction(
                kind="fact", content=" \n\t ", category=KEEP_EXPLICIT, worth_score=1.0
            ),
            cfg=cfg,
        )
        is False
    )


@pytest.mark.parametrize(
    "category",
    [WORTH_COMMONSENSE, WORTH_DERIVABLE, WORTH_SELF_GENERATED, WORTH_SMALL_TALK],
)
def test_should_keep_reject_category_low_score_dropped(category: str) -> None:
    """③ 拒收类别 且 分数低于阈值 → False（丢弃）；四类拒收均适用。"""
    cfg = MemoryWorthConfig(min_worth_score=0.5)
    assert (
        should_keep(
            MemoryExtraction(
                kind="fact", content="公共常识内容", category=category, worth_score=0.3
            ),
            cfg=cfg,
        )
        is False
    )


@pytest.mark.parametrize(
    "category",
    [WORTH_COMMONSENSE, WORTH_DERIVABLE, WORTH_SELF_GENERATED, WORTH_SMALL_TALK],
)
def test_should_keep_reject_category_high_score_kept(category: str) -> None:
    """④ 标了拒收类别却给高分（模型自相矛盾）→ **保守保留**（提示词漂移信号）。

    WHY 这是「宁漏不误丢」原则的代码化：一次矛盾判定不足以丢掉用户的事实，
    宁可多记一条，把矛盾暴露在日志里等提示词修好。
    """
    cfg = MemoryWorthConfig(min_worth_score=0.5)
    assert (
        should_keep(
            MemoryExtraction(
                kind="fact", content="也许是用户事实", category=category, worth_score=0.9
            ),
            cfg=cfg,
        )
        is True
    )


@pytest.mark.parametrize(
    "category",
    ["identity", "preference", "goal", "plan", "relation", "constraint", KEEP_EXPLICIT],
)
def test_should_keep_keep_category_always_kept(category: str) -> None:
    """⑤ 保留类别 → True，与 worth_score 无关（分数不是落地门）。"""
    cfg = MemoryWorthConfig(min_worth_score=0.5)
    assert (
        should_keep(
            MemoryExtraction(kind="fact", content="用户事实", category=category, worth_score=0.0),
            cfg=cfg,
        )
        is True
    )


def test_should_keep_threshold_boundary_inclusive() -> None:
    """阈值边界：score == threshold 视为达标保留（判定式用 `>=`，避免临界抖动）。"""
    cfg = MemoryWorthConfig(min_worth_score=0.5)
    assert (
        should_keep(
            MemoryExtraction(
                kind="fact", content="临界", category=WORTH_COMMONSENSE, worth_score=0.5
            ),
            cfg=cfg,
        )
        is True
    )
    assert (
        should_keep(
            MemoryExtraction(
                kind="fact", content="差一点", category=WORTH_COMMONSENSE, worth_score=0.49
            ),
            cfg=cfg,
        )
        is False
    )


def test_should_keep_unknown_category_kept() -> None:
    """未知类别（模型漂移）→ 保留：不在拒收集合内即不丢（宁可多记）。"""
    cfg = MemoryWorthConfig()
    assert (
        should_keep(
            MemoryExtraction(
                kind="fact", content="漂移类别", category="unknown_thing", worth_score=0.1
            ),
            cfg=cfg,
        )
        is True
    )


# —— 向后兼容：新字段缺省必须落到「保守保留」 ——


def test_extraction_defaults_are_conservative() -> None:
    """旧提示词/旧模型输出（只给 kind/content/importance）→ 默认 score=1.0、类别保留 → 落库。"""
    e = MemoryExtraction(kind="fact", content="用户叫小明", importance=0.8)
    assert e.worth_score == 1.0
    assert e.category == "identity"
    assert e.worth_reason == ""
    assert should_keep(e, cfg=MemoryWorthConfig()) is True


def test_extraction_worth_score_range_validated() -> None:
    """worth_score 越界拒绝（[0,1] 语义）。"""
    with pytest.raises(ValidationError):
        MemoryExtraction(kind="fact", content="x", worth_score=1.5)
    with pytest.raises(ValidationError):
        MemoryExtraction(kind="fact", content="x", worth_score=-0.1)


# —— 配置：默认值 + 边界 + 嵌套装配 ——


def test_worth_config_defaults_and_nested() -> None:
    """MemoryWorthConfig 默认（enabled / 阈值 0.5 / 每轮上限 5）；MemoryBehaviorConfig 嵌套同值。"""
    assert MemoryWorthConfig().model_dump() == {
        "enabled": True,
        "min_worth_score": 0.5,
        "memories_max_per_turn": 5,
    }
    assert MemoryBehaviorConfig().worth == MemoryWorthConfig()


def test_worth_config_validation() -> None:
    """阈值越界与非法上限拒绝。"""
    with pytest.raises(ValidationError):
        MemoryWorthConfig(min_worth_score=1.5)
    with pytest.raises(ValidationError):
        MemoryWorthConfig(min_worth_score=-0.1)
    with pytest.raises(ValidationError):
        MemoryWorthConfig(memories_max_per_turn=0)
