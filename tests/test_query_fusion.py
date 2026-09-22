"""agent/query/fusion.py —— RRF 融合纯函数单测（离线）。

覆盖：多路融合排序、跨路重复项累加、去重键构造、代表分数取最高原始分、
确定性 tie-break、top_n 截断与空输入。
"""

from __future__ import annotations

from agent.query import DEFAULT_RRF_K, RankedHit, dedup_key, reciprocal_rank_fusion


def _hit(key: str, *, score: float | None = None) -> RankedHit[str]:
    """构造一条带分数（或缺失分数）的命中。"""
    return RankedHit(key=key, item=f"item-{key}", score=score)


# —— 去重键 ——


def test_dedup_key_same_inputs_same_key() -> None:
    """同来源 + 同内容前缀 → 同键（多路返回同一片段时归并为一条）。"""
    assert dedup_key("doc1", "内容是……") == dedup_key("doc1", "内容是……")


def test_dedup_key_distinguishes_source_and_content() -> None:
    """不同来源或不同内容 → 不同键。"""
    assert dedup_key("doc1", "内容") != dedup_key("doc2", "内容")
    assert dedup_key("doc1", "内容甲") != dedup_key("doc1", "内容乙")


def test_dedup_key_content_prefix_cuts_long_text() -> None:
    """前缀长度可配：前缀以内的差异不影响归并（同一片段被不同长度的截断也能合并）。"""
    body = "前缀" * 100
    assert dedup_key("doc1", body + "尾部甲", content_prefix=16) == dedup_key(
        "doc1", body + "尾部乙", content_prefix=16
    )
    # 前缀覆盖到差异处时不再归并（防止「前缀设得过长等于整段比较」的误解）。
    assert dedup_key("doc1", body + "尾部甲", content_prefix=len(body) + 3) != dedup_key(
        "doc1", body + "尾部乙", content_prefix=len(body) + 3
    )


# —— RRF 融合 ——


def test_fusion_rewards_items_hit_by_multiple_lists() -> None:
    """被多路同时召回的内容更可信：累加 1/(k+rank) 后升到首位。"""
    fused = reciprocal_rank_fusion(
        [
            [_hit("a"), _hit("b")],
            [_hit("b"), _hit("c")],
        ]
    )
    assert [h.key for h in fused] == ["b", "a", "c"]


def test_fusion_keeps_best_raw_score_as_representative() -> None:
    """代表分数取**最高原始分**（不求和/不平均）—— 引用里要展示可解释的相关度。"""
    fused = reciprocal_rank_fusion(
        [
            [_hit("a", score=0.4)],
            [_hit("a", score=0.8)],
        ]
    )
    assert fused[0].score == 0.8


def test_fusion_score_none_when_all_missing() -> None:
    """各路都没有分数（asearch 兜底条目）→ 代表分数为 None。"""
    fused = reciprocal_rank_fusion([[_hit("a")], [_hit("b")]])
    assert {h.score for h in fused} == {None}


def test_fusion_prefers_scored_over_unscored_for_same_key() -> None:
    """同键：有分数的一路优先作为代表（None 不覆盖真实相似度）。"""
    fused = reciprocal_rank_fusion([[_hit("a")], [_hit("a", score=0.3)]])
    assert fused[0].score == 0.3


def test_fusion_is_deterministic_on_ties() -> None:
    """tie-break 确定性：融合分相同 → 先出现的路序 → 路内名次（跨运行可复现）。"""
    lists = [[_hit("x"), _hit("y")], [_hit("y"), _hit("x")]]
    first = [h.key for h in reciprocal_rank_fusion(lists)]
    second = [h.key for h in reciprocal_rank_fusion(lists)]
    assert first == second == ["x", "y"]


def test_fusion_respects_top_n() -> None:
    """top_n 截断（融合序前 N 条）。"""
    fused = reciprocal_rank_fusion([[_hit("a"), _hit("b"), _hit("c")]], top_n=2)
    assert [h.key for h in fused] == ["a", "b"]


def test_fusion_handles_empty_input() -> None:
    """空输入 / 空路 → 返回空列表（不抛）。"""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_fusion_k_is_passed_through() -> None:
    """k 影响融合分但不影响「多路命中优先」的相对结论（单调性）。"""
    lists = [[_hit("a"), _hit("b")], [_hit("b")]]
    assert [h.key for h in reciprocal_rank_fusion(lists, k=1)] == ["b", "a"]
    assert [h.key for h in reciprocal_rank_fusion(lists, k=DEFAULT_RRF_K)] == ["b", "a"]


def test_fusion_single_list_keeps_order() -> None:
    """单路输入即返回原序（名次不变）—— 与记忆侧「单查询不走融合」的取舍一致。"""
    fused = reciprocal_rank_fusion([[_hit("a"), _hit("b"), _hit("c")]])
    assert [h.key for h in fused] == ["a", "b", "c"]
