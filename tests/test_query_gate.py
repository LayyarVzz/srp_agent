"""agent/query/models.py + gate.py —— 查询理解数据模型与确定性预门控单测（离线）。

覆盖：变体池构造（保序/去重/截断/trim）、长度护栏、门控判据（空/极短/无实义字符/
寒暄命中/长句放行）、以及「门控不按长度猜闲聊」这条明确取舍。
"""

from __future__ import annotations

import pytest

from agent.core.config import QueryUnderstandingConfig
from agent.intent.models import Intent
from agent.query import QueryUnderstanding, QueryVariant, is_trivial_input, should_understand

CFG = QueryUnderstandingConfig()


# —— QueryUnderstanding.ranked_queries：变体池 ——


def test_ranked_queries_order_is_main_then_sub_then_synonym() -> None:
    """保序：主改写优先（总是首选检索），其后子查询、同义（仅补检用）。"""
    u = QueryUnderstanding(
        main_query="主查询",
        sub_queries=["子一", "子二"],
        synonyms=["同义一", "同义二"],
    )
    assert [(q.query, q.origin) for q in u.ranked_queries(max_variants=5)] == [
        ("主查询", QueryVariant.MAIN),
        ("子一", QueryVariant.SUB),
        ("子二", QueryVariant.SUB),
        ("同义一", QueryVariant.SYNONYM),
        ("同义二", QueryVariant.SYNONYM),
    ]


def test_ranked_queries_dedups_and_trims() -> None:
    """去重（含首尾空白归一后的重名）并按 max_variants 截断；空串不占位。"""
    u = QueryUnderstanding(
        main_query="主查询",
        sub_queries=[" 主查询 ", "", "子一"],
        synonyms=["子一", "同义"],
    )
    assert u.retrieval_queries(max_variants=3) == ["主查询", "子一", "同义"]
    assert u.retrieval_queries(max_variants=1) == ["主查询"]


def test_ranked_queries_empty_when_no_main_query() -> None:
    """主改写为空时变体池为空（子查询/同义不单独构成检索池）。"""
    u = QueryUnderstanding(main_query="   ", sub_queries=["子一"], synonyms=["同义"])
    assert u.ranked_queries(max_variants=5) == []


def test_hypothetical_not_in_variant_pool() -> None:
    """假设文档不进变体池：它是**嵌入输入**，不是检索查询（§2.6 隔离）。"""
    u = QueryUnderstanding(main_query="主查询", hypothetical_answer="假设答案文本")
    assert u.retrieval_queries(max_variants=5) == ["主查询"]


def test_defaults_are_retrieval_needed_true() -> None:
    """默认值得检索（保守：宁可多检索一次，也不因缺字段静默跳过）。"""
    u = QueryUnderstanding(main_query="x")
    assert u.retrieval_needed is True
    assert u.hypothetical_answer is None
    assert u.sub_queries == [] and u.synonyms == []


# —— 门控：结构性平凡输入 ——


@pytest.mark.parametrize(
    "text",
    ["", "   ", "。", "？", "!!", "~", "a", "1", "好"],
)
def test_trivial_inputs_skipped(text: str) -> None:
    """空 / 纯标点 / 长度 < min_query_chars / 无实义字符 → 跳过（零 LLM 成本）。"""
    assert is_trivial_input(text, min_query_chars=CFG.min_query_chars) is True
    assert should_understand(text, settings=CFG) is False


@pytest.mark.parametrize(
    "text",
    [
        "你好",
        "您好",
        "hi",
        "hello",
        "在吗",
        "谢谢",
        "多谢",
        "好的",
        "嗯嗯",
        "ok",
        "收到",
        "再见",
        "谢谢！",
    ],
)
def test_small_talk_skipped(text: str) -> None:
    """寒暄/确认/告别整句命中词表 → 跳过（尾随标点不影响命中）。"""
    assert should_understand(text, settings=CFG) is False


@pytest.mark.parametrize(
    "text",
    [
        "那它的上限呢",
        "我们公司年假怎么算",
        "帮我看看",
        "12加7等于多少",
        "你好，年假怎么算",
        "谢谢，那报销呢",
    ],
)
def test_real_queries_pass_gate(text: str) -> None:
    """真问题放行；**含寒暄但另有信息需求的长句也必须放行**（词表是整句精确匹配）。"""
    assert should_understand(text, settings=CFG) is True


def test_gate_does_not_guess_small_talk_by_length() -> None:
    """明确取舍：门控**不按长度猜闲聊** —— 长度分不出「好的呀」与「帮我看看」。

    这条是刻意的：误跳过请求会让改写与检索凭空失效（用户看不见的静默降级），
    比多花一次调用更糟；闲聊由 `retrieval_needed`（LLM 判）承担。
    """
    assert should_understand("帮我看看", settings=CFG, intent=Intent.CHAT) is True
    assert should_understand("好的呀", settings=CFG, intent=Intent.CHAT) is True


def test_gate_disabled_never_calls_llm() -> None:
    """enabled=False → 恒 False（零回归：不调 LLM、不产出门控常量由节点负责）。"""
    cfg = QueryUnderstandingConfig(enabled=False)
    assert should_understand("我们公司年假怎么算", settings=cfg) is False


def test_min_query_chars_configurable() -> None:
    """长度下限可配：调大后更短的输入被跳过。"""
    cfg = QueryUnderstandingConfig(min_query_chars=5)
    assert should_understand("你好呀", settings=cfg) is False
    assert should_understand("你好呀你好", settings=cfg) is True
