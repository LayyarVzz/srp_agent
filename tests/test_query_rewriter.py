"""agent/query/rewriter.py —— 查询改写器单测（离线 fake LLM）。

覆盖：结构化输出解析、失败/空结果返回 None（回退原始输入零回归）、prompt 组装
（HyDE 段按门控追加、上下文块带不可信声明、意图提示）、以及长度护栏接线。
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from agent.core.config import QueryUnderstandingConfig
from agent.intent.models import Intent, IntentContext
from agent.query import (
    REWRITE_PROMPT,
    QueryRewriter,
    QueryUnderstanding,
    QueryUnderstandingResult,
)
from tests.conftest import RecordingFakeChatModel, fake_structured_message, fake_text_message

CFG = QueryUnderstandingConfig()


def _rewriter(svc, **kwargs) -> QueryRewriter:
    """按配置默认值构造改写器（HyDE 总开关 + 短查询门控可覆盖）。"""
    return QueryRewriter(
        svc,
        enable_hypothetical=kwargs.get("enable_hypothetical", CFG.hypothetical_enabled),
        hypothetical_max_query_chars=kwargs.get("hyde_max", CFG.hyde_max_query_chars),
    )


def _understanding(**kwargs) -> QueryUnderstandingResult:
    return QueryUnderstandingResult(understanding=QueryUnderstanding(**kwargs))


async def test_rewrite_returns_understanding(make_llm_service) -> None:
    """结构化输出解析为 QueryUnderstanding（四项字段透传）。"""
    svc = make_llm_service(
        [
            fake_structured_message(
                _understanding(
                    main_query="用户所在公司的年假计算方式",
                    sub_queries=["年假按工龄计算", "出差补贴标准"],
                    synonyms=["带薪年假"],
                    hypothetical_answer="年假按工龄计算……",
                )
            )
        ]
    )
    result = await _rewriter(svc).rewrite("我们公司年假怎么算，另外出差补贴多少")
    assert result is not None
    assert result.main_query == "用户所在公司的年假计算方式"
    assert result.sub_queries == ["年假按工龄计算", "出差补贴标准"]
    assert result.synonyms == ["带薪年假"]
    assert result.hypothetical_answer == "年假按工龄计算……"


async def test_rewrite_failure_returns_none(make_llm_service) -> None:
    """LLM 调用异常（空迭代器）→ None（调用方回退原始输入，绝不抛）。"""
    svc = make_llm_service([])
    assert await _rewriter(svc).rewrite("我们公司年假怎么算") is None


async def test_rewrite_none_result_returns_none(make_llm_service) -> None:
    """模型未产出结构化输出（无 tool_calls）→ None，显式守卫不抛。"""
    svc = make_llm_service([fake_text_message("")])
    assert await _rewriter(svc).rewrite("我们公司年假怎么算") is None


async def test_rewrite_empty_understanding_returns_none(make_llm_service) -> None:
    """模型显式返回 understanding=None → None（视为无产出）。"""
    svc = make_llm_service([fake_structured_message(QueryUnderstandingResult())])
    assert await _rewriter(svc).rewrite("我们公司年假怎么算") is None


async def test_rewrite_empty_text_skips_llm(make_llm_service) -> None:
    """空白输入直接返回 None，不消耗 LLM 调用。"""
    svc = make_llm_service([])
    assert await _rewriter(svc).rewrite("   ") is None


# —— prompt 组装 ——


async def test_prompt_includes_context_and_intent(make_llm_service) -> None:
    """上下文块（摘要/关键信息）与意图提示进入 prompt；上下文声明为不可信数据。"""
    svc = make_llm_service(
        [fake_structured_message(_understanding(main_query="q"))],
        model_cls=RecordingFakeChatModel,
    )
    await _rewriter(svc).rewrite(
        "那它的上限呢",
        intent=Intent.CHAT,
        context=IntentContext(summary="用户在问年假", keyfacts=["用户是正式员工"]),
    )

    prompt = svc.chat_model.prompts[0]
    system_text = "".join(str(m.content) for m in prompt if isinstance(m, SystemMessage))
    assert "不可信数据" in system_text
    assert "用户在问年假" in system_text
    assert "用户是正式员工" in system_text
    assert "chat" in system_text  # 意图分类结果注入
    # 用户原话作为 HumanMessage（不被改写覆盖）。
    assert str(prompt[-1].content) == "那它的上限呢"


async def test_prompt_hyde_section_gated_by_length(make_llm_service) -> None:
    """HyDE 段仅在「短查询 + 总开关开」时追加；长查询不要求假设文档。"""
    svc = make_llm_service(
        [
            fake_structured_message(_understanding(main_query="短")),
            fake_structured_message(_understanding(main_query="长")),
        ],
        model_cls=RecordingFakeChatModel,
    )
    rewriter = _rewriter(svc, hyde_max=10)
    await rewriter.rewrite("短查询")
    await rewriter.rewrite("这是一个明显超过长度门控的很长很长的查询语句")

    short_text = "".join(str(m.content) for m in svc.chat_model.prompts[0])
    long_text = "".join(str(m.content) for m in svc.chat_model.prompts[1])
    assert "假设文档" in short_text
    assert "假设文档" not in long_text
    # 两段 prompt 都含基础改写判据（HyDE 只是附加段）。
    assert "主改写" in long_text


async def test_prompt_without_hyde_when_disabled(make_llm_service) -> None:
    """总开关关闭 → 任何长度都不要求假设文档。"""
    svc = make_llm_service(
        [fake_structured_message(_understanding(main_query="q"))],
        model_cls=RecordingFakeChatModel,
    )
    await _rewriter(svc, enable_hypothetical=False).rewrite("短查询")
    assert "假设文档" not in "".join(str(m.content) for m in svc.chat_model.prompts[0])


def test_prompt_declares_untrusted_and_semantic_boundaries() -> None:
    """提示词必须保留关键判据（改提示词时最容易被删掉的就是这些边界）。"""
    for marker in (
        "不可信数据",  # 安全约束
        "不得改变语义边界",  # 否定/条件/时间范围
        "不得引入用户没说过的新信息",  # 防编造
        "仅当输入包含多个可独立检索的信息需求时",  # 子查询节流
        "不得扩大或收窄语义范围",  # 同义边界
        "retrieval_needed",  # 是否值得检索
    ):
        assert marker in REWRITE_PROMPT, f"改写提示词缺少关键判据：{marker}"
