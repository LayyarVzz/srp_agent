"""agent/intent —— 意图分类器单测（离线，零网络）。

覆盖：规则兜底（关键词命中 / 默认 chat / 自定义关键词集 / 忽略上下文）、
LLM 分类器离线 happy path 与失败降级、分类 prompt 的 confidence 语义与上下文注入。
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from agent.intent.classifiers import (
    _CONTEXT_EMPTY_HINT,
    _CONTEXT_MAX_MESSAGES,
    _CONTEXT_MAX_TOTAL_CHARS,
    LLMIntentClassifier,
    RuleFallbackClassifier,
    render_intent_context,
)
from agent.intent.models import Intent, IntentContext, IntentResult
from agent.llm import LLMService
from tests.conftest import fake_structured_message, fake_text_message


async def test_rule_fallback_hits_keyword() -> None:
    """含工具关键词 → TOOL_USE（确定性兜底的主命中路径）。"""
    classifier = RuleFallbackClassifier()
    result = await classifier.classify([HumanMessage(content="帮我计算 3+5 等于多少")])
    assert result.intent == Intent.TOOL_USE
    assert result.confidence > 0.5


async def test_rule_fallback_defaults_to_chat() -> None:
    """无关键词 → 默认 CHAT（CLAUDE.md：兜底默认 chat）。"""
    classifier = RuleFallbackClassifier()
    result = await classifier.classify([HumanMessage(content="你好，介绍一下你自己")])
    assert result.intent == Intent.CHAT


async def test_rule_fallback_never_produces_plan() -> None:
    """v4.0 保守兜底：规则分类器永不产生 PLAN（复合任务特征不可靠，
    判成 TOOL_USE 走 ReAct 仍可用；判成 PLAN 若规划失败反而多一次开销）。"""
    classifier = RuleFallbackClassifier()
    result = await classifier.classify(
        [HumanMessage(content="把毕设资料找出来、总结成报告、翻译成英文")]
    )
    assert result.intent != Intent.PLAN


async def test_rule_fallback_custom_keywords() -> None:
    """自定义关键词集：命中新词 → TOOL_USE，未命中默认词 → CHAT。"""
    classifier = RuleFallbackClassifier(tool_keywords=["股票"])
    hit = await classifier.classify([HumanMessage(content="今天股票行情怎么样")])
    assert hit.intent == Intent.TOOL_USE
    miss = await classifier.classify([HumanMessage(content="帮我计算一下")])
    assert miss.intent == Intent.CHAT


async def test_llm_classifier_happy_path(
    make_llm_service: Callable[[list[AIMessage]], LLMService],
) -> None:
    """离线 fake 返回合法 IntentResult → 原样返回（结构化输出链路真实走通）。"""
    expected = IntentResult(intent=Intent.CHAT, confidence=0.9, reason="test")
    llm = make_llm_service([fake_structured_message(expected)])
    classifier = LLMIntentClassifier(llm)
    result = await classifier.classify([HumanMessage(content="你好")])
    assert result == expected


async def test_llm_classifier_can_produce_plan(
    make_llm_service: Callable[[list[AIMessage]], LLMService],
) -> None:
    """v4.0：LLM 分类器可产出 PLAN（复合任务 few-shot 已加入；结构化输出链路验证）。"""
    expected = IntentResult(intent=Intent.PLAN, confidence=0.9, reason="复合任务")
    llm = make_llm_service([fake_structured_message(expected)])
    classifier = LLMIntentClassifier(llm)
    result = await classifier.classify(
        [HumanMessage(content="把毕设资料找出来、总结成报告、翻译成英文")]
    )
    assert result.intent == Intent.PLAN


async def test_llm_classifier_low_confidence_passthrough(
    make_llm_service: Callable[[list[AIMessage]], LLMService],
) -> None:
    """v4.0 T2：分类器对**真模糊**请求（指代不明）产出低置信度 IntentResult →
    原样透传，供路由层触发澄清追问。

    WHY 用指代不明而非「缺参数」举例：缺参数属工具侧问题（触发源②），
    分类期不应判低置信（见 `test_build_prompt_declares_confidence_semantics`）。
    """
    expected = IntentResult(intent=Intent.CHAT, confidence=0.2, reason="指代不明")
    llm = make_llm_service([fake_structured_message(expected)])
    classifier = LLMIntentClassifier(llm)
    result = await classifier.classify([HumanMessage(content="就是那个，你懂的")])
    assert result == expected
    assert result.confidence < 0.5


def test_build_prompt_declares_confidence_semantics() -> None:
    """分类 prompt 必须区分「意图不可辨」与「信息不完整」。

    WHY：confidence 语义被重载（把「缺参数」也算成模糊）会造成两类后果——
    明确的功能请求（「帮我绑定飞书」）在工具执行前被抢跑成猜测式反问，且参数缺失
    （「帮我算一下」）永远走不到工具、拿不到接地追问。故此处同时断言：
    新语义在、真模糊锚点在、旧的低置信缺参数示范已移除。
    """
    prompt = LLMIntentClassifier._build_prompt([HumanMessage(content="帮我绑定飞书")])
    assert "confidence" in prompt
    assert "低置信度" in prompt
    # 新语义：缺参数不算模糊 + 对上文的回应算明确（续跑判定依据）。
    assert "缺参数不算模糊" in prompt
    assert "回答或确认你在上文中提出的问题" in prompt
    # 真模糊锚点必须保留（澄清触发源① 的保险：该问的仍要问）。
    assert "就是那个，你懂的" in prompt
    assert "（置信度 0.2）" in prompt
    assert "（置信度 0.35）" in prompt
    # 旧语义锚点：「缺参数 → 低置信」的错误示范必须已移除。
    assert "（置信度 0.3）" not in prompt


def test_build_prompt_includes_recent_context() -> None:
    """分类 prompt 必须带最近若干轮（含助理消息与工具结果），本轮输入单独呈现。"""
    messages = [
        HumanMessage(content="帮我绑定飞书"),
        AIMessage(content="已生成授权链接，完成授权后告诉我「我已完成授权」。"),
        HumanMessage(content="我已完成授权"),
    ]
    prompt = LLMIntentClassifier._build_prompt(messages)
    assert "助理：已生成授权链接" in prompt
    assert "用户：帮我绑定飞书" in prompt
    assert "用户最新消息：我已完成授权" in prompt
    # 本轮输入不作为上下文行重复出现（避免同一句出现两次干扰判定）。
    assert "\n用户：我已完成授权" not in prompt
    assert prompt.rstrip().endswith("用户最新消息：我已完成授权")


def test_build_prompt_injects_summary_and_keyfacts() -> None:
    """会话摘要/关键信息（IntentContext）必须进 prompt，且上下文块声明不可信。"""
    prompt = LLMIntentClassifier._build_prompt(
        [HumanMessage(content="你好")],
        IntentContext(summary="用户在办理飞书授权", keyfacts=["偏好中文"]),
    )
    assert "会话摘要：用户在办理飞书授权" in prompt
    assert "偏好中文" in prompt
    assert "不可信" in prompt


def test_build_prompt_context_includes_tool_result_name() -> None:
    """工具结果消息带工具名进上下文（判断「上一步做了什么工具」）。"""
    messages = [
        HumanMessage(content="帮我查日程"),
        ToolMessage(content="[工具结果已消费]", tool_call_id="c1", name="lark_calendar_get_agenda"),
        HumanMessage(content="再查一遍"),
    ]
    prompt = LLMIntentClassifier._build_prompt(messages)
    assert "工具结果(lark_calendar_get_agenda)" in prompt


def test_render_intent_context_empty_is_new_topic() -> None:
    """无上文（仅本轮输入）→ 显式「新话题」提示，而非空块。"""
    assert render_intent_context([HumanMessage(content="你好")]) == _CONTEXT_EMPTY_HINT


def test_render_intent_context_window_and_truncation() -> None:
    """上下文三重上限：条数窗口 / 单条截断 / 总长上限，且保留最新。"""
    messages = [HumanMessage(content=f"旧消息{i}") for i in range(10)]
    messages.append(AIMessage(content="A" * 400))
    messages.append(HumanMessage(content="本轮输入"))
    rendered = render_intent_context(messages)

    assert "旧消息0" not in rendered  # 超出窗口的旧消息被丢弃
    assert "A" * 400 not in rendered  # 单条超长被截断
    assert "…" in rendered
    assert "本轮输入" not in rendered  # 本轮输入不进上下文块
    assert len(rendered) <= _CONTEXT_MAX_TOTAL_CHARS
    assert rendered.count("\n") + 1 <= _CONTEXT_MAX_MESSAGES


async def test_rule_fallback_ignores_context() -> None:
    """规则兜底保持确定性：context 不参与关键词判定（同输入 → 同结果）。"""
    classifier = RuleFallbackClassifier()
    messages = [HumanMessage(content="你好，介绍一下你自己")]
    plain = await classifier.classify(messages)
    with_context = await classifier.classify(
        messages, IntentContext(summary="用户在绑定飞书", keyfacts=["偏好中文"])
    )
    assert with_context == plain


async def test_llm_classifier_falls_back_on_invalid_output(
    make_llm_service: Callable[[list[AIMessage]], LLMService],
) -> None:
    """LLM 输出非法（无工具调用结构）→ 归一化为 LLMError → 规则兜底。"""
    llm = make_llm_service([fake_text_message("这不是结构化输出")])
    classifier = LLMIntentClassifier(llm)
    result = await classifier.classify([HumanMessage(content="你好")])
    # 兜底后无关键词命中 → 默认 chat
    assert result.intent == Intent.CHAT


async def test_intent_result_confidence_bounds() -> None:
    """置信度越界 → Pydantic 校验失败（结构化输出的第一道护栏）。"""
    with pytest.raises(ValidationError):
        IntentResult(intent=Intent.CHAT, confidence=1.5, reason="越界")
