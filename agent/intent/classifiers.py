"""意图分类器实现：LLM 结构化输出 + 规则确定性兜底。

WHY 双层结构：LLM 分类失败/不可用/非法输出时，必须给出可复现的意图，
禁止让图路由悬空；规则兜底保证确定性与零网络。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from agent.errors import LLMError
from agent.intent.models import Intent, IntentClassifier, IntentResult
from agent.llm import LLMService

logger = logging.getLogger(__name__)

# 兜底启发式关键词：命中任一即判 TOOL_USE（构造时可注入自定义集合）。
DEFAULT_TOOL_KEYWORDS: frozenset[str] = frozenset(
    {"计算", "时间", "搜索", "查询", "天气", "日期", "转换", "换算"}
)

# LLM 意图分类 few-shot 示例（稳定结构化输出的锚点）。
_FEWSHOT_EXAMPLES: tuple[tuple[str, Intent], ...] = (
    ("你好，介绍一下你自己", Intent.CHAT),
    ("什么是虚拟数字人？", Intent.CHAT),
    ("帮我查一下昨天的新闻", Intent.TOOL_USE),
    ("计算 15 乘以 37 等于多少", Intent.TOOL_USE),
    ("现在几点了？", Intent.TOOL_USE),
)


def _latest_human_text(messages: Sequence[BaseMessage]) -> str:
    """取最近一条用户消息的文本（倒序遍历，取首个 human）。"""
    for msg in reversed(messages):
        if msg.type == "human":
            return str(msg.content)
    return ""


class RuleFallbackClassifier:
    """确定性兜底：关键词启发式，无命中默认 CHAT。"""

    def __init__(self, tool_keywords: Sequence[str] | None = None) -> None:
        self._tool_keywords = frozenset(tool_keywords or DEFAULT_TOOL_KEYWORDS)

    async def classify(self, messages: Sequence[BaseMessage]) -> IntentResult:
        text = _latest_human_text(messages)
        if any(keyword in text for keyword in self._tool_keywords):
            return IntentResult(
                intent=Intent.TOOL_USE,
                confidence=0.6,
                reason=f"关键词命中：{text[:50]}",
            )
        return IntentResult(
            intent=Intent.CHAT,
            confidence=0.5,
            reason="关键词未命中，默认 chat",
        )


class LLMIntentClassifier:
    """LLM 结构化意图分类；任何失败自动降级到规则兜底。"""

    def __init__(
        self,
        llm: LLMService,
        *,
        fallback: IntentClassifier | None = None,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or RuleFallbackClassifier()

    async def classify(self, messages: Sequence[BaseMessage]) -> IntentResult:
        try:
            result = await self._llm.ainvoke_structured(IntentResult, self._build_prompt(messages))
        except LLMError as exc:
            # 非法枚举值 / 解析失败 / 网络错误统一在此归一化为 LLMError 并降级。
            logger.warning("意图分类失败（%s），走规则兜底", exc)
            return await self._fallback.classify(messages)
        if result is None:
            # 某些解析路径对无工具调用的输出返回 None 而非抛错：同样视为失败。
            logger.warning("意图分类返回空结果，走规则兜底")
            return await self._fallback.classify(messages)
        return result

    @staticmethod
    def _build_prompt(messages: Sequence[BaseMessage]) -> str:
        user_text = _latest_human_text(messages)
        examples = "\n".join(
            f"- 用户：{text} → 意图：{intent.value}" for text, intent in _FEWSHOT_EXAMPLES
        )
        allowed = ", ".join(intent.value for intent in Intent)
        return (
            "你是意图分类器。判断用户消息属于哪种意图，只输出对应 JSON 结构（由调用方解析）。\n"
            f"可选意图：{allowed}\n"
            f"示例：\n{examples}\n"
            f"用户消息：{user_text}"
        )
