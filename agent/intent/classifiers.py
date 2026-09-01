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
# 复合任务（一句话含多个子任务/需多步编排）→ PLAN；
# 注意 RuleFallbackClassifier 不产生 PLAN（保守兜底：复合任务特征不可靠，
# 判成 TOOL_USE 走 ReAct 仍可用，判成 PLAN 若规划失败反而多一次开销）。
# 三元组 = (用户消息, 意图, 置信度)：明确请求给高置信（≥0.9）；
# 模糊/信息不足/指代不清的请求必须给低置信（<0.5），供路由层触发澄清追问（T2）。
_FEWSHOT_EXAMPLES: tuple[tuple[str, Intent, float], ...] = (
    ("你好，介绍一下你自己", Intent.CHAT, 0.95),
    ("什么是虚拟数字人？", Intent.CHAT, 0.95),
    ("帮我查一下昨天的新闻", Intent.TOOL_USE, 0.95),
    ("计算 15 乘以 37 等于多少", Intent.TOOL_USE, 0.95),
    ("现在几点了？", Intent.TOOL_USE, 0.95),
    ("把毕设资料找出来、总结成报告、翻译成英文", Intent.PLAN, 0.95),
    ("查一下今天的天气，然后把结果整理成英文摘要发给我", Intent.PLAN, 0.95),
    # 模糊请求 → 低置信度（意图可判但信息不足，须反问而非硬答）。
    ("帮我算一下", Intent.TOOL_USE, 0.3),  # 工具意图明确但缺算式等关键参数
    ("就是那个，你懂的", Intent.CHAT, 0.2),  # 指代不清，无法确定用户要什么
    ("把那个整理一下发给我", Intent.PLAN, 0.35),  # 复合任务但对象/目标不明
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
            f"- 用户：{text} → 意图：{intent.value}（置信度 {confidence}）"
            for text, intent, confidence in _FEWSHOT_EXAMPLES
        )
        allowed = ", ".join(intent.value for intent in Intent)
        return (
            "你是意图分类器。判断用户消息属于哪种意图，只输出对应 JSON 结构（由调用方解析）。\n"
            "confidence 表示你对本次分类的信心（不是任务难度）：\n"
            "- 请求明确、意图确定（含明确的工具/复合任务指令）→ 高置信度（≥0.9）；\n"
            "- 请求模糊、信息不足、指代不清（缺关键对象/参数/方向）→ 低置信度（<0.5），"
            "并在 reason 中说明模糊点——这类请求应被反问澄清，而不是自信地硬答或硬调用工具。\n"
            f"可选意图：{allowed}\n"
            f"示例：\n{examples}\n"
            f"用户消息：{user_text}"
        )
