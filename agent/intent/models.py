"""意图识别数据模型与分类器契约。

意图分类必须使用结构化输出（`IntentResult`），禁止自由文本返回意图；
分类失败必须有确定性兜底（见 `agent.intent.classifiers`）。
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class Intent(StrEnum):
    """粗粒度意图，作为图路由的第一个决策。"""

    CHAT = "chat"  # 普通交流 / 常识问题：直接回答
    TOOL_USE = "tool_use"  # 需要外部工具
    PLAN = "plan"  # 复合任务：需要显式多步规划
    # 扩展位：GREETING、MEMORY_RECALL、CHAINED_QUERY ...
    # 新增意图 = 枚举加值 + 可选 few-shot 条目，无需改路由边（仅 TOOL_USE/PLAN 被特判）。


class IntentResult(BaseModel):
    """意图分类结果（Pydantic 结构化输出）。

    `confidence` 的语义是**「本轮该做什么」是否可以判断**，不是任务难度、也不是用户
    这句话的信息是否完整：缺个别参数（如「帮我算一下」缺算式）属「可以判断」——工具会
    报参数错误并触发参数缺失澄清（触发源②），若在分类期就判低置信，会在工具执行前
    抢跑成一次无依据的猜测式追问。
    """

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class IntentContext(BaseModel):
    """意图分类可见的会话上下文（短期滚动摘要 + 会话关键信息）。

    WHY 独立模型而非拼接字符串：跨模块边界结构体必须结构化承载（CLAUDE.md）；且
    各实现可自行决定如何使用——LLM 分类器渲染进 prompt，规则兜底忽略（保持确定性）。
    """

    summary: str = ""
    keyfacts: list[str] = Field(default_factory=list)


class IntentClassifier(Protocol):
    """意图分类器契约：输入对话消息序列（+ 可选会话上下文），输出结构化 `IntentResult`。

    `context` 可选：缺省 None 表示调用方未提供会话上下文（老调用方与离线 fake 无需改动）。
    实现仍必须能从 `messages` 中读出最近若干轮的对话上下文——只判最后一句会让
    「对上一轮追问的简短回应」（如「好了」「我已完成授权」）被误判为模糊。
    """

    async def classify(
        self,
        messages: Sequence[BaseMessage],
        context: IntentContext | None = None,
    ) -> IntentResult: ...
