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
    # 扩展位：GREETING、MEMORY_RECALL、CHAINED_QUERY ...
    # 新增意图 = 枚举加值 + 可选 few-shot 条目，无需改路由边（仅 TOOL_USE 被特判）。


class IntentResult(BaseModel):
    """意图分类结果（Pydantic 结构化输出）。"""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class IntentClassifier(Protocol):
    """意图分类器契约：输入对话消息序列，输出结构化 `IntentResult`。"""

    async def classify(self, messages: Sequence[BaseMessage]) -> IntentResult: ...
