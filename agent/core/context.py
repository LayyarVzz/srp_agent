"""短期上下文管理数据模型。

滚动摘要（short_term_summary）与会话关键信息（session_keyfacts）跨节点/跨轮在
AgentState 中传递，属于跨模块边界结构体 → 必须定义为 Pydantic 模型。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionKeyFact(BaseModel):
    """会话关键信息：一条可在会话内逐条淘汰的要点。

    category 三选一 goal（当前目标）/ fact（已确认事实）/ todo（待办）；
    active 表示该项是否仍有效（已达成/矛盾/过期 → False，见 SUMMARY_PROMPT 遗忘规则）。
    `content` 必须脱离上下文可独立理解（第三人称陈述）。
    """

    content: str
    category: Literal["goal", "fact", "todo"] = "fact"
    active: bool = True


class ShortTermContext(BaseModel):
    """summarize_history 节点的一次结构化 LLM 输出：新摘要 + 关键信息列表。"""

    summary: str = ""
    keyfacts: list[SessionKeyFact] = Field(default_factory=list)
