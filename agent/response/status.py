"""状态枚举与状态事件（驱动数字人动画的流式轨迹）。

WHY 用 StrEnum：成员值即序列化字符串，经 LangGraph msgpack checkpointer 往返稳定，
且可直接写入 JSON/SSE。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class Status(StrEnum):
    """Agent 处理状态，随图节点流转逐条下发（供前端驱动数字人动画）。"""

    LISTENING = "listening"  # 倾听
    THINKING = "thinking"  # 思考
    RETRIEVING = "retrieving"  # 检索
    USING_TOOL = "using_tool"  # 调用工具
    SPEAKING = "speaking"  # 说话
    ERROR = "error"


class StatusEvent(BaseModel):
    """一条状态事件；`status_events` 按真实执行顺序累积成有序轨迹。"""

    status: Status
    tool_name: str | None = None
    message: str | None = None
