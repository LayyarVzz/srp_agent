"""Agent 图状态模型与节点名常量。

图编排状态遵循 dev-version1.0.md §3.1，字段标注 reducer 以支持跨节点累积。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from agent.errors import ErrorRecord
from agent.intent.models import Intent, IntentResult
from agent.memory.models import MemoryItem
from agent.response.models import AgentResponse
from agent.response.status import Status, StatusEvent
from agent.share.models import Citation
from agent.tools.models import ToolCallRecord, ToolResult

# —— 图节点名常量（节点名必须集中声明，禁止散落字符串字面量）——
NODE_LOAD_CONTEXT = "load_context"
NODE_TRIM_HISTORY = "trim_history"
NODE_CLASSIFY_INTENT = "classify_intent"
NODE_RECALL_MEMORY = "recall_memory"
NODE_CALL_MODEL = "call_model"
NODE_DISPATCH_TOOL = "dispatch_tool"
NODE_FALLBACK_CHAT = "fallback_chat"
NODE_GENERATE_ANSWER = "generate_answer"
NODE_VALIDATE_OUTPUT = "validate_output"
NODE_FORMAT_RESPONSE = "format_response"


class AgentState(TypedDict, total=False):
    """Agent 图状态"""

    # —— 会话作用域 ——
    session_id: str
    user_id: str

    # —— 会话上下文（checkpointer 累积，经 add_messages 合并）——
    messages: Annotated[list[BaseMessage], add_messages]
    input: str

    # —— 意图 ——
    intent: Intent
    intent_meta: IntentResult

    # —— 状态事件（流式下发）——
    status: Status
    status_events: Annotated[list[StatusEvent], operator.add]

    # —— 工具执行（循环）——
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    tool_result: ToolResult | None
    tool_iterations: int  # 普通覆盖字段：工具循环计数（上限语义）

    # —— 记忆 ——
    memory_context: list[MemoryItem]  # 普通覆盖：load_context 重置、recall_memory 合并

    # —— 输出 ——
    final_answer: str | None
    citations: Annotated[list[Citation], operator.add]
    finished_reason: str | None
    response: AgentResponse | None

    # —— 杂项 ——
    error: ErrorRecord | None
    config: dict
