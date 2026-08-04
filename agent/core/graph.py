"""Agent 图编排：节点、条件路由与图装配。

P1 交付 chat 路径完整可用；工具路径节点（decide_tool / dispatch_tool）为占位，
保证图结构与目标设计完全一致，P2 只填函数体、零改连线。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.core.config import AgentFrameworkConfig
from agent.core.state import (
    NODE_CLASSIFY_INTENT,
    NODE_DECIDE_TOOL,
    NODE_DISPATCH_TOOL,
    NODE_FALLBACK_CHAT,
    NODE_FORMAT_RESPONSE,
    NODE_GENERATE_ANSWER,
    NODE_LOAD_CONTEXT,
    NODE_TRIM_HISTORY,
    NODE_VALIDATE_OUTPUT,
    AgentState,
)
from agent.errors import ErrorRecord, LLMError
from agent.intent.classifiers import LLMIntentClassifier, RuleFallbackClassifier
from agent.intent.models import Intent
from agent.llm import LLMService
from agent.response.models import (
    FINISHED_REASON_COMPLETED,
    FINISHED_REASON_ERROR,
    FINISHED_REASON_FALLBACK,
    AgentResponse,
)
from agent.response.status import Status, StatusEvent
from agent.tools.models import (
    TOOL_ERROR_NO_TOOL,
    TOOL_ERROR_NOT_IMPLEMENTED,
    ToolCallRecord,
    ToolError,
    ToolResult,
)
from agent.tools.registry import InMemoryToolRegistry, ToolRegistry

logger = logging.getLogger(__name__)

# 助理人设（P1 固定；后续可按会话配置化）。
SYSTEM_PROMPT = (
    "你是 SRP 智能助理（由 3D 虚拟数字人承载）。回答简洁、友好、准确；"
    "引用任何知识来源时必须给出明确出处。"
)

# 降级话术（P1 固定文本，便于测试与演示；P5 后接入完整护栏与重试）。
_FALLBACK_NO_TOOL_TEXT = (
    "抱歉，我暂时没有可用的工具能力，无法完成这个请求。你可以换个问法，或直接和我聊天。"
)
_FALLBACK_TOOL_PENDING_TEXT = "当前工具能力还在开发中。你可以先直接和我聊天，或者换个问法。"
_FALLBACK_GENERIC_TEXT = "抱歉，我暂时无法回答这个问题，请稍后再试。"

# 兜底专用 system prompt：LLM 用自身知识作答；免责声明由系统统一追加。
# WHY 区分于 SYSTEM_PROMPT：fallback 场景无外部工具/知识库，需明确告知模型
# 使用自身知识、不确定就如实说明，避免对实时/私有信息编造。
_FALLBACK_SYSTEM_PROMPT = (
    "你是智能助理。当前无法调用任何外部工具或知识库，"
    "请用你自身积累的知识尽量准确、简洁地回答用户的问题；"
    "若涉及实时信息、私人数据或需要工具才能确证的内容，请如实说明你无法核实，不要编造或臆测；"
    "若知识不足，也请如实说明。请勿在回答中自行添加免责声明，免责声明由系统统一追加。"
)

# 兜底回答统一追加的免责声明（确定性固定后缀：可靠、可测、TTS 节奏稳定）。
_FALLBACK_DISCLAIMER_TEXT = "\n（以上内容基于 AI 自身知识生成，未经核实，请自行甄别。）"


def _degraded_fallback_text(state: AgentState) -> str:
    """LLM 不可用时的确定性兜底话术（按错误码/意图选择）。

    WHY 复用：fallback_chat（LLM 失败/空回复）降级时按分支选文案，
    避免三分支逻辑内联在节点里。generate_answer 的 generic 分支语义与此一致。
    """
    err = state.get("error")
    if err and err.code == TOOL_ERROR_NO_TOOL:
        return _FALLBACK_NO_TOOL_TEXT
    if state.get("intent") == Intent.TOOL_USE:
        return _FALLBACK_TOOL_PENDING_TEXT
    return _FALLBACK_GENERIC_TEXT


def set_status(
    status: Status,
    *,
    tool_name: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """构造状态增量：更新当前 status 并追加一条 StatusEvent。

    WHY 只返回新增事件：`status_events` 由 operator.add reducer 追加，
    若把已累积列表整表回写会造成重复。
    """
    return {
        "status": status,
        "status_events": [StatusEvent(status=status, tool_name=tool_name, message=message)],
    }


def _new_message_id(prefix: str) -> str:
    """生成带前缀的唯一消息 id。

    WHY 显式赋值：`trim_history` 依赖 `RemoveMessage(id)` 裁剪，LangChain 对新建
    消息不保证自动分配 id，显式 id 让裁剪始终可定位。
    """
    return f"{prefix}-{uuid.uuid4().hex}"


def build_agent_graph(
    llm: LLMService,
    config: AgentFrameworkConfig | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    tool_registry: ToolRegistry | None = None,
) -> CompiledStateGraph:
    """装配并编译 Agent 状态机（§3.4 完整图骨架）。

    WHY 闭包捕获依赖：LangGraph 节点签名固定为 `(state) -> dict`，把 llm/config/
    registry 经闭包注入，避免把运行依赖塞进 AgentState。
    """
    cfg = config or AgentFrameworkConfig.get_default()
    registry = tool_registry or InMemoryToolRegistry()
    intent_classifier = LLMIntentClassifier(llm, fallback=RuleFallbackClassifier())

    # —— 会话与上下文 ——
    async def load_context(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        updates = set_status(Status.THINKING, message="正在加载会话上下文")
        # 会话作用域兜底：优先对齐真实 checkpointer 线程（thread_id == session_id 契约），
        # 二者都缺时给一次性匿名 id——绝不复用固定常量 "default"：若调用方以 session_id
        # 派生 thread_id，固定兜底会让多个匿名会话挤进同一线程，checkpointer 跨会话串线。
        thread_id = (config.get("configurable") or {}).get("thread_id")
        updates["session_id"] = state.get("session_id") or thread_id or f"anon-{uuid.uuid4().hex}"
        updates["user_id"] = state.get("user_id") or "anonymous"
        raw_input = (state.get("input") or "").strip()
        # 输入长度护栏（CLAUDE.md 资源上限）：超长截断而非拒绝，防止超长输入失控。
        if len(raw_input) > cfg.graph.max_input_chars:
            raw_input = raw_input[: cfg.graph.max_input_chars]
            logger.warning("输入超长，已截断到 %d 字符", cfg.graph.max_input_chars)
        if raw_input:
            updates["input"] = raw_input
            updates["messages"] = [HumanMessage(content=raw_input, id=_new_message_id("h"))]
        return updates

    async def trim_history(state: AgentState) -> dict[str, Any]:
        # 轮数近似裁剪（无 tokenizer）：保留最近 N 轮（每轮 user+assistant 两条）。
        messages = state.get("messages") or []
        max_keep = cfg.graph.trim_keep_recent_rounds * 2
        if len(messages) <= max_keep:
            return {}
        # add_messages reducer 下只能用 RemoveMessage 删除，不能整表回写。
        remove_ids = [m.id for m in messages[:-max_keep] if m.id]
        if not remove_ids:
            return {}
        return {"messages": [RemoveMessage(id=mid) for mid in remove_ids]}

    # —— 意图 ——
    async def classify_intent(state: AgentState) -> dict[str, Any]:
        updates = set_status(Status.THINKING, message="正在识别意图")
        result = await intent_classifier.classify(state.get("messages") or [])
        updates["intent"] = result.intent
        updates["intent_meta"] = result
        return updates

    # —— 工具路径（P1 占位，P2 填函数体）——
    async def decide_tool(state: AgentState) -> dict[str, Any]:
        updates = set_status(Status.USING_TOOL, message="正在选择工具")
        specs = registry.list_specs()
        # P1 占位：不做 LLM 选工具，一律记为「未选中」并路由 fallback_chat。
        # P2 在此以 list_specs() 做结构化选工具，写 status="pending" 记录即可复用现有路由。
        if specs:
            logger.warning("P1 未实现工具选择（已注册 %d 个工具）", len(specs))
            updates["error"] = ErrorRecord(
                code=TOOL_ERROR_NOT_IMPLEMENTED, message="工具选择待 P2 实现"
            )
        else:
            updates["error"] = ErrorRecord(code=TOOL_ERROR_NO_TOOL, message="当前没有可用工具")
        updates["tool_calls"] = [
            ToolCallRecord(tool_name="", arguments={}, status="no_tool_matched")
        ]
        return updates

    async def dispatch_tool(state: AgentState) -> dict[str, Any]:
        # P1 占位（图结构可达，但 P1 中 decide_tool 恒走兜底，本节点不会被触发）。
        calls = state.get("tool_calls") or []
        last = calls[-1] if calls else None
        updates = {
            "tool_result": ToolResult(
                tool_name=last.tool_name if last else "",
                ok=False,
                error=ToolError(code=TOOL_ERROR_NOT_IMPLEMENTED, message="工具执行待 P2 实现"),
            ),
            "tool_iterations": (state.get("tool_iterations") or 0) + 1,
        }
        return updates

    # —— 回答生成与降级 ——
    async def fallback_chat(state: AgentState) -> dict[str, Any]:
        # 降级路径也发 SPEAKING，保证前端能感知「即将出话」。
        # 兜底优先用 LLM 自身知识作答（§3.2「道歉/知识回答/澄清提问」），
        # 系统统一追加免责声明；LLM 失败/空回复时回落到确定性话术，不比现状更差。
        updates = set_status(Status.SPEAKING, message="正在生成兜底回答")
        messages = state.get("messages") or []
        try:
            reply = await llm.ainvoke_text(
                [SystemMessage(content=_FALLBACK_SYSTEM_PROMPT), *messages]
            )
            reply = (reply or "").strip()
            if not reply:
                # 空回复兜底：比依赖 validate_output 更早拦截、日志更清晰。
                logger.warning("兜底回答为空，降级为固定话术")
                reply = _degraded_fallback_text(state)
            else:
                # 确定性追加免责声明（需求：需要自行甄别）。
                reply = f"{reply}{_FALLBACK_DISCLAIMER_TEXT}"
        except LLMError as exc:
            logger.warning("兜底回答生成失败（%s），降级为固定话术", exc)
            reply = _degraded_fallback_text(state)
        updates["final_answer"] = reply
        updates["finished_reason"] = FINISHED_REASON_FALLBACK
        updates["messages"] = [AIMessage(content=reply, id=_new_message_id("a"))]
        return updates

    async def generate_answer(state: AgentState) -> dict[str, Any]:
        updates = set_status(Status.SPEAKING, message="正在生成回答")
        messages = state.get("messages") or []
        try:
            reply = (
                await llm.ainvoke_text([SystemMessage(content=SYSTEM_PROMPT), *messages]) or ""
            ).strip()
        except LLMError as exc:
            logger.warning("回答生成失败（%s），降级话术", exc)
            reply = _FALLBACK_GENERIC_TEXT
            updates["finished_reason"] = FINISHED_REASON_ERROR
        if not reply:
            # 与 fallback_chat 一致：生成节点内拦截空回复并落定最终文案。
            # WHY 不能留给 validate_output：它只覆盖 final_answer，无法同步修正
            # 追加进 messages 的空 AIMessage，会导致 checkpoint 历史与最终回复不一致。
            logger.warning("回答为空，降级为固定话术")
            reply = _FALLBACK_GENERIC_TEXT
            updates["finished_reason"] = FINISHED_REASON_ERROR
        updates["final_answer"] = reply
        updates["messages"] = [AIMessage(content=reply, id=_new_message_id("a"))]
        return updates

    # —— 护栏与输出 ——
    async def validate_output(state: AgentState) -> dict[str, Any]:
        # P1 最小护栏：final_answer 非空即可；引用存在性 / 注入扫描属 P5。
        if not (state.get("final_answer") or "").strip():
            return {
                "final_answer": _FALLBACK_GENERIC_TEXT,
                "finished_reason": FINISHED_REASON_ERROR,
            }
        return {}

    async def format_response(state: AgentState) -> dict[str, Any]:
        # 组装 AgentResponse：只读累积列表（status_events / tool_calls），绝不整表回写，
        # 否则 operator.add reducer 会把历史事件重复追加一遍。
        return {
            "response": AgentResponse(
                session_id=state.get("session_id") or "",
                reply=state.get("final_answer") or "",
                citations=state.get("citations") or [],
                status_trace=state.get("status_events") or [],
                tool_trace=state.get("tool_calls") or [],
                finished_reason=state.get("finished_reason") or FINISHED_REASON_COMPLETED,
            )
        }

    # —— 条件路由（§3.3）——
    def route_intent(state: AgentState) -> str:
        # 仅 TOOL_USE 走工具路径；其余意图（含未来新增）自然走回答路径（§4.1 零改边）。
        if state.get("intent") == Intent.TOOL_USE:
            return NODE_DECIDE_TOOL
        return NODE_GENERATE_ANSWER

    def route_tool_choice(state: AgentState) -> str:
        # 已选工具（status="pending"，P2 写入）才派发；否则走兜底。
        calls = state.get("tool_calls") or []
        if calls and calls[-1].status == "pending":
            return NODE_DISPATCH_TOOL
        return NODE_FALLBACK_CHAT

    def route_after_tool(state: AgentState) -> str:
        result = state.get("tool_result")
        if result is None or not result.ok:
            return NODE_FALLBACK_CHAT
        if (state.get("tool_iterations") or 0) >= cfg.graph.max_tool_iterations:
            return NODE_GENERATE_ANSWER
        return NODE_DECIDE_TOOL

    # —— 装配（§3.4）——
    builder = StateGraph(AgentState)
    builder.add_node(NODE_LOAD_CONTEXT, load_context)
    builder.add_node(NODE_TRIM_HISTORY, trim_history)
    builder.add_node(NODE_CLASSIFY_INTENT, classify_intent)
    builder.add_node(NODE_DECIDE_TOOL, decide_tool)
    builder.add_node(NODE_DISPATCH_TOOL, dispatch_tool)
    builder.add_node(NODE_FALLBACK_CHAT, fallback_chat)
    builder.add_node(NODE_GENERATE_ANSWER, generate_answer)
    builder.add_node(NODE_VALIDATE_OUTPUT, validate_output)
    builder.add_node(NODE_FORMAT_RESPONSE, format_response)

    builder.set_entry_point(NODE_LOAD_CONTEXT)
    builder.add_edge(NODE_LOAD_CONTEXT, NODE_TRIM_HISTORY)
    builder.add_edge(NODE_TRIM_HISTORY, NODE_CLASSIFY_INTENT)

    builder.add_conditional_edges(
        NODE_CLASSIFY_INTENT,
        route_intent,
        {NODE_GENERATE_ANSWER: NODE_GENERATE_ANSWER, NODE_DECIDE_TOOL: NODE_DECIDE_TOOL},
    )
    builder.add_conditional_edges(
        NODE_DECIDE_TOOL,
        route_tool_choice,
        {NODE_DISPATCH_TOOL: NODE_DISPATCH_TOOL, NODE_FALLBACK_CHAT: NODE_FALLBACK_CHAT},
    )
    builder.add_conditional_edges(
        NODE_DISPATCH_TOOL,
        route_after_tool,
        {
            NODE_DECIDE_TOOL: NODE_DECIDE_TOOL,
            NODE_GENERATE_ANSWER: NODE_GENERATE_ANSWER,
            NODE_FALLBACK_CHAT: NODE_FALLBACK_CHAT,
        },
    )
    builder.add_edge(NODE_GENERATE_ANSWER, NODE_VALIDATE_OUTPUT)
    builder.add_edge(NODE_FALLBACK_CHAT, NODE_VALIDATE_OUTPUT)
    builder.add_edge(NODE_VALIDATE_OUTPUT, NODE_FORMAT_RESPONSE)
    builder.add_edge(NODE_FORMAT_RESPONSE, END)

    # §3.5：dev 默认 MemorySaver；thread_id == session_id 承载短期上下文。
    return builder.compile(checkpointer=checkpointer or MemorySaver())
