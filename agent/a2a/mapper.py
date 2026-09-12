"""A2A 语义映射（dev-version5.0.md §4.3 设计重心的实现）。

三组纯函数映射：
- peer 身份解析：peer 注册项 → 图入站 user_id（匿名命名空间 `a2a:<peer_id>`）；
- `AgentResponse` → 终态 `A2ATask`：reply → text Part，citations → metadata 附注；
- `chat_stream` 过程事件 → `message/stream` 的 status-update 帧（JSON-RPC 信封）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from agent.a2a.models import (
    A2AMessage,
    A2APeer,
    A2AStatusUpdate,
    A2ATask,
    A2ATaskState,
    A2ATaskStatus,
)
from agent.a2a.protocol import (
    A2A_ERROR_INVALID_REQUEST,
    JSONRPC_INVALID_REQUEST,
    A2AProtocolError,
    result_response,
)
from agent.response.models import (
    FINISHED_REASON_ERROR,
    AgentResponse,
    AnswerToken,
)
from agent.response.status import StatusEvent
from agent.tools.models import ToolCallRecord

# 匿名 peer 的 user_id 命名空间前缀（记忆/会话按该虚拟用户隔离，不混入人类用户数据）。
PEER_USER_PREFIX = "a2a:"


# —— peer 身份 ——


def resolve_peer(peers: dict[str, A2APeer], peer_id: str) -> A2APeer:
    """解析并校验 peer：未登记 / 已禁用 → a2a.invalid_request（防开放匿名滥用）。"""
    peer = peers.get(peer_id)
    if peer is None or not peer.enabled:
        raise A2AProtocolError(
            jsonrpc_code=JSONRPC_INVALID_REQUEST,
            a2a_code=A2A_ERROR_INVALID_REQUEST,
            message=f"未登记或已禁用的 peer: {peer_id}",
        )
    return peer


def peer_user_id(peer: A2APeer) -> str:
    """peer 的图入站身份：显式映射优先，否则匿名命名空间 `a2a:<id>`。"""
    return peer.user_id or f"{PEER_USER_PREFIX}{peer.id}"


# —— AgentResponse → 终态 Task ——


def task_from_response(task_id: str, response: AgentResponse) -> A2ATask:
    """把一轮 A2A 会话的最终回答组装为终态 Task。

    状态映射：`finished_reason == error` → failed（回答为降级文案）；
    其余（completed/partial/tool_limit/fallback/needs_clarification）→ completed
    （A2A 子集无「部分成功/澄清中」态，reply 原文承载，metadata 保留 finished_reason）。
    """
    failed = response.finished_reason == FINISHED_REASON_ERROR
    now = datetime.now(UTC)
    return A2ATask(
        id=task_id,
        session_id=task_id,
        state=A2ATaskState.FAILED if failed else A2ATaskState.COMPLETED,
        created_at=now,
        finished_at=now,
        message=A2AMessage.from_text(role="agent", text=response.reply),
        finished_reason=response.finished_reason,
        metadata={
            "citations": [c.model_dump(mode="json") for c in response.citations],
        },
    )


# —— chat_stream 事件 → 流式帧 ——


def status_update(
    *,
    task_id: str,
    session_id: str,
    state: A2ATaskState = A2ATaskState.WORKING,
    message: str | None = None,
    delta: str | None = None,
) -> A2AStatusUpdate:
    """构造过程进度帧 result（message/stream 的 SSE data 载荷）。"""
    return A2AStatusUpdate(
        taskId=task_id,
        contextId=session_id,
        status=A2ATaskStatus(state=state, message=message, delta=delta),
    )


def progress_frame(
    request_id: str | int | None,
    *,
    task_id: str,
    session_id: str,
    event: str,
    payload: object,
) -> dict[str, object]:
    """把 `chat_stream` 的过程事件翻译为 JSON-RPC 响应帧（SSE data 载荷）。

    - status → 状态说明文本（供调用方展示进度）；
    - tool → 工具调用摘要；
    - token → 回答增量（delta 字段，拼接 == 终态 reply）。
    """
    if event == "status":
        status_event = cast(StatusEvent, payload)
        text = status_event.message or f"状态：{status_event.status.value}"
        update = status_update(task_id=task_id, session_id=session_id, message=text)
    elif event == "tool":
        record = cast(ToolCallRecord, payload)
        update = status_update(
            task_id=task_id,
            session_id=session_id,
            message=f"工具 {record.tool_name}（{record.status}）",
        )
    elif event == "token":
        update = status_update(
            task_id=task_id, session_id=session_id, delta=cast(AnswerToken, payload).delta
        )
    else:  # 防御：未知事件类型不外发
        update = status_update(task_id=task_id, session_id=session_id)
    return result_response(
        request_id, update.model_dump(mode="json")
    ).model_dump(mode="json")


def task_frame(request_id: str | int | None, task: A2ATask) -> dict[str, object]:
    """终态 Task → JSON-RPC 响应帧（message/send 的响应体 / message/stream 的末帧）。"""
    return result_response(request_id, task.model_dump(mode="json")).model_dump(mode="json")
