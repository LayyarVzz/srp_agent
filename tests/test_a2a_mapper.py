"""agent/a2a/mapper.py 单测：peer 身份 / AgentResponse→Task / 事件→流式帧。"""

from __future__ import annotations

import pytest

from agent.a2a.mapper import (
    PEER_USER_PREFIX,
    peer_user_id,
    progress_frame,
    resolve_peer,
    task_frame,
    task_from_response,
)
from agent.a2a.models import A2APeer, A2ATaskState
from agent.a2a.protocol import (
    A2A_ERROR_INVALID_REQUEST,
    JSONRPC_INVALID_REQUEST,
    A2AProtocolError,
)
from agent.response.models import AgentResponse, AnswerToken
from agent.response.status import Status, StatusEvent
from agent.share.models import Citation
from agent.tools.models import ToolCallRecord

PEERS = {
    "trusted": A2APeer(id="trusted", user_id="svc-user"),
    "anon": A2APeer(id="anon"),
    "off": A2APeer(id="off", enabled=False),
}


def test_resolve_peer_ok() -> None:
    """已登记且启用的 peer 正常解析。"""
    assert resolve_peer(PEERS, "trusted").id == "trusted"


@pytest.mark.parametrize("peer_id", ["unknown", "off"])
def test_resolve_peer_rejects(peer_id: str) -> None:
    """未登记 / 已禁用 peer 一律 a2a.invalid_request（防开放匿名滥用）。"""
    with pytest.raises(A2AProtocolError) as exc_info:
        resolve_peer(PEERS, peer_id)
    assert exc_info.value.a2a_code == A2A_ERROR_INVALID_REQUEST
    assert exc_info.value.jsonrpc_code == JSONRPC_INVALID_REQUEST


def test_peer_user_id_mapping() -> None:
    """身份映射：显式映射优先；匿名 peer 落 `a2a:<id>` 命名空间。"""
    assert peer_user_id(PEERS["trusted"]) == "svc-user"
    assert peer_user_id(PEERS["anon"]) == f"{PEER_USER_PREFIX}anon"


def _response(**overrides: object) -> AgentResponse:
    defaults: dict[str, object] = {
        "session_id": "t1",
        "reply": "这是回答",
        "citations": [Citation(source_id="d1", source_title="文档一", snippet="片段")],
        "finished_reason": "completed",
    }
    defaults.update(overrides)
    return AgentResponse.model_validate(defaults)


def test_task_from_response_completed() -> None:
    """reply → agent text Part；citations 收敛进 metadata 附注。"""
    task = task_from_response("t1", _response())
    assert task.id == "t1"
    assert task.session_id == "t1"
    assert task.state is A2ATaskState.COMPLETED
    assert task.finished_at is not None
    assert task.message is not None
    assert task.message.role == "agent"
    assert task.message.text == "这是回答"
    assert task.finished_reason == "completed"
    citations = task.metadata["citations"]
    assert isinstance(citations, list) and len(citations) == 1
    assert citations[0]["source_id"] == "d1"
    assert citations[0]["source_title"] == "文档一"


def test_task_from_response_error_is_failed() -> None:
    """finished_reason=error → failed 态（reply 为降级文案，不丢弃）。"""
    task = task_from_response("t1", _response(finished_reason="error", reply="服务内部错误"))
    assert task.state is A2ATaskState.FAILED
    assert task.message is not None
    assert task.message.text == "服务内部错误"


def test_task_from_response_clarification_is_completed() -> None:
    """澄清追问在 A2A 子集收敛为 completed（reply 即反问正文，由 peer 回答）。"""
    task = task_from_response("t1", _response(finished_reason="needs_clarification"))
    assert task.state is A2ATaskState.COMPLETED


def test_progress_frame_status_event() -> None:
    """status 事件 → 状态说明文本（message 优先，缺省用状态名）。"""
    frame = progress_frame(
        "req-1",
        task_id="t1",
        session_id="t1",
        event="status",
        payload=StatusEvent(status=Status.THINKING, message="正在思考"),
    )
    assert frame["jsonrpc"] == "2.0"
    assert frame["id"] == "req-1"
    result = frame["result"]
    assert result["kind"] == "status-update"
    assert result["taskId"] == "t1"
    assert result["contextId"] == "t1"
    assert result["status"] == {"state": "working", "message": "正在思考", "delta": None}


def test_progress_frame_status_without_message() -> None:
    """status 事件无 message → 用状态值兜底（调用方始终有进度文本）。"""
    frame = progress_frame(
        "req-1",
        task_id="t1",
        session_id="t1",
        event="status",
        payload=StatusEvent(status=Status.USING_TOOL),
    )
    assert frame["result"]["status"]["message"] == "状态：using_tool"


def test_progress_frame_tool_event() -> None:
    """tool 事件 → 工具调用摘要。"""
    record = ToolCallRecord(tool_name="calculate", arguments={}, status="ok", result=None)
    frame = progress_frame("req-1", task_id="t1", session_id="t1", event="tool", payload=record)
    assert frame["result"]["status"]["message"] == "工具 calculate（ok）"


def test_progress_frame_token_event() -> None:
    """token 事件 → delta 增量（拼接 == 终态 reply 的预览契约）。"""
    frame = progress_frame(
        "req-1", task_id="t1", session_id="t1", event="token", payload=AnswerToken(delta="你")
    )
    assert frame["result"]["status"]["delta"] == "你"
    assert frame["result"]["status"]["message"] is None


def test_task_frame_envelope() -> None:
    """终态帧：result 即 Task 快照（message/send 响应体 / stream 末帧共用）。"""
    task = task_from_response("t1", _response())
    frame = task_frame("req-9", task)
    assert frame["id"] == "req-9"
    assert frame["result"]["id"] == "t1"
    assert frame["result"]["state"] == "completed"
    assert frame["result"]["message"]["role"] == "agent"
