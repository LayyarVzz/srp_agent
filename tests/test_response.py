"""agent/response —— 统一输出模型单测。"""

from __future__ import annotations

from agent.response.models import FINISHED_REASON_COMPLETED, AgentResponse
from agent.response.status import Status, StatusEvent
from agent.share.models import Citation


def test_status_event_defaults() -> None:
    """StatusEvent 可选字段默认为 None（SSE 可省略）。"""
    event = StatusEvent(status=Status.THINKING)
    assert event.tool_name is None
    assert event.message is None


def test_agent_response_defaults() -> None:
    """AgentResponse 空列表默认 + finished_reason 默认 completed。"""
    resp = AgentResponse(session_id="s1", reply="hi")
    assert resp.citations == []
    assert resp.status_trace == []
    assert resp.tool_trace == []
    assert resp.finished_reason == FINISHED_REASON_COMPLETED


def test_agent_response_full_fields() -> None:
    """AgentResponse 完整字段组装（状态轨迹 / 工具轨迹 / 引用）。"""
    resp = AgentResponse(
        session_id="s1",
        reply="答案",
        citations=[Citation(source_id="src-1", source_title="文档", snippet="片段")],
        status_trace=[StatusEvent(status=Status.SPEAKING)],
        finished_reason="fallback",
    )
    assert resp.citations[0].source_id == "src-1"
    assert resp.status_trace[0].status == Status.SPEAKING


def test_citation_score_optional() -> None:
    """Citation.score 可选：缺省 None，显式赋值可被存储。"""
    citation = Citation(source_id="x", source_title="t", snippet="s")
    assert citation.score is None
    citation_with_score = Citation(source_id="x", source_title="t", snippet="s", score=0.9)
    assert citation_with_score.score == 0.9
