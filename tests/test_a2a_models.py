"""agent/a2a/models.py 单测：协议模型与 AgentCard 构造。"""

from __future__ import annotations

from agent.a2a.models import (
    A2AMessage,
    A2APart,
    A2ATask,
    A2ATaskState,
    AgentCard,
    build_agent_card,
    is_terminal,
)


def test_task_state_terminal() -> None:
    """终态判定：completed/failed/canceled 为终态，submitted/working 不是。"""
    assert is_terminal(A2ATaskState.COMPLETED)
    assert is_terminal(A2ATaskState.FAILED)
    assert is_terminal(A2ATaskState.CANCELED)
    assert not is_terminal(A2ATaskState.SUBMITTED)
    assert not is_terminal(A2ATaskState.WORKING)


def test_message_from_text_and_text_property() -> None:
    """单段文本消息构造与 text 拼接（入站取文本 / 断言共用）。"""
    msg = A2AMessage.from_text(role="user", text="你好")
    assert msg.role == "user"
    assert msg.parts == [A2APart(text="你好")]
    assert msg.text == "你好"


def test_task_serialization_shape() -> None:
    """Task 序列化：state 输出字符串值，id 与 session_id 同值（一一对应契约）。"""
    task = A2ATask(id="t1", session_id="t1")
    data = task.model_dump(mode="json")
    assert data["id"] == "t1"
    assert data["session_id"] == "t1"
    assert data["state"] == "submitted"
    assert data["finished_at"] is None
    assert data["message"] is None


def test_build_agent_card() -> None:
    """AgentCard：url 由 base_url 拼装 /a2a；能力矩阵如实声明子集。"""
    card = build_agent_card(base_url="http://127.0.0.1:8002/")
    assert card.url == "http://127.0.0.1:8002/a2a"
    assert card.name == "srp-agent"
    assert card.capabilities == {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    }
    assert len(card.skills) == 1  # 单技能声明
    assert card.default_input_modes == ["text/plain"]
    # 卡片可完整 round-trip（发现端点响应契约）
    assert AgentCard.model_validate(card.model_dump(mode="json")) == card
