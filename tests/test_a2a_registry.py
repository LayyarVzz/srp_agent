"""agent/a2a/registry.py 单测：任务注册表状态机与取消语义。"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from agent.a2a.models import A2ATaskState
from agent.a2a.protocol import (
    A2A_ERROR_TASK_NOT_CANCELABLE,
    A2A_ERROR_TASK_NOT_FOUND,
    A2AProtocolError,
)
from agent.a2a.registry import A2ATaskRegistry
from agent.response.models import AgentResponse


def _response(session_id: str = "t1", reply: str = "答案") -> AgentResponse:
    return AgentResponse(session_id=session_id, reply=reply)


async def test_register_and_get() -> None:
    """登记即 submitted 态；id 即 session_id（task↔session 一一对应）。"""
    registry = A2ATaskRegistry()
    task = registry.register("t1")
    assert task.state is A2ATaskState.SUBMITTED
    assert registry.get("t1") is task


def test_get_unknown_raises_task_not_found() -> None:
    """未命中 → a2a.task_not_found（task/get / cancel 共用）。"""
    with pytest.raises(A2AProtocolError) as exc_info:
        A2ATaskRegistry().get("nope")
    assert exc_info.value.a2a_code == A2A_ERROR_TASK_NOT_FOUND


async def test_working_then_completed() -> None:
    """submitted → working → completed（AgentResponse 落终态 + finished_at）。"""
    registry = A2ATaskRegistry()
    registry.register("t1")
    registry.mark_working("t1")
    task = registry.mark_completed("t1", _response())
    assert task.state is A2ATaskState.COMPLETED
    assert task.finished_at is not None
    assert task.message is not None
    assert task.message.text == "答案"


async def test_mark_failed_records_error() -> None:
    """图运行中断 → failed，error 承载可读原因。"""
    registry = A2ATaskRegistry()
    registry.register("t1")
    registry.mark_failed("t1", "图运行异常")
    task = registry.get("t1")
    assert task.state is A2ATaskState.FAILED
    assert task.error == "图运行异常"


async def test_mark_completed_after_canceled_keeps_canceled() -> None:
    """取消与完成竞态：终态不可逆，先到者保留（canceled 不被 completed 覆盖）。"""
    registry = A2ATaskRegistry()
    registry.register("t1")
    registry.mark_canceled("t1")
    task = registry.mark_completed("t1", _response())
    assert task.state is A2ATaskState.CANCELED


async def test_record_preserves_created_at() -> None:
    """终态快照覆盖登记保留首次登记的 created_at（任务时间线不回跳）。"""
    registry = A2ATaskRegistry()
    created = registry.register("t1").created_at
    from agent.a2a.mapper import task_from_response

    registry.record(task_from_response("t1", _response()))
    assert registry.get("t1").created_at == created
    assert registry.get("t1").state is A2ATaskState.COMPLETED


async def test_cancel_working_task() -> None:
    """取消进行中任务：状态置 canceled，执行句柄被真实取消。"""
    registry = A2ATaskRegistry()
    registry.register("t1")
    registry.mark_working("t1")

    async def _forever() -> None:
        await asyncio.Event().wait()

    handle = asyncio.get_running_loop().create_task(_forever())
    registry.attach_handle("t1", handle)
    await asyncio.sleep(0)  # 让句柄真正开始执行
    task = registry.cancel("t1")
    assert task.state is A2ATaskState.CANCELED
    with contextlib.suppress(asyncio.CancelledError):
        await handle
    assert handle.cancelled()


async def test_cancel_terminal_raises_not_cancelable() -> None:
    """已终态任务不可取消 → a2a.task_not_cancelable。"""
    registry = A2ATaskRegistry()
    registry.register("t1")
    registry.mark_completed("t1", _response())
    with pytest.raises(A2AProtocolError) as exc_info:
        registry.cancel("t1")
    assert exc_info.value.a2a_code == A2A_ERROR_TASK_NOT_CANCELABLE


async def test_cancel_unknown_raises_task_not_found() -> None:
    with pytest.raises(A2AProtocolError) as exc_info:
        A2ATaskRegistry().cancel("nope")
    assert exc_info.value.a2a_code == A2A_ERROR_TASK_NOT_FOUND
