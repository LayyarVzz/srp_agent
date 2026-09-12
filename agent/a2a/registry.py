"""A2A 内存任务注册表：task/get 轮询与 task/cancel 的状态载体。

生命周期与进程一致（无持久化，非目标裁剪项）；task ↔ session 一一对应，
`id` 即 session_id。状态迁移只允许沿状态机前进（submitted → working → 终态），
由注册表集中收口，routes 层不做状态判断。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agent.a2a.mapper import task_from_response
from agent.a2a.models import A2ATask, A2ATaskState, is_terminal
from agent.a2a.protocol import (
    A2A_ERROR_TASK_NOT_CANCELABLE,
    JSONRPC_TASK_NOT_CANCELABLE,
    A2AProtocolError,
    task_not_found,
)
from agent.response.models import AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class _TaskEntry:
    """注册表条目：任务模型 + 执行句柄（message/stream 路径用于取消）。"""

    task: A2ATask
    handle: asyncio.Task[object] | None = field(default=None)


class A2ATaskRegistry:
    """进程内任务注册表（dict 载体，单事件循环访问，无需加锁）。"""

    def __init__(self) -> None:
        self._entries: dict[str, _TaskEntry] = {}

    # —— 查询 ——

    def get(self, task_id: str) -> A2ATask:
        """按 id 取任务快照；未命中 → a2a.task_not_found。"""
        entry = self._entries.get(task_id)
        if entry is None:
            raise task_not_found(task_id)
        return entry.task

    # —— 登记与状态迁移（message/stream 执行路径驱动）——

    def register(self, task_id: str) -> A2ATask:
        """登记新任务（submitted 态）；重复登记为防御性 no-op（返回现值）。"""
        entry = self._entries.get(task_id)
        if entry is not None:
            return entry.task
        entry = _TaskEntry(
            task=A2ATask(id=task_id, session_id=task_id, state=A2ATaskState.SUBMITTED)
        )
        self._entries[task_id] = entry
        return entry.task

    def attach_handle(self, task_id: str, handle: asyncio.Task[object]) -> None:
        """绑定执行句柄（供 task/cancel 取消进行中任务）。"""
        entry = self._entries.get(task_id)
        if entry is not None:
            entry.handle = handle

    def mark_working(self, task_id: str) -> None:
        self._transition(task_id, A2ATaskState.WORKING)

    def record(self, task: A2ATask) -> None:
        """以终态快照覆盖登记（保留首次登记的 created_at；message/send 路径亦可直接登记）。"""
        entry = self._entries.get(task.id)
        if entry is not None:
            if is_terminal(entry.task.state):
                return  # 已被取消等终态覆盖：竞态取先到者
            entry.task = task.model_copy(update={"created_at": entry.task.created_at})
            return
        self._entries[task.id] = _TaskEntry(task=task)

    def mark_completed(self, task_id: str, response: AgentResponse) -> A2ATask:
        """按 AgentResponse 落终态（completed/failed），返回任务快照。"""
        task = task_from_response(task_id, response)
        entry = self._entries.get(task_id)
        if entry is not None and is_terminal(entry.task.state):
            return entry.task  # 已被取消等终态覆盖：不回写
        self.record(task)
        return self.get(task_id)

    def mark_failed(self, task_id: str, message: str) -> None:
        """图运行中断 → failed（error 承载可读原因）。"""
        self._transition(task_id, A2ATaskState.FAILED, error=message)

    def mark_canceled(self, task_id: str) -> None:
        self._transition(task_id, A2ATaskState.CANCELED)

    # —— 取消 ——

    def cancel(self, task_id: str) -> A2ATask:
        """请求取消：终态 → a2a.task_not_cancelable；进行中 → canceled 并中断执行。

        取消经 asyncio 句柄生效（生成器取消语义：chat_stream 的 finally
        带外保存仍会执行）；无句柄的任务（登记后尚未绑定）仅迁移状态。
        """
        entry = self._entries.get(task_id)
        if entry is None:
            raise task_not_found(task_id)
        if is_terminal(entry.task.state):
            raise A2AProtocolError(
                jsonrpc_code=JSONRPC_TASK_NOT_CANCELABLE,
                a2a_code=A2A_ERROR_TASK_NOT_CANCELABLE,
                message=f"任务已终态（{entry.task.state.value}），不可取消: {task_id}",
            )
        if entry.handle is not None and not entry.handle.done():
            entry.handle.cancel()
        self.mark_canceled(task_id)
        return entry.task

    # —— 私有 ——

    def _transition(
        self,
        task_id: str,
        state: A2ATaskState,
        *,
        error: str | None = None,
    ) -> None:
        entry = self._entries.get(task_id)
        if entry is None:
            raise task_not_found(task_id)
        if is_terminal(entry.task.state):
            return  # 终态不可逆（防御：取消与完成的竞态取先到者）
        update: dict[str, object] = {"state": state}
        if state in (A2ATaskState.COMPLETED, A2ATaskState.FAILED, A2ATaskState.CANCELED):
            update["finished_at"] = datetime.now(UTC)
        if error is not None:
            update["error"] = error
        entry.task = entry.task.model_copy(update=update)
        logger.info("A2A 任务状态迁移：%s → %s", task_id, state.value)


__all__ = ["A2ATaskRegistry"]
