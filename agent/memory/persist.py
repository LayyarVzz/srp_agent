"""长期记忆带外保存。

非阻塞：长期记忆写入不得阻塞回答下发。
回答经 format_response 下发（图 END）后，入口层调用 `submit_memory_save`
以 fire-and-forget 后台任务执行「抽取 + 保存」；失败仅记日志、不影响主流程。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.messages import BaseMessage

from agent.memory.adapter import KNOWN_KINDS, MemoryStore
from agent.memory.extractor import MemoryExtractor
from agent.memory.models import MemoryItem

logger = logging.getLogger(__name__)

# 来源常量：会话对话抽取的记忆（要求 provenance 字段，禁止散落字面量）。
PROVENANCE_CONVERSATION = "conversation"

# 强引用集：asyncio 后台任务不持引用会被 GC 提前取消，保存引用防止 pending 任务被回收。
_background_tasks: set[asyncio.Task[None]] = set()


async def save_conversation_memory(
    messages: Sequence[BaseMessage],
    *,
    session_id: str,
    user_id: str,
    extractor: MemoryExtractor,
    store: MemoryStore,
) -> None:
    """抽取本轮值得记住的事实并逐条保存；内部吞掉一切异常，绝不抛出（尽力而为）。"""
    extractions = await extractor.extract(messages)  # extract 契约：永不抛
    for e in extractions:
        if e.kind not in KNOWN_KINDS:
            # 模型漂移信号：不丢内容、不 re-label，仅告警；召回端归入 other 组仍可达。
            logger.warning("抽取到未知记忆类型 kind=%s（召回时归入 other 组）", e.kind)
        item = MemoryItem(
            id=uuid4().hex,
            kind=e.kind,
            content=e.content,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            provenance=PROVENANCE_CONVERSATION,
            importance=e.importance,
        )
        try:
            await store.save(item)
            logger.debug(f"记忆带外保存成功（{item.content}）")
        except Exception as exc:
            logger.warning("记忆带外保存失败（kind=%s）：%s", e.kind, exc)


def submit_memory_save(
    messages: Sequence[BaseMessage],
    *,
    session_id: str,
    user_id: str,
    extractor: MemoryExtractor,
    store: MemoryStore,
) -> None:
    """fire-and-forget 触发带外保存；回答下发后调用一次，同步返回、不阻塞。"""
    if not messages:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("当前无事件循环，跳过记忆带外保存")
        return
    task = asyncio.create_task(
        _background_persist(
            messages, session_id=session_id, user_id=user_id, extractor=extractor, store=store
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_discard_task)


async def _background_persist(
    messages: Sequence[BaseMessage],
    *,
    session_id: str,
    user_id: str,
    extractor: MemoryExtractor,
    store: MemoryStore,
) -> None:
    """后台任务体：二次兜底，保证任务不带未处理异常退出。

    WHY 防告警：仅 `add_done_callback(discard)` 移除引用不构成「检索异常」；
    任务体自身永不抛出 + 回调显式 `task.exception()` 才能杜绝
    「Task exception was never retrieved」告警
    """
    try:
        await save_conversation_memory(
            messages, session_id=session_id, user_id=user_id, extractor=extractor, store=store
        )
    except Exception as exc:
        logger.warning("记忆带外保存任务异常：%s", exc)


def _discard_task(task: asyncio.Task[None]) -> None:
    """任务结束回调：从强引用集移除；显式消费可能残留的异常，杜绝 loop 关闭告警。"""
    _background_tasks.discard(task)
    if not task.cancelled():
        task.exception()


async def wait_pending_saves() -> None:
    """等待当前全部后台保存完成。"""
    tasks = tuple(_background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
