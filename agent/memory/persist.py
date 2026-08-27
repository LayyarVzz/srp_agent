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

from agent.core.config import DedupConfig
from agent.memory.adapter import KNOWN_KINDS, LONG_TERM_NAMESPACE, MemoryStore
from agent.memory.extractor import MemoryExtractor
from agent.memory.judge import MemoryRelationJudge
from agent.memory.models import (
    RELATION_EXACT,
    RELATION_OVERLAP,
    MemoryItem,
    SaveOutcome,
    normalize_content_hash,
)

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
    dedup: DedupConfig | None = None,
    judge: MemoryRelationJudge | None = None,
) -> None:
    """抽取本轮值得记住的事实并逐条保存；内部吞掉一切异常，绝不抛出（尽力而为）。

    `dedup` 为 None（或 `enabled=False`）时退回逐条 `save` 原行为；
    启用时走 `_save_deduped`（L1 content-hash + 语义候选；`judge` 非空则按事实三分类
    决策，否则退回 `store.upsert` 阈值路径）。两条路径都计算并落 `content_hash`：
    为未来再启用去重留指纹，不改变 recall 行为。
    """
    extractions = await extractor.extract(messages)  # extract 契约：永不抛
    # 保存过程对用户可观测（INFO，见 demo 默认级别）：抽取条数 → 逐条决策
    # （_save_deduped / judge 内部均 INFO 记录分支与三分类）→ 落库结果。
    logger.info(
        "记忆抽取完成：%d 条待保存（session=%s, user=%s）", len(extractions), session_id, user_id
    )
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
            content_hash=normalize_content_hash(e.content),
        )
        try:
            if dedup is not None and dedup.enabled:
                outcome = await _save_deduped(
                    item,
                    store=store,
                    judge=judge,
                    semantic_threshold=dedup.semantic_threshold,
                )
                # 决策明细已在上方 _save_deduped 内按分支 INFO 记录，此处记落库结果。
                logger.info(
                    "长期记忆保存 action=%s kind=%s id=%s（%s）",
                    outcome.action,
                    item.kind,
                    outcome.item.id,
                    item.content,
                )
            else:
                await store.save(item)
                logger.info(
                    "长期记忆保存（直存）kind=%s id=%s（%s）", item.kind, item.id, item.content
                )
        except Exception as exc:
            logger.warning("记忆带外保存失败（kind=%s）：%s", e.kind, exc)


async def _save_deduped(
    item: MemoryItem,
    *,
    store: MemoryStore,
    judge: MemoryRelationJudge | None,
    semantic_threshold: float,
) -> SaveOutcome:
    """带去重保存：L1 精确 →（有判定器）事实三分类 → 未命中新写（D4-2）。

    决策策略（防误并，宁漏并不误并）：
    - L1 content-hash 精确命中 → 直接合并（零 LLM 成本、零误并）；
    - 无判定器 / 判定失败 / 无语义候选 → 退回 `store.upsert` 阈值路径（零回归）；
    - 判定存在 exact_duplicate → 合并进最高分 exact（确定性重复）；
    - 恰一条可合并（exact/overlap）→ 合并进它（overlap 吸收新信息）；
    - 0 或 ≥2 条可合并 → 新写（组合/多事实，不强行并入任意一条防腐蚀既有事实）。
    全程非破坏：不删除既有记忆。
    """
    namespace = (item.user_id, LONG_TERM_NAMESPACE)
    exact, semantic = await store.find_dedup_candidates(item)
    if exact is not None:
        logger.info(
            "去重 L1 content-hash 精确命中 → 合并进 %s（%s）", exact.id, exact.content
        )
        return await store.merge(namespace, exact, item)
    if judge is None or not semantic:
        if judge is None:
            logger.info("未注入判定器 → 退回阈值路径（语义候选 %d 条）", len(semantic))
        else:
            logger.info("无语义候选（embeddings 不可用）→ 退回阈值路径")
        return await store.upsert(item, semantic_threshold=semantic_threshold)
    verdicts = await judge.judge(item, [m for m, _ in semantic])
    if not verdicts:
        logger.info("判定器返回空（失败/未判定）→ 退回阈值路径")
        return await store.upsert(item, semantic_threshold=semantic_threshold)
    # verdict.index 为 1-based 候选序号，对应 semantic 顺序（与判定 prompt 编号一致）。
    by_idx = {i + 1: (m, s) for i, (m, s) in enumerate(semantic)}
    mergeable: list[tuple[MemoryItem, float | None]] = []
    exacts: list[tuple[MemoryItem, float | None]] = []
    for v in verdicts:
        cand = by_idx.get(v.index)
        if cand is None:
            continue
        if v.relation == RELATION_EXACT:
            exacts.append(cand)
            mergeable.append(cand)
        elif v.relation == RELATION_OVERLAP:
            mergeable.append(cand)
    if exacts:  # 确定性重复存在 → 合并进最高分 exact（semantic 已按分数降序）
        logger.info(
            "存在 exact_duplicate → 合并进最高分 exact %s（%s）",
            exacts[0][0].id,
            exacts[0][0].content,
        )
        return await store.merge(namespace, exacts[0][0], item)
    if len(mergeable) == 1:  # 恰一条可合并 → 吸收新信息合并
        logger.info(
            "恰一条可合并（overlap_merge）→ 合并进 %s（%s）吸收新信息",
            mergeable[0][0].id,
            mergeable[0][0].content,
        )
        return await store.merge(namespace, mergeable[0][0], item)
    # 0 或 ≥2 条可合并（组合/多事实）→ 新写
    logger.info("可合并候选 %d 条（组合/多事实）→ 保守新写，不腐蚀既有事实", len(mergeable))
    await store.save(item)
    return SaveOutcome(action="inserted", item=item)


def submit_memory_save(
    messages: Sequence[BaseMessage],
    *,
    session_id: str,
    user_id: str,
    extractor: MemoryExtractor,
    store: MemoryStore,
    dedup: DedupConfig | None = None,
    judge: MemoryRelationJudge | None = None,
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
            messages,
            session_id=session_id,
            user_id=user_id,
            extractor=extractor,
            store=store,
            dedup=dedup,
            judge=judge,
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
    dedup: DedupConfig | None = None,
    judge: MemoryRelationJudge | None = None,
) -> None:
    """后台任务体：二次兜底，保证任务不带未处理异常退出。

    WHY 防告警：仅 `add_done_callback(discard)` 移除引用不构成「检索异常」；
    任务体自身永不抛出 + 回调显式 `task.exception()` 才能杜绝
    「Task exception was never retrieved」告警
    """
    try:
        await save_conversation_memory(
            messages,
            session_id=session_id,
            user_id=user_id,
            extractor=extractor,
            store=store,
            dedup=dedup,
            judge=judge,
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
