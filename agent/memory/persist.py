"""长期记忆带外保存。

非阻塞：长期记忆写入不得阻塞回答下发。
回答经 format_response 下发（图 END）后，入口层调用 `submit_memory_save`
以 fire-and-forget 后台任务执行「抽取 + 保存」；失败仅记日志、不影响主流程。

v6.0（T1）：抽取结果先过**值得性判定**（`should_keep`，单点决策）——
显式拒收常识/可推导/助手产出/寒暄（常识不必记），再走既有去重保存路径。
判定失败一律保守保留（宁漏不误丢，dev-version6.0.md §0.2）。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.messages import BaseMessage

from agent.core.config import DedupConfig, MemoryWorthConfig
from agent.memory.adapter import KNOWN_KINDS, LONG_TERM_NAMESPACE, MemoryStore
from agent.memory.extractor import MemoryExtractor
from agent.memory.judge import MemoryRelationJudge
from agent.memory.models import (
    RELATION_EXACT,
    RELATION_OVERLAP,
    WORTH_COMMONSENSE,
    WORTH_DERIVABLE,
    WORTH_SELF_GENERATED,
    WORTH_SMALL_TALK,
    MemoryExtraction,
    MemoryItem,
    SaveOutcome,
    normalize_content_hash,
)

logger = logging.getLogger(__name__)

# 来源常量：会话对话抽取的记忆（要求 provenance 字段，禁止散落字面量）。
PROVENANCE_CONVERSATION = "conversation"

# 拒收类别集合：命中者**且**分数低于阈值时丢弃（见 should_keep）。
# 集合只含「可判定」的硬负面类别 —— 不把「重要性低」这类连续量当落地门。
REJECT_CATEGORIES = frozenset(
    {WORTH_COMMONSENSE, WORTH_DERIVABLE, WORTH_SELF_GENERATED, WORTH_SMALL_TALK}
)

# 强引用集：asyncio 后台任务不持引用会被 GC 提前取消，保存引用防止 pending 任务被回收。
_background_tasks: set[asyncio.Task[None]] = set()


def should_keep(extraction: MemoryExtraction, *, cfg: MemoryWorthConfig) -> bool:
    """值得性判定：拒收类别 + 分数阈值（**唯一决策点**，判据禁止散落别处）。

    判定顺序（先类别后分数，可解释、可审计）：
    ① `enabled=False` → 恒 True（v5.1 零回归）；
    ② 内容为空 / 纯空白 → False（脏数据防御，与类别无关）；
    ③ 拒收类别**且** `worth_score < min_worth_score` → False（值得丢弃）；
    ④ 拒收类别但分数仍高（模型自相矛盾）→ **保守保留**：那是提示词漂移信号，
       宁可多记一条，也不因一次矛盾判定丢掉用户的事实（宁漏不误丢）；
    ⑤ 其余 → True。
    """
    if not cfg.enabled:
        return True
    if not extraction.content.strip():
        return False
    if extraction.category in REJECT_CATEGORIES:
        return extraction.worth_score >= cfg.min_worth_score
    return True


def _worth_rank(extractions: Sequence[MemoryExtraction]) -> list[MemoryExtraction]:
    """按 `worth_score` 降序排（稳定排序：同分保持模型输出顺序，跨运行可复现）。"""
    return sorted(extractions, key=lambda e: e.worth_score, reverse=True)


def _select_keepable(
    extractions: Sequence[MemoryExtraction], *, cfg: MemoryWorthConfig
) -> list[MemoryExtraction]:
    """逐条过值得性判定，再按每轮上限截断；**顺带打印两种丢弃原因**（不同的原因必须可区分）。

    WHY 先判后截：上限截断应以「值得记住的程度」为序，而不是模型输出顺序 ——
    否则一次话多就可能把最值得记的那条挤掉。

    WHY 日志在此处逐条打（而非调用方汇总后反推）：丢弃有两种**语义完全不同**的原因
    ——「不值得记」（类别+分数判定）与「超上限」（值得记但名额不够）。若只按
    「取过 should_keep 的差集」反推，会把被上限截断的条目误报成「不值得记」，
    直接误导复盘（看起来像模型判错了，其实是名额不够）。
    """
    worth_dropped: list[MemoryExtraction] = []
    kept: list[MemoryExtraction] = []
    for e in extractions:
        if should_keep(e, cfg=cfg):
            kept.append(e)
        else:
            worth_dropped.append(e)
    ranked = _worth_rank(kept)
    overflow = ranked[cfg.memories_max_per_turn :]
    if overflow:
        logger.info(
            "记忆条数超上限（上限 %d，候选 %d）→ 丢弃低价值 %d 条：%s",
            cfg.memories_max_per_turn,
            len(ranked),
            len(overflow),
            "；".join(f"{e.content}（{e.worth_score:.2f}）" for e in overflow),
        )
    for e in _worth_rank(worth_dropped):
        # 值得性丢弃：类别 + 分数 + 模型自述理由全量落日志（可审计、可复盘误丢）。
        logger.info(
            "记忆丢弃（不值得）category=%s score=%.2f reason=%s（%s）",
            e.category,
            e.worth_score,
            e.worth_reason or "无",
            e.content,
        )
    if worth_dropped or overflow:
        logger.info(
            "记忆值得性判定：候选 %d 条 → 落库 %d 条（不值得 %d、超上限 %d，阈值 %.2f）",
            len(extractions),
            len(ranked) - len(overflow),
            len(worth_dropped),
            len(overflow),
            cfg.min_worth_score,
        )
    return ranked[: cfg.memories_max_per_turn]


async def save_conversation_memory(
    messages: Sequence[BaseMessage],
    *,
    session_id: str,
    user_id: str,
    extractor: MemoryExtractor,
    store: MemoryStore,
    dedup: DedupConfig | None = None,
    judge: MemoryRelationJudge | None = None,
    worth: MemoryWorthConfig | None = None,
) -> None:
    """抽取本轮值得记住的事实，过值得性判定后逐条保存；内部吞掉一切异常（尽力而为）。

    `worth` 缺省用 `MemoryWorthConfig()` 默认值（`enabled=True`：判定开启且保守）；
    关闭判定（`worth.enabled=False`）即退回 v5.1「全部落库」行为。
    `dedup` 为 None（或 `enabled=False`）时退回逐条 `save` 原行为；
    启用时走 `_save_deduped`（L1 content-hash + 语义候选；`judge` 非空则按事实三分类
    决策，否则退回 `store.upsert` 阈值路径）。
    """
    extractions = await extractor.extract(messages)  # extract 契约：永不抛
    worth_cfg = worth or MemoryWorthConfig()
    # 保存过程对用户可观测（INFO，见 demo 默认级别）：抽取条数 → 值得性判定与上限截断
    # （_select_keepable 内逐条打丢弃原因）→ 逐条保存决策（_save_deduped / judge 内部均 INFO）。
    logger.info(
        "记忆抽取完成：%d 条待判定（session=%s, user=%s）", len(extractions), session_id, user_id
    )
    for e in _select_keepable(extractions, cfg=worth_cfg):
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


def _score_rank_key(pair: tuple[MemoryItem, float | None]) -> tuple[bool, float]:
    """合并目标排序键：有真实分数者优先、分数降序，score=None 兜底条目排最后。

    WHY 显式键：`exacts`/`mergeable` 的构建顺序来自模型 verdict 输出（可能乱序），
    需按语义分数重排，保证 `[0]` 即最相似候选；None 分数（asearch 兜底条目）
    视为「未知相似度」，排在有分者之后。
    """
    score = pair[1]
    return (score is not None, score if score is not None else -1.0)


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
        logger.info("去重 L1 content-hash 精确命中 → 合并进 %s（%s）", exact.id, exact.content)
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
    # 合并目标按分数降序取最高分：exacts/mergeable 的构建顺序来自「模型 verdict 输出」
    # （可能乱序 index），而非 semantic 的 asearch 降序——必须显式重排，
    # 才能保证 [0] 即最相似候选（score=None 兜底条目排最后）。
    exacts.sort(key=_score_rank_key, reverse=True)
    mergeable.sort(key=_score_rank_key, reverse=True)
    if exacts:  # 确定性重复存在 → 合并进最高分 exact（已按语义分数降序）
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
    worth: MemoryWorthConfig | None = None,
) -> None:
    """fire-and-forget 触发带外保存；回答下发后调用一次，同步返回、不阻塞。

    `worth` 由装配层从 `cfg.memory.worth` 传入（框架行为项）；缺省用默认值。
    """
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
            worth=worth,
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
    worth: MemoryWorthConfig | None = None,
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
            worth=worth,
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
