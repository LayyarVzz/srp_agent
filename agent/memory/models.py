"""记忆数据模型。

每条 `MemoryItem` 必须携带 kind / session_id / user_id / timestamp / provenance；
召回结果必须带来源。保存去重的纯函数、关系判定模型与结果模型也在此（D4）。
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agent.share.models import Citation

# —— 合并判定关系常量（判定器输出；禁止散落字符串字面量）——

RELATION_EXACT = "exact_duplicate"  # 完全重复：同事实/同事件，无新信息
RELATION_OVERLAP = "overlap_merge"  # 部分重叠应合并：同事实但补充了新信息
RELATION_NOT_DUPLICATE = "not_duplicate"  # 相关但不重复：不同事件/对象/时空不兼容
# 可合并关系集合：exact / overlap 任一 → 进入合并决策（见 persist._save_deduped）。
MERGEABLE_RELATIONS = frozenset({RELATION_EXACT, RELATION_OVERLAP})


def normalize_content_hash(content: str) -> str:
    """对记忆内容做零误并风险的确定性指纹（SHA-256）。

    归一化只做「去空白/去标点/统一小写」这类格式层归一：任意两条归一化后相同的文本
    一定携带同一条事实（仅格式/大小写差异），因此精确层误并风险为零；
    不做停用词/实体替换等语义层改动——那是 L2 语义查重的职责，避免把
    「意思接近但不同」的文本在精确层就误并。
    CJK 字符 `isalnum()` 为 True 故保留；全角/半角标点与空白全部剔除。
    """
    normalized = "".join(ch for ch in content if ch.isalnum()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MemoryItem(BaseModel):
    """一条长期记忆（事实 / 片段 / 偏好…），必须携带会话与用户作用域。"""

    id: str
    kind: str  # fact / episode / preference / ...
    content: str
    session_id: str
    user_id: str
    timestamp: datetime
    provenance: str  # 来源（会话、工具等）
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    # 归一化内容的 SHA-256 指纹（保存去重 L1 精确层键）；旧数据默认空串，向后兼容。
    content_hash: str = ""


class SaveOutcome(BaseModel):
    """一次带去重保存的结果：新写（inserted）或合并（merged）。"""

    action: Literal["inserted", "merged"]
    item: MemoryItem  # 落库后的条目（merged 时为合并结果）


class MemoryRecallResult(BaseModel):
    """记忆召回结果：记忆条目 + 来源引用。"""

    items: list[MemoryItem] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)


class MemoryExtraction(BaseModel):
    """一次结构化抽取结果（来自 LLM，可独立理解）。

    `kind` 为宽松 str（不约束 Literal）：
    强约束时模型偶发输出超纲值会导致整批校验失败被丢弃（降级保险，见 kind 策略）。
    `importance` 由模型给出，作为 v2.0 非语义召回的确定性排序键。
    """

    kind: str  # fact / episode / preference（召回端未知值归入 other 组）
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryExtractionResult(BaseModel):
    """整批抽取结果；没有值得记住的内容时为空列表。"""

    memories: list[MemoryExtraction] = Field(default_factory=list)


class CandidateVerdict(BaseModel):
    """判定器对单条候选记忆的关系判定。

    `index` 为候选在 prompt 中的编号（1-based），供判定器引用；
    `relation` 三分类（完全重复 / 部分重叠应合并 / 相关但不重复）；
    `reason` 简述事实性判定依据（时间/地点/人物/数量/事件/观点倾向/否定词等），
    仅作审计与日志，不参与决策。
    """

    index: int
    relation: Literal["exact_duplicate", "overlap_merge", "not_duplicate"]
    reason: str = ""


class MergeDecisionResult(BaseModel):
    """判定器对全部候选的整体结构化输出；判定失败/无判定时为空列表。"""

    verdicts: list[CandidateVerdict] = Field(default_factory=list)
