"""多路召回融合（RRF，v6.0 T2，dev-version6.0.md §2.7 D7）。

**纯函数**：不碰 IO、不依赖框架，记忆召回（agent 侧）与 RAG 补检（工具拦截器）
共用同一份实现 —— 两处各自的融合逻辑一旦分家，排序行为就会漂移。

为什么用 RRF 而不是「分数加权平均」：
- 各路分数**不可比**（不同查询的 cosine 分布不同、RAG 与 pgvector 的尺度也不同），
  加权平均需要标定权重且难以解释；
- RRF 只看**名次**，天然对各路分数尺度免疫，k 平滑掉头部噪声，是无标定场景的稳健选择。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# RRF 平滑常数（标准取值 60）：k 越大，名次差异被压得越平，头部越不激进。
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RankedHit[T]:
    """一条带分数与去重键的召回结果（对调用方通用，不绑定具体业务模型）。"""

    key: str  # 去重键（记忆=条目 id；RAG=source.id + 内容前缀）
    item: T  # 原始结果对象
    score: float | None  # 原始相似度（可缺省 —— asearch 兜底条目没有分数）


def dedup_key(*parts: str, content_prefix: int = 64) -> str:
    """构造去重键：来源标识 + **内容前缀**（同一片段被多路召回时归并为一条）。

    调用方通常传 `(source_id, content)`：`source_id` 原样参与（区分同源不同片段要靠内容），
    内容取前 `content_prefix` 字符 —— 不同路返回的同一片段长度可能不同，用前缀即可稳定归并。
    前缀必须同时作用于**所有**输入片段，否则内容整段进键、该参数形同虚设。
    """
    if not parts:
        return ""
    head, *rest = parts
    body = "".join(rest)[:content_prefix]
    return f"{head.strip()}::{body}"


def reciprocal_rank_fusion[T](
    result_lists: Sequence[Sequence[RankedHit[T]]],
    *,
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> list[RankedHit[T]]:
    """多路结果融合：按 `Σ 1/(k + rank)` 降序返回**去重后**的结果。

    - `rank` 是该结果在**其所在路**中的名次（0-based，各路按传入顺序即已排序）；
    - 同一 key 出现在多路 → 分数累加（被多路同时召回的内容更可信，这是 RRF 的核心信号）；
    - 保留**最高原始分数**作为代表分数（不求和/不平均：引用里要展示可解释的相关度，
      融合分是排序产物、不该冒充相似度，与记忆侧 `Citation.score` 口径一致）；
    - tie-break 确定性：融合分相同 → 首次出现的路序 → 路内名次（保证跨运行可复现）。
    """
    fused: dict[str, float] = {}
    best_score: dict[str, float | None] = {}
    rep: dict[str, T] = {}
    first_seen: dict[str, tuple[int, int]] = {}

    for list_index, hits in enumerate(result_lists):
        for rank, hit in enumerate(hits):
            fused[hit.key] = fused.get(hit.key, 0.0) + 1.0 / (k + rank + 1)
            prev = best_score.get(hit.key)
            if prev is None or (hit.score is not None and hit.score > prev):
                best_score[hit.key] = hit.score
            if hit.key not in first_seen:
                first_seen[hit.key] = (list_index, rank)
                rep[hit.key] = hit.item

    ordered = sorted(
        fused,
        key=lambda key: (-fused[key], first_seen[key]),
    )
    if top_n is not None:
        ordered = ordered[:top_n]
    return [RankedHit(key=key, item=rep[key], score=best_score[key]) for key in ordered]
