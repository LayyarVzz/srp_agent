"""查询理解数据模型（v6.0 T2，dev-version6.0.md §3.2）。

一次结构化 LLM 调用的产物：主改写 / 子查询 / 同义 / HyDE 假设文档。
四项字段各自有明确下游消费方（§2.4 D4），**无消费方的字段不产出** ——
所以这里没有「候选查询列表」这类无人消费的容器。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field


class QueryVariant(StrEnum):
    """变体来源标记（供日志与融合归因；禁止散落字符串字面量）。"""

    MAIN = "main"  # 主改写
    SUB = "sub"  # 子问题
    SYNONYM = "synonym"  # 同义/近义表述


@dataclass(frozen=True)
class RankedQuery:
    """带来源标记的检索查询（变体池元素）。"""

    query: str
    origin: QueryVariant


class QueryUnderstanding(BaseModel):
    """一次查询理解结果（结构化 LLM 输出；空字段表示未产出）。

    `retrieval_needed=False` 的准确语义是「**本轮不需要主题检索**」：
    跳过 fact/episode 长期记忆召回与 RAG 检索实参改写；
    preference 身份预加载不受其影响（它不是按主题检索，且既有测试与语义都要求每轮进行）。
    """

    retrieval_needed: bool = True  # 是否值得做主题检索（False → 跳过召回的 fact/episode 部分）
    # 主检索查询（补全指代 + 规范表达）。空串 = 模型未产出有效改写：
    # 结构化 schema 不能把本字段设为 required（模型可能只给出 retrieval_needed=False），
    # 下游 `ranked_queries` 与 `rewriter` 都以「空串」承载「无产出」语义。
    main_query: str = ""
    sub_queries: list[str] = Field(default_factory=list)  # 子问题（复合任务；≤ sub_query_max）
    synonyms: list[str] = Field(default_factory=list)  # 同义/近义表述（扩大召回；≤ synonym_max）
    hypothetical_answer: str | None = None  # HyDE 假设文档（仅门控通过时产出）
    reason: str = ""  # 改写依据（审计用；不进 prompt、不落库）

    def ranked_queries(self, *, max_variants: int) -> list[RankedQuery]:
        """去重后的变体池（保序）：主改写 → 子查询 → 同义，按 `max_variants` 截断。

        WHY 保序且带来源：主改写最可信（总是首选检索），子查询/同义仅在召回不足时补检，
        下游据此判断「哪个是首轮查询、哪些是补检变体」；来源同时进日志便于归因。

        WHY 主改写为空时返回空池（而不是回退子查询/同义）：子查询与同义是**相对主改写**
        的补充，没有主改写就没有「首轮查询」；此时让调用方回退用户原话（v5.1 行为）比
        拿一个只该用于补检的变体去替换用户查询更安全。
        """
        if not self.main_query.strip():
            return []
        pool: list[RankedQuery] = []
        seen: set[str] = set()
        candidates = [
            (self.main_query, QueryVariant.MAIN),
            *((q, QueryVariant.SUB) for q in self.sub_queries),
            *((q, QueryVariant.SYNONYM) for q in self.synonyms),
        ]
        for text, origin in candidates:
            norm = text.strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            pool.append(RankedQuery(query=norm, origin=origin))
        return pool[:max_variants]

    def retrieval_queries(self, *, max_variants: int) -> list[str]:
        """变体池的纯文本视图（记忆召回等多路检索直接消费）。"""
        return [item.query for item in self.ranked_queries(max_variants=max_variants)]


class QueryUnderstandingResult(QueryUnderstanding):
    """结构化 LLM 输出的**扁平**载体（本模型即结构化调用 schema）。

    WHY 不用「外层容器 + 可选内层对象」的嵌套形态（旧实现：`understanding: X | None`）：
    嵌套 `anyOf: [object, null]` 会让部分兼容端点（实测阿里云 MaaS + deepseek-v4.1-flash）
    **失控生成**——`tool_calls` 照样产出，但在 tool call 之前先写一大段自由文本，一路跑到
    端点默认 completion 上限 8192，单次请求 40~60s，直接撞 `request_timeout` 并触发重试，
    表现为「一个节点卡 120~180s、最终还回退成原始输入」。
    实测对照：嵌套形态 6/6 次跑满 8192（40~60s），扁平形态 0/6 次（2~4s，约 200 token）。
    故 schema 保持**对象根 + 全原始类型字段**，「无产出」由 `rewriter` 以
    「实例为空 / main_query 为空串」判定。

    WHY 继承而非重复声明字段：结构化 schema 与领域模型必须**同源**，否则改一处漏一处。
    本类不额外声明字段（`produced` 这类客户端信号会污染 schema，故不引入）。
    """
