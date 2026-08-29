"""Agent 模块公共数据模型（跨子模块共享）。

WHY 独立成模块：`Citation` 被 tools（``ToolResult.citations``）、memory
（``MemoryRecallResult.sources``）、response（``AgentResponse.citations``）三方引用，
若放任意一个子模块都会与其它子模块形成循环导入。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """来源引用：回答中引用的任何来源必须存在于本次检索结果集内（护栏校验）。"""

    source_id: str
    source_title: str
    snippet: str
    score: float | None = Field(default=None, description="相关性得分（如有）")
    url: str | None = None


class MemoryRecallConfig(BaseModel):
    """语义召回混合重排参数（core/config 与 memory/adapter 共用）。

    两套权重按召回职责区分（权重元组与 `adapter._rerank_signals` 返回的三元组
    一一对应，score / importance / recency）：`content_weights` 是内容召回
    （fact/episode，query 主导）的默认权重；`preference_weights` 是偏好预加载
    （importance 主导，身份先验）的权重——辨析「任务上下文」与「用户先验」
    两类召回。未来加 lexical(BM25) 信号时两处同步扩为四元组。
    """

    # 内容召回默认权重（w_score / w_importance / w_recency）：query 语义主导。
    # 和为 1 不强制：线性加权缩放不改变排序，仅约束每项非负。
    content_weights: tuple[float, float, float] = (0.6, 0.25, 0.15)

    # 偏好预加载权重：importance 主导（身份先验），query 语义只做辅助决胜。
    preference_weights: tuple[float, float, float] = (0.2, 0.6, 0.2)

    # 语义预取放大倍数：asearch limit = top_k * recall_fetch_factor，留出混合重排空间。
    recall_fetch_factor: int = Field(default=4, ge=1)

    # 近因半衰期（天）：recency = 1/(1 + age_days/half_life)，越大衰减越慢。
    recency_half_life_days: float = Field(default=30.0, ge=1.0)

    @model_validator(mode="after")
    def _validate_weights_non_negative(self) -> MemoryRecallConfig:
        """两套权重每项必须非负；元组长度已由 `tuple[float, float, float]`
        类型注解结构性约束为 3。"""
        if any(w < 0 for w in (*self.content_weights, *self.preference_weights)):
            raise ValueError("content_weights / preference_weights 每项必须 >= 0")
        return self
