"""指标路由：`GET /metrics`（进程内轻量指标，plan §5.4）。

WHY 不在 `/api/v1` 下：这是**运维/答辩视图**，不是产品接口 —— 放根路径与 `/healthz`
同一族，将来若接抓取端也不必改路径。

WHY 不做鉴权：内容只有计数与分布（无消息原文、无用户标识、无密钥）—— 与事件表不同，
它不是「按用户隔离的数据」，而是进程自述。若暴露在公网，按 TrustedHost / 网关侧收紧。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_metrics
from app.metrics import MetricsRegistry
from app.models import MetricsResponse

router = APIRouter(tags=["metrics"])

MetricsDep = Annotated[MetricsRegistry, Depends(get_metrics)]


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(registry: MetricsDep) -> MetricsResponse:
    """当前进程的指标快照（读内存计数，不查库、不阻塞）。"""
    return MetricsResponse(**registry.snapshot())
