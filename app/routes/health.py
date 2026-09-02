"""健康检查：GET /healthz（存活探针，部署/联调自检）。"""

from __future__ import annotations

from fastapi import APIRouter

from settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, object]:
    """存活探针"""
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.environment,
        "llm_configured": bool(settings.llm_api_key.get_secret_value()),
    }
