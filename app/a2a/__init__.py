"""A2A HTTP 端点包（T5 Server 入口层：薄路由，业务在 agent/a2a 与 runtime）。"""

from app.a2a.routes import router

__all__ = ["router"]
