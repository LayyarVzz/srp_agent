"""FastAPI 应用工厂：lifespan 装配 AgentRuntime、挂中间件/异常处理器/路由。

Windows 注意：`ensure_selector_event_loop()` 必须在任何事件循环创建之前调用，
放本模块导入期=——漏掉它 Postgres 异步连接会报
`Psycopg cannot use the 'ProactorEventLoop'`（纯 SQLite memory 路径不受影响）。

根 `main.py` 仅 re-export 本模块的 `app`（保持 `uvicorn main:app` 兼容）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from agent.memory import wait_pending_saves
from agent.runtime import AgentRuntime
from agent.share.eventloop import ensure_selector_event_loop
from app.errors import register_exception_handlers
from app.routes import chat, health, sessions
from settings import configure_logging, get_settings

# Windows：psycopg 异步需 SelectorEventLoop，须在 uvicorn 建 loop 之前设置（模块导入期）。
ensure_selector_event_loop()


def create_app(*, runtime: AgentRuntime | None = None) -> FastAPI:
    """应用工厂。`runtime` 供测试注入 fake 组合根；生产传 None 由 lifespan 装配。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime or await AgentRuntime.create()
        try:
            yield
        finally:
            await wait_pending_saves()  # 先排干带外记忆保存，再关连接池
            await app.state.runtime.aclose()

    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(
        title=settings.app_name,
        description="数字人 Agent 交互服务（MVP）：会话管理 + 文字/语音交互（SSE）",
        version="0.1.0",
        lifespan=lifespan,
    )
    # 中间件决策：CORS 必须、TrustedHost 推荐；
    # GZip 不用（SSE 流式不该压缩，会缓冲破坏实时性）。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-User-Id"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # dev；部署收紧
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(sessions.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    if runtime is not None:
        # 测试注入：ASGITransport 不触发 lifespan，直接预置 app.state.runtime。
        app.state.runtime = runtime
    return app


app = create_app()
