"""FastAPI 应用工厂：lifespan 装配 AgentRuntime、挂中间件/异常处理器/路由。

Windows 注意：`ensure_selector_event_loop()` 必须在任何事件循环创建之前调用，
放本模块导入期=——漏掉它 Postgres 异步连接会报
`Psycopg cannot use the 'ProactorEventLoop'`（纯 SQLite memory 路径不受影响）。

日志：配置统一经 `shared.logging`（与 3 个 MCP 服务同一 formatter / 脱敏 / 关联标识口径）；
`RequestContextMiddleware` 在入口注入 `X-Request-Id`，使 api→agent→MCP 全链路可 grep。

事件落库（C5）：`EventStoreSink` 订阅结构化事件、带外批量写 `interaction_events`
（不阻塞请求）；`/api/v1/logs/recent` 与 sink 共享同一仓库实例，故接口读到的事件与
stdout 日志是同一份事实。

指标（Phase D）：`MetricsRegistry` 订阅**同一条事件总线**、在内存里折叠成计数与分布
（`/metrics`）—— 与事件字段同名同源，不引入第二套口径。

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
from app.a2a import routes as a2a
from app.errors import register_exception_handlers
from app.metrics import MetricsRegistry
from app.request_context import REQUEST_ID_HEADER_OUT, RequestContextMiddleware
from app.routes import chat, health, logs, metrics, sessions
from settings import get_settings
from shared.events_sink import EventStoreSink, build_event_sink
from shared.events_store import EventRepository
from shared.logging import (
    LoggingConfig,
    ServiceName,
    configure_logging,
    subscribe_events,
    unsubscribe_events,
)

# Windows：psycopg 异步需 SelectorEventLoop，须在 uvicorn 建 loop 之前设置（模块导入期）。
ensure_selector_event_loop()

# 允许外部携带/读取的关联头：CORS 下不回 expose，前端就拿不到 trace_id 去对日志。
_REQUEST_ID_HEADERS = ["X-Request-Id", "X-User-Id"]


def create_app(
    *,
    runtime: AgentRuntime | None = None,
    event_repository: EventRepository | None = None,
    event_sink: EventStoreSink | None = None,
) -> FastAPI:
    """应用工厂。

    `runtime` / `event_repository` 供测试注入（fake 组合根 / 隔离库）；
    生产传 None 由本函数按 DSN 装配事件存储、由 lifespan 装配 runtime。
    """

    settings = get_settings()
    # 事件存储装配（进程级单例）：按 DSN 裁决 dev=SQLite memory / prod=Postgres。
    # 仓库在此（而非 lifespan）构造并挂到 app.state：路由（读）与 sink（写）必须共享
    # 同一实例，SQLite memory 下各建一份会得到两个互不相见的库。
    if event_repository is not None:
        sink = event_sink or EventStoreSink(event_repository)
        repository = event_repository
    elif event_sink is not None:
        repository = event_sink.repository
    else:
        sink, repository = build_event_sink(settings.database_url)
    app_state_event_repository = repository
    app_state_event_sink = sink
    # 指标注册表（进程级）：与事件 sink 吃同一条事件总线（口径同源，见 app/metrics.py）。
    app_state_metrics = MetricsRegistry(service=ServiceName.API)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 建表（幂等）+ 启动带外写线程 + 订阅事件：装配期完成，避免首个请求时踩空表。
        await app_state_event_repository.setup()
        subscribe_events(app_state_event_sink.send)
        subscribe_events(app_state_metrics.observe)
        app_state_event_sink.start()
        app.state.runtime = runtime or await AgentRuntime.create()
        try:
            yield
        finally:
            # 顺序：先排干带外记忆保存，再排干事件 sink（含排空队列），最后关 runtime。
            # WHY sink 在 runtime 之前关：runtime.aclose() 自身会产生事件（关闭路径也要
            # 可观测），队列必须先还活着，否则最后一批事件被丢。
            await wait_pending_saves()
            await app.state.runtime.aclose()
            await app_state_event_sink.aclose()
            # 注销订阅：监听器注册表是**进程级全局**的，残留会让下一个 app（测试/重启）
            # 继续把事件写进已关停的注册表（与 sink 同一口径）。
            unsubscribe_events(app_state_metrics.observe)

    # 日志单点配置：service=api（事件按进程归属），格式/级别来自 LOG_* 环境项。
    configure_logging(LoggingConfig.from_settings(settings, service=ServiceName.API))
    app = FastAPI(
        title=settings.app_name,
        description="数字人 Agent 交互服务（MVP）：会话管理 + 文字/语音交互（SSE）",
        version="0.1.0",
        lifespan=lifespan,
    )
    # 事件仓库挂在 state 上（路由依赖读它；测试直接覆盖该属性即可换库）。
    app.state.event_repository = app_state_event_repository
    app.state.event_sink = app_state_event_sink
    app.state.metrics = app_state_metrics
    # 中间件决策：CORS 必须、TrustedHost 推荐；
    # GZip 不用（SSE 流式不该压缩，会缓冲破坏实时性）。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[*_REQUEST_ID_HEADERS, "Content-Type"],
        # 不回 expose_headers，浏览器侧 JS 读不到 X-Request-Id → 前端报错时无法提供 trace_id。
        expose_headers=[REQUEST_ID_HEADER_OUT],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # dev；部署收紧
    # 关联上下文（最外层：异常路径与第三方库日志也要带 trace_id）。
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(sessions.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(logs.router, prefix="/api/v1")
    # 指标走根路径（运维/答辩视图，与 /healthz 同族；见 app/routes/metrics.py）。
    app.include_router(metrics.router)
    # A2A 端点挂根路径（/.well-known/agent.json 与 /a2a，协议约定的绝对路径）。
    app.include_router(a2a.router)
    if runtime is not None:
        # 测试注入：ASGITransport 不触发 lifespan，直接预置 app.state.runtime。
        app.state.runtime = runtime
    return app


app = create_app()
