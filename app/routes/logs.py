"""日志/事件查询路由：读 `interaction_events` 表（C5 起由库承载，不再是进程内 deque）。

WHY 查库而非读进程内缓存：容器里 api 可能多 worker / 多副本，进程内 deque 只能看到
**自己那个进程**的事件，前端联调时会「一半请求查得到、一半查不到」。事件表是唯一
能看到全量最近事件的读源。

归属隔离：`/recent` 恒按 `X-User-Id` 过滤（事件含 session_id / 工具名 / 记忆动作，
跨用户可见即信息泄漏，与记忆、会话元数据同一口径）；`/trace/{id}` 是排查口，trace_id
本身是服务端发出的不可猜随机值（`req_` + 16 字节 hex），持有它即视为有权查看该链路。

自观测噪声：`/recent` 的结果**排除本次查询自身的 trace**。查事件这个动作也会产生
`request.received` / `request.finished`，不排除的话第一屏永远是「你正在查日志」这件事
自己，且排在最新的位置把真正要看的业务事件挤下去（一次查询最多自占 2 条）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

from app.deps import get_event_repository
from app.models import RecentLogsResponse, TraceEventsResponse
from app.routes.sessions import require_user_id
from shared.events_store import EventRepository
from shared.logging import current_trace_id

router = APIRouter(tags=["logs"])

RepositoryDep = Annotated[EventRepository, Depends(get_event_repository)]
UserHeader = Annotated[str | None, Header()]

# 本次查询自身最多贡献 2 条事件（received + finished），多取 2 条再剔除即可凑满 limit。
_SELF_EVENT_OVERHEAD = 2


@router.get("/logs/recent", response_model=RecentLogsResponse)
async def recent_logs(
    repository: RepositoryDep,
    x_user_id: UserHeader = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> RecentLogsResponse:
    """最近交互事件（按时间降序，仅本人）：前端联调排查问题用，不做前端主链路。

    排除本次查询自身的 trace（见模块 docstring）：多取 `_SELF_EVENT_OVERHEAD` 条后剔除，
    保证调用方仍能拿到接近 `limit` 条业务事件。
    """
    user_id = require_user_id(x_user_id)
    own_trace = current_trace_id()
    fetched = await repository.recent(user_id=user_id, limit=limit + _SELF_EVENT_OVERHEAD)
    events = [e for e in fetched if e.trace_id != own_trace][:limit]
    return RecentLogsResponse(limit=limit, events=events)


@router.get("/logs/trace/{trace_id}", response_model=TraceEventsResponse)
async def trace_events(
    trace_id: str,
    repository: RepositoryDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> TraceEventsResponse:
    """按 trace_id 取一条请求的全链路事件（升序 = 真实发生顺序）。

    排查典型用法：前端拿响应头 `X-Request-Id` 调本接口，一条链路从
    `request.received` 到 `request.finished`（含工具/意图/回答）全在眼前。
    """
    events = await repository.list_by_trace(trace_id=trace_id, limit=limit)
    return TraceEventsResponse(trace_id=trace_id, events=events)
