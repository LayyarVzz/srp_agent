"""FastAPI 依赖：从 app.state 取 AgentRuntime（组合根）与事件仓库（观测存储）。

Agent 装配统一由 `agent/runtime.py`（AgentRuntime.create）在 lifespan 完成，
路由层只消费组合根暴露的会话与对话编排方法，不接触 Agent 内部实现。

`get_event_repository` 返回的是 `create_app` 装配的**进程级**事件仓库：与带外 sink
共享同一实例（同一连接池），保证「写入的事件」与「接口读到的事件」是同一份数据。
"""

from __future__ import annotations

from fastapi import Request

from agent.runtime import AgentRuntime
from shared.events_store import EventRepository


def get_runtime(request: Request) -> AgentRuntime:
    """获取应用级 AgentRuntime（lifespan 装配到 app.state.runtime）。"""
    return request.app.state.runtime


def get_event_repository(request: Request) -> EventRepository:
    """获取应用级事件仓库（`create_app` 直接装配到 app.state；读写共享同一实例）。"""
    return request.app.state.event_repository
