"""FastAPI 依赖：从 app.state 取 AgentRuntime（组合根）。

Agent 装配统一由 `agent/runtime.py`（AgentRuntime.create）在 lifespan 完成，
路由层只消费组合根暴露的会话与对话编排方法，不接触 Agent 内部实现。
"""

from __future__ import annotations

from fastapi import Request

from agent.runtime import AgentRuntime


def get_runtime(request: Request) -> AgentRuntime:
    """获取应用级 AgentRuntime（lifespan 装配到 app.state.runtime）。"""
    return request.app.state.runtime
