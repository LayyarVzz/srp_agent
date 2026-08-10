"""记忆存储工厂：按配置构造 langgraph BaseStore（dev/prod 切换点）。"""

from __future__ import annotations

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from agent.core.config import MemoryBehaviorConfig


def build_store(cfg: MemoryBehaviorConfig) -> BaseStore:
    """按 store_type 构造默认存储；postgres 需显式注入（P4-1 未配 pg 依赖）。

    WHY 快速失败：生产切 postgres 却忘传 store 时，比静默退回 InMemoryStore 更安全。
    """
    if cfg.store_type == "postgres":
        raise NotImplementedError(
            "P4-1 未接入 PostgresStore，请在 build_agent_graph(store=...) 显式传入"
        )
    return InMemoryStore()
