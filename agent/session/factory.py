"""会话后端装配工厂：独立于记忆后端，按 DSN 构造会话元数据存储 + 业务层。

对外暴露统一入口 `build_session_backend`：传入 `database_url` 则构造 Postgres
`sessions` 表仓库，无则降级 SQLite memory（与 `build_memory_backends` 的 DSN 裁决
镜像）。会话元数据与记忆**各自独立装配、各自 aclose**，由运行时层（runtime /
demo / 测试）组合两者——本模块不依赖 `agent.memory`，无横向耦合。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.core.config import SessionBehaviorConfig
from agent.session.manager import SessionManager
from agent.session.repository import (
    SessionRepository,
    build_session_repository,
)


@dataclass
class SessionBackend:
    """会话后端集合：repository（SQLAlchemy 存储）+ manager（业务层）。

    `aclose()` 关闭 repository 连接池；manager 无独立资源，随 repository 回收。
    """

    repository: SessionRepository
    manager: SessionManager

    async def aclose(self) -> None:
        """关闭会话仓库连接池（进程退出 / 测试收尾）。"""
        await self.repository.aclose()


async def build_session_backend(
    database_url: str | None = None,
    *,
    config: SessionBehaviorConfig | None = None,
) -> SessionBackend:
    """会话模块自装入：dev=SQLite memory / prod=Postgres（DSN 有无裁决，镜像记忆后端）。

    WHY 独立 async 入口：与 `build_memory_backends` 对称，运行时层一行组合、teardown
    集中；装配细节（setup 建表、manager 构造）收敛在本模块，调用方无需了解
    `sessions` 表与 TTL 实现。
    """
    repository = build_session_repository(database_url)
    await repository.setup()
    return SessionBackend(repository=repository, manager=SessionManager(repository, config=config))
