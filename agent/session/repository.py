"""会话元数据独立存储层（SQLAlchemy，dev=SQLite memory / prod=Postgres）。

会话**消息**由 LangGraph checkpointer 承载（thread_id == session_id）；本模块只负责
会话身份元数据（session_id↔user_id 映射 + created_at + expires_at）的持久化。

WHY 独立 `sessions` 表而非 langgraph `store` 表：会话与长期记忆生命周期不同
（会话短命高频增删/精确点查，记忆长期积累/语义检索），分表便于各自索引与生命周期
管理；langgraph Store 表名硬编码且按 namespace 分区，无法为会话单独建索引/清理策略。
TTL 由 `expires_at` 承载、过期在读
路径过滤，不再依赖 langgraph Store 的 TTL sweeper。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import DateTime, Index, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from agent.session.models import SessionContext

# 会话列表查询（WHERE user_id + ORDER BY created_at DESC）的复合索引名。
SESSION_USER_CREATED_INDEX = "ix_sessions_user_created"


class SessionRepository(Protocol):
    """会话元数据存储契约（结构型协议；测试可注入假实现）。

    全部方法 async；归属过滤（恒带 user_id）与过期过滤（expires_at <= now 视同
    不存在）是各实现必须满足的语义，Manager 据此做「未命中/越权统一 not_found」。
    """

    async def create(self, ctx: SessionContext, *, ttl_minutes: int | None = None) -> None:
        """登记一条会话元数据；`ttl_minutes` 非 None 时写入过期时间。"""

    async def get(self, *, user_id: str, session_id: str) -> SessionContext | None:
        """按归属取会话（恒带 user_id 过滤 + 排除已过期），未命中返回 None。"""

    async def list(self, *, user_id: str, limit: int = 20) -> list[SessionContext]:
        """按 created_at 降序列出该用户会话（排除已过期），最多 limit 条。"""

    async def delete(self, *, user_id: str, session_id: str) -> bool:
        """删除会话元数据（归属 + 未过期过滤），返回是否真正删除。"""

    async def setup(self) -> None:
        """幂等建表（装配期调用一次）。"""

    async def aclose(self) -> None:
        """关闭连接池（进程退出 / 测试收尾）。"""


class _Base(DeclarativeBase):
    """SQLAlchemy declarative 基类（会话元数据表所属）。"""


class SessionRow(_Base):
    """`sessions` 表行模型：身份 + 归属 + 生命周期。

    PK=session_id（唯一标识）；user_id 恒随行存储——归属校验在存储层天然成立，
    杜绝跨用户点查；expires_at 承载 TTL，NULL 表示永不过期。
    """

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index(SESSION_USER_CREATED_INDEX, user_id, created_at.desc()),)


class SQLAlchemySessionRepository:
    """SQLAlchemy 实现的会话元数据仓库（同一实现服务 dev/prod）。

    过期过滤在读路径执行（get/list/delete 均排除 `expires_at <= now`）：TTL 生命周期
    完全由本仓库承载，不再依赖 langgraph Store 的 TTL sweeper；过期行惰性残存，由
    上层可选清理/迁移处置（本阶段仅保证「过期视同不存在」的读语义）。
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: AsyncEngine | None = None,
    ) -> None:
        """构造仓库。显式传 engine 覆盖 DSN（测试隔离用），否则按 database_url 建 engine。"""
        self._engine = engine or _build_engine(database_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def create(self, ctx: SessionContext, *, ttl_minutes: int | None = None) -> None:
        """登记一条会话元数据；TTL 折算为 expires_at 落库（NULL 表示永不过期）。"""
        expires_at = (
            ctx.created_at + timedelta(minutes=ttl_minutes) if ttl_minutes is not None else None
        )
        async with self._session_factory() as session:
            session.add(
                SessionRow(
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    created_at=ctx.created_at,
                    expires_at=expires_at,
                )
            )
            await session.commit()

    async def get(self, *, user_id: str, session_id: str) -> SessionContext | None:
        """按 PK 点查 + Python 侧归属/过期校验（两方言时间语义安全，见 `_expired`）。"""
        async with self._session_factory() as session:
            row = await session.get(SessionRow, session_id)
            if row is None or row.user_id != user_id or self._expired(row):
                return None
            return self._to_context(row)

    async def list(self, *, user_id: str, limit: int = 20) -> list[SessionContext]:
        """按 user 过滤 + created_at 降序；过期在 Python 侧剔除后再截断 limit。

        WHY 不在 SQL 里过滤过期：SQLite 的字符串时间比较在 aware/naive 混存时
        不可靠，统一取回后规整比较，语义跨方言一致（每 user 会话量小，成本可忽略）。
        """
        async with self._session_factory() as session:
            stmt = (
                select(SessionRow)
                .where(SessionRow.user_id == user_id)
                .order_by(SessionRow.created_at.desc())
            )
            rows = (await session.execute(stmt)).scalars().all()
        return [self._to_context(r) for r in rows if not self._expired(r)][:limit]

    async def delete(self, *, user_id: str, session_id: str) -> bool:
        """删除会话元数据（归属 + 未过期过滤），返回是否真正删除。

        WHY 返回 bool 而非抛错：Manager 据此把「未命中 / 越权 / 已过期」统一映射为
        not_found（防枚举），存储层保持无状态、可被测试直接断言。
        """
        async with self._session_factory() as session:
            row = await session.get(SessionRow, session_id)
            if row is None or row.user_id != user_id or self._expired(row):
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def setup(self) -> None:
        """幂等建表（`create_all` 仅在缺表时执行），装配期调用一次。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def aclose(self) -> None:
        """关闭连接池（进程退出 / 测试收尾）。"""
        await self._engine.dispose()

    # —— 私有辅助 ——

    def _to_context(self, row: SessionRow) -> SessionContext:
        """行 → SessionContext（created_at 时区规整为 UTC，与 Manager 生成端一致）。"""
        return SessionContext(
            session_id=row.session_id,
            user_id=row.user_id,
            created_at=_to_utc(row.created_at),
        )

    def _expired(self, row: SessionRow) -> bool:
        """过期判断：无过期时间永不过期；过期时间需时区规整后与当前 UTC 比较。

        WHY 时区规整：SQLite 的 DateTime(timezone=True) 读回可能丢 tz（naive），
        Postgres 恒 aware；统一视为 UTC 再比较，避免 naive/aware 比较抛 TypeError。
        """
        if row.expires_at is None:
            return False
        return _to_utc(row.expires_at) <= datetime.now(UTC)


def _to_utc(value: datetime) -> datetime:
    """时区规整：naive 视为 UTC，aware 统一转 UTC（存储端用 aware UTC 写）。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _build_engine(database_url: str | None) -> AsyncEngine:
    """按 DSN 有无构造异步 engine：无 → SQLite memory；有 → Postgres（psycopg 异步驱动）。

    WHY StaticPool + check_same_thread=False：`:memory:` 数据库与连接一一对应，默认
    连接池换连接即「换库」导致数据消失；StaticPool 固定单连接且允许跨线程共享，
    保证 dev 单进程内语义一致。
    """
    if database_url is None:
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    # DSN 统一转 postgresql+psycopg 方言（兼容 postgres:// 别名），复用 psycopg[binary]。
    url = database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return create_async_engine(url)


def build_session_repository(database_url: str | None = None) -> SQLAlchemySessionRepository:
    """统一构造会话仓库：有 DSN → Postgres；无 → SQLite memory（镜像记忆后端的 DSN 裁决）。

    WHY 独立于 build_memory_backends：会话仓库不依赖记忆后端，任何装配入口
    （demo / app / factory）均可独立构造并注入 SessionManager。
    """
    return SQLAlchemySessionRepository(database_url)
