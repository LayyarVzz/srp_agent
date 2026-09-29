"""交互事件持久层（`interaction_events` 表；dev=SQLite memory / prod=Postgres）。

WHY 独立于 `shared/logging.py`：日志底座必须保持**纯标准库**（容器/CLI/MCP 子进程都能
只 import 它），而本模块依赖 SQLAlchemy。二者关系是「单向依赖」——
`sink.py` 消费 `shared.logging.LogEvent`，日志底座对存储零感知。

WHY 事件表与「日志」并存（而非只留 stdout）：stdout 是**进程级**的，容器重启/多副本
之后无法回溯；`/api/v1/logs/recent` 需要跨进程读同一份「最近发生了什么」。二者是
互补口径，不是重复：stdout 供 `docker compose logs`/采集器，事件表供接口与事后追查。

WHY 不变量：列名与 `LogEvent` 字段**同名同义**（event/trace_id/session_id/... ），
自由 `fields` 收敛进 `data`（JSON）。这样事件表既能被 SQL 直接聚合（按 event/状态/
耗时），又不必为每个新事件改表结构 —— 新增事件只在 `LogEvent.fields` 里加键即可。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, DateTime, Index, Integer, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool, StaticPool

from shared.logging import LogEvent

# 最近事件查询（WHERE user_id + ORDER BY ts DESC）的复合索引名。
EVENT_USER_TS_INDEX = "ix_interaction_events_user_ts"
# 按 trace 捞一条请求的全部事件（api→agent→MCP 全链路）的索引名。
EVENT_TRACE_INDEX = "ix_interaction_events_trace"
# 单次批量写入上限：一次事务写太多会让观测写放大到影响业务库的锁竞争。
MAX_BATCH_SIZE = 500


class EventRecord(BaseModel):
    """事件表的一行（读路径的边界结构体；写入侧直接用 `LogEvent`）。

    WHY 单独模型而非复用 `LogEvent`：`LogEvent` 是「刚发生的事件」的写入契约（带
    service/level 等运行时语义），`EventRecord` 是「已落库的事实」的读取契约（带
    主键 id 与写入时间）。两者字段相近但语义不同，混用会让读接口被写入侧改动波及。
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    ts: datetime
    event: str
    level: str
    service: str
    trace_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    status: str | None = None
    code: str | None = None
    duration_ms: int | None = None
    tool_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventRepository(Protocol):
    """事件存储契约（结构型协议；测试可注入假实现）。

    写入是**批量**的：带外 sink 一次排干队列后单事务写入，避免「每个事件一次事务」
    把观测开销放大到与业务写竞争。读路径恒带 `user_id` 归属过滤 —— 事件含会话 id
    与工具名，跨用户可见即信息泄漏，与记忆/会话同口径隔离。
    """

    async def append(self, events: list[LogEvent]) -> None:
        """批量插入事件（空列表为 no-op）。"""

    async def recent(self, *, user_id: str | None = None, limit: int = 50) -> list[EventRecord]:
        """按 ts 降序取最近事件；`user_id` 非空时只返回该用户的事件。"""

    async def list_by_trace(self, *, trace_id: str, limit: int = 100) -> list[EventRecord]:
        """按 trace_id 取一条请求链路的事件（升序：读起来即真实发生顺序）。"""

    async def count(self) -> int:
        """事件总行数（多 worker 写入去重/幂等断言的观测口）。"""

    async def setup(self) -> None:
        """幂等建表（装配期调用一次）。"""

    async def aclose(self) -> None:
        """关闭连接池（进程退出 / 测试收尾）。"""


class _Base(DeclarativeBase):
    """SQLAlchemy declarative 基类（事件表所属；独立 metadata，不与会话表耦合）。"""


class InteractionEventRow(_Base):
    """`interaction_events` 表行模型。

    `id` 用 UUID4 字符串（**不是** 自增）：多 worker（uvicorn `--workers N` / 多副本）
    各自独立写入同一张表，自增需要在写前取序列（多余往返），并发下也更易冲突；
    UUID4 让每个进程都能无协调地写入，`count()` 因此是「行数」而非「最大序号」。
    """

    __tablename__ = "interaction_events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[str] = mapped_column(Text, nullable=False)
    service: Mapped[str] = mapped_column(Text, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(Text, nullable=True)
    code: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 自由字段（对应 `LogEvent.fields`）：新增事件不改表结构，避免每次观测迭代都上迁移。
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index(EVENT_USER_TS_INDEX, user_id, ts.desc()),
        Index(EVENT_TRACE_INDEX, trace_id),
    )


class SQLAlchemyEventRepository:
    """SQLAlchemy 实现的事件仓库（同一实现服务 dev/prod，与 `SLQAlchemySessionRepository` 同构）。

    读路径统一在 Python 侧做 limit 截断与行→模型转换，SQL 只负责过滤与排序：
    两端方言（SQLite/Postgres）的时间语义差异不会漏到调用方。
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: AsyncEngine | None = None,
    ) -> None:
        """构造仓库。显式传 engine 覆盖 DSN（测试隔离用），否则按 database_url 建 engine。"""
        self._engine = engine or _build_engine(_plain_dsn(database_url))
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def append(self, events: list[LogEvent]) -> None:
        """批量插入事件：单事务写入整批（空批直接返回，避免空事务）。"""
        if not events:
            return
        rows = [_to_row(e) for e in events[:MAX_BATCH_SIZE]]
        async with self._session_factory() as session:
            session.add_all(rows)
            await session.commit()

    async def recent(self, *, user_id: str | None = None, limit: int = 50) -> list[EventRecord]:
        """按 ts 降序取最近事件；`user_id` 非空时只返回该用户事件（归属隔离）。"""
        stmt = select(InteractionEventRow)
        if user_id is not None:
            stmt = stmt.where(InteractionEventRow.user_id == user_id)
        stmt = stmt.order_by(InteractionEventRow.ts.desc()).limit(max(1, limit))
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [_to_record(r) for r in rows]

    async def list_by_trace(self, *, trace_id: str, limit: int = 100) -> list[EventRecord]:
        """按 trace_id 升序取链路事件（升序 = 真实发生顺序，排查时无需倒着读）。"""
        stmt = (
            select(InteractionEventRow)
            .where(InteractionEventRow.trace_id == trace_id)
            .order_by(InteractionEventRow.ts.asc())
            .limit(max(1, limit))
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [_to_record(r) for r in rows]

    async def count(self) -> int:
        """事件总行数（供多 worker 写入的去重/幂等断言）。"""
        from sqlalchemy import func

        async with self._session_factory() as session:
            total = await session.scalar(select(func.count()).select_from(InteractionEventRow))
        return int(total or 0)

    async def setup(self) -> None:
        """幂等建表（`create_all` 仅在缺表时执行），装配期调用一次。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def aclose(self) -> None:
        """关闭连接池（进程退出 / 测试收尾）。"""
        await self._engine.dispose()


def _to_row(event: LogEvent) -> InteractionEventRow:
    """`LogEvent` → 表行：列取同名语义字段，自由 `fields` 进 `data`（JSON）。

    直接用 `event.fields` 而不是 `event.summary()`：summary 是「给人 grep 的扁平
    dict」（含 timestamp 字符串），反解析回列既多余又易漂移；`fields` 才是原始自由字段。
    """
    return InteractionEventRow(
        id=event.id,
        ts=event.ts,
        event=event.event,
        level=event.level,
        service=event.service,
        trace_id=event.trace_id,
        session_id=event.session_id,
        user_id=event.user_id,
        status=event.status,
        code=event.code,
        duration_ms=event.duration_ms,
        tool_name=event.tool_name,
        data=dict(event.fields),
    )


def _to_record(row: InteractionEventRow) -> EventRecord:
    """表行 → `EventRecord`（ts 时区规整为 UTC，与写入端一致）。"""
    return EventRecord(
        id=row.id,
        ts=_to_utc(row.ts),
        event=row.event,
        level=row.level,
        service=row.service,
        trace_id=row.trace_id,
        session_id=row.session_id,
        user_id=row.user_id,
        status=row.status,
        code=row.code,
        duration_ms=row.duration_ms,
        tool_name=row.tool_name,
        data=dict(row.data or {}),
    )


def _to_utc(value: datetime) -> datetime:
    """时区规整：naive 视为 UTC，aware 统一转 UTC（存储端用 aware UTC 写）。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _build_engine(database_url: str | None) -> AsyncEngine:
    """按 DSN 有无构造异步 engine：无 → SQLite memory；有 → 给定 DSN（Postgres/文件 SQLite）。

    **统一 NullPool**（与 `agent/session/repository.py` 的 StaticPool 有意不同）：事件表的
    写入方是**带外线程自己的事件循环**，读方在主循环 —— 连接池把连接绑在「建它的那个循环」
    上，跨循环复用即报错。NullPool 每次操作新建连接、用完即关，天然跨循环安全；事件写入是
    低频批量（一批一次连接），连接开销可忽略，换来的正确性更重要。

    例外：`database_url is None`（dev 默认的内存库）必须 StaticPool —— `:memory:`
    与连接一一对应，换连接即换库、数据会凭空消失。
    """
    if database_url is None:
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    url = database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return create_async_engine(url, poolclass=NullPool)


def build_event_repository(database_url: str | None = None) -> SQLAlchemyEventRepository:
    """统一构造事件仓库：有 DSN → Postgres；无 → SQLite memory（镜像会话/记忆后端裁决）。"""
    return SQLAlchemyEventRepository(_plain_dsn(database_url))


def _plain_dsn(database_url: str | None) -> str | None:
    """DSN 解包：`settings.database_url` 是 `SecretStr`（防误打印），存储层要的是明文串。

    只有存储装配点做这一次解包（`get_secret_value()`），后续任何日志都拿不到明文 ——
    比在每个调用点 `str(...)`（会得到 `**********`）更不容易踩错。
    """
    if database_url is None:
        return None
    value = (
        database_url.get_secret_value()
        if hasattr(database_url, "get_secret_value")
        else str(database_url)
    )
    return value or None
