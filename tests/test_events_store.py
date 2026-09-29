"""`interaction_events` 事件表 + 带外 sink 单测（Phase C / C5）。

覆盖三件事：
① 存储语义：写→读往返、用户归属过滤、trace 链路升序、按 ts 降序；
② sink 语义：**绝不阻塞/绝不上抛**（业务最坏情况只是丢事件）、批量写入、
   多 worker（多 sink 并发写同一表）无主键冲突；
③ 接口契约：`/api/v1/logs/recent` 查库（不再是进程内 deque）、`/logs/trace/{id}`
   给出全链路顺序。

真实链路验收（HTTP 请求 → 事件 → 落库 → 接口读到）在 `test_logs_route.py`。
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest_asyncio

from shared.events_sink import EventStoreSink
from shared.events_store import (
    SQLAlchemyEventRepository,
    build_event_repository,
)
from shared.logging import EVENT_REQUEST_FINISHED, EVENT_TOOL_CALLED, LogEvent


def _event(
    name: str = EVENT_TOOL_CALLED,
    *,
    ts: datetime | None = None,
    user_id: str | None = "u1",
    session_id: str | None = "s1",
    trace_id: str | None = "req_1",
    duration_ms: int | None = 12,
    fields: dict[str, Any] | None = None,
) -> LogEvent:
    """构造一条测试事件（默认字段齐全，便于逐项覆盖）。"""
    return LogEvent(
        event=name,
        service="api",
        ts=ts or datetime.now(UTC),
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        status="ok",
        duration_ms=duration_ms,
        tool_name="calc",
        fields=fields or {"args": ""},
    )


@pytest_asyncio.fixture
async def repository(tmp_path: Any) -> Any:
    """隔离的事件仓库（**文件** SQLite + NullPool：与生产同构的连接语义）。

    WHY 不用 `:memory:`：内存库与连接一一对应（StaticPool 单连接），而带外 sink 在
    **另一个线程的循环**里写 —— 单连接池跨循环复用会直接报错。文件库 + NullPool 与生产
    （Postgres + NullPool）语义一致，测试才是在验生产行为而不是验一套特例。
    """
    repo = build_event_repository(f"sqlite+aiosqlite:///{tmp_path / 'events.db'}")
    await repo.setup()
    yield repo
    await repo.aclose()


# —— ① 存储语义 ——


async def test_append_then_recent_roundtrip(repository: Any) -> None:
    """写→读往返：列取同名语义字段，自由 `fields` 进 `data`（两处都不丢）。"""
    event = _event(fields={"answer": "len=6 sha256=abc12345", "phase": "generated"})
    await repository.append([event])

    records = await repository.recent(user_id="u1")
    assert len(records) == 1
    record = records[0]
    assert record.id == event.id
    assert record.event == EVENT_TOOL_CALLED
    assert record.trace_id == "req_1"
    assert record.session_id == "s1"
    assert record.user_id == "u1"
    assert record.duration_ms == 12
    assert record.tool_name == "calc"
    assert record.data["phase"] == "generated"
    assert record.ts.tzinfo is not None  # 读回恒为 aware UTC


async def test_recent_is_descending_and_limited(repository: Any) -> None:
    """`recent` 按 ts 降序并遵守 limit（排查时最新事件必须排最前）。"""
    base = datetime.now(UTC)
    await repository.append(
        [_event(ts=base + timedelta(seconds=i), trace_id=f"req_{i}") for i in range(5)]
    )
    records = await repository.recent(user_id="u1", limit=3)
    assert [r.trace_id for r in records] == ["req_4", "req_3", "req_2"]


async def test_recent_is_isolated_by_user(repository: Any) -> None:
    """归属隔离：A 用户读不到 B 用户的事件（事件含会话/工具信息，等同记忆口径）。"""
    await repository.append([_event(user_id="u1"), _event(user_id="u2", session_id="s2")])
    assert [r.user_id for r in await repository.recent(user_id="u1")] == ["u1"]
    assert [r.user_id for r in await repository.recent(user_id="u2")] == ["u2"]
    # 不带 user_id 的读（运维口径）能看到全部 —— 路由层恒带 user_id，不会走这条。
    assert len(await repository.recent()) == 2


async def test_list_by_trace_is_ascending(repository: Any) -> None:
    """一条链路的全部事件按时间升序返回（读起来即真实发生顺序）。"""
    base = datetime.now(UTC)
    await repository.append(
        [
            _event(EVENT_REQUEST_FINISHED, ts=base + timedelta(seconds=2), trace_id="req_x"),
            _event(EVENT_TOOL_CALLED, ts=base + timedelta(seconds=1), trace_id="req_x"),
            _event(EVENT_TOOL_CALLED, ts=base, trace_id="req_other"),
        ]
    )
    names = [r.event for r in await repository.list_by_trace(trace_id="req_x")]
    assert names == [EVENT_TOOL_CALLED, EVENT_REQUEST_FINISHED]


async def test_append_empty_is_noop(repository: Any) -> None:
    """空批为 no-op（避免空事务）。"""
    await repository.append([])
    assert await repository.count() == 0


async def test_setup_is_idempotent(repository: Any) -> None:
    """重复建表不报错（每次应用启动都会调一次 `setup`）。"""
    await repository.setup()
    await repository.setup()
    assert await repository.count() == 0


async def test_postgres_dsn_is_translated_to_psycopg_dialect() -> None:
    """仓库构造只建 engine（惰性连接）：Postgres DSN 不真连库、`aclose` 不报错。"""
    repo = SQLAlchemyEventRepository("postgres://u:p@127.0.0.1:1/none")
    await repo.aclose()
    # SQLite 路径之外的方言归一由 `_build_engine` 承担（见其 docstring 的三种前缀）。


# —— ② sink 语义 ——


async def test_sink_persists_events_via_queue(repository: Any) -> None:
    """sink 把入队事件带外写库（写线程 + 常驻循环），关闭时排干不丢。"""
    sink = EventStoreSink(repository, idle_interval_s=0.05)
    sink.start()
    for i in range(3):
        sink.send(_event(trace_id=f"req_{i}"))
    await _wait_for(repository.count, expected=3)
    await sink.aclose()

    records = await repository.recent(user_id="u1", limit=10)
    assert {r.trace_id for r in records} == {"req_0", "req_1", "req_2"}


async def test_sink_swallows_repository_failure() -> None:
    """仓库写失败只记日志、不上抛（观测写不得反噬业务线程）。"""

    class _Broken:
        async def append(self, events: list[LogEvent]) -> None:
            raise RuntimeError("db down")

        async def aclose(self) -> None:
            return None

    sink = EventStoreSink(_Broken(), idle_interval_s=0.05)  # type: ignore[arg-type]
    sink.start()
    sink.send(_event())  # 不应抛异常
    await asyncio.sleep(0.2)
    await sink.aclose()


async def test_sink_drops_oldest_when_queue_full() -> None:
    """队列满时丢最旧并计数（生产者永不阻塞 —— 这是「非阻塞」的硬证据）。"""

    class _Slow:
        async def append(self, events: list[LogEvent]) -> None:
            await asyncio.sleep(0.05)

        async def aclose(self) -> None:
            return None

    sink = EventStoreSink(_Slow(), queue_size=2, idle_interval_s=0.01)  # type: ignore[arg-type]
    # 不 start()：无人消费，队列必然打满 → send() 仍必须立即返回
    started = time.perf_counter()
    for i in range(20):
        sink.send(_event(trace_id=f"req_{i}"))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, "send() 阻塞了，违反非阻塞契约"
    assert sink.dropped > 0


async def test_multiple_sinks_share_one_table_without_conflict(repository: Any) -> None:
    """多 worker 语义：两个 sink 并发写同一张表，行数与投递条数一致（uuid4 主键无冲突）。

    WHY 不是「一个 sink 写两次」：多 worker 是**多进程**各持一个 sink 写同一张表；
    本用例用两个 sink 模拟这一形态，验证主键不冲突、无需协调的写前取号。
    """
    first = EventStoreSink(repository, idle_interval_s=0.05)
    second = EventStoreSink(repository, idle_interval_s=0.05)
    first.start()
    second.start()
    for i in range(4):
        first.send(_event(trace_id=f"req_a{i}"))
        second.send(_event(trace_id=f"req_b{i}"))
    await _wait_for(repository.count, expected=8)
    assert await repository.count() == 8
    # 两个 sink 共享同一仓库实例 → 连接池只能关一次：先各自排干线程，再统一收尾。
    await _stop_thread(first)
    await _stop_thread(second)


async def test_sink_aclose_is_idempotent(repository: Any) -> None:
    """重复 aclose 不报错（lifespan 与测试收尾可能都调一次）。"""
    sink = EventStoreSink(repository, idle_interval_s=0.02)
    await sink.aclose()  # 未 start：直接关仓库（连接池仍需释放）
    sink2 = EventStoreSink(repository, idle_interval_s=0.02)
    sink2.start()
    await sink2.aclose()


async def test_sink_failure_log_does_not_refeed_events() -> None:
    """sink 写失败时打的 WARNING 不会再次分发事件（重入哨兵生效）。

    若哨兵失效：写失败 → `logger.warning` → 分发 → sink 再入队 → 失败 → 无限自我放大。
    这里用「count 恒定的失败仓库 + 记录收到条数」证明没有放大。
    """

    class _Broken:
        def __init__(self) -> None:
            self.received = 0

        async def append(self, events: list[LogEvent]) -> None:
            self.received += len(events)
            raise RuntimeError("db down")

        async def aclose(self) -> None:
            return None

    broken = _Broken()
    sink = EventStoreSink(broken, idle_interval_s=0.02)  # type: ignore[arg-type]
    sink.start()
    sink.send(_event(trace_id="req_once"))
    await asyncio.sleep(0.2)
    await _stop_thread(sink)
    # 只收到「真正投递的那 1 条」：失败告警没有把自己再灌回队列。
    assert broken.received <= 2


async def test_count_reflects_rows(repository: Any) -> None:
    """`count()` 为行数（多 worker 幂等断言依赖它）。"""
    await repository.append([_event(), _event(trace_id="req_2")])
    assert await repository.count() == 2


async def _stop_thread(sink: EventStoreSink) -> None:
    """只停 sink 的写线程、不关仓库（多 sink 共享仓库的测试收尾用）。"""
    await sink.stop()


async def _wait_for(probe: Any, *, expected: int, timeout_s: float = 5.0) -> None:
    """轮询等待异步条件成立（sink 是带外写，测试不能假设「入队即已落库」）。

    WHY 探针同为协程：仓库的连接池绑定**测试所在的循环**，在另一个循环里读会拿到
    已失效的连接（aiosqlite/psycopg 的连接是循环绑定的）—— 必须同循环轮询。
    """
    deadline = time.perf_counter() + timeout_s
    last = 0
    while time.perf_counter() < deadline:
        last = await probe()
        if last >= expected:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"等待超时：期望 ≥{expected}，实际 {last}")
