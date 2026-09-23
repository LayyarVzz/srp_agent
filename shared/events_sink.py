"""事件带外落库 sink（非阻塞队列 + 专职线程）。

**为什么必须带外**：`subscribe_events` 的监听器在**业务线程/事件循环内同步调用**，
监听器里直接 `await` 写库等于把观测开销塞进请求关键路径（DB 抖动 → 回答变慢），
而 CLAUDE.md 的硬约束是「记忆/观测写入不得阻塞主链路」。

**设计**：`send()` 只做 `put_nowait`（O(1)、无锁等待）→ 专职守护线程从队列排干、
单事务批量写库。因此：

- 业务侧最坏情况：队列满 → 丢**最旧**事件并记 WARNING（观测数据可丢，回答不可慢）；
- 写库在**独立线程 + 常驻事件循环**（`run_coroutine_threadsafe`）：连接池绑定该循环，
  不会出现「每次 `asyncio.run` 新建循环 → 池里连接跨循环失效」的隐患；
- 关闭时先排干队列再 `aclose()`，容器 SIGTERM 下最后一批事件不丢。

**多 worker**：每个 worker 进程各有一个 sink，写同一张 `interaction_events` 表；
`LogEvent.id` 是 uuid4，故无主键冲突、无需协调 —— 这正是选择 uuid4 的原因。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from shared.events_store import EventRepository, build_event_repository
from shared.logging import LogEvent

logger = logging.getLogger(__name__)

# 队列容量：按峰值「每请求约 10 条事件」估算，4096 条 ≈ 400 个并发请求的缓冲量。
# 超出即丢最旧 —— 宁可丢观测数据也不让请求线程等待。
DEFAULT_QUEUE_SIZE = 4096
# 单批最大条数（与仓库 MAX_BATCH_SIZE 同量级，避免一次事务过大）。
DEFAULT_BATCH_SIZE = 200
# 排空轮询间隔（秒）：有事件时按批写，空闲时以该间隔醒来（兼顾延迟与空转）。
DEFAULT_IDLE_INTERVAL_S = 0.5
# 关闭时排干队列的最长等待（秒）：容器 SIGTERM 后要尽快退出，不能无限等。
DEFAULT_DRAIN_TIMEOUT_S = 5.0


class EventStoreSink:
    """把事件异步写进 `interaction_events` 的带外 sink（`subscribe_events` 的监听器）。

    `send()` 签名与 `EventListener` 一致，可直接注册；`start()`/`aclose()` 由应用
    lifespan 调用。线程在首次 `start()` 时启动（不在 `__init__` 里起线程 —— 测试
    构造 sink 不该产生副作用线程）。
    """

    def __init__(
        self,
        repository: EventRepository,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        idle_interval_s: float = DEFAULT_IDLE_INTERVAL_S,
        drain_timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    ) -> None:
        self._repository = repository
        self._queue_size = queue_size
        # 队列在写线程建循环时创建（`asyncio.Queue` 绑定创建它的循环）；`start()` 前
        # 入队的事件无处可放，故 start 之前 `send` 直接丢弃并计数（不静默）。
        self._queue: asyncio.Queue[LogEvent | None] | None = None
        self._batch_size = batch_size
        self._idle_interval_s = idle_interval_s
        self._drain_timeout_s = drain_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        # 丢弃计数（队列满时累加）：暴露给测试与运维，避免「悄悄丢数据」。
        self.dropped = 0

    @property
    def repository(self) -> EventRepository:
        """本 sink 的写入仓库（读路径/测试按同一实例取回刚写入的事件）。"""
        return self._repository

    # —— 生产者侧（业务线程 / 事件循环内，同步调用，绝不可阻塞）——

    def send(self, event: LogEvent) -> None:
        """监听器入口：入队即返回；队列满/dropped 丢最旧并计数（观测可降级，请求不可慢）。"""
        loop, queue = self._loop, self._queue
        if loop is None or queue is None or loop.is_closed():
            self.dropped += 1
            return  # 未 start / 已关闭：没有消费方，直接丢弃（不静默：计数可见）
        try:
            # 跨线程投递必须走 call_soon_threadsafe：直接 put_nowait 不会唤醒正在 await
            # 的消费者，且非线程安全（仅靠 GIL 侥幸不炸）。
            loop.call_soon_threadsafe(self._enqueue, event)
        except RuntimeError:  # 关闭竞态：循环刚好关掉
            self.dropped += 1

    def _enqueue(self, event: LogEvent) -> None:
        """（循环内执行）入队；满则丢最旧，保持「最近窗口」语义。"""
        queue = self._queue
        if queue is None:  # pragma: no cover  # 仅在关闭竞态下发生
            return
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            self.dropped += 1
        try:
            queue.get_nowait()  # 丢最旧
        except asyncio.QueueEmpty:  # pragma: no cover  # 竞态：消费者刚好取走
            pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover  # 极端竞态，放弃这条
            return
        if self.dropped == 1 or self.dropped % 100 == 0:
            logger.warning("事件队列已满，开始丢弃最旧事件（累计 %d 条）", self.dropped)

    # —— 生命周期 ——

    def start(self) -> None:
        """启动专职写库线程（幂等）。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="srp-event-store", daemon=True)
        self._thread.start()
        # 等循环就绪：`aclose()` 需要在循环上跑 `aclose()` 协程，避免竞态空转。
        self._ready.wait(timeout=self._drain_timeout_s)

    async def aclose(self) -> None:
        """先 `stop()`（排干队列 + 停写线程），再关仓库连接池（幂等）。

        排干语义：消费循环收到哨兵后**立即返回**（哨兵只可能在「上一条已写入」之后被
        `await queue.get()` 取出），故线程 join 成功即代表队列已空、已写尽。
        """
        await self.stop()
        # 无论是否 start 过都要关连接池（否则 aiosqlite/asyncpg 会留资源告警）。
        await self._repository.aclose()

    async def stop(self) -> None:
        """停写线程：投哨兵 → 等线程退出（队列已排干）。不关仓库（多 sink 共享时用）。"""
        thread, loop = self._thread, self._loop
        if thread is not None and loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._enqueue, None)  # 哨兵
            except RuntimeError:  # pragma: no cover  # 关闭竞态
                pass
            await asyncio.to_thread(thread.join, self._drain_timeout_s)
        self._thread = None
        self._loop = None
        self._queue = None

    # —— 消费者侧（专职线程）——

    def _run(self) -> None:
        """线程主体：建常驻循环并 `run_forever`，消费协程在**该循环内**运行。

        WHY 必须 `run_forever`（而不是同步阻塞取队列）：写库是 `await` 的，协程只能由
        事件循环驱动 —— 若线程里同步 `queue.get()` 阻塞，`run_coroutine_threadsafe`
        排进去的协程永远不会被执行（表现为 future 超时、事件静默丢失）。故这里让循环
        常驻，用 `asyncio.Queue` 在循环内消费。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=self._queue_size)
        self._ready.set()
        try:
            loop.run_until_complete(self._consume())
        finally:
            loop.close()

    async def _consume(self) -> None:
        """消费循环：排干一批 → 批量写库 → 再排干，直到收到哨兵。"""
        while True:
            batch = await self._drain_batch()
            if batch is None:  # 哨兵：退出（本批已写完）
                return
            if batch:
                await self._persist(batch)

    async def _drain_batch(self) -> list[LogEvent] | None:
        """取一批事件：先 await 第一条（带超时），随后非阻塞取到 batch_size 或排空。

        返回 None 表示收到哨兵（关闭信号）。空闲期用 `wait_for` 超时唤醒，既不空转
        也能及时看到 `aclose()` 投下的哨兵。
        """
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=self._idle_interval_s)
        except TimeoutError:
            return []
        if first is None:
            return None
        batch = [first]
        while len(batch) < self._batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                # 哨兵在批中：本批先写完，退出由下一轮处理（重新入队哨兵）。
                self._queue.put_nowait(None)
                return batch
            batch.append(item)
        return batch

    async def _persist(self, batch: list[LogEvent]) -> None:
        """批量写库；失败只记日志（绝不反噬主链路，也绝不重试放大）。"""
        try:
            await self._repository.append(batch)
        except Exception as exc:  # 观测写失败不得上抛
            logger.warning("事件落库失败（丢弃 %d 条）：%s", len(batch), exc)


def build_event_sink(
    database_url: str | None = None,
    *,
    repository: EventRepository | None = None,
    **kwargs: Any,
) -> tuple[EventStoreSink, EventRepository]:
    """构造「sink + 仓库」并返回两者：路由读库、sink 写库，共享同一仓库实例。

    WHY 成对返回：读（`/api/v1/logs/recent`）与写（sink）必须看到**同一个**库；
    分别构造两个仓库在 SQLite memory 下会各自拿到**独立的空库**（`StaticPool` 每实例
    一条连接），表现为「接口永远查不到刚发生的事件」。
    """
    resolved = repository or build_event_repository(database_url)
    return EventStoreSink(resolved, **kwargs), resolved
