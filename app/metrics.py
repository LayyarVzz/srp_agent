"""进程内轻量指标（`GET /metrics` 的数据源）。

WHY 消费 C3 的**事件总线**而非在各处手埋点：事件字段就是指标口径（plan §5.4 ——
「指标名与结构化事件字段同名」）。两套口径迟早分叉，而分叉后最难查的情形是
「看板说健康、日志在报错」。故本模块只做一件事：把已经下发的事件**折叠成计数与分布**。

WHY 进程内计数而非 Prometheus/Grafana：单实例演示/答辩规模下，时序库 + 抓取端 +
仪表盘属过度工程（plan §5.4 明确不做）。代价是**多副本时指标只代表本进程** ——
响应里带 `service` / `pid` 就是为了让人一眼看出「这不是全局数」，而不是误读。

口径陷阱（C 阶段实施中确认）：`request.finished` **有两条**（入口中间件记 HTTP 往返、
Agent runtime 记交互往返 `source="agent"`），故聚合必须按 `fields.source` 分开 ——
混在一条延迟分布里，HTTP 往返与交互往返会互相污染（一个含 SSE 下发、一个不含）。
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, deque
from datetime import UTC, datetime
from typing import Final

from shared.logging import (
    EVENT_ANSWER_GENERATED,
    EVENT_INTENT_CLASSIFIED,
    EVENT_MEMORY_SAVED,
    EVENT_REQUEST_FINISHED,
    EVENT_REQUEST_RECEIVED,
    EVENT_TOOL_CALLED,
    LogEvent,
)

logger = logging.getLogger(__name__)

# `request.finished` 的 source 语义（与 agent/runtime.py、app/request_context.py 的取值一致）。
SOURCE_AGENT: Final = "agent"
SOURCE_HTTP: Final = "http"

# 工具/请求的成功标记。
STATUS_OK: Final = "ok"

# 终态缺失时的归一桶名（见 `_on_answer_generated`：复用路径的事件可能不带终态）。
UNSPECIFIED: Final = "unspecified"

# 耗时样本窗口容量：长期运行进程不得无限增长。窗口外的样本仍计入 count/sum（累计精确），
# 只有百分位是按最近 `_WINDOW` 条计算的 —— 对「最近表现」而言这恰恰是想要的口径。
_WINDOW: Final = 2048


class DurationWindow:
    """耗时分布：**精确累计**（count/sum/max）+ **滑动窗口百分位**。

    WHY 不用「全量样本列表算百分位」：那是一个只增不减的内存泄漏（每次请求一条样本，
    长跑几天就是几百万个整数）；而只留窗口又会让「历史最慢」消失 —— 故两者并存。
    """

    def __init__(self, maxlen: int = _WINDOW) -> None:
        self._samples: deque[int] = deque(maxlen=maxlen)
        self.count = 0
        self.total_ms = 0
        self.max_ms = 0

    def observe(self, duration_ms: int) -> None:
        """记录一次耗时（负值防御性归零：耗时不该为负，但脏输入不得污染统计）。"""
        value = max(0, int(duration_ms))
        self._samples.append(value)
        self.count += 1
        self.total_ms += value
        self.max_ms = max(self.max_ms, value)

    def percentile(self, pct: float) -> float:
        """窗口内百分位（最近邻取值，不插值：指标要的是「实际观测到的那个值」）。"""
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
        return float(ordered[index])

    @property
    def avg_ms(self) -> float:
        """窗口外样本也计入的**全局**均值（与 count/sum 同口径，避免两个平均互相矛盾）。"""
        return round(self.total_ms / self.count, 1) if self.count else 0.0

    def stats(self) -> dict[str, float | int]:
        """一次快照（供 `/metrics` 直接下发）。"""
        return {
            "count": self.count,
            "avg_ms": self.avg_ms,
            "p50_ms": self.percentile(50),
            "p95_ms": self.percentile(95),
            "max_ms": self.max_ms,
            "window": len(self._samples),
        }


class MetricsRegistry:
    """把 C3 事件折叠成进程内指标（`observe` 即事件监听器）。

    线程/协程安全性：事件分发是**同进程同步调用**（`shared.logging._dispatch`），
    计数操作都是单条 dict/int 自增，无需锁；跨线程写入场景下 CPython 的 GIL 保证
    单条操作原子（不做「读-改-写」复合操作，故不会丢计数）。
    """

    def __init__(self, *, service: str = "", started_at: datetime | None = None) -> None:
        self.service = service
        self.started_at = started_at or datetime.now(UTC)
        self._started_monotonic = time.monotonic()

        # 请求/响应
        self.requests = 0
        self.requests_by_source: Counter[str] = Counter()
        self.responses_by_source: Counter[str] = Counter()
        self.responses_by_status: Counter[str] = Counter()
        self.errors = 0
        self.finished_reasons: Counter[str] = Counter()
        self.latency_by_source: dict[str, DurationWindow] = {}

        # 工具
        self.tools_by_name: Counter[str] = Counter()
        self.tools_by_status: Counter[str] = Counter()
        self.tool_latency = DurationWindow()

        # 意图 / 记忆 / token
        self.intents: Counter[str] = Counter()
        self.memory_by_action: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()

    # —— 事件入口（`subscribe_events(registry.observe)`）——

    def observe(self, event: LogEvent) -> None:
        """折叠一条结构化事件（**任何异常都不得外抛**：可观测性不得反噬主链路）。

        `shared.logging._dispatch` 已对监听器异常做兜底告警，这里仍显式防御 ——
        「指标错一行、对话链路跟着断」是本项目反复拒绝的失败方向。
        """
        try:
            handler = _EVENT_HANDLERS.get(event.event)
            if handler is not None:
                handler(self, event)
        except Exception as exc:  # 兜底：指标统计失败不得影响分发链路上的其他监听器
            logger.warning("指标折叠失败（事件=%s，指标缺失不影响主链路）：%s", event.event, exc)

    # —— 各事件的分支处理（按事件名分发，字段口径见 C3 事件定义）——

    def _on_request_received(self, event: LogEvent) -> None:
        self.requests += 1
        self.requests_by_source[str(event.fields.get("source") or "")] += 1

    def _on_request_finished(self, event: LogEvent) -> None:
        source = str(event.fields.get("source") or "")
        self.responses_by_source[source] += 1
        status = event.status or ""
        self.responses_by_status[status] += 1
        # 错误口径：显式 error 终态，或带错误码（`code` 非空）——两者都算失败的那一轮。
        if status == "error" or event.code:
            self.errors += 1
        if event.duration_ms is not None:
            self.latency_by_source.setdefault(source, DurationWindow()).observe(event.duration_ms)
        usage = event.fields.get("tokens")
        if isinstance(usage, dict):
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage.get(name)
                if isinstance(value, int | float):
                    self.tokens[name] += int(value)

    def _on_tool_called(self, event: LogEvent) -> None:
        status = event.status or ""
        name = event.tool_name or ""
        self.tools_by_status[status] += 1
        self.tools_by_name[name] += 1
        if event.duration_ms is not None:
            self.tool_latency.observe(event.duration_ms)

    def _on_intent_classified(self, event: LogEvent) -> None:
        self.intents[str(event.fields.get("intent") or event.status or "")] += 1

    def _on_answer_generated(self, event: LogEvent) -> None:
        # `answer.generated` 的 status 即 finished_reason（见 log_answer_generated）。
        # 空串归一到 `unspecified`：`generate_answer` 的**复用路径**（call_model 已直接产出
        # 文本 → 早返回）不重算终态，事件里 `finished_reason` 可能为空。不归一的话，分布里会
        # 出现一个名为 "" 的桶 —— 看板上读不出它是什么，而它其实是「终态由上游决定」
        # （响应侧此时为 completed）。C4 侧的补全留作后续；这里先保证指标可读。
        reason = str(event.fields.get("finished_reason") or event.status or "")
        self.finished_reasons[reason or UNSPECIFIED] += 1

    def _on_memory_saved(self, event: LogEvent) -> None:
        self.memory_by_action[str(event.fields.get("action") or event.status or "")] += 1

    # —— 快照 ——

    @property
    def uptime_s(self) -> float:
        """进程启动至今秒数（单调钟：不受系统时间调整影响）。"""
        return round(time.monotonic() - self._started_monotonic, 1)

    def snapshot(self) -> dict[str, object]:
        """当前指标快照（直接对应 `MetricsResponse` 字段，避免二次映射）。"""
        agent_latency = self.latency_by_source.get(SOURCE_AGENT)
        http_latency = self.latency_by_source.get(SOURCE_HTTP)
        agent_responses = self.responses_by_source.get(SOURCE_AGENT, 0)
        total_responses = sum(self.responses_by_source.values())
        tools_total = sum(self.tools_by_status.values())
        return {
            "service": self.service,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "uptime_s": self.uptime_s,
            "requests": self.requests,
            "requests_by_source": dict(self.requests_by_source),
            "responses": total_responses,
            "responses_by_source": dict(self.responses_by_source),
            "responses_by_status": dict(self.responses_by_status),
            "errors": self.errors,
            "error_rate": round(self.errors / total_responses, 4) if total_responses else 0.0,
            "finished_reasons": dict(self.finished_reasons),
            # 交互往返（source=agent）是「一轮对话」的真实耗时；HTTP 往返含 SSE 下发，
            # 两者分列（混在一起会把 HTTP 开销算进 Agent 延迟）。
            "latency": agent_latency.stats() if agent_latency else None,
            "http_latency": http_latency.stats() if http_latency else None,
            "interactions": agent_responses,
            "tools": {
                "total": tools_total,
                "by_status": dict(self.tools_by_status),
                "by_name": dict(self.tools_by_name),
                "success_rate": (
                    round(self.tools_by_status.get(STATUS_OK, 0) / tools_total, 4)
                    if tools_total
                    else 0.0
                ),
                "latency": self.tool_latency.stats() if self.tool_latency.count else None,
            },
            "intents": dict(self.intents),
            "tokens": dict(self.tokens),
            "memory": {
                "saved": sum(self.memory_by_action.values()),
                "by_action": dict(self.memory_by_action),
            },
        }


# 事件名 → 处理函数（模块级映射：新增事件只改这一处，`observe` 保持零分支增长）。
_EVENT_HANDLERS = {
    EVENT_REQUEST_RECEIVED: MetricsRegistry._on_request_received,
    EVENT_REQUEST_FINISHED: MetricsRegistry._on_request_finished,
    EVENT_TOOL_CALLED: MetricsRegistry._on_tool_called,
    EVENT_INTENT_CLASSIFIED: MetricsRegistry._on_intent_classified,
    EVENT_ANSWER_GENERATED: MetricsRegistry._on_answer_generated,
    EVENT_MEMORY_SAVED: MetricsRegistry._on_memory_saved,
}

__all__ = [
    "SOURCE_AGENT",
    "SOURCE_HTTP",
    "STATUS_OK",
    "UNSPECIFIED",
    "DurationWindow",
    "MetricsRegistry",
]
