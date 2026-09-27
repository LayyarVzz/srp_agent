"""`/metrics` 轻量指标单测（离线）：事件折叠口径 + 路由 + 有界内存。

口径（plan §5.4）：指标名与 C3 结构化事件字段**同名同源** —— 指标只由事件总线折叠而来，
不额外埋点；因此本文件的用例都是「喂事件 → 断言计数」，而不是打桩业务函数。
"""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.metrics import SOURCE_AGENT, SOURCE_HTTP, DurationWindow, MetricsRegistry
from shared.logging import (
    EVENT_ANSWER_GENERATED,
    EVENT_INTENT_CLASSIFIED,
    EVENT_MEMORY_SAVED,
    EVENT_REQUEST_FINISHED,
    EVENT_REQUEST_RECEIVED,
    EVENT_TOOL_CALLED,
    LogEvent,
)

# —— 事件构造（字段口径与 C3 各便捷函数一致）——


def _received(*, source: str = SOURCE_AGENT) -> LogEvent:
    return LogEvent(event=EVENT_REQUEST_RECEIVED, status="received", fields={"source": source})


def _finished(
    *,
    source: str = SOURCE_AGENT,
    status: str = "completed",
    code: str | None = None,
    duration_ms: int | None = 100,
    tokens: dict[str, int] | None = None,
) -> LogEvent:
    fields: dict[str, object] = {"source": source}
    if tokens is not None:
        fields["tokens"] = tokens
    return LogEvent(
        event=EVENT_REQUEST_FINISHED,
        status=status,
        code=code,
        duration_ms=duration_ms,
        fields=fields,
    )


def _tool(name: str = "clock", *, status: str = "ok", duration_ms: int | None = 20) -> LogEvent:
    return LogEvent(
        event=EVENT_TOOL_CALLED,
        status=status,
        tool_name=name,
        duration_ms=duration_ms,
        fields={},
    )


# —— 折叠口径 ——


def test_folds_request_lifecycle() -> None:
    """请求/响应/错误/终态/延迟/token 各归各的口径。"""
    registry = MetricsRegistry(service="api")
    for _ in range(3):
        registry.observe(_received())
    registry.observe(_finished(duration_ms=120, tokens={"input_tokens": 10, "total_tokens": 12}))
    registry.observe(
        _finished(duration_ms=370, tokens={"input_tokens": 5, "total_tokens": 7}),
    )
    registry.observe(_finished(status="error", code="llm_error.request", duration_ms=50))

    snap = registry.snapshot()
    assert snap["requests"] == 3
    assert snap["requests_by_source"] == {SOURCE_AGENT: 3}
    assert snap["responses"] == 3
    assert snap["errors"] == 1  # 显式 error 终态
    assert snap["responses_by_status"] == {"completed": 2, "error": 1}
    assert snap["error_rate"] == round(1 / 3, 4)
    latency = snap["latency"]
    assert latency["count"] == 3
    assert latency["max_ms"] == 370
    assert latency["avg_ms"] == 180.0
    assert snap["tokens"] == {"input_tokens": 15, "total_tokens": 19}


def test_error_without_error_status_still_counted() -> None:
    """带错误码但终态不是 error 的那一轮也算失败（`code` 非空即失败）。"""
    registry = MetricsRegistry()
    registry.observe(_finished(status="completed", code="guardrail_error.output"))
    assert registry.snapshot()["errors"] == 1


def test_agent_and_http_latency_are_separated() -> None:
    """交互往返（agent）与 HTTP 往返（http）**分列**：混在一起会把 SSE 下发算进 Agent 延迟。"""
    registry = MetricsRegistry()
    registry.observe(_finished(source=SOURCE_AGENT, duration_ms=200))
    registry.observe(_finished(source=SOURCE_HTTP, duration_ms=900))
    snap = registry.snapshot()
    assert snap["latency"]["max_ms"] == 200
    assert snap["http_latency"]["max_ms"] == 900
    assert snap["interactions"] == 1  # 只有 agent 那一轮算「一次交互」
    assert snap["responses"] == 2  # 但失败率的分母是全部收尾


def test_folds_tools_intents_answers_memory() -> None:
    """工具成功率 / 意图分布 / finished_reason 分布 / 记忆动作分布。"""
    registry = MetricsRegistry()
    registry.observe(_tool("clock", duration_ms=10))
    registry.observe(_tool("clock", duration_ms=30))
    registry.observe(_tool("search_knowledge", status="error", duration_ms=90))
    registry.observe(
        LogEvent(event=EVENT_INTENT_CLASSIFIED, status="tool_use", fields={"intent": "tool_use"})
    )
    registry.observe(
        LogEvent(
            event=EVENT_ANSWER_GENERATED,
            status="fallback",
            fields={"finished_reason": "fallback"},
        )
    )
    registry.observe(
        LogEvent(event=EVENT_MEMORY_SAVED, status="inserted", fields={"action": "inserted"})
    )

    snap = registry.snapshot()
    assert snap["tools"]["total"] == 3
    assert snap["tools"]["by_name"] == {"clock": 2, "search_knowledge": 1}
    assert snap["tools"]["by_status"] == {"ok": 2, "error": 1}
    assert snap["tools"]["success_rate"] == round(2 / 3, 4)
    assert snap["tools"]["latency"]["max_ms"] == 90
    assert snap["intents"] == {"tool_use": 1}
    assert snap["finished_reasons"] == {"fallback": 1}
    assert snap["memory"] == {"saved": 1, "by_action": {"inserted": 1}}


def test_unknown_event_is_ignored() -> None:
    """未知事件名不计数也不报错（新增事件先上线、指标后跟进时不得炸）。"""
    registry = MetricsRegistry()
    registry.observe(LogEvent(event="something.new", fields={}))
    assert registry.snapshot()["requests"] == 0


def test_observe_never_raises_on_broken_payload() -> None:
    """折叠过程出错**不得外抛**（可观测性不得反噬主链路），且不影响后续事件折叠。"""
    registry = MetricsRegistry()

    class _Boom:
        """字段访问即抛错的伪事件（模拟事件结构漂移 / 字段类型异常）。"""

        event = EVENT_REQUEST_RECEIVED

        @property
        def fields(self) -> dict[str, object]:
            raise RuntimeError("fields 炸了")

    registry.observe(_Boom())  # type: ignore[arg-type]
    registry.observe(_received())  # 后续合法事件照常折叠
    assert registry.snapshot()["requests"] == 2


def test_all_c3_events_are_covered() -> None:
    """**口径守卫**：C3 的每个结构化事件都必须有明确归属（计数或显式忽略）。

    WHY：新增事件（如 v5.1 的绑定引导）若悄悄不进指标，看板会「少了那一类」而没有任何
    报错；这条断言把「遗漏」变成红灯，逼一次显式决策。
    """
    from app.metrics import _EVENT_HANDLERS

    c3_events = {
        EVENT_REQUEST_RECEIVED,
        EVENT_INTENT_CLASSIFIED,
        EVENT_TOOL_CALLED,
        EVENT_ANSWER_GENERATED,
        EVENT_MEMORY_SAVED,
        EVENT_REQUEST_FINISHED,
    }
    assert set(_EVENT_HANDLERS) == c3_events


# —— 有界内存与分位数 ——


def test_duration_window_is_bounded_but_totals_stay_exact() -> None:
    """样本窗口有上限（长跑不涨内存），而 count/avg/max 保持**全局精确**。"""
    window = DurationWindow(maxlen=10)
    for value in range(100):
        window.observe(value)
    assert window.count == 100
    assert window.max_ms == 99  # 历史最慢不因出窗而消失
    assert window.avg_ms == round(sum(range(100)) / 100, 1)
    assert window.stats()["window"] == 10  # 百分位只用最近 10 条


def test_percentile_uses_window_samples() -> None:
    """百分位取窗口内最近邻值（不插值：指标要的是实际观测到的那个值）。"""
    window = DurationWindow(maxlen=100)
    for value in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        window.observe(value)
    assert window.percentile(50) == 50.0
    assert window.percentile(95) == 100.0


def test_duration_window_defends_against_negative() -> None:
    """负耗时防御性归零（脏输入不得让「最慢」变成负数、把看板带偏）。"""
    window = DurationWindow()
    window.observe(-5)
    assert window.max_ms == 0
    assert window.avg_ms == 0.0


def test_empty_window_stats() -> None:
    """零样本时百分位/均值为 0（响应里 `latency` 另行置 None，见 snapshot）。"""
    assert DurationWindow().percentile(95) == 0.0
    assert MetricsRegistry().snapshot()["latency"] is None


# —— 路由 ——


async def test_metrics_route_returns_snapshot() -> None:
    """`GET /metrics` 下发当前进程指标（读内存计数，不查库）。"""
    app = create_app()
    app.state.metrics.observe(_received())
    app.state.metrics.observe(_finished(duration_ms=42))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == "api"
    assert body["requests"] == 1
    assert body["responses"] == 1
    assert body["latency"]["max_ms"] == 42
    assert body["pid"] > 0
    assert body["uptime_s"] >= 0


async def test_metrics_route_on_fresh_app() -> None:
    """全新进程：计数为 0、分布为空、延迟为 None（而不是让前端拿到 null 报错）。"""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics")
    body = resp.json()
    assert resp.status_code == 200
    assert body["requests"] == 0
    assert body["latency"] is None
    assert body["tools"]["success_rate"] == 0.0
    assert body["memory"] == {"saved": 0, "by_action": {}}


def test_create_app_does_not_subscribe_at_construction() -> None:
    """订阅只发生在 lifespan（构造期不订阅）。

    WHY 单列：监听器注册表是**进程级全局**的 —— 构造期订阅会让「建了 app 但没跑 lifespan」
    的场景（测试、多 worker fork）留下永久订阅，事件随后写进已废弃的注册表。
    """
    from shared.logging import listener_count

    before = listener_count()
    create_app()
    assert listener_count() == before
