"""E2 轨迹 golden set（Phase D 评测）：固定输入 → 期望**执行路径**的离线硬门禁。

WHY 单独立于既有单测：E1（770+ 例单测）验的是「各节点各自对不对」；本文件验的是
「一批**真实会遇到的输入**跑出来的整条轨迹是否符合预期」——意图、工具序列、降级与否、
状态轨迹、引用来源。轨迹级回归是「重构没改坏任何单点、但整体路径变了」这类事故的
唯一防线（例如某次改动让 CHAT 轮悄悄多调一次 LLM：单测全绿，成本却翻倍）。

三类断言（plan §5.5）：
1. **重组断言**：`intent` / 工具名序列 / `finished_reason` / 是否降级；
2. **成本断言**：本轮 LLM 调用次数（token 花钱的显式可见化）；
3. **引用断言**：`citations ⊆ 本次检索集`（护栏不变量 —— 回答里引用的来源必须真的检索过）。

零真 LLM / 零外网：全部经 `tests/conftest.py` 的离线 fake 模型与假工具驱动。
**已知边界**：`ToolResult.citations` 目前全仓库无一处赋值（RAG→引用 的接线尚未实现，
属他人模块），故今天的「本次检索集」实际只来自**长期记忆召回**；本文件的引用断言按
「检索集 = 种子记忆 id ∪ 轨迹里出现过的工具结果 citations」计算，等 RAG 侧接线落地后
该断言自动开始覆盖工具来源，无需改测试。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from agent.core.config import AgentFrameworkConfig, AgentGraphConfig, SubagentConfig
from agent.core.graph import _FALLBACK_GENERIC_TEXT, build_agent_graph
from agent.core.models import PlanResult, PlanStep
from agent.intent.models import Intent, IntentResult
from agent.memory import MemoryStore, wait_pending_saves
from agent.memory.models import MemoryItem
from agent.response.models import (
    FINISHED_REASON_COMPLETED,
    FINISHED_REASON_ERROR,
    FINISHED_REASON_FALLBACK,
    FINISHED_REASON_NEEDS_CLARIFICATION,
    FINISHED_REASON_TOOL_LIMIT,
    ClarifyResult,
)
from agent.response.status import Status
from shared.logging import (
    EVENT_ANSWER_GENERATED,
    EVENT_INTENT_CLASSIFIED,
    EVENT_MEMORY_SAVED,
    EVENT_REQUEST_FINISHED,
    EVENT_TOOL_CALLED,
    RecordingListener,
)
from tests.conftest import (
    RecordingFakeChatModel,
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
    tool_call_messages,
    understand_message,
)

# 未绑定飞书的工具错误前缀（v5.1 §6.3：走确定性绑定引导，而非降级）。
_UNBOUND_MSG = "tool_error.lark_unbound: 用户尚未绑定飞书账号"

# 图内（run_graph 直驱）能观测到的事件集合：**不含** HTTP/runtime 边界与带外保存事件。
_GRAPH_LEVEL_EVENTS = frozenset(
    {EVENT_INTENT_CLASSIFIED, EVENT_TOOL_CALLED, EVENT_ANSWER_GENERATED}
)


# —— 用例模型 ——


@dataclass(frozen=True)
class TraceCase:
    """一条轨迹 golden 用例：输入 + 期望轨迹。"""

    name: str
    text: str
    messages: tuple[AIMessage, ...]
    tools: tuple[Any, ...] = ()
    seeds: tuple[MemoryItem, ...] = ()
    config: AgentFrameworkConfig | None = None
    user_id: str = "anonymous"
    expect_intent: str = Intent.CHAT.value
    expect_tools: tuple[str, ...] = ()
    expect_tool_statuses: tuple[str, ...] | None = None
    expect_reply: str | None = None
    expect_finished_reason: str = FINISHED_REASON_COMPLETED
    expect_degraded: bool = False
    expect_citations: frozenset[str] = frozenset()
    expect_statuses: tuple[str, ...] = (Status.THINKING, Status.SPEAKING)
    expect_clarification: bool = False
    # 是否应产出 `answer.generated` 事件：澄清 / 绑定引导等**确定性路径**不生成回答、
    # 不经 `generate_answer`，故必须为 False（写成 True 会掩盖「路径悄悄退化成 LLM 答」）。
    expect_answer_event: bool = True
    expect_llm_calls: int | None = None
    note: str = ""
    tags: tuple[str, ...] = field(default=())


def _seed(memory_id: str, kind: str, content: str, *, importance: float = 0.6) -> MemoryItem:
    """种子长期记忆（provenance=seed，时间戳取当前时刻）。"""
    return MemoryItem(
        id=memory_id,
        kind=kind,
        content=content,
        session_id="s-seed",
        user_id="anonymous",
        timestamp=datetime.now(UTC),
        provenance="seed",
        importance=importance,
    )


# 上下文：「我上次说…」是**问句** → 通过查询理解门控，故脚本必须含 understand 消息。
# WHY 写明这条坑：漏写它会让回答消息被 `understand_query` 当结构化输出消费掉，
# 末次 call_model 因脚本耗尽而失败 → 终态变 error —— golden set 的「结束原因」断言
# 正是靠这个把「脚本与路径错配」当场暴露出来（而不是悄悄测了个降级路径）。
_FOLLOW_UP_Q = "我上次说我读几年级来着？"


# —— golden 用例表（**输入 → 期望路径**；改行为必须同步改这里，这就是门禁的含义）——

CASES: tuple[TraceCase, ...] = (
    # ① 直答家族
    TraceCase(
        name="chat_greeting_direct",
        text="你好",
        messages=tuple(chat_turn_messages(Intent.CHAT, "你好！")),
        expect_llm_calls=2,  # 意图分类 + 回答
        note="寒暄短输入：确定性门控跳过查询理解（省一次 LLM 调用）",
        tags=("chat", "cost"),
    ),
    TraceCase(
        name="chat_knowledge_no_memory",
        text="什么是投影仪？",
        messages=tuple(chat_turn_messages(Intent.CHAT, "投影仪是一种显示设备", understand=True)),
        expect_llm_calls=3,  # 意图 + 查询理解 + 回答
        note="知识型问题：走查询理解，但长期记忆为空 → 无引用、无 RETRIEVING",
        tags=("chat", "no-citation"),
    ),
    TraceCase(
        name="chat_preference_preloaded",
        text="你好",
        messages=tuple(chat_turn_messages(Intent.CHAT, "好的，我会简洁回答")),
        seeds=(_seed("m-pref-1", "preference", "用户偏好简洁回答", importance=0.9),),
        expect_citations=frozenset({"m-pref-1"}),
        expect_statuses=(Status.THINKING, Status.SPEAKING),  # 预加载不下发 RETRIEVING
        expect_llm_calls=2,
        note="每轮身份预加载：偏好进 citations，但不产生检索状态帧",
        tags=("citation", "memory"),
    ),
    TraceCase(
        name="chat_fact_recall",
        text=_FOLLOW_UP_Q,
        messages=tuple(chat_turn_messages(Intent.CHAT, "你读初中三年级", understand=True)),
        seeds=(_seed("m-fact-1", "fact", "用户是初中三年级学生"),),
        expect_citations=frozenset({"m-fact-1"}),
        expect_statuses=(Status.THINKING, Status.RETRIEVING, Status.SPEAKING),
        expect_llm_calls=3,  # 意图 + 查询理解 + 回答（问句过门控）
        note="按需召回：事实类记忆进 citations，并下发 RETRIEVING",
        tags=("citation", "memory"),
    ),
    TraceCase(
        name="chat_clarify_low_confidence",
        text="就是那个你懂的",
        messages=(
            fake_structured_message(
                IntentResult(intent=Intent.CHAT, confidence=0.3, reason="模糊")
            ),
            understand_message(),
            fake_structured_message(
                ClarifyResult(question="你是想查询 A 还是 B？", options=["查 A", "查 B"])
            ),
        ),
        expect_finished_reason=FINISHED_REASON_NEEDS_CLARIFICATION,
        expect_clarification=True,
        expect_answer_event=False,  # 澄清话术是确定性产出，不经 generate_answer
        expect_statuses=(Status.THINKING, Status.CLARIFYING),
        expect_llm_calls=3,
        note="低置信意图 → 澄清追问（不降级、不给含糊回答）",
        tags=("clarify",),
    ),
    TraceCase(
        name="chat_empty_reply_degraded",
        text="你好",
        messages=tuple(chat_turn_messages(Intent.CHAT, "   ")),
        # 实测口径：空回答 → 固定兜底话术 + finished_reason=error（degraded=True）。
        expect_finished_reason=FINISHED_REASON_ERROR,
        expect_degraded=True,
        expect_reply=_FALLBACK_GENERIC_TEXT,
        note="模型产出全空白 → 确定性兜底话术（degraded=True）且终态 error，不得下发空回答",
        tags=("chat", "degraded"),
    ),
    TraceCase(
        name="chat_reply_whitespace_stripped",
        text="你好",
        messages=tuple(chat_turn_messages(Intent.CHAT, "  你好！  ")),
        expect_reply="你好！",
        note="首尾空白净文本下发（前端不出现「空一行」的诡异气泡）",
        tags=("chat",),
    ),
    TraceCase(
        name="chat_rule_fallback_classification",
        text="你好",
        # 意图结构化输出失败（模型给了纯文本、无 tool_call）→ 确定性规则兜底 CHAT。
        messages=(
            fake_text_message("（模型没按结构化格式输出）"),
            fake_text_message("兜底后照样回答"),
        ),
        expect_reply="兜底后照样回答",
        expect_llm_calls=2,
        note="意图分类结构化失败 → 规则兜底 CHAT（不中断、不降级）",
        tags=("chat", "fallback-classifier"),
    ),
    # ③ 计划家族（v4.0 串行语义：T7 默认开启并行，此处显式钉住串行）——
    TraceCase(
        name="plan_two_steps_serial",
        text="先总结资料再翻译成英文",
        messages=(
            fake_structured_message(
                IntentResult(intent=Intent.PLAN, confidence=0.95, reason="test")
            ),
            understand_message(),
            fake_structured_message(
                PlanResult(
                    summary="先总结资料，再翻译成英文",
                    steps=[
                        PlanStep(goal="总结毕设资料", tool=None),
                        PlanStep(goal="把总结翻译成英文", tool="translate"),
                    ],
                )
            ),
            fake_text_message("资料已总结"),
            AIMessage(content="", tool_calls=[{"name": "translate", "args": {}, "id": "t1"}]),
            fake_text_message("整合后的最终回答"),
        ),
        tools=(make_fake_tool("translate", content="English summary"),),
        config=AgentFrameworkConfig(subagents=SubagentConfig(enabled=False)),
        expect_intent=Intent.PLAN.value,
        expect_tools=("translate",),  # 变换步（tool=None）不产生工具记录
        expect_reply="整合后的最终回答",
        expect_statuses=(Status.THINKING, Status.PLANNING, Status.USING_TOOL, Status.SPEAKING),
        expect_llm_calls=6,  # 意图 + 查询理解 + 规划 + 变换步 + 工具步 + 整合
        note="复合任务：PLANNING → 串行步骤执行 → 整合回答；变换步不占工具轨迹",
        tags=("plan", "cost"),
    ),
    # ② 工具家族
    TraceCase(
        name="tool_single_ok",
        text="现在几点",
        messages=tuple(tool_call_messages([[{"name": "clock", "args": {}, "id": "c1"}]], "十点")),
        tools=(make_fake_tool("clock", content="10:00"),),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("clock",),
        expect_statuses=(Status.THINKING, Status.USING_TOOL, Status.SPEAKING),
        expect_llm_calls=4,  # 意图 + 查询理解 + 选工具 + 终答
        note="单工具成功路径",
        tags=("tool", "cost"),
    ),
    TraceCase(
        name="tool_two_parallel_in_one_batch",
        text="算 3+5 并看时间",
        messages=tuple(
            tool_call_messages(
                [
                    [
                        {"name": "calc", "args": {"expression": "3+5"}, "id": "c1"},
                        {"name": "clock", "args": {}, "id": "c2"},
                    ]
                ],
                "结果是 8，现在 10 点",
            )
        ),
        tools=(make_fake_tool("calc", content="8"), make_fake_tool("clock", content="10:00")),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("calc", "clock"),
        note="同一批两个 tool_calls：轨迹按调用顺序各记一条（并行不丢记录）",
        tags=("tool",),
    ),
    TraceCase(
        name="tool_two_iterations_serial",
        text="连续两次计算",
        messages=tuple(
            tool_call_messages(
                [
                    [{"name": "calc", "args": {"expression": "1+1"}, "id": "c1"}],
                    [{"name": "calc", "args": {"expression": "2+2"}, "id": "c2"}],
                ],
                "两次都算完了",
            )
        ),
        tools=(make_fake_tool("calc", content="ok"),),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("calc", "calc"),
        note="多轮工具循环：两轮都进轨迹",
        tags=("tool",),
    ),
    TraceCase(
        name="tool_execution_error_fallback",
        text="帮我算 1/0",
        messages=tuple(
            tool_call_messages(
                [[{"name": "calc", "args": {"expression": "1/0"}, "id": "c1"}]], "占位"
            )
        ),
        tools=(make_fake_tool("calc", fail_with=RuntimeError("boom")),),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("calc",),
        expect_finished_reason=FINISHED_REASON_FALLBACK,
        expect_degraded=True,
        note="工具执行失败 → 确定性降级（不能让请求失败）",
        tags=("tool", "degraded"),
    ),
    TraceCase(
        name="tool_unknown_hallucinated",
        text="用幽灵工具查一下",
        messages=tuple(tool_call_messages([[{"name": "ghost", "args": {}, "id": "c1"}]], "占位")),
        tools=(make_fake_tool("clock", content="10:00"),),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("ghost",),
        expect_finished_reason=FINISHED_REASON_FALLBACK,
        expect_degraded=True,
        note="模型幻觉工具名 → unknown_tool → 降级（错误码可路由）",
        tags=("tool", "degraded"),
    ),
    TraceCase(
        name="tool_iteration_limit",
        text="连续计算很多次",
        messages=tuple(
            tool_call_messages(
                [
                    [{"name": "calc", "args": {"expression": "1+1"}, "id": "c1"}],
                    [{"name": "calc", "args": {"expression": "2+2"}, "id": "c2"}],
                ],
                "上限回答",
            )
        ),
        tools=(make_fake_tool("calc", content="ok"),),
        config=AgentFrameworkConfig(graph=AgentGraphConfig(max_tool_iterations=2)),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("calc", "calc"),
        expect_finished_reason=FINISHED_REASON_TOOL_LIMIT,
        note="达工具迭代上限 → tool_limit（不再回环；资源上限护栏）",
        tags=("tool", "limit"),
    ),
    TraceCase(
        name="tool_unbound_lark_guides_binding",
        text="看看我的日程",
        messages=(
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="test")
            ),
            understand_message(),
            AIMessage(
                content="",
                tool_calls=[{"name": "lark_calendar", "args": {}, "id": "c1"}],
            ),
            fake_text_message("终答"),
        ),
        tools=(make_fake_tool("lark_calendar", fail_with=RuntimeError(_UNBOUND_MSG)),),
        user_id="user-a",
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("lark_calendar",),
        expect_finished_reason=FINISHED_REASON_NEEDS_CLARIFICATION,
        expect_clarification=True,
        expect_answer_event=False,  # 绑定引导为确定性话术（LLM 序列里的「终答」不得被用上）
        expect_statuses=(Status.THINKING, Status.USING_TOOL, Status.CLARIFYING),
        note="未绑定飞书 → 确定性绑定引导（澄清原语），**不是** fallback —— "
        "把「你还没绑定」说成「我答不上来」是错误降级",
        tags=("tool", "lark", "clarify"),
    ),
    TraceCase(
        name="tool_batch_mixed_success_failure",
        text="同时算 3+5 和看时间",
        messages=tuple(
            tool_call_messages(
                [
                    [
                        {"name": "calc", "args": {"expression": "3+5"}, "id": "c1"},
                        {"name": "clock", "args": {}, "id": "c2"},
                    ]
                ],
                "占位",
            )
        ),
        tools=(
            make_fake_tool("calc", content="8"),
            make_fake_tool("clock", fail_with=RuntimeError("boom")),
        ),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("calc", "clock"),
        expect_tool_statuses=("ok", "error"),  # 同批部分失败：两条记录各自留痕
        expect_finished_reason=FINISHED_REASON_FALLBACK,
        expect_degraded=True,
        note="一批里部分成功部分失败 → 仍走确定性降级，但**轨迹保留成功那条**（不因降级丢证据）",
        tags=("tool", "degraded"),
    ),
    TraceCase(
        name="tool_ok_with_fact_recall",
        text="查一下日程，我上次说我是几年级？",
        messages=tuple(
            tool_call_messages([[{"name": "clock", "args": {}, "id": "c1"}]], "十点，你读初三")
        ),
        tools=(make_fake_tool("clock", content="10:00"),),
        seeds=(_seed("m-fact-2", "fact", "用户是初中三年级学生"),),
        expect_intent=Intent.TOOL_USE.value,
        expect_tools=("clock",),
        expect_citations=frozenset({"m-fact-2"}),
        expect_statuses=(Status.THINKING, Status.RETRIEVING, Status.USING_TOOL, Status.SPEAKING),
        note="工具轮同时召回长期记忆：工具与记忆来源并存于同一轨迹",
        tags=("tool", "citation"),
    ),
)


# —— 断言主体（每条用例跑真图，零真 LLM）——


def _retrieval_set(response: Any, seeds: Sequence[MemoryItem]) -> set[str]:
    """本次检索集 = 种子记忆 id ∪ 轨迹里出现过的工具结果 citations。

    WHY 从轨迹而非「按用例声明」反推：不变量要验的是「回答引用的来源真的被检索过」，
    检索集必须来自**实际执行的证据**（工具结果携带的 citations），而不是测试自己写的期望。
    """
    retrieved = {item.id for item in seeds}
    for record in response.tool_trace:
        result = getattr(record, "result", None)
        for citation in getattr(result, "citations", None) or []:
            retrieved.add(citation.source_id)
    return retrieved


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
async def test_golden_trace(case: TraceCase, make_llm_service: Any, run_graph: Any) -> None:
    """一条 golden 用例：跑真图 → 逐项断言轨迹。"""
    store = InMemoryStore()
    for item in case.seeds:
        await MemoryStore(store).save(item)

    service = make_llm_service(list(case.messages), model_cls=RecordingFakeChatModel)
    graph = build_agent_graph(service, case.config, tools=list(case.tools), store=store)
    listener = RecordingListener().install()
    try:
        response, _chunks = await run_graph(graph, text=case.text, user_id=case.user_id)
    finally:
        listener.uninstall()
    await wait_pending_saves()

    assert response is not None, f"{case.name}: 图未产出 AgentResponse"

    # ① 重组断言：意图
    intents = listener.find(EVENT_INTENT_CLASSIFIED)
    assert len(intents) == 1, f"{case.name}: 意图分类应恰好一次"
    assert intents[0].status == case.expect_intent, f"{case.name}: 意图判定不符（{case.note}）"

    # ① 重组断言：工具序列 / 结束原因 / 降级
    assert tuple(r.tool_name for r in response.tool_trace) == case.expect_tools, (
        f"{case.name}: 工具序列不符（{case.note}）"
    )
    if case.expect_tool_statuses is not None:
        assert tuple(r.status for r in response.tool_trace) == case.expect_tool_statuses, (
            f"{case.name}: 工具成败序列不符"
        )
    if case.expect_reply is not None:
        assert response.reply == case.expect_reply, (
            f"{case.name}: 回答内容不符（{response.reply!r}）"
        )
    assert response.finished_reason == case.expect_finished_reason, f"{case.name}: 结束原因不符"
    answers = listener.find(EVENT_ANSWER_GENERATED)
    if case.expect_answer_event:
        assert answers, f"{case.name}: 缺少 answer.generated 事件"
        assert bool(answers[-1].fields.get("degraded")) is case.expect_degraded, (
            f"{case.name}: 降级标记不符"
        )
    else:
        # 确定性路径（澄清 / 绑定引导）**不得**产出 answer.generated：它意味着回答走了 LLM，
        # 而这两条路径的可用性恰恰依赖「不经 LLM」（LLM 失败时也不许退化成「我答不上来」）。
        assert not answers, f"{case.name}: 确定性路径不应产出 answer.generated"

    # ① 重组断言：状态轨迹顺序与澄清载荷
    statuses = tuple(e.status for e in response.status_trace)
    for expected in case.expect_statuses:
        assert expected in statuses, f"{case.name}: 状态轨迹缺 {expected}（实得 {statuses}）"
    assert (response.clarification is not None) is case.expect_clarification

    # ② 成本断言：LLM 调用次数（新增一次调用 = 成本翻倍，必须显式可见）
    if case.expect_llm_calls is not None:
        assert len(service.chat_model.prompts) == case.expect_llm_calls, (
            f"{case.name}: LLM 调用次数不符（实得 {len(service.chat_model.prompts)}）"
        )

    # ③ 引用断言（护栏不变量）：引用必须真在本次检索集内，且来源可读
    retrieved = _retrieval_set(response, case.seeds)
    cited = {c.source_id for c in response.citations}
    assert cited == set(case.expect_citations), f"{case.name}: 引用集合不符（实得 {cited}）"
    assert cited <= retrieved, (
        f"{case.name}: 引用了本次**未检索到**的来源 {cited - retrieved}（护栏不变量被破坏）"
    )
    for citation in response.citations:
        assert citation.source_id and citation.snippet, f"{case.name}: 引用缺少来源或片段"

    # 事件侧同源：工具调用事件数与轨迹一致（事件与响应两条口径不得分叉）
    assert len(listener.find(EVENT_TOOL_CALLED)) == len(case.expect_tools)
    assert EVENT_INTENT_CLASSIFIED in listener.names()


# —— 用例表自身的守卫（防止 golden set 退化）——


def test_golden_set_shape() -> None:
    """用例集规模与覆盖门槛：golden set 是**门禁**，规模塌陷等于门禁失效。"""
    assert len(CASES) >= 12, "轨迹 golden 用例过少"
    names = [c.name for c in CASES]
    assert len(set(names)) == len(names), "用例名重复（参数化 id 必须唯一）"
    tags = {tag for case in CASES for tag in case.tags}
    # 关键路径必须各有覆盖：直答 / 工具 / 计划 / 降级 / 澄清 / 引用 / 成本 / 上限
    for required in ("chat", "tool", "plan", "degraded", "clarify", "citation", "cost", "limit"):
        assert required in tags, f"golden set 缺少「{required}」类用例"
    assert len(CASES) >= 16, "用例数少于 Phase D 要求的下限（20-30 例的目标区间）"


def test_golden_set_has_citation_and_degraded_coverage() -> None:
    """两类**最贵**的回归必须有对应用例：引用不变量（合规）与降级（可用性）。"""
    assert any(case.expect_citations for case in CASES), "无引用断言用例"
    assert any(case.expect_degraded for case in CASES), "无降级断言用例"
    assert any(case.expect_finished_reason == FINISHED_REASON_FALLBACK for case in CASES)
    assert any(case.expect_finished_reason == FINISHED_REASON_TOOL_LIMIT for case in CASES)
    assert any(case.expect_finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION for case in CASES)


def test_trace_scope_is_graph_only() -> None:
    """轨迹范围**只到图级**（不含 HTTP / runtime 边界事件）。

    WHY 写明：`request.received` / `request.finished` 由中间件与 `AgentRuntime.chat_stream`
    下发，本文件以 `run_graph` 直驱图、不经 HTTP，故**断言里不得出现这两个事件**
    （出现即说明有人把期望写成了「应该有」而不是「实际有」）。这两条边界事件由
    C4 的事件测试与 `/metrics` 测试各自覆盖。
    """
    assert EVENT_MEMORY_SAVED not in _GRAPH_LEVEL_EVENTS  # 带外保存，非图内
    assert EVENT_REQUEST_FINISHED not in _GRAPH_LEVEL_EVENTS
    assert {EVENT_INTENT_CLASSIFIED, EVENT_TOOL_CALLED, EVENT_ANSWER_GENERATED} <= (
        _GRAPH_LEVEL_EVENTS
    )
