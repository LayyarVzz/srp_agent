"""飞书未绑定的确定性引导测试（v5.1 §6.3：`tool_error.lark_unbound` 路由）。

核心断言（这是 v5.1 相对 v5.0 的关键行为差异）：
**「你还没绑定飞书」绝不能走 `fallback_chat`** —— 把「你没绑定」说成
「我答不上来」是错误降级，用户拿不到该做的动作。

覆盖：
- ReAct 路径（plan 为空）与 PLAN 路径都路由到绑定引导；
- 引导话术**不经 LLM**（纯确定性、可复现）；
- `finished_reason=needs_clarification`（澄清原语语义，非 fallback）；
- 未绑定消息里的绑定入口链接被提取并拼进话术（用户必须拿到入口）；
- 其他工具失败**仍**走 fallback_chat（零回归：只有未绑定这一条改道）。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from agent.core.config import AgentFrameworkConfig, SubagentConfig
from agent.core.models import PlanResult, PlanStep
from agent.core.state import NODE_LARK_BIND_GUIDE
from agent.intent.models import Intent, IntentResult
from agent.response.models import FINISHED_REASON_FALLBACK, FINISHED_REASON_NEEDS_CLARIFICATION
from agent.response.status import Status
from agent.tools.lark_scope import is_lark_unbound_message
from agent.tools.models import TOOL_ERROR_LARK_UNBOUND as MODELS_LARK_UNBOUND_CODE
from shared.lark.errors import (
    LARK_UNBOUND_GUIDE,
    LARK_UNBOUND_LINK_LABEL,
    LARK_UNBOUND_PREFIX,
    TOOL_ERROR_LARK_UNBOUND,
    unbound_message,
)
from tests.conftest import (
    fake_structured_message,
    fake_text_message,
    make_fake_tool,
    understand_message,
)

# 一条带绑定入口的未绑定工具消息（服务侧真实产出形态）。
_LINK = "https://accounts.feishu.cn/oauth/v1/device/verify?flow_id=f1&user_code=AB"
_UNBOUND_MSG = unbound_message("未绑定飞书账号（作用域 user-a）", verification_uri_complete=_LINK)

_TOOL_NAME = "lark_calendar_get_agenda"


def _tool_call(name: str = _TOOL_NAME) -> AIMessage:
    """构造一条带工具调用的 AI 消息（模拟 call_model 的产出）。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": "c1", "type": "tool_call"}],
    )


def _graph_with_tool(
    build_graph: Any,
    *,
    error_text: str | None = None,
    content: str = "ok",
) -> Any:
    """构造调用 lark 工具的 ReAct 图（error_text 非空则该工具执行失败）。"""
    tool = make_fake_tool(
        _TOOL_NAME,
        content=content,
        fail_with=RuntimeError(error_text) if error_text is not None else None,
    )
    # 消息序列契约（TOOL_USE 轮）：意图 → 查询理解（v6.0 T2，门控放行时一次结构化调用）
    # → call_model 的工具调用 → 终答文本。
    messages: list[BaseMessage] = [
        fake_structured_message(
            IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="单工具调用")
        ),
        understand_message(),
        _tool_call(),
        fake_text_message("终答"),
    ]
    return build_graph(messages, tools=[tool])


def _plan_intent_message() -> AIMessage:
    """构造 PLAN 意图的分类产出（plan 开启/关闭两条路径共用）。"""
    return fake_structured_message(
        IntentResult(intent=Intent.PLAN, confidence=0.95, reason="复合任务")
    )


def _plan_messages(*, tool_call: bool = True) -> list[BaseMessage]:
    """PLAN 模式一轮的消息序列：意图(PLAN) → 查询理解 → 计划 → （工具调用）→ 终答。"""
    messages: list[BaseMessage] = [
        _plan_intent_message(),
        understand_message(),
        fake_structured_message(
            PlanResult(summary="查日程", steps=[PlanStep(goal="查日程", tool=_TOOL_NAME)])
        ),
    ]
    if tool_call:
        messages.append(_tool_call())
    messages.append(fake_text_message("终答"))
    return messages


def _serial_cfg() -> AgentFrameworkConfig:
    """子代理关闭的框架配置：本文件验证「未绑定 → 引导」的串行语义路径。

    WHY 显式关闭：T7 默认开启并行扇出，而扇出后是 join 汇聚——那条路径同样经
    `route_after_join → _plan_failure_target` 覆盖（路由逻辑共用），此处钉住串行即可。
    """
    return AgentFrameworkConfig(subagents=SubagentConfig(enabled=False))


def _visited(chunks: list[dict[str, Any]]) -> list[str]:
    """从 updates 流还原访问过的节点序列。"""
    return [node for chunk in chunks for node in chunk]


# —— 核心：未绑定 → 确定性引导（非 fallback）——


async def test_unbound_tool_routes_to_binding_guide_not_fallback(build_graph, run_graph) -> None:
    """未绑定 → 绑定引导：finished_reason=needs_clarification，且**不是** fallback。"""
    graph = _graph_with_tool(build_graph, error_text=_UNBOUND_MSG)
    response, chunks = await run_graph(graph, text="看看我的日程", user_id="user-a")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
    assert response.finished_reason != FINISHED_REASON_FALLBACK
    visited = _visited(chunks)
    assert NODE_LARK_BIND_GUIDE in visited
    assert "fallback_chat" not in visited  # 关键：不得降级


async def test_binding_guide_text_is_deterministic_and_actionable(build_graph, run_graph) -> None:
    """引导话术为服务侧常量（含绑定动作）且**带上绑定入口链接**。"""
    graph = _graph_with_tool(build_graph, error_text=_UNBOUND_MSG)
    response, _ = await run_graph(graph, text="看看我的日程", user_id="user-a")

    assert response is not None
    assert LARK_UNBOUND_GUIDE in response.reply
    assert _LINK in response.reply  # 用户必须能直接拿到入口
    assert response.clarification is not None
    assert response.clarification.question == response.reply


async def test_binding_guide_does_not_use_llm_fallback_text(build_graph, run_graph) -> None:
    """引导**不经 LLM**：LLM 序列里的兜底文本不得出现在引导话术里。

    WHY 关键设计断言：若引导走 LLM，则 LLM 失败时会退化成 fallback（「我答不上来」），
    恰好是 §6.3 要禁止的降级。故引导必须是纯确定性路径 —— 这里用「兜底文本不得出现」
    来钉住该性质。
    """
    graph = _graph_with_tool(build_graph, error_text=_UNBOUND_MSG)
    response, _ = await run_graph(graph, text="看看我的日程", user_id="user-a")
    assert response is not None
    assert "终答" not in response.reply
    assert LARK_UNBOUND_GUIDE in response.reply


async def test_binding_guide_emits_clarifying_status(build_graph, run_graph) -> None:
    """引导下发 CLARIFYING 状态事件（前端数字人可感知「需要用户动作」）。"""
    graph = _graph_with_tool(build_graph, error_text=_UNBOUND_MSG)
    response, _ = await run_graph(graph, text="看看我的日程", user_id="user-a")
    assert response is not None
    assert Status.CLARIFYING in [e.status for e in response.status_trace]


async def test_binding_guide_without_link_still_guides(build_graph, run_graph) -> None:
    """未绑定消息没有链接（无进行中的授权流程）→ 仍给完整引导，且不留空标签。"""
    graph = _graph_with_tool(build_graph, error_text=unbound_message("未绑定飞书账号"))
    response, _ = await run_graph(graph, text="看看我的日程", user_id="user-a")
    assert response is not None
    assert LARK_UNBOUND_GUIDE in response.reply
    assert LARK_UNBOUND_LINK_LABEL not in response.reply
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION


# —— 零回归：其他失败仍走既有语义 ——


async def test_other_tool_failure_still_falls_back(build_graph, run_graph) -> None:
    """普通工具失败（非未绑定）→ 仍走 fallback_chat（既有语义零回归）。"""
    graph = _graph_with_tool(build_graph, error_text="连接超时")
    response, chunks = await run_graph(graph, text="看看我的日程", user_id="user-a")

    assert response is not None
    assert response.finished_reason == FINISHED_REASON_FALLBACK
    assert NODE_LARK_BIND_GUIDE not in _visited(chunks)


async def test_successful_tool_call_is_unaffected(build_graph, run_graph) -> None:
    """工具成功 → 正常工具循环，不触发绑定引导（未绑定判定不误伤成功路径）。"""
    graph = _graph_with_tool(build_graph, content='{"events": []}')
    response, chunks = await run_graph(graph, text="看看我的日程", user_id="user-a")
    assert response is not None
    assert response.reply.startswith("终答")
    assert NODE_LARK_BIND_GUIDE not in _visited(chunks)


# —— 识别契约（前缀常量与错误码成对）——


def test_prefix_constant_matches_error_code() -> None:
    """错误码常量与消息前缀必须成对（模型侧与图侧各一份，改一处必须改另一处）。"""
    assert MODELS_LARK_UNBOUND_CODE == TOOL_ERROR_LARK_UNBOUND
    assert _UNBOUND_MSG.startswith(f"{TOOL_ERROR_LARK_UNBOUND}: ")
    assert LARK_UNBOUND_PREFIX == f"{TOOL_ERROR_LARK_UNBOUND}: "


@pytest.mark.parametrize(
    "text",
    ["tool_error.lark_unbound:", "tool_error.lark_unbound: 未绑定"],
)
def test_prefix_only_messages_are_recognized(text: str) -> None:
    """前缀即契约：只要求前缀存在（服务侧文案可变，前缀不可变）。"""
    assert is_lark_unbound_message(text)


def test_ordinary_message_is_not_recognized() -> None:
    """普通错误消息不得被误判为未绑定（否则会把正常失败改道成绑定引导）。"""
    assert not is_lark_unbound_message("普通错误：连接超时")
    assert not is_lark_unbound_message("")


# —— PLAN 路径同样引导（不能只在 ReAct 路径生效）——


async def test_plan_path_also_routes_to_binding_guide(build_graph, run_graph) -> None:
    """PLAN 模式的未绑定 → 绑定引导，且**先于重规划**（重规划解决不了「用户没授权」）。"""
    tool = make_fake_tool(_TOOL_NAME, fail_with=RuntimeError(_UNBOUND_MSG))
    graph = build_graph(_plan_messages(), config=_serial_cfg(), tools=[tool])
    response, chunks = await run_graph(graph, text="帮我安排一下今天的日程", user_id="user-a")

    assert response is not None
    visited = _visited(chunks)
    assert NODE_LARK_BIND_GUIDE in visited, f"PLAN 路径未走绑定引导：{visited}"
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
    assert "replan_task" not in visited  # 不白烧一轮重规划预算


async def test_plan_config_disabled_falls_back_to_react_guide(build_graph, run_graph) -> None:
    """plan 功能关闭（零回归配置）时未绑定同样引导（ReAct 路径已覆盖，此处钉住交叉）。"""
    cfg = AgentFrameworkConfig()
    cfg.plan.enabled = False
    tool = make_fake_tool(_TOOL_NAME, fail_with=RuntimeError(_UNBOUND_MSG))
    graph = build_graph(
        # plan 关闭 → 序列不含计划消息（意图 → 查询理解 → 工具调用 → 终答）。
        [_plan_intent_message(), understand_message(), _tool_call(), fake_text_message("终答")],
        config=cfg,
        tools=[tool],
    )
    response, chunks = await run_graph(graph, text="看看我的日程", user_id="user-a")

    assert response is not None
    assert NODE_LARK_BIND_GUIDE in _visited(chunks)
    assert response.finished_reason == FINISHED_REASON_NEEDS_CLARIFICATION
