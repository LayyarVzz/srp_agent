"""v4.0 T2 澄清式追问 —— 模型 / 配置 / 触发判定 单测（离线，确定性）。

覆盖（dev-version4.0.md §4.1–§4.4、§8）：
- `ClarifyResult` / `Clarification` / `AgentResponse.clarification` 契约与默认值；
- `Status.CLARIFYING` / `FINISHED_REASON_NEEDS_CLARIFICATION` / `TOOL_ERROR_MISSING_ARGUMENT` 常量；
- `ClarifyConfig` 默认值与边界校验；
- 触发源纯函数判定：① 低置信（严格小于、规则兜底 0.5 不触发、已追问过不触发、关闭不触发）；
  ② 参数缺失（enabled / clarify_asked / 错误码三个条件）；
- 参数缺失信号检测（`_is_tool_invocation_error`）与降级话术映射。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.core.config import AgentFrameworkConfig, ClarifyConfig
from agent.core.graph import (
    _degraded_fallback_text,
    _is_tool_invocation_error,
    _should_clarify_low_confidence,
    _should_clarify_missing_argument,
)
from agent.core.state import AgentState
from agent.errors import ErrorRecord
from agent.intent.models import IntentResult
from agent.response.models import (
    FINISHED_REASON_NEEDS_CLARIFICATION,
    AgentResponse,
    Clarification,
    ClarifyResult,
)
from agent.response.status import Status
from agent.tools.models import (
    TOOL_ERROR_EXECUTION,
    TOOL_ERROR_MISSING_ARGUMENT,
    ToolError,
)

# —— 模型契约 ——


def test_clarify_result_defaults() -> None:
    """ClarifyResult 可选字段：options 默认空列表（纯开放反问）。"""
    result = ClarifyResult(question="你是想查 A 还是 B？")
    assert result.options == []


def test_clarification_defaults_and_roundtrip() -> None:
    """Clarification 响应契约：字段 + model_dump 往返（checkpointer msgpack 兼容）。"""
    clar = Clarification(question="你想查哪个？", options=["天气", "新闻"])
    assert clar.options == ["天气", "新闻"]
    assert Clarification.model_validate(clar.model_dump()) == clar


def test_agent_response_clarification_field() -> None:
    """AgentResponse 新增 clarification 字段：默认 None，显式赋值可往返。"""
    resp = AgentResponse(session_id="s1", reply="question")
    assert resp.clarification is None
    resp2 = AgentResponse(
        session_id="s1",
        reply="question",
        finished_reason=FINISHED_REASON_NEEDS_CLARIFICATION,
        clarification=Clarification(question="question", options=["A"]),
    )
    assert resp2.clarification is not None
    assert resp2.clarification.question == "question"
    assert AgentResponse.model_validate(resp2.model_dump()) == resp2


# —— 常量 ——


def test_status_clarifying_constant() -> None:
    assert Status.CLARIFYING == "clarifying"


def test_finished_reason_needs_clarification_constant() -> None:
    assert FINISHED_REASON_NEEDS_CLARIFICATION == "needs_clarification"


def test_missing_argument_error_code_constant() -> None:
    assert TOOL_ERROR_MISSING_ARGUMENT == "tool_error.missing_argument"


# —— ClarifyConfig ——


def test_clarify_config_defaults() -> None:
    """spec 默认：enabled / min_confidence=0.5（严格小于）/ max_asks_per_turn=1。"""
    cfg = ClarifyConfig()
    assert cfg.enabled is True
    assert cfg.min_confidence == 0.5
    assert cfg.max_asks_per_turn == 1
    # 聚合配置嵌套同值（单点定义）。
    assert AgentFrameworkConfig().clarify == ClarifyConfig()


def test_clarify_config_validation() -> None:
    """边界校验：置信度区间 [0,1]、追问上限 ≥1。"""
    with pytest.raises(ValidationError):
        ClarifyConfig(min_confidence=1.5)
    with pytest.raises(ValidationError):
        ClarifyConfig(min_confidence=-0.1)
    with pytest.raises(ValidationError):
        ClarifyConfig(max_asks_per_turn=0)


# —— 触发源①：意图低置信（纯函数判定） ——


def _low_conf_state(confidence: float, *, asked: bool = False) -> AgentState:
    return {
        "intent_meta": IntentResult(intent="chat", confidence=confidence, reason="test"),
        "clarify_asked": asked,
    }


def test_low_confidence_triggers_below_threshold() -> None:
    """置信度严格小于阈值（0.4 < 0.5）→ 触发。"""
    assert _should_clarify_low_confidence(_low_conf_state(0.4), enabled=True, min_confidence=0.5)


def test_low_confidence_boundary_not_triggered() -> None:
    """等于阈值（0.5）不触发：规则兜底 CHAT=0.5 恰不反问（§4.1 注释）。"""
    assert not _should_clarify_low_confidence(
        _low_conf_state(0.5), enabled=True, min_confidence=0.5
    )


def test_low_confidence_high_confidence_not_triggered() -> None:
    assert not _should_clarify_low_confidence(
        _low_conf_state(0.9), enabled=True, min_confidence=0.5
    )


def test_low_confidence_applies_to_any_intent() -> None:
    """触发源①对任意意图生效（v4.0 §4.1 扩展）：模糊的 TOOL_USE/PLAN 请求同样反问。

    判定只看 `intent_meta.confidence`，与意图值无关——规则兜底 TOOL_USE=0.6 仍不触发。
    """
    for intent_value in ("chat", "tool_use", "plan"):
        state: AgentState = {
            "intent_meta": IntentResult(intent=intent_value, confidence=0.3, reason="模糊"),
        }
        assert _should_clarify_low_confidence(state, enabled=True, min_confidence=0.5), (
            f"intent={intent_value} 低置信应触发澄清"
        )


def test_low_confidence_disabled_not_triggered() -> None:
    """澄清关闭（enabled=False）→ 不触发（零回归）。"""
    assert not _should_clarify_low_confidence(
        _low_conf_state(0.3), enabled=False, min_confidence=0.5
    )


def test_low_confidence_already_asked_not_triggered() -> None:
    """本轮已追问过（clarify_asked）→ 不触发（防澄清循环 §4.3）。"""
    assert not _should_clarify_low_confidence(
        _low_conf_state(0.3, asked=True), enabled=True, min_confidence=0.5
    )


def test_low_confidence_missing_meta_not_triggered() -> None:
    """intent_meta 缺失（防御）→ 不触发。"""
    assert not _should_clarify_low_confidence({}, enabled=True, min_confidence=0.5)


# —— 触发源②：工具参数缺失（纯函数判定） ——


def _missing_arg_state(*, asked: bool = False) -> AgentState:
    return {
        "error": ErrorRecord(code=TOOL_ERROR_MISSING_ARGUMENT, message="zone: Field required"),
        "clarify_asked": asked,
    }


def test_missing_argument_triggers() -> None:
    assert _should_clarify_missing_argument(_missing_arg_state(), enabled=True)


def test_missing_argument_other_error_code_not_triggered() -> None:
    """非 missing_argument 错误码（execution）→ 不触发（仍走 fallback 降级）。"""
    state: AgentState = {"error": ErrorRecord(code=TOOL_ERROR_EXECUTION, message="boom")}
    assert not _should_clarify_missing_argument(state, enabled=True)


def test_missing_argument_disabled_not_triggered() -> None:
    assert not _should_clarify_missing_argument(_missing_arg_state(), enabled=False)


def test_missing_argument_already_asked_not_triggered() -> None:
    """已追问过 → 不二次追问（转 fallback_chat，§4.3）。"""
    assert not _should_clarify_missing_argument(_missing_arg_state(asked=True), enabled=True)


def test_missing_argument_no_error_not_triggered() -> None:
    assert not _should_clarify_missing_argument({}, enabled=True)


# —— 参数缺失信号检测 ——


def test_is_tool_invocation_error_positive() -> None:
    """langgraph ToolNode 对参数校验失败的 ToolMessage content 形态（实测）。"""
    content = (
        "Error: ToolInvocationError(\"Error invoking tool 'clock' with kwargs {} "
        'with error:\\n zone: Field required\\n Please fix the error and try again.")\n'
        " Please fix your mistakes."
    )
    assert _is_tool_invocation_error(content)


def test_is_tool_invocation_error_negative_execution() -> None:
    """执行失败（Error executing tool）不误判为参数缺失。"""
    content = "Error executing tool 'fetch' with kwargs {'url': 'x'} with error:\n boom"
    assert not _is_tool_invocation_error(content)


def test_is_tool_invocation_error_negative_normal() -> None:
    """正常工具返回内容不误判。"""
    assert not _is_tool_invocation_error("ok 结果")


# —— 降级话术映射 ——


def test_degraded_fallback_text_missing_argument_uses_tool_error_text() -> None:
    """missing_argument 的确定性兜底用「工具错误」话术（请求因参数缺失未完成）。"""
    state: AgentState = {
        "error": ToolError(code=TOOL_ERROR_MISSING_ARGUMENT, message="zone required")
    }
    text = _degraded_fallback_text(state)
    assert "工具调用出错" in text
