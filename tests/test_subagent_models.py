"""agent/core 子代理数据模型与配置（T7）——离线单测。

覆盖：SubagentResult 默认值 / SubagentConfig 边界约束（ge/le）/
Status.DELEGATING 序列化值 / AgentFrameworkConfig.subagents 装配默认。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.core.config import AgentFrameworkConfig, SubagentConfig
from agent.core.models import SubagentResult
from agent.response.status import Status


def test_subagent_result_defaults() -> None:
    """ok 结果：error 缺省 None、summary/tool_summary 缺省空串（整合渲染按空跳过）。"""
    result = SubagentResult(step_index=0, ok=True)
    assert result.summary == ""
    assert result.error is None
    assert result.tool_summary == ""


def test_subagent_result_failure_carries_error() -> None:
    """失败结果：error 必填语义由调用方保证，模型本身允许携带失败信息。"""
    result = SubagentResult(step_index=1, ok=False, summary="", error="工具执行失败")
    assert result.ok is False
    assert result.error == "工具执行失败"


def test_subagent_config_defaults_match_design_doc() -> None:
    """默认值与 dev-version5.0.md §8.4 一致：enabled=True / max_parallel=3 / 单代理上限 3。"""
    cfg = SubagentConfig()
    assert cfg.enabled is True
    assert cfg.max_parallel == 3
    assert cfg.per_subagent_max_tool_calls == 3


@pytest.mark.parametrize("field", ["max_parallel", "per_subagent_max_tool_calls"])
def test_subagent_config_rejects_zero(field: str) -> None:
    """并行上限/迭代上限下界为 1（ge=1）：0 值直接构造失败。"""
    with pytest.raises(ValidationError):
        SubagentConfig(**{field: 0})


def test_subagent_config_max_parallel_upper_bound() -> None:
    """批内并行上限 le=6：超限构造失败（防无界并发）。"""
    with pytest.raises(ValidationError):
        SubagentConfig(max_parallel=7)


def test_framework_config_aggregates_subagents() -> None:
    """AgentFrameworkConfig 装配 subagents 子配置（get_default 与生产一致）。"""
    cfg = AgentFrameworkConfig.get_default()
    assert isinstance(cfg.subagents, SubagentConfig)
    assert cfg.subagents.enabled is True


def test_delegating_status_serialization_value() -> None:
    """DELEGATING 成员值即序列化字符串（StrEnum，SSE/Checkpointer 往返稳定）。"""
    assert Status.DELEGATING.value == "delegating"
    assert str(Status.DELEGATING) == "delegating"
