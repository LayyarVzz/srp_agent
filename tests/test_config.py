"""agent/core/config.py —— 框架行为配置单测（离线，不依赖 .env / 网络）。"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from agent.core.config import (
    AgentFrameworkConfig,
    DedupConfig,
    LLMBehaviorConfig,
    LLMConfig,
    LLMProvider,
    MemoryBehaviorConfig,
    QueryUnderstandingConfig,
    RetrievalConfig,
)
from agent.share.models import MemoryRecallConfig


def test_framework_defaults() -> None:
    """框架聚合默认值应覆盖编排/LLM/工具/记忆四个面。"""
    fw = AgentFrameworkConfig.get_default()
    assert fw.graph.max_tool_iterations == 3
    assert fw.graph.trim_token_budget == 8000
    assert fw.tools.mcp_timeout_s == 10.0
    assert fw.tools.mcp_max_retries == 2
    assert fw.tools.mcp_max_content_chars == 10_000
    assert fw.memory.top_k == 5
    assert fw.memory.store_type == "in_memory"
    assert fw.memory.preload_profile is True
    # 语义召回混合重排参数（spec 默认值：两套职责权重）。
    assert fw.memory.recall == MemoryRecallConfig()
    assert fw.memory.recall.content_weights == (0.6, 0.25, 0.15)
    assert fw.memory.recall.preference_weights == (0.2, 0.6, 0.2)
    assert fw.memory.recall.recall_fetch_factor == 4
    assert fw.memory.recall.recency_half_life_days == 30.0


def test_deepseek_preset_effective() -> None:
    cfg = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key=SecretStr("sk-x"))
    assert cfg.effective_base_url == "https://api.deepseek.com/v1"
    assert cfg.effective_model == "deepseek-v4-flash"


def test_qwen_preset_effective() -> None:
    cfg = LLMConfig(provider=LLMProvider.QWEN, api_key=SecretStr("sk-x"))
    assert cfg.effective_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert cfg.effective_model == "qwen-plus"


def test_explicit_override_wins_over_preset() -> None:
    """显式 base_url/model 覆盖优先于预设（可指向任意 OpenAI 兼容端点）。"""
    cfg = LLMConfig(
        provider=LLMProvider.QWEN,
        api_key=SecretStr("sk-x"),
        base_url="http://localhost:11434/v1",
        model="llama3",
    )
    assert cfg.effective_base_url == "http://localhost:11434/v1"
    assert cfg.effective_model == "llama3"


def test_structured_extra_body_deepseek_v4() -> None:
    """DeepSeek V4 需关闭思考模式：结构化输出请求附加 thinking=disabled。"""
    cfg = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key=SecretStr("sk-x"))
    assert cfg.structured_extra_body == {"thinking": {"type": "disabled"}}


def test_structured_extra_body_qwen_none() -> None:
    """Qwen 不识别 thinking 参数：一律不附加 extra_body。"""
    cfg = LLMConfig(provider=LLMProvider.QWEN, api_key=SecretStr("sk-x"))
    assert cfg.structured_extra_body is None


def test_structured_extra_body_non_v4_none() -> None:
    """非 V4 模型（deepseek-chat 等）无思考模式限制：不附加 extra_body。"""
    cfg = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key=SecretStr("sk-x"), model="deepseek-chat")
    assert cfg.structured_extra_body is None


def test_structured_extra_body_disabled_flag() -> None:
    """关闭全局 disable_thinking 后不再附加思考模式参数。"""
    cfg = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        api_key=SecretStr("sk-x"),
        disable_thinking=False,
    )
    assert cfg.structured_extra_body is None


def test_openai_compatible_requires_override() -> None:
    """开放端点无预设，缺 base_url/model 时惰性抛错（不影响构造）。"""
    cfg = LLMConfig(provider=LLMProvider.OPENAI_COMPATIBLE, api_key=SecretStr("sk-x"))
    with pytest.raises(ValueError, match="base_url"):
        _ = cfg.effective_base_url
    with pytest.raises(ValueError, match="model"):
        _ = cfg.effective_model


def test_from_runtime_merges_behavior() -> None:
    """from_runtime 用行为默认，可显式覆盖，并注入运行选择。"""
    behavior = LLMBehaviorConfig(temperature=0.2, structured_method="json_mode")
    cfg = LLMConfig.from_runtime(
        provider=LLMProvider.DEEPSEEK,
        api_key="sk-runtime",
        model="deepseek-v4-flash",
        behavior=behavior,
    )
    assert cfg.provider is LLMProvider.DEEPSEEK
    assert cfg.api_key.get_secret_value() == "sk-runtime"
    assert cfg.temperature == 0.2
    assert cfg.structured_method == "json_mode"
    assert cfg.effective_model == "deepseek-v4-flash"


def test_framework_config_ignores_env(monkeypatch) -> None:
    """框架行为配置是纯 Pydantic，绝不读取环境变量。"""
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("LLM_API_KEY", "sk-env")
    cfg = LLMConfig()
    assert cfg.provider is LLMProvider.DEEPSEEK
    assert cfg.api_key.get_secret_value() == ""


def test_recall_config_defaults_and_nested() -> None:
    """MemoryRecallConfig spec 默认；MemoryBehaviorConfig 嵌套同值（单点定义、零重复）。"""
    assert MemoryRecallConfig().model_dump() == {
        "content_weights": (0.6, 0.25, 0.15),
        "preference_weights": (0.2, 0.6, 0.2),
        "recall_fetch_factor": 4,
        "recency_half_life_days": 30.0,
        # v6.0 T2：多查询 RRF 融合平滑常数（单查询路径不使用融合）。
        "rrf_k": 60,
    }
    assert MemoryBehaviorConfig().recall == MemoryRecallConfig()


def test_query_understanding_config_defaults_and_nested() -> None:
    """QueryUnderstandingConfig / RetrievalConfig spec 默认；聚合配置嵌套同值。"""
    assert QueryUnderstandingConfig().model_dump() == {
        "enabled": True,
        "min_query_chars": 2,
        "sub_query_max": 3,
        "synonym_max": 2,
        "max_variant_chars": 200,
        "hypothetical_enabled": True,
        "hyde_max_query_chars": 30,
    }
    assert RetrievalConfig().model_dump() == {
        "multi_query_enabled": True,
        "max_variants": 3,
        "weak_score_threshold": 0.3,
        "multi_query_budget_s": 5.0,
        "rrf_k": 60,
        "override_model_query": True,
    }
    framework = AgentFrameworkConfig()
    assert framework.query_understanding == QueryUnderstandingConfig()
    assert framework.retrieval == RetrievalConfig()


def test_query_understanding_config_validation() -> None:
    """越界值拒绝：子查询/同义上限、变体长度预算、HyDE 门控长度、融合上限与预算。"""
    with pytest.raises(ValidationError):
        QueryUnderstandingConfig(sub_query_max=9)
    with pytest.raises(ValidationError):
        QueryUnderstandingConfig(synonym_max=-1)
    with pytest.raises(ValidationError):
        QueryUnderstandingConfig(max_variant_chars=4)
    with pytest.raises(ValidationError):
        QueryUnderstandingConfig(hyde_max_query_chars=0)
    with pytest.raises(ValidationError):
        RetrievalConfig(max_variants=9)
    with pytest.raises(ValidationError):
        RetrievalConfig(weak_score_threshold=1.5)
    with pytest.raises(ValidationError):
        RetrievalConfig(multi_query_budget_s=0)


def test_dedup_config_defaults_and_nested() -> None:
    """DedupConfig spec 默认（enabled=True、保守阈值 0.92）；MemoryBehaviorConfig 嵌套同值。"""
    assert DedupConfig().model_dump() == {"enabled": True, "semantic_threshold": 0.92}
    assert MemoryBehaviorConfig().dedup == DedupConfig()
    # 阈值越界拒绝（[0,1] 语义相似度区间）。
    with pytest.raises(ValidationError):
        DedupConfig(semantic_threshold=1.5)
    with pytest.raises(ValidationError):
        DedupConfig(semantic_threshold=-0.1)


def test_recall_config_weights_validation() -> None:
    """两套权重的负项都拒绝；权重和不必为 1（线性缩放不改变排序）。"""
    with pytest.raises(ValueError, match=">= 0"):
        MemoryRecallConfig(content_weights=(-0.1, 0.25, 0.15))
    with pytest.raises(ValueError, match=">= 0"):
        MemoryRecallConfig(preference_weights=(-0.1, 0.6, 0.2))
    # 元组长度由类型注解约束：4 元组 → ValidationError。
    with pytest.raises(ValidationError):
        MemoryRecallConfig(content_weights=(0.6, 0.25, 0.15, 0.0))
    # 和不为 1 合法。
    assert MemoryRecallConfig(content_weights=(1.0, 0.0, 0.0)).content_weights == (1.0, 0.0, 0.0)
