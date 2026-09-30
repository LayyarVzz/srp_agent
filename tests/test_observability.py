"""观测（Phase D）单测：配置面 + 观测底座。

**全程离线**：不连 Langfuse Cloud（CI 无凭据、无外网）。凡需要 handler 的用例一律注入
fake（`FakeCallbackHandler` 或直接注入工厂），真 SDK 只用于「类型/构造」这类零网络断言。

口径（与 plan-docker-observability.md §5 一致）：
- 未配置 `LANGFUSE_*` → `Observability` 为 `None`，图 config **逐字等于 C4 的 `[usage]`**
  （观测零贡献；注意文档原话「不含 callbacks 键」已被 C4 的 token 累计 handler 改写）；
- 配置齐备 → 归因/掩码/关闭路径均可离线断言；
- 观测报错**不得冒泡**进对话链路（fail-open）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from pydantic import SecretStr, ValidationError

from agent.intent.models import Intent
from settings import RuntimeSettings
from shared.logging import MASKED
from shared.observability import (
    _ENV_KEYS,
    DEFAULT_BASE_URL,
    FULLY_MASKED,
    OUT_OF_BAND_TRACE_NAME,
    TRACE_NAME,
    Observability,
    ObservabilityConfig,
    install_observability,
    mask_payload,
    reset_observability,
)
from tests.conftest import chat_turn_messages, make_fake_tool, tool_call_messages

# —— 测试替身（**禁止真连云端**：CI 无凭据、无外网）——


class FakeHandler(BaseCallbackHandler):
    """假 langchain 回调：只证明「挂上去了/没挂」，不产生 span、不开后台线程。"""


class FakeClient:
    """假 SDK 客户端：记录 flush/shutdown 调用，便于断言关闭路径的健壮性。"""

    def __init__(
        self, *, flush_error: Exception | None = None, shutdown_error: Exception | None = None
    ) -> None:
        self.flush_calls = 0
        self.shutdown_calls = 0
        self._flush_error = flush_error
        self._shutdown_error = shutdown_error

    def flush(self) -> None:
        self.flush_calls += 1
        if self._flush_error is not None:
            raise self._flush_error

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self._shutdown_error is not None:
            raise self._shutdown_error


def _fake_handler_factory(_config: ObservabilityConfig) -> BaseCallbackHandler:
    return FakeHandler()


def _enabled_settings(**overrides: Any) -> RuntimeSettings:
    """配置齐备的 settings（假凭据；不连任何端点）。"""
    values: dict[str, Any] = {
        "langfuse_enabled": True,
        "langfuse_public_key": SecretStr("pk-lf-test"),
        "langfuse_secret_key": SecretStr("sk-lf-test"),
    }
    values.update(overrides)
    return RuntimeSettings(_env_file=None, **values)


@pytest.fixture
def clean_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 `os.environ` 里的观测变量（env 投影是**进程级副作用**，用例必须可回滚）。

    `monkeypatch.delenv` 会记录原值，并在用例结束时连**用例中途写入的值**一起还原。
    """
    for env_name, _ in _ENV_KEYS:
        monkeypatch.delenv(env_name, raising=False)


@pytest.fixture(autouse=True)
def _reset_observability_singleton() -> Any:
    """进程级单例逐用例重置（否则前一个用例的装配会串到后一个）。"""
    reset_observability()
    yield
    reset_observability()


# —— 配置面（settings.py ↔ LANGFUSE_*）——


def test_langfuse_defaults_off() -> None:
    """默认关闭：未配置即零回调（观测不得成为对话链路前置条件）。"""
    s = RuntimeSettings(_env_file=None)
    assert s.langfuse_enabled is False
    assert s.langfuse_public_key.get_secret_value() == ""
    assert s.langfuse_secret_key.get_secret_value() == ""
    assert s.langfuse_base_url == "https://cloud.langfuse.com"  # EU 区域默认
    assert s.langfuse_environment is None  # 缺省对齐 ENVIRONMENT（解析在观测层）
    assert s.langfuse_release is None
    assert s.langfuse_sample_rate == 1.0
    assert s.langfuse_timeout_s == 5
    assert s.langfuse_flush_timeout_s == 5.0


def test_langfuse_env_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LANGFUSE_*` 环境变量可完整注入（pydantic-settings 大小写不敏感映射）。"""
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "dev")
    monkeypatch.setenv("LANGFUSE_RELEASE", "srp-agent@abc1234")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("LANGFUSE_TIMEOUT", "9")
    s = RuntimeSettings(_env_file=None)
    assert s.langfuse_enabled is True
    assert s.langfuse_public_key.get_secret_value() == "pk-lf-test"
    assert s.langfuse_base_url == "https://us.cloud.langfuse.com"
    assert s.langfuse_environment == "dev"
    assert s.langfuse_release == "srp-agent@abc1234"
    assert s.langfuse_sample_rate == 0.25
    assert s.langfuse_timeout_s == 9


def test_langfuse_secret_keys_masked() -> None:
    """密钥是 SecretStr：repr/str 不得出现明文（与 LLM_API_KEY 同口径）。"""
    s = RuntimeSettings(langfuse_secret_key=SecretStr("sk-lf-very-secret"))
    assert "sk-lf-very-secret" not in str(s.langfuse_secret_key)
    assert "sk-lf-very-secret" not in repr(s.langfuse_secret_key)


def test_blank_env_means_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """空串等同未配置。

    WHY 单列一条：compose 里 `${LANGFUSE_TRACING_ENVIRONMENT:-}` 展开就是空串 ——
    不归一化会让「宿主没配 .env」的容器在启动期校验失败（默认关闭的功能反而把服务搞挂）。
    """
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "")
    monkeypatch.setenv("LANGFUSE_RELEASE", "   ")
    s = RuntimeSettings(_env_file=None)
    assert s.langfuse_environment is None
    assert s.langfuse_release is None


@pytest.mark.parametrize("bad", ["Langfuse-prod", "PROD", "prod env", "a" * 41])
def test_invalid_environment_rejected_at_startup(bad: str) -> None:
    """环境标记不合法 → 启动即失败。

    WHY 不交给 SDK：SDK 只丢弃并告警（trace 照发但少标记，云端分不清哪套部署 = 静默失真）。
    """
    with pytest.raises(ValidationError):
        RuntimeSettings(_env_file=None, langfuse_environment=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_sample_rate_out_of_range_rejected(bad: float) -> None:
    """采样率越界 → 启动即失败（与 SDK 的 ValueError 同口径，但更早、更可读）。"""
    with pytest.raises(ValidationError):
        RuntimeSettings(_env_file=None, langfuse_sample_rate=bad)


# —— 观测底座：装配门槛（未配置 = 零回调）——


def test_disabled_returns_none() -> None:
    """未开观测 → `None`（不建客户端、不写 env、不给图挂回调）。"""
    assert Observability.from_settings(RuntimeSettings(_env_file=None)) is None


def test_enabled_without_keys_returns_none() -> None:
    """开了开关但缺凭据 → 仍 `None`。

    WHY 值得单测：SDK 对缺 key 的行为是「禁用客户端 + 告警、不抛错」，会留下一个
    静默不工作的观测面；本项目把它归到「未配置」，与零回路口径一致。
    """
    assert Observability.from_settings(_enabled_settings(langfuse_secret_key=SecretStr(""))) is None


def test_env_projection_names_match_sdk_constants() -> None:
    """投影用的 env 名必须与 SDK 自己的常量**逐字相同**。

    WHY 对着 SDK 校验而不是拷一份字符串：变量名拼错是本功能最隐蔽的失败 ——
    SDK 读不到就当默认值，观测「看起来开着、实际发错环境/发错区域」，没有报错可查。
    """
    from langfuse._client import environment_variables as sdk_env

    sdk_names = {getattr(sdk_env, name) for name in dir(sdk_env) if name.isupper()}
    for env_name, _ in _ENV_KEYS:
        assert env_name in sdk_names, f"{env_name} 不是 SDK 认识的环境变量名"


def test_from_settings_resolves_environment_and_release(clean_langfuse_env: None) -> None:
    """环境标记缺省对齐 `ENVIRONMENT`；release 缺省 `app_name@<env>`（显式给值免串味 CI）。"""
    config = ObservabilityConfig.from_settings(_enabled_settings())
    assert config is not None
    assert config.environment == "dev"
    assert config.release == "srp-agent@dev"
    assert config.base_url == DEFAULT_BASE_URL

    explicit = ObservabilityConfig.from_settings(
        _enabled_settings(langfuse_environment="test", langfuse_release="build-42")
    )
    assert explicit is not None
    assert explicit.environment == "test"
    assert explicit.release == "build-42"


def test_blank_base_url_falls_back_to_default(clean_langfuse_env: None) -> None:
    """`LANGFUSE_BASE_URL=` 空串（compose `${VAR:-}` 展开）→ 回默认区域端点，不是空端点。"""
    config = ObservabilityConfig.from_settings(_enabled_settings(langfuse_base_url="  "))
    assert config is not None
    assert config.base_url == DEFAULT_BASE_URL


def test_env_projected_before_client_build(clean_langfuse_env: None) -> None:
    """**投影必须发生在建客户端之前**，且值取自 settings（业务代码不读 env）。"""
    seen: dict[str, str | None] = {}

    def _client_factory(_config: ObservabilityConfig) -> FakeClient:
        seen["public_key"] = __import__("os").environ.get("LANGFUSE_PUBLIC_KEY")
        seen["enabled"] = __import__("os").environ.get("LANGFUSE_TRACING_ENABLED")
        seen["environment"] = __import__("os").environ.get("LANGFUSE_TRACING_ENVIRONMENT")
        seen["host"] = __import__("os").environ.get("LANGFUSE_HOST")
        return FakeClient()

    obs = Observability.from_settings(
        _enabled_settings(),
        handler_factory=_fake_handler_factory,
        client_factory=_client_factory,
    )
    assert obs is not None
    assert seen["public_key"] == "pk-lf-test"
    assert seen["enabled"] == "true"
    assert seen["environment"] == "dev"
    # 废弃旧名刻意不投影：同设两者时 SDK 静默取 BASE_URL，投影它反而制造误判空间。
    assert seen["host"] is None


def test_describe_contains_no_secret() -> None:
    """装配摘要可进日志，但绝不含密钥明文。"""
    config = ObservabilityConfig.from_settings(_enabled_settings())
    assert config is not None
    described = config.describe()
    assert "pk-lf-test" not in described
    assert "sk-lf-test" not in described


# —— 观测底座：回调 / 归因 / 掩码 / 关闭 ——


def test_callbacks_empty_after_close() -> None:
    """aclose 后不再产出回调（fail-safe：不返回绑在已关闭客户端上的 handler）。"""
    obs = Observability(
        ObservabilityConfig(enabled=True),
        client=FakeClient(),
        handler_factory=_fake_handler_factory,
    )
    assert len(obs.langchain_callbacks()) == 1
    assert all(isinstance(h, FakeHandler) for h in obs.langchain_callbacks())


async def test_callbacks_new_per_request_and_empty_when_closed() -> None:
    """**每请求新建** handler（并发多会话下复用同一 handler 会串 run 状态）。"""
    client = FakeClient()
    obs = Observability(
        ObservabilityConfig(enabled=True), client=client, handler_factory=_fake_handler_factory
    )
    first, second = obs.langchain_callbacks()[0], obs.langchain_callbacks()[0]
    assert first is not second

    await obs.aclose()
    assert obs.closed is True
    assert obs.langchain_callbacks() == []


def test_trace_metadata_keys() -> None:
    """归因键名是 **SDK 契约**（写错等于没有归因，且不会有任何报错）。"""
    obs = Observability(ObservabilityConfig(enabled=True), client=FakeClient())
    metadata = obs.trace_metadata(session_id="s-1", user_id="u-1", trace_id="req_abc")
    assert metadata["langfuse_session_id"] == "s-1"
    assert metadata["langfuse_user_id"] == "u-1"
    assert metadata["langfuse_trace_name"] == TRACE_NAME
    # 本地 trace_id 关联键：不带 langfuse_ 前缀（它是普通 span metadata，不是归因指令）
    assert metadata["request_trace_id"] == "req_abc"
    # 未提供时不写空键（避免云端出现 request_trace_id="" 的噪声）
    assert "request_trace_id" not in obs.trace_metadata(session_id="s", user_id="u")


def test_mask_recurses_and_keeps_json_serializable() -> None:
    """掩码递归生效且结果可 JSON 序列化（SDK 对 mask 返回值的硬要求）。"""
    payload = {
        "prompt": "帮我绑定飞书 api_key=sk-abcdefgh12345678",
        "meta": {
            "authorization": "Bearer abcdefgh12345678",
            "user_access_token": "u-1234567890abcdef",
            "nested": [{"refresh_token": "r-1234567890abcdef"}],
            "when": {"not": "serializable-object"},
        },
    }
    masked = mask_payload(data=payload)
    dumped = json.dumps(masked, ensure_ascii=False)  # 不可序列化会在此抛错
    assert "sk-abcdefgh12345678" not in dumped
    assert "u-1234567890abcdef" not in dumped
    assert "r-1234567890abcdef" not in dumped
    assert "abcdefgh12345678" not in dumped
    assert "帮我绑定飞书" in dumped  # 非敏感正文保留（排查要看得到上下文）


def test_mask_failure_discards_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """掩码函数自身失败 → 整车占位（宁可丢内容也不泄漏未掩码数据）。"""
    import shared.observability as module

    def _boom(_text: str) -> str:
        raise RuntimeError("mask 崩了")

    monkeypatch.setattr(module, "mask_text", _boom)
    assert mask_payload(data={"prompt": "敏感原文"}) == FULLY_MASKED


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "apiKey",
        "authorization",
        "access_token",
        "refresh_token",
        "user_access_token",
        "app_secret",
        "password",
        "session_key",
        "private_key",
    ],
)
def test_mask_masks_sensitive_keys_recursively(key: str) -> None:
    """**键级**凭据必须打码（值里没有字段名可依，只靠正则兜底会整批漏过去）。"""
    masked = mask_payload(data={"outer": [{"inner": {key: "plainvalue123"}}]})
    dumped = json.dumps(masked, ensure_ascii=False)
    assert "plainvalue123" not in dumped
    assert MASKED in dumped


def test_mask_keeps_ordinary_fields() -> None:
    """非敏感字段保留原值（排查要看得到上下文，全掩码等于没有观测）。"""
    masked = mask_payload(data={"tool": "search_knowledge", "count": 3, "ok": True})
    assert masked == {"tool": "search_knowledge", "count": 3, "ok": True}


async def test_aclose_flush_and_shutdown_swallow_errors() -> None:
    """关闭路径：flush + shutdown 都调用；两者抛错都只记日志、不冒泡；幂等。"""
    client = FakeClient(flush_error=RuntimeError("网络不可达"), shutdown_error=RuntimeError("已关"))
    obs = Observability(ObservabilityConfig(enabled=True), client=client)
    await obs.aclose()  # 不得抛错
    assert client.flush_calls == 1
    assert client.shutdown_calls == 1
    await obs.aclose()  # 幂等
    assert client.flush_calls == 1


async def test_aclose_flush_timeout_does_not_hang() -> None:
    """flush 卡死 → 超时后照常关停（退出路径不得被云端拖住）。"""
    import time

    class _HangingClient(FakeClient):
        def flush(self) -> None:
            self.flush_calls += 1
            time.sleep(5)

    client = _HangingClient()
    obs = Observability(ObservabilityConfig(enabled=True, flush_timeout_s=0.05), client=client)
    await obs.aclose()
    assert client.shutdown_calls == 1


# —— 进程级单例（runtime 装配入口）——


def test_install_singleton_and_rebuild_after_close(
    clean_langfuse_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """单例：重复 install 复用同一句柄；已关闭的句柄**不复用**（复用 = 静默失去观测）。"""
    import shared.observability as module

    monkeypatch.setattr(module, "_default_client_factory", lambda _config: FakeClient())
    settings = _enabled_settings()
    first = install_observability(settings)
    assert first is not None
    assert install_observability(settings) is first  # 复用

    first._closed = True  # 模拟 runtime 已 aclose
    second = install_observability(settings)
    assert second is not None
    assert second is not first


def test_install_disabled_returns_none(clean_langfuse_env: None) -> None:
    """未配置 → 单例为 `None`（调用方据此不给图挂任何观测键）。"""
    assert install_observability(RuntimeSettings(_env_file=None)) is None


# —— 注入点（agent/runtime.py）：图 config 挂回调 + 归因，一处注入覆盖全图 ——


class RecordingHandler(AsyncCallbackHandler):
    """记录回调事件的假 handler（**真实 langchain 回调协议**，不连云端）。

    WHY 用真 AsyncCallbackHandler 而非打桩：这一节要验的正是「回调是否真的随 graph
    config 传播到节点内的 LLM / ToolNode」（plan §7.5 不确定项 5 —— `agent/llm.py`
    不传 config，全靠 langchain 的回调上下文继承）。打桩就验不到传播。
    """

    def __init__(self) -> None:
        self.llm_starts = 0
        self.chain_runs: list[str] = []
        self.tool_runs: list[str] = []
        self.tool_ends: list[Any] = []

    async def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self.llm_starts += 1

    async def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.llm_starts += 1

    async def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or ""
        self.chain_runs.append(str(name))

    async def on_tool_start(self, serialized: Any, input_str: Any, **kwargs: Any) -> None:
        self.tool_runs.append(str((serialized or {}).get("name") or ""))

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self.tool_ends.append(output)


class _CapturingGraph:
    """包一层真图，捕获 `chat_stream` 实际交给 `astream` 的 config。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.configs: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        self.configs.append(dict(kwargs.get("config") or {}))
        async for item in self._inner.astream(*args, **kwargs):
            yield item


@pytest.fixture
def runtime_factory(build_graph: Any) -> Any:
    """构造注入 fake LLM 的 AgentRuntime（可挂观测句柄；会话用 SQLite memory 后端）。"""

    async def _make(
        messages: list[Any],
        *,
        tools: list[Any] | None = None,
        observability: Any = None,
        extractor: Any = None,
    ) -> Any:
        from langgraph.store.memory import InMemoryStore

        from agent.core.config import AgentFrameworkConfig
        from agent.memory import MemoryStore
        from agent.runtime import AgentRuntime
        from agent.session import build_session_backend

        backend = await build_session_backend(None)
        return AgentRuntime(
            graph=_CapturingGraph(build_graph(messages, tools=tools)),
            sessions=backend.manager,
            memory_store=MemoryStore(InMemoryStore()),
            cfg=AgentFrameworkConfig.get_default(),
            _session_backend=backend,
            observability=observability,
            extractor=extractor,
        )

    return _make


async def test_disabled_keeps_graph_config_byte_identical(runtime_factory: Any) -> None:
    """**零回归判据**：未配置观测 → 图 config 只有 C4 的 `[usage]`，且没有 metadata 键。

    注：plan §5.2 原话是「未配置 → 图 config 不含 callbacks 键」——该措辞在 C4 落地后
    已不成立（token 累计 handler 一直占着这个键）。可验证的等价口径是：观测对 config
    **零贡献**（callbacks 逐字等于 C4 时期、不新增 metadata 键）。
    """
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好"), observability=None)
    try:
        async for _ in runtime.chat_stream(user_id="u1", session_id="s1", text="你好"):
            pass
        config = runtime.graph.configs[0]
        assert "metadata" not in config
        callbacks = config["callbacks"]
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], UsageMetadataCallbackHandler)
    finally:
        await runtime.aclose()


async def test_enabled_attaches_callbacks_and_attribution(runtime_factory: Any) -> None:
    """配置观测 → 图 config 同时带 token handler + 观测 handler + 归因 metadata。"""
    obs = Observability(
        ObservabilityConfig(enabled=True, environment="dev"),
        client=FakeClient(),
        handler_factory=lambda _config: RecordingHandler(),
    )
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好"), observability=obs)
    try:
        async for _ in runtime.chat_stream(user_id="u1", session_id="s1", text="你好"):
            pass
        config = runtime.graph.configs[0]
        assert len(config["callbacks"]) == 2
        assert isinstance(config["callbacks"][0], UsageMetadataCallbackHandler)
        assert isinstance(config["callbacks"][1], RecordingHandler)
        metadata = config["metadata"]
        assert metadata["langfuse_session_id"] == "s1"
        assert metadata["langfuse_user_id"] == "u1"
        assert metadata["langfuse_trace_name"] == TRACE_NAME
    finally:
        await runtime.aclose()


async def test_callbacks_reach_node_level_llm_and_tools(runtime_factory: Any) -> None:
    """**实测 §7.5-5**：handler 挂在 graph config 上，是否真能收到节点内 LLM / 工具回调。

    `agent/llm.py` 与图节点都不传 config（全靠 langchain 的回调上下文继承）—— 这条不实测
    就只能假设；一旦不继承，trace 会退化成「只有一条根 span」，且没有任何报错。
    """
    handler = RecordingHandler()
    obs = Observability(
        ObservabilityConfig(enabled=True),
        client=FakeClient(),
        handler_factory=lambda _config: handler,
    )
    messages = tool_call_messages([[{"name": "clock", "args": {}, "id": "call_1"}]], "现在十点")
    runtime = await runtime_factory(
        messages, tools=[make_fake_tool("clock", content="10:00")], observability=obs
    )
    try:
        async for _ in runtime.chat_stream(user_id="u1", session_id="s1", text="现在几点"):
            pass
    finally:
        await runtime.aclose()

    assert handler.llm_starts > 0, "节点内 LLM 调用未收到回调（trace 会退化成单 span）"
    assert handler.tool_runs == ["clock"], f"ToolNode 工具调用未收到回调：{handler.tool_runs}"
    # on_tool_end 收到的是 ToolMessage 对象（Langfuse handler 正是据此把
    # `status == "error"` 的工具结果自动置 span level=ERROR）——故这里按 content 断言。
    assert [getattr(item, "content", item) for item in handler.tool_ends] == ["10:00"]
    assert any(handler.chain_runs), "未收到任何 chain 回调（节点名无从归因）"


# —— 带外保存观测接线（O3）：callbacks + metadata 显式传入 submit_memory_save ——


def _capture_submit(monkeypatch: Any, captured: dict[str, Any]) -> None:
    """把 `agent.runtime.submit_memory_save` 换成记录器（不真正保存，只捕获入参）。"""
    import agent.runtime as runtime_module

    def _fake_submit(messages: Any, **kwargs: Any) -> None:
        captured["messages"] = messages
        captured.update(kwargs)

    monkeypatch.setattr(runtime_module, "submit_memory_save", _fake_submit)


async def test_out_of_band_save_receives_callbacks_and_metadata(
    runtime_factory: Any, monkeypatch: Any
) -> None:
    """观测已配置 → 带外保存携带新回调 + 归因 metadata（独立 trace 名，归因键同对话）。

    带外 task 的回调上下文不继承（task 上下文来自 chat_stream 调用方），不显式传 config
    就是 Langfuse 零 span —— 本用例锁定「有回调且 metadata 正确」的接线契约。
    """
    made: list[RecordingHandler] = []

    def _factory(_config: ObservabilityConfig) -> RecordingHandler:
        made.append(RecordingHandler())
        return made[-1]

    obs = Observability(
        ObservabilityConfig(enabled=True), client=FakeClient(), handler_factory=_factory
    )
    runtime = await runtime_factory(
        chat_turn_messages(Intent.CHAT, "你好！"), observability=obs, extractor=object()
    )
    captured: dict[str, Any] = {}
    _capture_submit(monkeypatch, captured)
    try:
        async for _ in runtime.chat_stream(user_id="u1", session_id="s1", text="我叫小明"):
            pass
    finally:
        await runtime.aclose()

    # 带外用**新建**的 handler（每请求新建契约）：图 config 拿第 1 个，带外拿第 2 个。
    assert captured["callbacks"] == [made[1]]
    metadata = captured["metadata"]
    assert metadata["langfuse_session_id"] == "s1"
    assert metadata["langfuse_user_id"] == "u1"
    assert metadata["langfuse_trace_name"] == OUT_OF_BAND_TRACE_NAME
    # 归因键与对话 trace 同值：本地 trace_id 关联键两边同源（测试无请求上下文时同为缺省）。
    assert (
        metadata.get("request_trace_id")
        == runtime.graph.configs[0]["metadata"].get("request_trace_id")
    )
    # 对话 trace 本体不受影响：仍是默认 trace 名。
    assert runtime.graph.configs[0]["metadata"]["langfuse_trace_name"] == TRACE_NAME


async def test_out_of_band_save_without_observability_stays_none(
    runtime_factory: Any, monkeypatch: Any
) -> None:
    """未配置观测 → 带外保存 callbacks/metadata 均为 None（行为与接线前逐字相同）。"""
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好！"), extractor=object())
    captured: dict[str, Any] = {}
    _capture_submit(monkeypatch, captured)
    try:
        async for _ in runtime.chat_stream(user_id="u1", session_id="s1", text="我叫小明"):
            pass
    finally:
        await runtime.aclose()

    assert captured["callbacks"] is None
    assert captured["metadata"] is None


async def test_trivial_input_skips_out_of_band_save(
    runtime_factory: Any, monkeypatch: Any
) -> None:
    """平凡输入（寒暄/确认/无意义字符）→ 跳过带外抽取（§7.1，每轮寒暄省一次抽取调用）。"""
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好！"), extractor=object())
    captured: dict[str, Any] = {}
    _capture_submit(monkeypatch, captured)
    try:
        async for _ in runtime.chat_stream(user_id="u1", session_id="s1", text="好的"):
            pass
    finally:
        await runtime.aclose()

    assert "messages" not in captured  # submit_memory_save 未被调用


async def test_aclose_flushes_after_pending_saves(runtime_factory: Any, monkeypatch: Any) -> None:
    """关闭顺序：排干带外保存 → **flush 观测** → 其余后端（顺序反了会丢最后一批 span）。"""
    order: list[str] = []

    class _Client(FakeClient):
        def flush(self) -> None:
            order.append("observability.flush")
            super().flush()

    class _Backend:
        async def aclose(self) -> None:
            order.append("session")

    async def _fake_wait_pending_saves() -> None:
        order.append("pending_saves")

    import agent.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "wait_pending_saves", _fake_wait_pending_saves)
    obs = Observability(ObservabilityConfig(enabled=True), client=_Client())
    runtime = await runtime_factory(chat_turn_messages(Intent.CHAT, "你好"), observability=obs)
    runtime._session_backend = _Backend()
    await runtime.aclose()
    assert order == ["pending_saves", "observability.flush", "session"]
