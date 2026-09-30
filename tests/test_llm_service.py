"""agent/llm.py —— LLMService 单测（离线，零网络）。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from agent.core.config import LLMConfig, LLMProvider
from agent.errors import LLM_ERROR_AUTH, LLM_ERROR_REQUEST, LLMError
from agent.llm import LLMService, merge_ai_message_chunks
from tests.conftest import StructuredFakeChatModel as _StructuredFakeModel
from tests.conftest import make_fake_tool


class DemoIntent(BaseModel):
    intent: str
    confidence: float
    reason: str


def _fake_tool_calls_model(tool_calls: list[dict]) -> _StructuredFakeModel:
    """构造产出指定 tool_calls 的 AIMessage 的 fake（bind_tools 多工具路径）。"""
    msg = AIMessage(content="", tool_calls=tool_calls)
    return _StructuredFakeModel(messages=iter([msg]))


def _fake_tool_model(result: DemoIntent) -> _StructuredFakeModel:
    """构造产出指定 tool-call AIMessage 的 fake（tool name 必须等于 schema 类名）。"""
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "DemoIntent", "args": result.model_dump(), "id": "call_1"}],
    )
    return _StructuredFakeModel(messages=iter([msg]))


def _fake_text_model(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def test_builds_chatopenai_from_config() -> None:
    """合并配置 → ChatOpenAI，端点/模型经预设解析。"""
    cfg = LLMConfig(provider=LLMProvider.QWEN, api_key=SecretStr("sk-x"))
    model = LLMService(config=cfg).chat_model
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "qwen-plus"
    assert model.openai_api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_missing_api_key_raises() -> None:
    cfg = LLMConfig(api_key=SecretStr(""))
    service = LLMService(config=cfg)
    with pytest.raises(LLMError) as exc:
        _ = service.chat_model
    assert exc.value.code == LLM_ERROR_AUTH


def test_structured_model_returns_runnable() -> None:
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    runnable = LLMService(config=cfg).structured_model(DemoIntent)
    assert isinstance(runnable, Runnable)


async def test_ainvoke_structured_offline() -> None:
    """离线验收：构造注入 fake，走真实 with_structured_output → bind_tools → 解析链路。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    expected = DemoIntent(intent="chat", confidence=0.9, reason="test")
    service = LLMService(config=cfg, chat_model=_fake_tool_model(expected))
    result = await service.ainvoke_structured(DemoIntent, "hi")
    assert isinstance(result, DemoIntent)
    assert result == expected


class _SpyStructured:
    """记录 ainvoke 收到入参的假结构化 Runnable（只验透传，不验解析）。"""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.seen: dict[str, Any] = {}

    async def ainvoke(self, prompt: Any, *, config: Any = None, **kwargs: Any) -> Any:
        self.seen = {"prompt": prompt, "config": config}
        return self._result


async def test_ainvoke_structured_forwards_config() -> None:
    """显式 config 原样透传给底层结构化 Runnable（带外路径挂观测回调的唯一入口，O3）。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    expected = DemoIntent(intent="chat", confidence=0.9, reason="test")
    service = LLMService(config=cfg, chat_model=_fake_tool_model(expected))
    spy = _SpyStructured(expected)
    service.structured_model = lambda _schema: spy  # type: ignore[method-assign]
    config: dict[str, Any] = {"callbacks": [], "metadata": {"langfuse_session_id": "s1"}}
    assert await service.ainvoke_structured(DemoIntent, "hi", config=config) is expected
    assert spy.seen == {"prompt": "hi", "config": config}


async def test_ainvoke_structured_default_config_is_none() -> None:
    """不传 config → 底层收到 None（与现状逐字同路，现有调用方零改动）。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    expected = DemoIntent(intent="chat", confidence=0.9, reason="test")
    service = LLMService(config=cfg, chat_model=_fake_tool_model(expected))
    spy = _SpyStructured(expected)
    service.structured_model = lambda _schema: spy  # type: ignore[method-assign]
    await service.ainvoke_structured(DemoIntent, "hi")
    assert spy.seen == {"prompt": "hi", "config": None}


async def test_ainvoke_structured_with_config_offline() -> None:
    """带 config 走真实 with_structured_output 链路仍正常解析（回调随 config 生效不打断解析）。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    expected = DemoIntent(intent="chat", confidence=0.9, reason="test")
    service = LLMService(config=cfg, chat_model=_fake_tool_model(expected))
    result = await service.ainvoke_structured(DemoIntent, "hi", config={"callbacks": []})
    assert result == expected


async def test_ainvoke_text_offline() -> None:
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    service = LLMService(config=cfg, chat_model=_fake_text_model("你好"))
    text = await service.ainvoke_text("hi")
    assert text == "你好"


async def test_astream_text_offline_streams_deltas() -> None:
    """流式文本：增量按空白切分逐段产出，拼接 == 完整文本（ainvoke 同源）。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    service = LLMService(config=cfg, chat_model=_fake_text_model("你好 世界！"))
    parts = [part async for part in service.astream_text("hi")]
    assert len(parts) > 1  # 真流式：含空白切分出的多段增量
    assert "".join(parts) == "你好 世界！"


async def test_astream_text_error_normalized() -> None:
    """流式文本失败（空迭代器耗竭）→ 归一化为 LLMError(llm_error.request)。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    service = LLMService(config=cfg, chat_model=GenericFakeChatModel(messages=iter([])))
    with pytest.raises(LLMError) as exc:
        async for _ in service.astream_text("hi"):
            pass
    assert exc.value.code == LLM_ERROR_REQUEST


# —— 工具绑定流式（call_model 路径：content 增量 + tool_call 还原）——


async def test_astream_tools_content_deltas_and_merge() -> None:
    """流式工具调用：content 直答逐增量产出，merge_ai_message_chunks 还原等价 AIMessage。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    model = _StructuredFakeModel(messages=iter([AIMessage(content="现在是 14:30。")]))
    service = LLMService(config=cfg, chat_model=model)
    chunks = [chunk async for chunk in service.astream_tools([make_fake_tool("calc")], "hi")]
    contents = "".join(str(chunk.content) for chunk in chunks if isinstance(chunk.content, str))
    assert contents == "现在是 14:30。"
    resp = merge_ai_message_chunks(chunks)
    assert isinstance(resp, AIMessage)
    assert str(resp.content) == "现在是 14:30。"
    assert not resp.tool_calls  # 直答：无工具调用


async def test_astream_tools_tool_calls_roundtrip() -> None:
    """流式工具调用：tool_calls 经 chunk 聚合后可还原（name/args/id 与输入一致）。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    calls = [{"name": "calc", "args": {"expression": "1+1"}, "id": "c1"}]
    service = LLMService(config=cfg, chat_model=_fake_tool_calls_model(calls))
    chunks = [chunk async for chunk in service.astream_tools([make_fake_tool("calc")], "hi")]
    resp = merge_ai_message_chunks(chunks)
    assert isinstance(resp, AIMessage)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "calc"
    assert resp.tool_calls[0]["id"] == "c1"
    assert resp.tool_calls[0]["args"] == {"expression": "1+1"}


async def test_ainvoke_tools_stream_aggregates_tool_calls() -> None:
    """ainvoke_tools（聚合 astream_tools）语义与直接 ainvoke 等价：还原 tool_calls。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    calls = [{"name": "calc", "args": {"expression": "2+2"}, "id": "c2"}]
    service = LLMService(config=cfg, chat_model=_fake_tool_calls_model(calls))
    resp = await service.ainvoke_tools([make_fake_tool("calc")], "hi")
    assert isinstance(resp, AIMessage)
    assert resp.tool_calls[0]["name"] == "calc"
    assert resp.tool_calls[0]["args"] == {"expression": "2+2"}


def test_merge_ai_message_chunks_keeps_content_and_tool_calls() -> None:
    """异常形态（content 先于 tool_calls）聚合后 content 与 tool_calls 同时保留。

    WHY 锁定图语义：同一次调用 content 与 tool_calls 同现时，content 已按预览
    外发（前端以 tool 事件重置），聚合出的 AIMessage 仍需携带两者——content
    进 messages 历史、tool_calls 驱动 dispatch_tool 分流。
    """
    chunks = [
        AIMessageChunk(content="先交代一下", id="m"),
        AIMessageChunk(
            content="",
            tool_call_chunks=[ToolCallChunk(name="calc", args="{}", id="call_x", index=0)],
            id="m",
        ),
    ]
    resp = merge_ai_message_chunks(chunks)
    assert isinstance(resp, AIMessage)
    assert resp.content == "先交代一下"
    assert resp.tool_calls[0]["name"] == "calc"
    assert resp.tool_calls[0]["id"] == "call_x"


async def test_astream_tools_error_normalized() -> None:
    """流式工具调用失败（空迭代器耗竭）→ 归一化为 LLMError(llm_error.request)。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    service = LLMService(config=cfg, chat_model=_StructuredFakeModel(messages=iter([])))
    with pytest.raises(LLMError) as exc:
        async for _ in service.astream_tools([make_fake_tool("calc")], "hi"):
            pass
    assert exc.value.code == LLM_ERROR_REQUEST


async def test_structured_after_text_keeps_thinking_disabled(monkeypatch) -> None:
    """回归：统一关思考后，结构化输出复用已缓存的 chat_model，且该模型本身已关思考。

    WHY 语义变化：原实现「普通对话保留思考 + 结构化专用模型关思考」的双模型切换
    已废弃；现 chat_model 懒构造即带 `structured_extra_body`（thinking=disabled），
    text/tool/structured 三条路径共用同一模型，缓存污染 bug 从根上消失。
    """
    cfg = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key=SecretStr("sk-x"))
    service = LLMService(config=cfg)
    # 等价于跑过一次 ainvoke_text：触发 chat_model 懒构造并缓存（已统一关思考）。
    _ = service.chat_model
    assert service.chat_model.extra_body == cfg.structured_extra_body
    # 思考确实被关闭（两口径同时下发，见 LLMConfig.structured_extra_body 的 WHY）。
    assert service.chat_model.extra_body["enable_thinking"] is False
    assert service.chat_model.extra_body["thinking"] == {"type": "disabled"}

    built: list[dict | None] = []
    fake = _fake_tool_model(DemoIntent(intent="chat", confidence=1.0, reason="x"))

    def fake_build(*, extra_body: dict | None = None):
        built.append(extra_body)
        return fake

    monkeypatch.setattr(service, "_build_chat_model", fake_build)
    runnable = service.structured_model(DemoIntent)
    # 复用已缓存模型，不再构造任何专用模型。
    assert built == []
    assert isinstance(runnable, Runnable)


async def test_injected_model_bypasses_extra_body(monkeypatch) -> None:
    """注入自定义模型时，结构化输出复用注入模型，不额外构造 extra_body 模型。"""
    cfg = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key=SecretStr("sk-x"))
    demo = DemoIntent(intent="chat", confidence=1.0, reason="x")
    service = LLMService(config=cfg, chat_model=_fake_tool_model(demo))
    built: list[dict | None] = []

    def fake_build(*, extra_body: dict | None = None):
        built.append(extra_body)
        return _fake_text_model("should not be used")

    monkeypatch.setattr(service, "_build_chat_model", fake_build)
    runnable = service.structured_model(DemoIntent)
    assert built == []  # 未构造任何专用模型
    assert isinstance(runnable, Runnable)


# —— 多工具绑定（ToolNode 生态：bind_tools 产出 AIMessage.tool_calls）——


async def test_tool_model_binds_tools_and_produces_tool_calls() -> None:
    """tool_model 返回 Runnable，注入 fake 时复用注入模型并产出带 tool_calls 的 AIMessage。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    service = LLMService(
        config=cfg,
        chat_model=_fake_tool_calls_model(
            [{"name": "calc", "args": {"expression": "1+1"}, "id": "c1"}]
        ),
    )
    runnable = service.tool_model([make_fake_tool("calc")])
    assert isinstance(runnable, Runnable)

    resp = await runnable.ainvoke("hi")
    assert isinstance(resp, AIMessage)
    assert resp.tool_calls[0]["name"] == "calc"
    assert resp.tool_calls[0]["args"] == {"expression": "1+1"}


async def test_tool_model_carries_thinking_disabled_extra_body(monkeypatch) -> None:
    """tool_model 复用同一模型：构造时即带「关闭思考」extra_body（provider 无关）。

    实测托管端点（阿里云 MaaS）会忽略不认识的字段，故 auto 口径两种都下发；
    断言与 `LLMConfig.structured_extra_body` 同源，避免两处漂移。
    """
    cfg = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key=SecretStr("sk-x"))
    service = LLMService(config=cfg)
    built: list[dict | None] = []
    fake = _fake_tool_calls_model([{"name": "calc", "args": {}, "id": "c1"}])

    def fake_build(*, extra_body: dict | None = None):
        built.append(extra_body)
        return fake

    monkeypatch.setattr(service, "_build_chat_model", fake_build)
    runnable = service.tool_model([make_fake_tool("calc")])
    assert built == [cfg.structured_extra_body]
    assert built[0] is not None  # 思考确实被关闭（不是空 body）
    assert isinstance(runnable, Runnable)


async def test_tool_model_injected_model_reuses_injected(monkeypatch) -> None:
    """注入自定义模型时，tool_model 复用注入模型，不额外构造 extra_body 模型。"""
    cfg = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key=SecretStr("sk-x"))
    service = LLMService(
        config=cfg,
        chat_model=_fake_tool_calls_model([{"name": "calc", "args": {}, "id": "c1"}]),
    )
    built: list[dict | None] = []

    def fake_build(*, extra_body: dict | None = None):
        built.append(extra_body)
        return _fake_text_model("should not be used")

    monkeypatch.setattr(service, "_build_chat_model", fake_build)
    runnable = service.tool_model([make_fake_tool("calc")])
    assert built == []  # 未构造任何专用模型
    assert isinstance(runnable, Runnable)


async def test_ainvoke_tools_error_normalized() -> None:
    """bind_tools 调用失败（空迭代器耗竭）→ 归一化为 LLMError(llm_error.request)。"""
    cfg = LLMConfig(api_key=SecretStr("sk-x"))
    service = LLMService(config=cfg, chat_model=_StructuredFakeModel(messages=iter([])))
    with pytest.raises(LLMError) as exc:
        await service.ainvoke_tools([make_fake_tool("calc")], "hi")
    assert exc.value.code == LLM_ERROR_REQUEST
