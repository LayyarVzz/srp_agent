"""图级用例：意图分类的会话上下文注入（v5.2，本地未跟踪测试）。

覆盖 C2：`classify_intent` 把最近若干轮对话 + 短期摘要/关键信息交给分类器，
使「对上一轮追问/要求的简短回应」不再因脱上下文而被判模糊。

WHY 独立文件：`/tests` 被 .gitignore 忽略，本文件与 tests/test_intent.py 的断言变更
一律留在本地，不入仓库（与本次「只提交生产代码」的口径一致）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.core.config import AgentFrameworkConfig, AgentGraphConfig
from agent.core.context import ShortTermContext
from agent.core.graph import build_agent_graph
from agent.intent.models import Intent, IntentResult
from agent.llm import LLMService
from agent.response.models import ClarifyResult
from tests.conftest import (
    RecordingFakeChatModel,
    chat_turn_messages,
    fake_structured_message,
    fake_text_message,
    understand_message,
)

_CLASSIFY_MARKER = "你是意图分类器"


def _classify_prompts(model: RecordingFakeChatModel) -> list[str]:
    """取出所有「意图分类」调用的 prompt 全文（按调用顺序）。"""
    return [
        "\n".join(str(getattr(m, "content", "")) for m in prompt)
        for prompt in model.prompts
        if any(_CLASSIFY_MARKER in str(getattr(m, "content", "")) for m in prompt)
    ]


async def test_classify_prompt_carries_previous_turn_context(
    make_llm_service: Callable[..., LLMService],
    run_graph: Callable[..., Any],
) -> None:
    """第 2 轮分类 prompt 必须带上第 1 轮的助理消息与用户消息（续跑判定依据）。"""
    service = make_llm_service(
        [
            # v6.0 T2：「帮我绑定飞书」过确定性门控 → 意图消息后带一条查询理解。
            *chat_turn_messages(
                Intent.CHAT,
                "已生成授权链接，完成授权后告诉我「我已完成授权」。",
                understand=True,
            ),
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="续跑绑定")
            ),
            # 第 2 轮「我已完成授权」同样过门控（脚本按调用顺序：意图 → 查询理解 → 回答）。
            understand_message(),
            fake_text_message("好的，我来完成绑定。"),
        ],
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(service)
    await run_graph(graph, text="帮我绑定飞书", session_id="s1")
    await run_graph(graph, text="我已完成授权", session_id="s1")

    prompts = _classify_prompts(service.chat_model)
    assert len(prompts) == 2
    second = prompts[1]
    # 上一轮助理的指令（含用户该说什么）必须在场——这是「用户照做」得以被理解的唯一依据。
    assert "助理：已生成授权链接" in second
    assert "用户：帮我绑定飞书" in second
    assert "用户最新消息：我已完成授权" in second
    # 上下文块带不可信声明（安全约束）。
    assert "不可信" in second


async def test_classify_prompt_carries_short_term_summary(
    make_llm_service: Callable[..., LLMService],
    run_graph: Callable[..., Any],
) -> None:
    """滚动摘要产出后必须进分类 prompt（IntentContext 接线生效）。

    触发路径：把保留窗口压到 1 轮，使第 2 轮入口的 trim_history 裁掉旧消息
    → summarize_history 产出摘要 → 同一轮 classify_intent 消费该摘要。
    """
    service = make_llm_service(
        [
            # v6.0 T2：第 1 轮「帮我绑定飞书」过确定性门控 → 意图消息后带一条查询理解。
            *chat_turn_messages(Intent.CHAT, "好的，已收到。", understand=True),
            fake_structured_message(
                ShortTermContext(summary="用户在办理飞书账号授权", keyfacts=[])
            ),
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="续跑绑定")
            ),
            # 第 2 轮「我已完成授权」同样过门控（摘要消息由 summarize_history 先消费）。
            understand_message(),
            fake_text_message("好的，我来完成绑定。"),
        ],
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(
        service,
        AgentFrameworkConfig(graph=AgentGraphConfig(trim_keep_recent_rounds=1)),
    )
    await run_graph(graph, text="帮我绑定飞书", session_id="s1")
    await run_graph(graph, text="我已完成授权", session_id="s1")

    prompts = _classify_prompts(service.chat_model)
    assert len(prompts) == 2
    assert "会话摘要：用户在办理飞书账号授权" in prompts[1]


async def test_classify_prompt_marks_new_topic_without_context(
    make_llm_service: Callable[..., LLMService],
    run_graph: Callable[..., Any],
) -> None:
    """首轮（无上文）→ 上下文块显式标注「新话题」，不产生空块。"""
    service = make_llm_service(
        chat_turn_messages(Intent.CHAT, "你好，有什么可以帮你？"),
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(service)
    await run_graph(graph, text="你好", session_id="s1")

    prompts = _classify_prompts(service.chat_model)
    assert len(prompts) == 1
    assert "（无上文，本轮为新话题）" in prompts[0]


# —— C3：澄清追问接地（触发原因 + 同一份上下文 + 禁止泛问/编选项）——

_CLARIFY_MARKER = "你是对话澄清器"


def _clarify_prompts(model: RecordingFakeChatModel) -> list[str]:
    """取出所有「澄清追问」调用的 prompt 全文（按调用顺序）。"""
    return [
        "\n".join(str(getattr(m, "content", "")) for m in prompt)
        for prompt in model.prompts
        if any(_CLARIFY_MARKER in str(getattr(m, "content", "")) for m in prompt)
    ]


async def test_clarify_prompt_carries_trigger_and_context(
    make_llm_service: Callable[..., LLMService],
    run_graph: Callable[..., Any],
) -> None:
    """低置信澄清：prompt 必须带「为什么问」（分类器 reason）+ 对话上下文 + 反泛问约束。"""
    service = make_llm_service(
        [
            # v6.0 T2：第 1 轮「帮我翻译一下」过确定性门控（长度 6 ≥ min_query_chars=2
            # 且不命中寒暄词表）→ 意图消息后带一条查询理解。
            *chat_turn_messages(Intent.CHAT, "好的，已记录你的偏好。", understand=True),
            fake_structured_message(
                IntentResult(
                    intent=Intent.CHAT, confidence=0.3, reason="指代不明：不确定要翻哪个对象"
                )
            ),
            # 第 2 轮「那个」同样**过门控**：门控只按「空/纯标点/长度 < 2/整句命中寒暄词表」
            # 跳过，「那个」两项都不命中 → 仍会调一次查询理解（实测调用序：
            # IntentResult → QueryUnderstandingResult → ClarifyResult），脚本必须按此排布。
            understand_message(),
            fake_structured_message(
                ClarifyResult(question="你指的是哪份资料？", options=["毕设资料", "周报"])
            ),
        ],
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(service)
    await run_graph(graph, text="帮我翻译一下", session_id="s1")
    response, _ = await run_graph(graph, text="那个", session_id="s1")

    assert response is not None
    assert response.clarification is not None
    prompts = _clarify_prompts(service.chat_model)
    assert len(prompts) == 1
    prompt = prompts[0]
    # 触发原因（分类器给出的模糊点）必须在场——这是反问能落到缺口上的前提。
    assert "触发原因：意图置信度低（0.30）：指代不明：不确定要翻哪个对象" in prompt
    # 同一份上下文块（带不可信声明）也必须在场。
    assert "助理：好的，已记录你的偏好。" in prompt
    assert "不可信" in prompt
    # 反泛问 / 反编选项约束。
    assert "禁止无信息量的泛问" in prompt
    assert "禁止编造不存在的分支" in prompt


class _ClockArgs(BaseModel):
    """带必填参数的假工具 schema：`zone` 缺失 → ToolNode 参数校验失败。"""

    zone: str = Field(description="时区")


def _schema_tool(name: str = "clock") -> StructuredTool:
    """构造带 args_schema 的假工具（真实 MCP 工具形态：参数缺失产生 ToolInvocationError）。"""

    async def _run(zone: str) -> str:
        return f"time in {zone}"

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=f"{name} 假工具（必填 zone）",
        args_schema=_ClockArgs,
    )


async def test_clarify_prompt_carries_missing_argument_trigger(
    make_llm_service: Callable[..., LLMService],
    run_graph: Callable[..., Any],
) -> None:
    """参数缺失澄清：prompt 的触发原因必须指出「哪个工具缺哪个参数」（来自工具校验）。"""
    service = make_llm_service(
        [
            fake_structured_message(
                IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="工具")
            ),
            # v6.0 T2：「帮我查一下时间」过门控 → 意图分类后补一条查询理解（其后才是工具调用）。
            understand_message(),
            AIMessage(content="", tool_calls=[{"name": "clock", "args": {}, "id": "c1"}]),
            fake_structured_message(
                ClarifyResult(
                    question="请问要查询哪个时区的时间？", options=["UTC", "Asia/Shanghai"]
                )
            ),
        ],
        model_cls=RecordingFakeChatModel,
    )
    graph = build_agent_graph(service, tools=[_schema_tool()])
    response, _ = await run_graph(graph, text="帮我查一下时间", session_id="s1")

    assert response is not None
    assert response.clarification is not None
    prompts = _clarify_prompts(service.chat_model)
    assert len(prompts) == 1
    assert "触发原因：工具 clock 的参数缺失" in prompts[0]
    # 缺口字段来自工具 schema 校验（zone 必填），而非模型猜测。
    assert "zone" in prompts[0]
