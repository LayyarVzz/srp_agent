"""agent/memory/extractor.py —— MemoryExtractor 单测（离线 fake LLM）。

覆盖：结构化抽取往返、空结果、None 守卫、LLM 失败兜底、prompt 组装、输入截断。
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import HumanMessage

from agent.memory.extractor import MemoryExtractor
from agent.memory.models import MemoryExtraction, MemoryExtractionResult
from tests.conftest import RecordingFakeChatModel, fake_structured_message, fake_text_message


async def test_extract_returns_structured_items(make_llm_service) -> None:
    """结构化输出解析为抽取列表，kind/content/importance 正确。"""
    svc = make_llm_service(
        [
            fake_structured_message(
                MemoryExtractionResult(
                    memories=[
                        MemoryExtraction(kind="fact", content="用户叫小明", importance=0.8),
                        MemoryExtraction(kind="preference", content="用户偏好简洁", importance=0.9),
                    ]
                )
            )
        ]
    )
    extractor = MemoryExtractor(svc)
    result = await extractor.extract([HumanMessage(content="我叫小明，喜欢简洁")])

    assert len(result) == 2
    assert result[0].kind == "fact"
    assert result[0].content == "用户叫小明"
    assert result[0].importance == 0.8
    assert result[1].kind == "preference"
    assert result[1].importance == 0.9


async def test_extract_empty_result(make_llm_service) -> None:
    """模型判定无可记内容 → 返回空列表。"""
    svc = make_llm_service([fake_structured_message(MemoryExtractionResult())])
    extractor = MemoryExtractor(svc)
    assert await extractor.extract([HumanMessage(content="现在几点了？")]) == []


class _RecordingLLM:
    """记录 config 透传的假 LLM（只验 ainvoke_structured 收到的 config，不验抽取逻辑）。"""

    def __init__(self) -> None:
        self.configs: list[object] = []

    async def ainvoke_structured(
        self, schema: object, prompt: object, *, config: object = None
    ) -> MemoryExtractionResult:
        self.configs.append(config)
        return MemoryExtractionResult()


async def test_extract_forwards_config_to_llm() -> None:
    """config 显式透传给结构化调用（带外观测回调挂载点，O3）；缺省为 None 与现状同路。"""
    llm = _RecordingLLM()
    extractor = MemoryExtractor(llm)  # type: ignore[arg-type]
    config = {"callbacks": [], "metadata": {"langfuse_session_id": "s1"}}
    await extractor.extract([HumanMessage(content="x")], config=config)
    assert llm.configs == [config]
    await extractor.extract([HumanMessage(content="x")])
    assert llm.configs[-1] is None


async def test_extract_none_result_returns_empty(make_llm_service) -> None:
    """无工具调用时 with_structured_output 返回 None：显式守卫，不抛、返回 []。"""
    svc = make_llm_service([fake_text_message("")])  # 无 tool_calls → 解析为 None
    extractor = MemoryExtractor(svc)
    assert await extractor.extract([HumanMessage(content="hi")]) == []


async def test_extract_llm_failure_returns_empty(make_llm_service) -> None:
    """LLM 调用异常（空迭代器）→ 兜底返回 []，绝不抛出。"""
    svc = make_llm_service([])
    extractor = MemoryExtractor(svc)
    assert await extractor.extract([HumanMessage(content="hi")]) == []


async def test_extract_builds_system_prompt(make_llm_service) -> None:
    """prompt 组装：首条为 SystemMessage(EXTRACT_PROMPT)，后随对话消息。"""
    svc = make_llm_service(
        [
            fake_structured_message(
                MemoryExtractionResult(memories=[MemoryExtraction(kind="fact", content="x")])
            )
        ],
        model_cls=RecordingFakeChatModel,
    )
    extractor = MemoryExtractor(svc)
    await extractor.extract([HumanMessage(content="hi")])

    prompt_messages = svc.chat_model.prompts[0]
    assert prompt_messages[0].type == "system"
    assert "记忆抽取器" in str(prompt_messages[0].content)
    assert str(prompt_messages[1].content) == "hi"


async def test_extract_truncates_recent_messages(make_llm_service) -> None:
    """小 max_input_chars：旧消息被截断、最新消息保留（保证至少一轮上下文）。"""
    svc = make_llm_service(
        [fake_structured_message(MemoryExtractionResult())],
        model_cls=RecordingFakeChatModel,
    )
    extractor = MemoryExtractor(svc, max_input_chars=10)
    await extractor.extract(
        [
            HumanMessage(content="这是很早以前的超长旧消息" * 5),
            HumanMessage(content="新消息"),
        ]
    )

    prompt_messages = svc.chat_model.prompts[0]
    contents = [str(m.content) for m in prompt_messages]
    assert "新消息" in contents
    assert "很早以前" not in contents


# —— 空结果可观测（O5-F4）：区分「未产出候选」与「自报拒收」两种空 ——


async def test_extract_empty_without_rejections_logs_signal(
    make_llm_service, caplog: pytest.LogCaptureFixture
) -> None:
    """空结果且无自报拒收 → INFO 信号（漏提排查入口：与「无记忆可存」可区分）。"""


    svc = make_llm_service([fake_structured_message(MemoryExtractionResult())])
    with caplog.at_level(logging.INFO, logger="agent.memory.extractor"):
        assert await MemoryExtractor(svc).extract([HumanMessage(content="现在几点了？")]) == []
    assert "无候选产出" in caplog.text


async def test_extract_empty_with_rejections_no_signal(
    make_llm_service, caplog: pytest.LogCaptureFixture
) -> None:
    """模型自报拒收的空 → 不打「无候选产出」信号（两种空语义不同，不混报）。"""


    svc = make_llm_service(
        [
            fake_structured_message(
                MemoryExtractionResult(rejected_count=2, rejected_categories=["commonsense"])
            )
        ]
    )
    with caplog.at_level(logging.INFO, logger="agent.memory.extractor"):
        assert await MemoryExtractor(svc).extract([HumanMessage(content="地球绕太阳转吗")]) == []
    assert "无候选产出" not in caplog.text
    assert "自报拒收 2 条" in caplog.text


# —— 解析后净化（O7-F10）：失控输出不落库、不进日志/trace ——


async def test_extract_truncates_overlong_worth_reason(make_llm_service) -> None:
    """worth_reason 超 200 字 → 截断（纯审计字段），条目本身保留。"""
    svc = make_llm_service(
        [
            fake_structured_message(
                MemoryExtractionResult(
                    memories=[
                        MemoryExtraction(
                            kind="fact", content="用户叫小明", worth_reason="审计" * 150
                        )
                    ]
                )
            )
        ]
    )
    result = await MemoryExtractor(svc).extract([HumanMessage(content="我叫小明")])

    assert len(result) == 1
    assert result[0].worth_reason == "审计" * 100  # 200 字截断
    assert result[0].content == "用户叫小明"  # 内容不受审计字段超限影响


async def test_extract_drops_degenerate_content_keeps_rest(make_llm_service) -> None:
    """content 循环重复 → 丢弃该条；同批好内容不受影响（净化不整批连坐）。"""
    degenerate = "注意：本条与下周要出差同句出现，用户自述职业身份，属稳定事实。保留。" * 8
    svc = make_llm_service(
        [
            fake_structured_message(
                MemoryExtractionResult(
                    memories=[
                        MemoryExtraction(kind="fact", content=degenerate, importance=0.9),
                        MemoryExtraction(kind="fact", content="用户叫小明"),
                    ]
                )
            )
        ]
    )
    result = await MemoryExtractor(svc).extract([HumanMessage(content="x")])

    assert [m.content for m in result] == ["用户叫小明"]


async def test_extract_truncates_overlong_non_repeating_content(make_llm_service) -> None:
    """content 非重复但超 500 字 → 截断到 500（不丢弃，与循环重复的处理相区分）。"""
    long_unique = "".join(f"第{i}条记忆内容" for i in range(120))  # 远超 500 且无循环子串
    svc = make_llm_service(
        [
            fake_structured_message(
                MemoryExtractionResult(
                    memories=[MemoryExtraction(kind="fact", content=long_unique)]
                )
            )
        ]
    )
    result = await MemoryExtractor(svc).extract([HumanMessage(content="x")])

    assert len(result) == 1
    assert len(result[0].content) == 500
    assert result[0].content == long_unique[:500]


async def test_extract_passes_normal_output_verbatim(make_llm_service) -> None:
    """正常输出逐字透传：净化不改动任何字段（含中文短句与正常 reason）。"""
    normal = MemoryExtraction(
        kind="fact", content="用户是一名程序员", worth_reason="用户自述职业身份", importance=0.8
    )
    svc = make_llm_service(
        [fake_structured_message(MemoryExtractionResult(memories=[normal]))]
    )
    result = await MemoryExtractor(svc).extract([HumanMessage(content="我是一名程序员")])

    assert result == [normal]
    assert result[0].content == "用户是一名程序员"
    assert result[0].worth_reason == "用户自述职业身份"


def test_is_degenerate_boundaries() -> None:
    """滑窗判定的边界：短文本不判、恰好重复窗口判、非循环长文不判。"""
    from agent.memory.extractor import _is_degenerate

    assert _is_degenerate("短句") is False  # 不足一个窗口
    assert _is_degenerate("abcdefgh" * 8) is True  # 周期性重复（窗口 16 命中 3+ 次）
    assert _is_degenerate("".join(f"第{i}条" for i in range(60))) is False  # 非循环长文
