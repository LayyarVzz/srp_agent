"""agent/memory/extractor.py —— MemoryExtractor 单测（离线 fake LLM）。

覆盖：结构化抽取往返、空结果、None 守卫、LLM 失败兜底、prompt 组装、输入截断。
"""

from __future__ import annotations

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
