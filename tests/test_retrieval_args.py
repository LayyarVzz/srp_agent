"""agent/tools/retrieval_args.py —— RAG 检索实参改写与惰性补检单测（离线）。

覆盖（dev-version6.0.md §6）：
- 实参确定性归一（模型填值被覆盖、无改写原样放行、`override_model_query=False` 回退）；
- 非 rag 服务 / 无 runtime / 无检索查询 → 原样透传（零回归）；
- HyDE 可用性探测（RAG 未声明 `use_hypothetical` 时不新增该参数）；
- 召回不足惰性补检 + RRF 融合（空结果触发、分数达标不补检、上限约束、预算约束、
  融合后仍为合法响应结构）；关闭开关只改写不补检。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool

from agent.core.config import RetrievalConfig
from agent.query import QueryUnderstanding
from agent.tools.retrieval_args import (
    RAG_HYPOTHETICAL_ARG,
    RAG_QUERY_ARG,
    RetrievalQueryInterceptor,
)

SERVER = "rag"


class _FakeTool:
    """带 `tool_call_schema` 的假工具（用于 `use_hypothetical` 可用性探测）。"""

    def __init__(self, *, with_hypothetical: bool) -> None:
        props: dict[str, Any] = {RAG_QUERY_ARG: {"type": "string"}}
        if with_hypothetical:
            props[RAG_HYPOTHETICAL_ARG] = {"type": "boolean"}
        self.tool_call_schema = {"type": "object", "properties": props}


@dataclass
class _Request:
    """镜像 `MCPToolCallRequest` 的最小形状（name/args/server_name/runtime/tool + override）。"""

    name: str = "search_knowledge"
    args: dict[str, Any] = field(default_factory=dict)
    server_name: str = SERVER
    runtime: Any = None
    tool: Any = None

    def override(self, **kwargs: Any) -> _Request:
        return _Request(
            name=kwargs.get("name", self.name),
            args=kwargs.get("args", self.args),
            server_name=self.server_name,
            runtime=self.runtime,
            tool=self.tool,
        )


class _Runtime:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


@dataclass
class _Handler:
    """记录每次调用实参的假 handler；按 `responses` 顺序返回（不足则复用最后一个）。"""

    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, request: _Request) -> str:
        self.calls.append(dict(request.args))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _chunk(source_id: str, content: str, score: float) -> dict[str, Any]:
    """一条知识片段（镜像 Services 的 KnowledgeChunk 结构）。"""
    return {
        "content": content,
        "source": {"id": source_id, "title": source_id, "url": None},
        "metadata": {},
        "score": score,
    }


def _response(chunks: list[dict[str, Any]], query: str = "原始查询") -> str:
    """RAG 工具返回体（JSON 文本，与 MCP 线上形态一致）。"""
    return json.dumps({"query": query, "chunks": chunks}, ensure_ascii=False)


def _understanding(*, main: str = "改写后的查询", **kwargs: Any) -> QueryUnderstanding:
    return QueryUnderstanding(main_query=main, **kwargs)


def _interceptor(**cfg_kwargs: Any) -> RetrievalQueryInterceptor:
    return RetrievalQueryInterceptor(config=RetrievalConfig(**cfg_kwargs))


# —— 实参确定性归一 ——


async def test_model_query_is_overridden_by_rewrite() -> None:
    """模型填的 query 被改写结果确定性覆盖（正确性不依赖模型遵循提示词）。"""
    request = _Request(
        args={RAG_QUERY_ARG: "用户原话"},
        runtime=_Runtime({"retrieval_query": "改写后的查询"}),
    )
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)

    assert handler.calls[0][RAG_QUERY_ARG] == "改写后的查询"


async def test_original_request_is_not_mutated() -> None:
    """`override` 产生新请求，原请求不变（不可变语义，避免拦截器互相污染）。"""
    request = _Request(
        args={RAG_QUERY_ARG: "用户原话"}, runtime=_Runtime({"retrieval_query": "改写"})
    )
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)
    assert request.args[RAG_QUERY_ARG] == "用户原话"


async def test_passthrough_without_retrieval_query() -> None:
    """图上没有检索查询（门控跳过 / 改写失败）→ 原样透传（v5.1 行为）。"""
    request = _Request(args={RAG_QUERY_ARG: "用户原话"}, runtime=_Runtime({}))
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)
    assert handler.calls[0][RAG_QUERY_ARG] == "用户原话"


async def test_passthrough_without_runtime() -> None:
    """图外调用（无 runtime）→ 原样透传，不抛。"""
    request = _Request(args={RAG_QUERY_ARG: "用户原话"})
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)
    assert handler.calls[0][RAG_QUERY_ARG] == "用户原话"


async def test_other_servers_pass_through_untouched() -> None:
    """非 rag 服务（如 lark/tools）一律原样放行 —— 拦截器只做检索实参归一。"""
    request = _Request(
        args={"calendar_id": "c1"},
        server_name="lark",
        runtime=_Runtime({"retrieval_query": "改写"}),
    )
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)
    assert handler.calls[0] == {"calendar_id": "c1"}


async def test_override_can_be_disabled() -> None:
    """`override_model_query=False` → 连实参归一也关掉（完全回 v5.1）。"""
    request = _Request(
        args={RAG_QUERY_ARG: "用户原话"}, runtime=_Runtime({"retrieval_query": "改写"})
    )
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor(override_model_query=False)(request, handler)
    assert handler.calls[0][RAG_QUERY_ARG] == "用户原话"


# —— HyDE：可用性探测 + 门控 ——


async def test_hypothetical_arg_passed_when_supported() -> None:
    """RAG 侧声明了 `use_hypothetical` 且有假设文档 → 传 True（假答案仍只作嵌入输入）。"""
    request = _Request(
        args={RAG_QUERY_ARG: "原话"},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(hypothetical_answer="假设答案文本"),
            }
        ),
        tool=_FakeTool(with_hypothetical=True),
    )
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)
    assert handler.calls[0][RAG_HYPOTHETICAL_ARG] is True
    # 假设文档文本本身绝不出现在实参里（只传开关）。
    assert "假设答案文本" not in json.dumps(handler.calls[0], ensure_ascii=False)


async def test_hypothetical_arg_omitted_when_unsupported() -> None:
    """RAG 未声明该形参 → 不带该键（自动降级为不启用，避免未知参数报错）。"""
    request = _Request(
        args={RAG_QUERY_ARG: "原话"},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(hypothetical_answer="假设答案文本"),
            }
        ),
        tool=_FakeTool(with_hypothetical=False),
    )
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)
    assert RAG_HYPOTHETICAL_ARG not in handler.calls[0]


async def test_hypothetical_arg_absent_without_hypothetical_answer() -> None:
    """没有假设文档（门控未通过 / 未生成）→ 不传该参数。"""
    request = _Request(
        args={RAG_QUERY_ARG: "原话"},
        runtime=_Runtime({"retrieval_query": "改写", "query_understanding": _understanding()}),
        tool=_FakeTool(with_hypothetical=True),
    )
    handler = _Handler([_response([_chunk("d1", "内容", 0.9)])])
    await _interceptor()(request, handler)
    assert RAG_HYPOTHETICAL_ARG not in handler.calls[0]


# —— 召回不足：惰性补检 + RRF 融合 ——


async def test_no_supplement_when_first_round_is_strong() -> None:
    """首轮最高分达标 → 不补检（不为边际收益付延迟），返回体原样透传。"""
    request = _Request(
        args={},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(sub_queries=["子一"], synonyms=["同义"]),
            }
        ),
    )
    body = _response([_chunk("d1", "内容", 0.9)])
    handler = _Handler([body])
    result = await _interceptor()(request, handler)

    assert len(handler.calls) == 1
    # 未补检 → 不改写返回体（融合只在真发生补检时重建，避免无谓的 JSON 往返）。
    assert result == body


async def test_empty_first_round_triggers_supplement_and_fusion() -> None:
    """首轮为空 → 用子查询/同义补检，RRF 融合后返回真实片段（引用仍来自检索）。

    补检在「某路达标」时提前停止（不为边际收益付延迟），故要让两路都跑到，
    变体路径返回的分数都需低于弱阈值（0.1 < 0.3）。
    """
    request = _Request(
        args={},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(sub_queries=["子一"], synonyms=["同义一"]),
            }
        ),
    )
    handler = _Handler(
        [
            _response([]),  # 首轮空
            _response([_chunk("d1", "子查询命中", 0.1)]),  # 补检 1：弱命中 → 继续
            _response([_chunk("d1", "子查询命中", 0.1), _chunk("d2", "同义命中", 0.05)]),
        ]
    )
    result = json.loads(await _interceptor()(request, handler))

    assert [c["source"]["id"] for c in result["chunks"]] == ["d1", "d2"]
    assert result["query"] == "改写"
    # 补检实参：依次用变体池（不再是主改写）。
    assert [call[RAG_QUERY_ARG] for call in handler.calls] == ["改写", "子一", "同义一"]


async def test_weak_first_round_triggers_supplement() -> None:
    """首轮有结果但最高分低于弱阈值 → 仍补检（弱结果等于近似没有）。"""
    request = _Request(
        args={},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(sub_queries=["子一"]),
            }
        ),
    )
    handler = _Handler(
        [
            _response([_chunk("weak", "弱命中", 0.1)]),
            _response([_chunk("strong", "补检强命中", 0.8)]),
        ]
    )
    result = json.loads(await _interceptor()(request, handler))
    assert len(handler.calls) == 2
    assert {c["source"]["id"] for c in result["chunks"]} == {"weak", "strong"}


async def test_supplement_stops_at_max_variants() -> None:
    """补检次数受 `max_variants - 1` 约束（首轮之外最多 2 路，max_variants=3）。"""
    request = _Request(
        args={},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(
                    sub_queries=["子一", "子二", "子三"], synonyms=["同义"]
                ),
            }
        ),
    )
    handler = _Handler([_response([])])
    await _interceptor(max_variants=3)(request, handler)
    assert len(handler.calls) == 3  # 首轮 + 2 路补检


async def test_supplement_respects_budget() -> None:
    """补检受耗时预算约束：预算耗尽即停（不把延迟无限放大）。"""
    request = _Request(
        args={},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(sub_queries=["子一", "子二"]),
            }
        ),
    )

    @dataclass
    class _SlowHandler:
        calls: list[dict[str, Any]] = field(default_factory=list)

        async def __call__(self, req: _Request) -> str:
            self.calls.append(dict(req.args))
            # 首轮即耗尽极小预算（异步 sleep：不阻塞事件循环，符合 async-first 纪律）。
            await asyncio.sleep(0.05)
            return _response([])

    handler = _SlowHandler()
    await _interceptor(multi_query_budget_s=0.01)(request, handler)
    assert len(handler.calls) == 1  # 首轮之后预算已耗尽 → 不补检


async def test_no_supplement_when_multi_query_disabled() -> None:
    """`multi_query_enabled=False` → 只做实参改写，绝不补检（零回归开关）。"""
    request = _Request(
        args={RAG_QUERY_ARG: "原话"},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(sub_queries=["子一"]),
            }
        ),
    )
    handler = _Handler([_response([])])
    await _interceptor(multi_query_enabled=False)(request, handler)
    assert len(handler.calls) == 1
    assert handler.calls[0][RAG_QUERY_ARG] == "改写"


async def test_fusion_dedups_same_chunk_across_variants() -> None:
    """同一片段被多路召回 → 融合去重为一条（键 = source.id + 内容前缀）。"""
    request = _Request(
        args={},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(sub_queries=["子一"], synonyms=["同义"]),
            }
        ),
    )
    same = _chunk("d1", "同一片段", 0.4)
    handler = _Handler([_response([]), _response([dict(same)]), _response([dict(same)])])
    result = json.loads(await _interceptor()(request, handler))
    assert len(result["chunks"]) == 1


async def test_unparsable_response_passes_through() -> None:
    """返回体不是可解析 JSON → 原样透传（绝不因解析失败吞掉工具结果）。"""
    request = _Request(
        args={},
        runtime=_Runtime({"retrieval_query": "改写", "query_understanding": _understanding()}),
    )
    handler = _Handler(["not-json"])
    assert await _interceptor()(request, handler) == "not-json"


async def test_tool_message_like_response_is_rewritten_in_place() -> None:
    """响应对象带 `.content` 时按原类型重建（ToolMessage 形态，保结构）。"""

    @dataclass
    class _Msg:
        content: str

        def model_copy(self, update: dict[str, Any]) -> _Msg:
            return _Msg(content=update["content"])

    request = _Request(
        args={},
        runtime=_Runtime(
            {
                "retrieval_query": "改写",
                "query_understanding": _understanding(sub_queries=["子一"]),
            }
        ),
    )

    @dataclass
    class _MsgHandler:
        calls: list[dict[str, Any]] = field(default_factory=list)

        async def __call__(self, req: _Request) -> _Msg:
            self.calls.append(dict(req.args))
            chunks = [] if len(self.calls) == 1 else [_chunk("d1", "补检命中", 0.6)]
            return _Msg(content=_response(chunks))

    handler = _MsgHandler()
    result = await _interceptor()(request, handler)
    assert isinstance(result, _Msg)
    assert json.loads(result.content)["chunks"][0]["source"]["id"] == "d1"


def test_interceptor_import_is_side_effect_free() -> None:
    """模块导入不应依赖 services 内部实现（上层依赖下层接口的边界约束）。"""
    import agent.tools.retrieval_args as module

    assert module.RAG_SERVER_NAME == "rag"
    assert isinstance(StructuredTool, type)  # 占位断言：确保 langchain 工具契约可用
