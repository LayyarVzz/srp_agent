"""共享测试设施：离线 fake LLM / 假工具工厂与图构建 fixture。

沿用 `GenericFakeChatModel` 注入模式：fake 覆盖 `bind_tools`，使 `with_structured_output`
走工具调用解析路径；消息按序放入同一迭代器，供图在单次运行内依次消费
（先 classify 结构化、后每次 call_model 消费一条 AI、终答文本被末次 call_model 消费）。
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field, SecretStr

from agent.core.config import AgentFrameworkConfig, LLMConfig
from agent.core.graph import build_agent_graph
from agent.intent.models import Intent, IntentResult
from agent.llm import LLMService
from agent.memory import MemoryStore
from agent.query.models import QueryUnderstanding, QueryUnderstandingResult
from agent.response.models import AgentResponse
from agent.runtime import AgentRuntime
from agent.session import build_session_backend
from app.main import create_app

# Windows 下 psycopg 异步驱动不兼容 ProactorEventLoop：pytest-asyncio 自动创建的事件循环
# 必须是 SelectorEventLoop，Postgres 集成测试（test_postgres_integration.py）才能连通。
# 须在任何事件循环创建之前设置，故放 conftest 模块导入期。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class StructuredFakeChatModel(GenericFakeChatModel):
    """离线 fake：覆盖 bind_tools，使 with_structured_output 走工具调用解析路径。

    WHY 重写 `_stream`：基类 GenericFakeChatModel 只对 additional_kwargs 里的
    function_call 做流式分块，现代 `tool_calls` 字段不会被流式出来；本实现把
    content 按空白切分（与基类一致），并把每条 tool_call 补成一个完整
    ToolCallChunk 增量块，使 call_model 的 astream_tools 聚合后能还原出
    AIMessage.tool_calls（图内 call_model 已全部切到流式调用）。

    WHY `bind_tools` 返回**浅拷贝**而非 `self`：`with_structured_output` 的实现是
    `self.bind_tools([schema])` 后再挂解析器，而 langchain 的绑定是可组合的
    （在已有绑定之上叠加）。若这里返回 `self`，同一实例上的结构化绑定会**跨调用累积**
    —— 表现为「图里第一次结构化调用（意图分类）之后，第二次（如 T2 查询理解）会拿到
    上一次绑定的工具名」而解析失败。真实 provider（ChatOpenAI）的 bind_tools 本就返回
    新对象，故这是 fake 保真度问题，不是生产缺陷。
    浅拷贝共享 `prompts` / `route_iters` 等记录字段的**同一列表对象**，
    故既有 prompt / 路由断言仍然生效。
    """

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self.model_copy(deep=False)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        # 复用基类 _generate（RecordingFakeChatModel 的 prompt 记录在此触发），
        # 随后把单条消息流式化为 AIMessageChunk 序列。
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        content = message.content
        if content:
            if not isinstance(content, str):
                msg = "Expected content to be a string."
                raise ValueError(msg)
            content_chunks = re.split(r"(\s)", content)
            for idx, token in enumerate(content_chunks):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=token, id=message.id))
                if idx == len(content_chunks) - 1 and not message.additional_kwargs:
                    chunk.message.chunk_position = "last"
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        if message.tool_calls:
            tool_chunks = [
                ToolCallChunk(
                    name=str(tc.get("name") or ""),
                    args=json.dumps(tc.get("args") or {}, ensure_ascii=False),
                    id=str(tc.get("id") or ""),
                    index=i,
                )
                for i, tc in enumerate(message.tool_calls)
            ]
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_call_chunks=tool_chunks, id=message.id)
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk


class RecordingFakeChatModel(StructuredFakeChatModel):
    """记录每次调用的 prompt 消息序列（供护栏 / 工具结果注入断言）。

    WHY 声明为 Pydantic 字段：GenericFakeChatModel 是 BaseChatModel（Pydantic 模型），
    运行时 `self.prompts = []` 会触发「无此字段」错误；`exclude=True` 避免序列化。
    """

    prompts: list[list[BaseMessage]] = Field(default_factory=list, exclude=True)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.prompts.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RoutedFakeChatModel(RecordingFakeChatModel):
    """按 prompt 特征文本路由消息脚本的 fake（T7 并行 Send 分支的确定性测试基建）。

    WHY 路由而非共享迭代器：并行扇出的多个子代理分支从同一迭代器消费的顺序由
    asyncio 调度决定（不确定）；按 marker 匹配 prompt 后各分支消费各自路由的
    独立脚本，测试与分支调度顺序解耦。匹配顺序 = routes 列表顺序（先特异后一般）；
    未命中任何 marker 时回退基类迭代器（通常放「意图分类 + 最终整合文本」），
    与 `RecordingFakeChatModel` 的逐条消费契约一致。
    """

    routes: list[tuple[str, list[AIMessage]]] = Field(default_factory=list, exclude=True)
    route_iters: dict[int, Any] = Field(default_factory=dict, exclude=True)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.prompts.append(list(messages))
        text = "\n".join(str(getattr(m, "content", "")) for m in messages)
        for idx, (marker, script) in enumerate(self.routes):
            if marker in text:
                iterator = self.route_iters.get(idx)
                if iterator is None:
                    iterator = iter(script)
                    self.route_iters[idx] = iterator
                try:
                    message = next(iterator)
                except StopIteration as exc:  # 脚本耗尽 = 测试编排错误，显式失败
                    msg = f"路由脚本已耗尽：marker={marker!r}"
                    raise AssertionError(msg) from exc
                return ChatResult(generations=[ChatGeneration(message=message)])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


_fake_call_seq = itertools.count(1)


def fake_structured_message(result: BaseModel) -> AIMessage:
    """构造产出指定结构化结果的 AIMessage（tool name 必须等于 schema 类名）。

    WHY tool_call_id 逐条唯一：图内 `add_messages` 按**消息 id** 去重，而 tool_call_id
    固定会让「同轮第二次结构化调用」与第一次的 tool_call 复用同一个 id
    （LangChain 的 tool_call 幂等合并语义），表现为第二次调用拿到**上一次**的模型输出。
    v6.0 起同一轮会有多次结构化调用（意图分类 + 查询理解 + 澄清），故必须唯一。
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": type(result).__name__,
                "args": result.model_dump(),
                "id": f"call_{next(_fake_call_seq)}",
            }
        ],
    )


def fake_text_message(text: str) -> AIMessage:
    """构造产出指定文本的 AIMessage。"""
    return AIMessage(content=text)


def understand_message(
    main_query: str = "改写后的查询",
    *,
    retrieval_needed: bool = True,
    sub_queries: list[str] | None = None,
    synonyms: list[str] | None = None,
    hypothetical_answer: str | None = None,
) -> AIMessage:
    """构造一次查询理解（T2）的结构化输出消息。

    WHY 需要 helper：v6.0 起每轮在意图分类之后新增一次结构化调用（`understand_query`，
    通过门控时执行）。凡按「调用顺序」编排脚本的测试都必须在意图消息之后插入这条；
    统一 helper 保证各处口径一致（默认 `retrieval_needed=True`，
    使召回 / 澄清 / 规划等既有路径照常执行，测试只多一条消息、断言不变）。
    """
    return fake_structured_message(
        QueryUnderstandingResult(
            understanding=QueryUnderstanding(
                retrieval_needed=retrieval_needed,
                main_query=main_query,
                sub_queries=sub_queries or [],
                synonyms=synonyms or [],
                hypothetical_answer=hypothetical_answer,
            )
        )
    )


def chat_turn_messages(intent: Intent, reply: str, *, understand: bool = False) -> list[AIMessage]:
    """一轮对话所需的 LLM 消息序列：结构化（意图）[+ 查询理解] + 文本（回答）。

    `understand=False`（默认）：适用于 CHAT 短输入 —— 它被确定性门控跳过，
    `understand_query` 不调 LLM，脚本里不能多出这条消息（否则被后续调用误消费）。
    `understand=True`：适用于会通过门控的输入（如 TOOL_USE 意图、较长文本），
    在意图消息之后插入一条查询理解消息。
    """
    messages = [
        fake_structured_message(IntentResult(intent=intent, confidence=0.95, reason="test"))
    ]
    if understand:
        messages.append(understand_message())
    messages.append(fake_text_message(reply))
    return messages


def tool_call_messages(
    tool_calls_seq: Sequence[list[dict[str, Any]]], reply: str
) -> list[AIMessage]:
    """工具型会话消息序列：意图(TOOL_USE) + 查询理解 + 每次 call_model 的 tool_calls + 终答文本。

    消息数契约：fake 一次 LLM 调用消费一条。classify 消费意图、`understand_query`
    消费查询理解（TOOL_USE 意图必然通过门控）、每次 call_model 消费一条 AI
    （带 tool_calls 或最终文本）、dispatch_tool 不消费。末尾 `reply` 被末次
    call_model 消费（产出终答文本 AI），generate_answer 复用该文本。
    """
    return [
        fake_structured_message(
            IntentResult(intent=Intent.TOOL_USE, confidence=0.95, reason="test")
        ),
        understand_message(),
        *[AIMessage(content="", tool_calls=list(calls)) for calls in tool_calls_seq],
        fake_text_message(reply),
    ]


def make_fake_tool(
    name: str,
    *,
    content: str = "ok",
    fail_with: Exception | None = None,
    recorder: list[dict[str, Any]] | None = None,
) -> BaseTool:
    """构造可配置成败的假 BaseTool（StructuredTool），供 ToolNode 执行与断言。

    `recorder` 收集每次调用的参数（dict）；`fail_with` 非空时工具执行抛异常，
    ToolNode（handle_tool_errors=True）会把异常归一为 ToolMessage(status="error")。

    WHY infer_schema=False：args_schema 缺省时 LangChain 对纯 `extra="allow"` 模型
    会误判为「无参数工具」并丢弃入参；无 schema 时 dict 输入原样透传给 coroutine，
    `run_manager`/`config` 绑定命名参数、不进入业务 kwargs。
    """

    async def _run(
        *,
        run_manager: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> str:
        if recorder is not None:
            recorder.append(kwargs)
        if fail_with is not None:
            raise fail_with
        return content

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=f"{name} 假工具",
        infer_schema=False,
    )


@pytest.fixture
def framework_config() -> AgentFrameworkConfig:
    """框架行为默认配置（与生产一致）。"""
    return AgentFrameworkConfig.get_default()


@pytest.fixture
def make_llm_service() -> Callable[..., LLMService]:
    """构造注入离线 fake 的 LLMService（msg 序列按调用顺序被图消费）。

    `model_cls` 可换为 `RecordingFakeChatModel` 等子类（记录 prompt 供断言）。
    """

    def _make(
        messages: list[AIMessage],
        *,
        model_cls: type[StructuredFakeChatModel] = StructuredFakeChatModel,
    ) -> LLMService:
        cfg = LLMConfig(api_key=SecretStr("sk-x"))
        return LLMService(config=cfg, chat_model=model_cls(messages=iter(messages)))

    return _make


@pytest.fixture
def build_graph(
    make_llm_service: Callable[..., LLMService],
) -> Callable[..., Any]:
    """按 LLM 消息序列构建编译后的 Agent 图（可覆盖 config / tools / store）。"""

    def _build(
        messages: list[AIMessage],
        *,
        config: AgentFrameworkConfig | None = None,
        tools: list[BaseTool] | None = None,
        store: BaseStore | None = None,
    ) -> Any:
        return build_agent_graph(make_llm_service(messages), config, tools=tools, store=store)

    return _build


@pytest.fixture
def run_graph() -> Callable[..., Any]:
    """以 stream_mode="updates" 驱动一次会话，返回 (最终 AgentResponse, 全部 chunk)。"""

    async def _run(
        graph: Any,
        *,
        text: str,
        session_id: str = "s1",
        user_id: str | None = None,
    ) -> tuple[AgentResponse | None, list[dict]]:
        chunks: list[dict] = []
        response: AgentResponse | None = None
        _input: dict[str, str] = {"input": text, "session_id": session_id}
        if user_id is not None:
            _input["user_id"] = user_id
        async for chunk in graph.astream(
            _input,
            config={"configurable": {"thread_id": session_id}},
            stream_mode="updates",
        ):
            chunks.append(chunk)
            # 空更新的节点（如 trim_history/validate_output）会产出 {node: None}。
            for node_updates in chunk.values():
                if node_updates and "response" in node_updates:
                    response = node_updates["response"]
        return response, chunks

    return _run


# —— FastAPI 层离线测试设施（注入 fake LLM 的 AgentRuntime + 应用，不依赖真实 LLM/MCP）——


@pytest.fixture
def api_runtime_factory(
    build_graph: Callable[..., Any],
) -> Callable[..., Any]:
    """构造注入 fake LLM 的 AgentRuntime（会话用 SQLite memory 后端，零配置）。

    与 test_runtime.py 的 runtime_factory 同构，供 API 测试构造组合根。
    """

    async def _make(
        messages: list[AIMessage],
        *,
        tools: list[BaseTool] | None = None,
    ) -> AgentRuntime:
        graph = build_graph(messages, tools=tools)
        backend = await build_session_backend(None)
        return AgentRuntime(
            graph=graph,
            sessions=backend.manager,
            memory_store=MemoryStore(InMemoryStore()),
            cfg=AgentFrameworkConfig.get_default(),
            _session_backend=backend,  # 让 aclose() 关闭会话仓库连接池
        )

    return _make


@pytest.fixture
def api_app_factory(api_runtime_factory: Callable[..., Any]) -> Callable[..., Any]:
    """构造注入 fake 组合根的 FastAPI 应用（lifespan 不跑，app.state.runtime 预置）。

    返回 (app, runtime)：测试负责在 finally 中 `await runtime.aclose()`。
    """

    async def _make(
        messages: list[AIMessage],
        *,
        tools: list[BaseTool] | None = None,
    ) -> tuple[Any, AgentRuntime]:
        runtime = await api_runtime_factory(messages, tools=tools)
        app = create_app(runtime=runtime)
        return app, runtime

    return _make
