"""agent/tools/lark_scope.py 单测：飞书工具作用域注入（离线，零 MCP、零网络）。

覆盖 dev-version5.1.md §5.4 的三条硬要求：
1. **注入**：lark 工具调用的 `_lark_scope` = 图状态 `user_id`（`ToolNode` 透传）；
2. **覆盖**：模型在 args 里瞎填的 `_lark_scope` 被**无条件覆盖**；
3. **不越界**：非 lark 服务的调用原样放行（零影响）；
另加「LLM 不可见」断言：真实服务端 schema 不含 `_lark_scope`（并保留兜底剥离能力），
且模型视图与执行视图互不污染（注入的身份仍被工具接受）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, TypedDict

from fastmcp import Client, FastMCP
from langchain_core.messages import AIMessage
from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools.lark_scope import (
    LARK_SCOPE_ARG,
    UNKNOWN_SCOPE,
    LarkScopeInterceptor,
    is_lark_unbound_message,
    visible_tools,
)
from shared.lark.errors import LARK_UNBOUND_PREFIX

# 飞书工具服务端签名（含注入参数）的真实形状：args_schema 是 JSON schema dict。
LARK_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "string"},
        LARK_SCOPE_ARG: {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": [],
}


class _Runtime:
    """替身 ToolRuntime：只保留拦截器读取的 `state` 属性。"""

    def __init__(self, state: dict[str, Any] | None) -> None:
        self.state = state


class _Request:
    """替身 MCPToolCallRequest（name/args/server_name/runtime + override）。"""

    def __init__(
        self,
        *,
        name: str,
        args: dict[str, Any],
        server_name: str,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.args = args
        self.server_name = server_name
        self.headers: dict[str, Any] | None = None
        self.runtime = _Runtime(state)

    def override(self, **overrides: Any) -> _Request:
        clone = _Request(
            name=self.name,
            args=dict(self.args),
            server_name=self.server_name,
            state=getattr(self.runtime, "state", None),
        )
        clone.name = overrides.get("name", self.name)
        clone.args = overrides.get("args", dict(self.args))
        clone.headers = overrides.get("headers", self.headers)
        return clone


class _Handler:
    """记录实际下发的 args（= 真正上 MCP 线的载荷）。"""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def __call__(self, request: _Request) -> str:
        self.seen.append(dict(request.args))
        return "ok"


async def test_injects_user_id_into_lark_tool_args() -> None:
    """图状态 user_id → `_lark_scope`（V51-M1 的注入侧）。"""
    interceptor = LarkScopeInterceptor()
    handler = _Handler()
    request = _Request(
        name="lark_calendar_get_agenda",
        args={"start": "2026-09-12"},
        server_name="lark_mcp",
        state={"messages": [], "user_id": "user-a"},
    )
    await interceptor(request, handler)
    assert handler.seen == [{"start": "2026-09-12", LARK_SCOPE_ARG: "user-a"}]


async def test_overrides_model_supplied_scope() -> None:
    """模型瞎填的 `_lark_scope` 被无条件覆盖（身份不由模型决定，§5.2）。"""
    interceptor = LarkScopeInterceptor()
    handler = _Handler()
    request = _Request(
        name="lark_calendar_get_agenda",
        args={LARK_SCOPE_ARG: "user-b", "start": "2026-09-12"},
        server_name="lark_mcp",
        state={"user_id": "user-a"},
    )
    await interceptor(request, handler)
    assert handler.seen[0][LARK_SCOPE_ARG] == "user-a"


async def test_identity_isolation_across_calls() -> None:
    """连续两个不同用户的调用 → 各自的作用域（互不串号）。"""
    interceptor = LarkScopeInterceptor()
    handler = _Handler()
    for user_id in ("user-a", "user-b"):
        await interceptor(
            _Request(
                name="lark_calendar_get_agenda",
                args={},
                server_name="lark_mcp",
                state={"user_id": user_id},
            ),
            handler,
        )
    assert [args[LARK_SCOPE_ARG] for args in handler.seen] == ["user-a", "user-b"]


async def test_non_lark_server_passes_through_untouched() -> None:
    """非 lark 服务（tools_mcp / rag / a2a_mcp）原样放行，不注入任何键。"""
    interceptor = LarkScopeInterceptor()
    handler = _Handler()
    request = _Request(
        name="calculate",
        args={"expression": "1+1"},
        server_name="tools_mcp",
        state={"user_id": "user-a"},
    )
    await interceptor(request, handler)
    assert handler.seen == [{"expression": "1+1"}]


async def test_missing_user_id_falls_back_to_unknown_scope() -> None:
    """图状态无 user_id → 哨兵作用域（服务侧必然判未绑定 → 引导绑定），不退化为匿名调用。"""
    interceptor = LarkScopeInterceptor()
    handler = _Handler()
    request = _Request(name="lark_docs_read", args={"doc": "x"}, server_name="lark_mcp", state={})
    await interceptor(request, handler)
    assert handler.seen[0][LARK_SCOPE_ARG] == UNKNOWN_SCOPE
    assert UNKNOWN_SCOPE  # 非空（空串会退化成无作用域调用）


def test_real_mcp_tools_declare_scope_and_no_as_user() -> None:
    """真实服务端契约：每个飞书工具都声明了 `_lark_scope`，且 `as_user` 已删除。

    WHY 打真实服务端（进程内 FastMCP 客户端，零网络、零 CLI）：这条把两个事实钉死 ——
    ①FastMCP **会**把下划线前缀形参（`_lark_scope`）暴露进 `inputSchema`，
    故「LLM 不可见」**不能**指望服务端隐藏，必须由 agent 侧 `visible_tools()` 剥离；
    ②v5.1 已整体删除 bot 身份开关 `as_user`（V51-M4）。
    """
    from fastmcp import Client

    from services.lark_mcp.server import mcp

    async def _list() -> list[Any]:
        async with Client(mcp) as client:
            return await client.list_tools()

    listed = asyncio.run(_list())
    assert listed, "飞书工具面不应为空"
    for tool in listed:
        properties = set((tool.inputSchema or {}).get("properties", {}))
        assert LARK_SCOPE_ARG in properties, f"{tool.name} 未声明内部作用域参数"
        assert "as_user" not in properties, f"{tool.name} 仍暴露已废弃的 as_user"


def test_visible_tools_hide_scope_from_model_view() -> None:
    """`visible_tools` 剥掉模型视图的 `_lark_scope`，且**不污染执行侧原工具**。

    WHY 关键：同一工具对象既供 `bind_tools` 又供 `ToolNode`；若就地改写，
    执行侧丢掉形参会让注入的实参无处落地（或反之让模型看到内部参数）。
    两条断言分别锁住「模型不可见」与「执行不受影响」。
    """
    original = StructuredTool(
        name="lark_calendar_get_agenda",
        description="查询日程",
        args_schema=dict(LARK_TOOL_SCHEMA),
        coroutine=_noop,
    )
    visible = visible_tools([original])
    assert len(visible) == 1
    model_view = visible[0].tool_call_schema
    assert LARK_SCOPE_ARG not in model_view["properties"]
    assert "start" in model_view["properties"]
    # 原工具未被改写：执行侧仍声明 `_lark_scope`（拦截器注入的实参可落地）。
    assert LARK_SCOPE_ARG in original.args_schema["properties"]
    assert visible[0] is not original
    assert visible[0].name == original.name


def test_visible_tools_passes_non_lark_tools_through_identically() -> None:
    """非飞书工具（无 scope）原样返回同一对象：零拷贝、零行为差异（零回归）。"""
    plain = StructuredTool(
        name="calculate",
        description="计算",
        args_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
        coroutine=_noop,
    )
    assert visible_tools([plain]) == [plain]
    assert visible_tools([plain])[0] is plain


def test_is_lark_unbound_message_prefix_detection() -> None:
    """未绑定识别锚点：仅常量前缀命中（跨 MCP 线后异常退化为文本）。"""
    assert is_lark_unbound_message(f"{LARK_UNBOUND_PREFIX}用户 user-a 尚未绑定飞书")
    assert not is_lark_unbound_message("lark-cli im +messages-send 失败：boom")
    assert not is_lark_unbound_message("")


def test_tool_node_passes_user_id_into_tool_runtime_state() -> None:
    """机制层验证：`ToolNode` 输入 dict 的额外键会进入 `ToolRuntime.state`。

    WHY 这是整条身份链路的**前置条件**：上一版设计依赖 `ToolRuntime` /
    `InjectedToolArg`（对 MCP 远程工具**结构性不可用**），改用客户端拦截器后，
    拦截器读的是 `request.runtime.state` —— 该 state 必须真的带上图状态的 `user_id`
    （图内 `dispatch_tool` 以 `{"messages": ..., "user_id": ...}` 调 ToolNode）。
    仅传 `messages` 的旧写法会让**所有用户的作用域退化为同一值**（隔离整体失效），
    故此处把「额外键确实透传」钉死，防止回退。

    WHY 走编译图而非裸 `ainvoke`：`InjectedToolArg` 需要 LangGraph 运行时上下文，
    裸调 `ToolNode.ainvoke(dict)` 会因缺 runtime 配置报错 —— 图内调用才与生产一致。
    """
    captured: dict[str, Any] = {}

    async def _tool(
        start: str | None = None,
        runtime: Annotated[Any, InjectedToolArg()] = None,
    ) -> str:
        captured["runtime_state"] = getattr(runtime, "state", None)
        captured["start"] = start
        return json.dumps({"ok": True})

    tool = StructuredTool(
        name="probe",
        description="探测运行时状态",
        args_schema={"type": "object", "properties": {"start": {"type": "string"}}},
        coroutine=_tool,
    )
    node = ToolNode([tool], handle_tool_errors=True)

    class _State(TypedDict, total=False):
        messages: Annotated[list[Any], add_messages]
        user_id: str | None

    builder = StateGraph(_State)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "probe", "args": {"start": "2026-09-12"}, "id": "call-1", "type": "tool_call"}
        ],
    )
    out = asyncio.run(graph.ainvoke({"messages": [ai], "user_id": "user-a"}))
    assert out["messages"][-1].status != "error", out["messages"][-1].content
    # 图状态 user_id 确实随 dict 输入透传进 ToolRuntime.state（拦截器的数据来源）。
    assert captured["runtime_state"]["user_id"] == "user-a"
    assert captured["start"] == "2026-09-12"


def test_interceptor_is_a_valid_adapter_tool_interceptor() -> None:
    """装配契约：`LarkScopeInterceptor` 必须是 adapters 认得的 `ToolCallInterceptor`。

    WHY 单独断言：工厂把它传给 `MultiServerMCPClient(tool_interceptors=[...])`，
    若签名/协议不符，装配期不会报错而是在调用期静默不注入（身份隔离整体失效）。
    """
    from langchain_mcp_adapters.interceptors import ToolCallInterceptor

    assert isinstance(LarkScopeInterceptor(), ToolCallInterceptor)


def test_injected_scope_reaches_server_tool_as_argument() -> None:
    """服务端契约（进程内 FastMCP，零网络）：`_lark_scope` 是工具形参并能被读取。

    WHY 这条是「注入 → 服务端取到作用域」的兑现点：agent 侧注入的实参名必须与
    服务端签名一致（`_lark_scope`），否则作用域在 MCP 线上被丢弃 → 所有用户都会
    退化成同一身份（越权风险）。用真实 `services.lark_mcp.server` 的组件形状
    （FastMCP + 同名工具 + 同签名）验证参数**能**被读取。
    """
    seen: dict[str, Any] = {}

    async def _run(scope: str | None, start: str | None) -> str:
        seen["scope"] = scope
        seen["start"] = start
        return json.dumps({"ok": True})

    server = FastMCP("lark-contract")

    @server.tool(name="lark_calendar_get_agenda")
    async def _agenda(start: str | None = None, _lark_scope: str | None = None) -> str:
        return await _run(_lark_scope, start)

    async def _call() -> Any:
        async with Client(server) as client:
            return await client.call_tool(
                "lark_calendar_get_agenda",
                {"start": "2026-09-12", LARK_SCOPE_ARG: "user-a"},
            )

    asyncio.run(_call())
    assert seen["scope"] == "user-a"
    assert seen["start"] == "2026-09-12"


async def _noop(**kwargs: Any) -> str:
    """占位协程（本测试只关心 schema，不执行工具）。"""
    return str(kwargs)
