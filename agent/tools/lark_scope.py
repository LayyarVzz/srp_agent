"""飞书工具调用拦截器：把「谁在调用」注入到 MCP 工具实参（v5.1，dev-version5.1.md §5）。

**问题**：工具是 MCP 远程工具，装配期一次性 `get_tools()`；图内 `ToolNode` 执行时
`tool_node.ainvoke({"messages": ...})` 不带 `user_id`，而 `ToolRuntime` / `InjectedToolArg`
等注入机制对 MCP 适配出的远程 `StructuredTool` **结构性不可用**（只注入本地 Python 工具）。

**方案（D5.1.2）**：`langchain-mcp-adapters` 的 `ToolCallInterceptor` —— 在
**客户端校验之后、直接上 MCP 线之前**改写 `request.args`，无条件覆盖 `_lark_scope`。

- 用 `args` 而非 `headers`：`headers` 只在 HTTP 传输生效（stdio 被静默忽略），
  `args` 两种传输都通 → 作为**唯一**主通道，避免双机制漂移；
- 拦截器注入是安全的：即便模型在 `_lark_scope` 里瞎填，也会被**无条件覆盖**；
- 另经 `visible_tools()` 给 LLM 绑定**剥离 `_lark_scope` 的副本**（原对象继续用于
  执行）——FastMCP 会把该形参声明进 `inputSchema`（实测下划线前缀**不会**被隐藏），
  故「LLM 不可见」必须由 agent 侧显式剥离保证，不能指望服务端不暴露。

作用域值即图状态 `user_id`（`AgentState.user_id`，由入口层按会话归属校验后传入）
—— 与记忆/会话同一 `user_id` 口径，天然按用户隔离、互不可见。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool

from shared.lark import LARK_MCP_SERVER_NAME
from shared.lark.errors import LARK_UNBOUND_PREFIX, TOOL_ERROR_LARK_UNBOUND

logger = logging.getLogger(__name__)

# 注入实参名：以下划线开头表示「内部参数」，与飞书工具的业务参数区分。
LARK_SCOPE_ARG = "_lark_scope"

# 未知作用域哨兵：图状态缺 user_id 时使用（服务侧必然视为未绑定 → 引导绑定），
# 绝不用空串/None —— 那会退化成「无作用域」的歧义调用。
UNKNOWN_SCOPE = "__unknown__"


def is_lark_unbound_message(content: str) -> bool:
    """判定 ToolMessage 内容是否为「飞书未绑定」（确定性前缀识别）。

    WHY 靠常量前缀：异常跨 MCP 线后退化成文本，前缀（`tool_error.lark_unbound:`）
    是 agent 侧识别该语义的唯一可靠依据（见 shared/lark/errors.py）。
    """
    return LARK_UNBOUND_PREFIX.strip() in content


def _scope_from_request(request: Any) -> str:
    """从拦截请求的 LangGraph runtime 中取出作用域（图状态 user_id）。

    `ToolNode` 对普通 dict 输入直接透传（`_extract_state` 对 dict 返回 input as-is），
    故 `runtime.state["user_id"]` 即本轮会话用户；缺失时返回哨兵（→ 未绑定引导）。
    """
    runtime = getattr(request, "runtime", None)
    state = getattr(runtime, "state", None)
    if isinstance(state, dict):
        user_id = state.get("user_id")
        if isinstance(user_id, str) and user_id:
            return user_id
    logger.warning(
        "lark 工具调用缺少作用域（图状态无 user_id），按未绑定处理：tool=%s",
        getattr(request, "name", "<unknown>"),
    )
    return UNKNOWN_SCOPE


class LarkScopeInterceptor:
    """仅作用于 lark_mcp 服务：把 `_lark_scope` 无条件覆盖为该调用的用户身份。

    非 lark 服务与 lark 的绑定工具（绑定的就是调用者本人）一律原样放行 ——
    拦截器只做「补身份」，不改业务参数、不做鉴权决策。
    """

    def __init__(self, *, server_name: str = LARK_MCP_SERVER_NAME) -> None:
        self._server_name = server_name

    async def __call__(self, request: Any, handler: Any) -> Any:
        if getattr(request, "server_name", None) != self._server_name:
            return await handler(request)
        scope = _scope_from_request(request)
        args = dict(getattr(request, "args", None) or {})
        # 无条件覆盖（含模型瞎填的值）：身份只由调用方决定，不由模型决定。
        args[LARK_SCOPE_ARG] = scope
        return await handler(request.override(args=args))


def visible_tools(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """返回**供 LLM 绑定**的工具副本：飞书工具去掉 `_lark_scope`，其余原样返回。

    WHY 需要「另一份」而不是改写原工具：同一个工具对象承担两个职责 ——
    `bind_tools`（给模型看的声明）与 `ToolNode`（执行实体）。执行侧必须保留
    `_lark_scope` 形参（拦截器注入的实参要能被 MCP 服务接受，服务端 FastMCP
    确实把它声明在 `inputSchema` 里 —— 实测非下划线参数同样暴露），而模型侧
    必须看不到它（§5.4-2：LLM 不可见、不可填，避免模型试图自行指定他人身份）。
    故此处返回**剥离后的副本**专供绑定，原对象继续用于执行，两侧互不影响。

    WHY 不在执行侧也剥离：`ToolNode` 按 `_lark_scope` 之外的实参查表执行，
    但 MCP 服务端按自己的签名校验；把服务端声明的参数从客户端 schema 抹掉
    只会让「模型可见」与「服务端接受」出现第三种形态，增加无谓的耦合面。

    非飞书工具（无 `_lark_scope`）原样返回同一对象 —— 零拷贝、零行为差异。
    """
    result: list[BaseTool] = []
    for tool in tools:
        schema = tool.tool_call_schema
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict) or LARK_SCOPE_ARG not in properties:
            result.append(tool)
            continue
        visible_schema: dict[str, Any] = {
            **schema,
            "properties": {k: v for k, v in properties.items() if k != LARK_SCOPE_ARG},
        }
        required = schema.get("required")
        if isinstance(required, list) and LARK_SCOPE_ARG in required:
            visible_schema["required"] = [name for name in required if name != LARK_SCOPE_ARG]
        # 只改「给模型看的那份」：失败的默认值不应导致模型受益于内部参数的存在。
        clone = tool.model_copy(deep=False)
        clone.args_schema = visible_schema
        result.append(clone)
        logger.debug("飞书工具 %s 的模型视图已隐藏内部参数 %s", tool.name, LARK_SCOPE_ARG)
    return result


__all__ = [
    "LARK_SCOPE_ARG",
    "TOOL_ERROR_LARK_UNBOUND",
    "UNKNOWN_SCOPE",
    "LarkScopeInterceptor",
    "is_lark_unbound_message",
    "visible_tools",
]
