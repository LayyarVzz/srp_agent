"""飞书绑定域共享基础设施（v5.1，dev-version5.1.md）。

放在根级 `shared/` 而非 `agent/lark/` 的理由（对设计文档 §10 的落点修正）：
`lark_bindings` 表 / OAuth 客户端 / 令牌加解密的**消费方是 `services/lark_mcp`（容器内）**，
而 CLAUDE.md 硬约束「模块间只允许上层依赖下层接口」——`services` 反向 import `agent`
是越界。本包只依赖 stdlib + pydantic + sqlalchemy + httpx，两个消费方各自独立部署。
"""

from __future__ import annotations

from shared.lark.errors import (
    LARK_UNBOUND_PREFIX,
    TOOL_ERROR_LARK_UNBOUND,
    LarkBoundError,
    LarkCliError,
    LarkUnboundError,
    unbound_message,
)

# MCP 服务名常量（单一来源）：agent 侧拦截器据此判定「是否飞书工具」
# （agent 不 import services，故常量落在共享层；services/lark_mcp/client_config.py 复用之）。
LARK_MCP_SERVER_NAME = "lark_mcp"

__all__ = [
    "LARK_MCP_SERVER_NAME",
    "LARK_UNBOUND_PREFIX",
    "TOOL_ERROR_LARK_UNBOUND",
    "LarkBoundError",
    "LarkCliError",
    "LarkUnboundError",
    "unbound_message",
]
