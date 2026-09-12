"""A2A 协议适配模块（T5 Server 纯逻辑层，app/a2a 只做 HTTP 转发）。

核心子集：AgentCard 发现 / message/send / message/stream(SSE) / task/get /
task/cancel / text Part；能力矩阵显式裁剪见 models.py 模块注释。

WHY 本包 __init__ 只 re-export models（不拉起 mapper/protocol/registry）：
`agent/core/config.py` 需要 `agent.a2a.models.A2APeer`，若包初始化连带导入
mapper（依赖 response/tools 子包），会形成 config ↔ tools 的循环导入。
其余符号请直接从子模块导入（`agent.a2a.protocol` / `agent.a2a.mapper` /
`agent.a2a.registry`）。
"""

from agent.a2a.models import (
    A2A_AGENT_DESCRIPTION,
    A2A_AGENT_NAME,
    A2A_AGENT_VERSION,
    A2AMessage,
    A2APart,
    A2APeer,
    A2ASkill,
    A2AStatusUpdate,
    A2ATask,
    A2ATaskState,
    A2ATaskStatus,
    AgentCard,
    build_agent_card,
    is_terminal,
)

__all__ = [
    "A2A_AGENT_DESCRIPTION",
    "A2A_AGENT_NAME",
    "A2A_AGENT_VERSION",
    "A2AMessage",
    "A2APart",
    "A2APeer",
    "A2ASkill",
    "A2AStatusUpdate",
    "A2ATask",
    "A2ATaskState",
    "A2ATaskStatus",
    "AgentCard",
    "build_agent_card",
    "is_terminal",
]
