"""A2A 协议数据模型（T5 Server 出入站契约，纯 Pydantic）。

A2A（Agent-to-Agent）核心子集（能力矩阵显式裁剪，dev-version5.0.md §4.1）：
AgentCard 发现、Task 生命周期、Message/Part(text)。file / function-call Part、
多技能协商、推送 webhook 本版不支持——模型上不留字段，避免「假装支持」。

语义映射契约（dev-version5.0.md §4.3，映射函数见 mapper.py）：
- task_id == session_id（一一对应，`thread_id == session_id` 契约保持）；
- `AgentResponse.reply` → agent 角色 text Part；citations 序列化为 Task.metadata 附注。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# —— AgentCard 常量（卡片内容静态声明；url 按请求 base_url 动态拼装）——
A2A_AGENT_NAME = "srp-agent"
A2A_AGENT_DESCRIPTION = (
    "虚拟数字人 Agent（A2A 核心子集）：文本问答 + MCP 工具调用；"
    "支持 message/send、message/stream(SSE)、task/get、task/cancel 与 text Part"
)
A2A_AGENT_VERSION = "0.1.0"


class A2ATaskState(StrEnum):
    """任务状态机（与 A2A 规范状态值对齐）：submitted → working → 终态。"""

    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# 终态集合：task/get 轮询与 cancel 判定的依据。
_TERMINAL_STATES = frozenset(
    {A2ATaskState.COMPLETED, A2ATaskState.FAILED, A2ATaskState.CANCELED}
)


def is_terminal(state: A2ATaskState) -> bool:
    """是否终态（completed/failed/canceled）：终态任务不可再取消。"""
    return state in _TERMINAL_STATES


class A2APart(BaseModel):
    """消息部件（本版仅 text）：一段 UTF-8 文本。"""

    kind: Literal["text"] = "text"
    text: str


class A2AMessage(BaseModel):
    """A2A 消息（user=入站提问 / agent=出站回答）。"""

    role: Literal["user", "agent"]
    parts: list[A2APart] = Field(default_factory=list)

    @classmethod
    def from_text(cls, *, role: Literal["user", "agent"], text: str) -> A2AMessage:
        """以单段文本构造消息（本项目语义映射的唯一用法）。"""
        return cls(role=role, parts=[A2APart(text=text)])

    @property
    def text(self) -> str:
        """全部 text part 拼接（入站取文本 / 测试断言共用）。"""
        return "".join(part.text for part in self.parts if part.kind == "text")


class A2ATask(BaseModel):
    """A2A 任务（task ↔ session 一一对应：id 即 session_id）。

    `message` 仅在终态承载 agent 回答（text Part）；`metadata` 是本项目
    附注字段（citations / finished_reason），供调用方回溯答案来源。
    """

    id: str  # == session_id（uuid4 hex）
    session_id: str  # 显式冗余：把「task↔session 一一对应」契约落在数据上
    state: A2ATaskState = A2ATaskState.SUBMITTED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    message: A2AMessage | None = None
    finished_reason: str | None = None  # AgentResponse.finished_reason 透传
    error: str | None = None  # failed 态的可读原因
    metadata: dict[str, object] | None = None


class A2ATaskStatus(BaseModel):
    """任务状态快照（过程帧承载 working 进度说明 / 回答增量）。"""

    state: A2ATaskState = A2ATaskState.WORKING
    message: str | None = None  # 状态/工具进度说明（供调用方展示）
    delta: str | None = None  # 回答 token 增量（拼接 == 终态 reply）


class A2AStatusUpdate(BaseModel):
    """message/stream 过程帧 result（status-update：A2A 流式子集形态）。"""

    kind: Literal["status-update"] = "status-update"
    taskId: str
    contextId: str  # 即 session_id（task↔session 一一对应的流式侧体现）
    status: A2ATaskStatus = Field(default_factory=A2ATaskStatus)


class A2APeer(BaseModel):
    """远端调用方（peer）注册项：无 X-User-Id 的入站身份映射。

    `user_id` 为空 → 匿名命名空间 `a2a:<id>`（记忆/会话按该虚拟用户隔离，
    不混入人类用户数据）；配置显式映射可给可信 peer 固定身份。
    """

    id: str
    user_id: str | None = None
    enabled: bool = True


class A2ASkill(BaseModel):
    """AgentCard 技能声明（单技能：通用文本协作）。"""

    id: str = "general-assistance"
    name: str = "通用文本协助"
    description: str = "接收文本任务，返回文本回答（可调用内部工具）"
    tags: list[str] = Field(default_factory=lambda: ["text", "chat"])


class AgentCard(BaseModel):
    """A2A 发现名片：`GET /.well-known/agent.json` 的响应体。

    capabilities 如实声明本版能力矩阵：streaming=True（SSE）、
    pushNotifications=False（无推送 webhook）、stateTransitionHistory=False
    （task 只暴露当前状态，不含历史迁移记录）。
    """

    name: str = A2A_AGENT_NAME
    description: str = A2A_AGENT_DESCRIPTION
    url: str  # JSON-RPC 端点（<base>/a2a），由请求 base_url 动态拼装
    version: str = A2A_AGENT_VERSION
    protocol_version: str = "0.3.0"  # 对齐 A2A 规范版本的子集声明
    preferred_transport: str = "JSONRPC"
    default_input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    capabilities: dict[str, bool] = Field(
        default_factory=lambda: {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        }
    )
    skills: list[A2ASkill] = Field(default_factory=lambda: [A2ASkill()])


def build_agent_card(*, base_url: str) -> AgentCard:
    """按请求来源拼装 AgentCard（url 指向本实例的 /a2a 端点）。"""
    return AgentCard(url=f"{base_url.rstrip('/')}/a2a")
