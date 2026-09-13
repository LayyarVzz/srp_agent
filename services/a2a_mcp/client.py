"""A2A 远端智能体客户端：发现 / message/send / task 轮询 / message/stream（T6 核心）。

与 agent/a2a（本仓库 Server 子集）协议形态对称：AgentCard 发现 → message/send
（非终态则 task/get 轮询至终态或超时）→ 终态 Task 提取 reply；stream 路径消费
`message/stream` 的 SSE 帧（status-update 过程帧 + 终态 Task 帧）。

WHY 自包含不 import agent 包：services/* 独立打包部署（镜像只拷贝 services/），
协议编解码在本模块用最小 dict 形态实现（与 Server 侧 JSON-RPC 信封约定一致）。

错误契约：远端不可达 / 协议错误 / 任务 failed / 超时 → `A2AClientError`
（FastMCP 转错误工具结果，Agent 侧归一 tool_error.execution）。
远端回答回流 = ToolMessage 不可信数据（既有声明机制覆盖，禁止拼入 system 指令）。
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 任务终态集合（与 A2A 规范状态值对齐；agent/a2a/models.A2ATaskState 同源）。
_TERMINAL_STATES = frozenset({"completed", "failed", "canceled"})

_SSE_DATA_PREFIX = "data: "

# JSON-RPC 请求 id（进程内自增即可：客户端与远端无跨进程关联需求）。
_request_ids = itertools.count(1)


class A2AClientError(Exception):
    """A2A 调用失败（远端不可达 / 协议错误 / 任务失败 / 超时）。"""


class A2AAgentResult(BaseModel):
    """一次远端智能体调用的结构化结果（MCP 工具输出契约）。"""

    agent_id: str
    task_id: str | None = None
    state: str  # completed / failed / canceled（A2A 终态）
    reply: str = ""  # agent 角色 text part 拼接
    progress: list[str] = Field(default_factory=list)  # 过程进度摘要（供调用方展示）


def _rpc_body(method: str, params: dict[str, object]) -> dict[str, object]:
    """构造 JSON-RPC 2.0 请求体（信封与 agent/a2a/protocol.py 约定一致）。"""
    return {"jsonrpc": "2.0", "id": next(_request_ids), "method": method, "params": params}


def _message_params(message: str) -> dict[str, object]:
    """A2A message 入参：user 角色 + 单 text part（Server 侧 _require_message_text 同构）。"""
    return {"message": {"role": "user", "parts": [{"kind": "text", "text": message}]}}


def _reply_of(task: dict[str, Any]) -> str:
    """提取 Task.message 的全部 text part 拼接（AgentResponse.reply 的镜像映射）。"""
    message = task.get("message")
    if not isinstance(message, dict):
        return ""
    parts = message.get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, dict) and part.get("kind", "text") == "text"
    )


def _extract_result(payload: dict[str, Any]) -> dict[str, Any]:
    """JSON-RPC 响应 → result；error 信封 → A2AClientError（data.code 承载 a2a.* 码）。"""
    error = payload.get("error")
    if isinstance(error, dict):
        data = error.get("data") if isinstance(error.get("data"), dict) else {}
        a2a_code = str(data.get("code") or "a2a.error")
        raise A2AClientError(f"远端错误 {a2a_code}: {error.get('message')}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise A2AClientError("远端响应缺少 result 对象")
    return result


class A2AClient:
    """远端 A2A 智能体客户端（asyncio + httpx；transport 供测试注入离线传输）。

    `peer_id` 是本方对远端声明的调用方身份（`X-A2a-Peer-Id` 请求头，与
    Server 侧 peer 注册表约定一致）：远端未登记该 peer → a2a.invalid_request。
    """

    def __init__(
        self,
        *,
        request_timeout_s: float = 120.0,
        poll_interval_s: float = 1.0,
        peer_id: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._request_timeout_s = request_timeout_s
        self._poll_interval_s = poll_interval_s
        self._peer_id = peer_id
        self._transport = transport

    def _http_client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(self._request_timeout_s)
        headers = {"X-A2a-Peer-Id": self._peer_id} if self._peer_id else None
        return httpx.AsyncClient(timeout=timeout, headers=headers, transport=self._transport)

    async def get_agent_card(self, base_url: str) -> dict[str, Any]:
        """发现远端 AgentCard（`GET /.well-known/agent.json`），返回原始 dict。"""
        url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        try:
            async with self._http_client() as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise A2AClientError(f"远端 AgentCard 获取失败 {url}: {exc}") from exc
        card = resp.json()
        if not isinstance(card, dict):
            raise A2AClientError("远端 AgentCard 不是 JSON 对象")
        return card

    async def call_agent(
        self, *, base_url: str, agent_id: str, message: str
    ) -> A2AAgentResult:
        """同步调用：message/send → 非终态则 task/get 轮询至终态（总预算内）。"""
        url = f"{base_url.rstrip('/')}/a2a"
        try:
            async with self._http_client() as client:
                resp = await client.post(
                    url, json=_rpc_body("message/send", _message_params(message))
                )
                resp.raise_for_status()
                task = _extract_task_from_response(resp)
                task = await self._await_terminal(client, url, task)
        except httpx.HTTPError as exc:
            raise A2AClientError(f"远端智能体不可达 {url}: {exc}") from exc
        return _result_from_task(agent_id=agent_id, task=task)

    async def call_agent_stream(
        self, *, base_url: str, agent_id: str, message: str
    ) -> A2AAgentResult:
        """流式调用：消费 message/stream SSE 帧，聚合过程进度与终态 reply。"""
        url = f"{base_url.rstrip('/')}/a2a"
        progress: list[str] = []
        task: dict[str, Any] | None = None
        try:
            async with self._http_client() as client:
                async with client.stream(
                    "POST", url, json=_rpc_body("message/stream", _message_params(message))
                ) as resp:
                    resp.raise_for_status()
                    async for frame in _iter_sse_frames(resp):
                        result = _extract_result(frame)
                        if result.get("kind") == "status-update":
                            task_id = result.get("taskId")
                            status = result.get("status")
                            text = status.get("message") if isinstance(status, dict) else None
                            if isinstance(text, str) and text:
                                progress.append(text)
                            if task_id and task is None:
                                task = {"id": task_id, "state": "working"}
                            continue
                        if isinstance(result.get("state"), str):  # 终态 Task 帧
                            task = result
        except httpx.HTTPError as exc:
            raise A2AClientError(f"远端智能体不可达 {url}: {exc}") from exc
        if task is None:
            raise A2AClientError("远端流式响应未返回终态 Task")
        result = _result_from_task(agent_id=agent_id, task=task)
        result.progress = progress
        return result

    # —— 私有 ——

    async def _await_terminal(
        self, client: httpx.AsyncClient, url: str, task: dict[str, Any]
    ) -> dict[str, Any]:
        """task/get 轮询至终态；总预算耗尽 → A2A 超时错误（重试由 MCP 连接层承担）。"""
        deadline = time.monotonic() + self._request_timeout_s
        task_id = str(task.get("id") or "")
        while task.get("state") not in _TERMINAL_STATES:
            if time.monotonic() > deadline:
                budget = self._request_timeout_s
                raise A2AClientError(f"远端任务超时未终态（>{budget}s）: {task_id}")
            await asyncio.sleep(self._poll_interval_s)
            resp = await client.post(url, json=_rpc_body("task/get", {"id": task_id}))
            resp.raise_for_status()
            task = _extract_task_from_response(resp)
        return task


def _extract_task_from_response(resp: httpx.Response) -> dict[str, Any]:
    """解析 JSON-RPC 响应 → Task dict（error 信封 → A2AClientError）。"""
    return _extract_result(resp.json())


def _result_from_task(*, agent_id: str, task: dict[str, Any]) -> A2AAgentResult:
    """终态 Task → 结构化结果；failed 态 → A2AClientError（reply 为降级/错误文案）。"""
    state = str(task.get("state") or "")
    reply = _reply_of(task)
    if state == "failed":
        raise A2AClientError(f"远端任务失败: {task.get('error') or reply or '未知原因'}")
    return A2AAgentResult(
        agent_id=agent_id,
        task_id=str(task.get("id")) if task.get("id") else None,
        state=state,
        reply=reply,
    )


async def _iter_sse_frames(resp: httpx.Response) -> Any:
    """逐帧解析 `event: message` + `data: <json>` 的 SSE 流（yield JSON-RPC 帧 dict）。"""
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        try:
            yield json.loads(line[len(_SSE_DATA_PREFIX) :])
        except ValueError as exc:
            raise A2AClientError(f"远端 SSE 帧不是合法 JSON: {line[:80]!r}") from exc
