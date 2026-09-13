"""A2A HTTP 端点（T5）：AgentCard 发现 + JSON-RPC 2.0 单端点（薄路由）。

只做参数校验、peer 身份 fail-fast 与 JSON/SSE 编码转发，无业务逻辑——
任务编排委托 `AgentRuntime.run_a2a_task(_stream)`，协议纯逻辑在 agent/a2a。

对外面（dev-version5.0.md §4.2）：
- `GET /.well-known/agent.json` → AgentCard（url 动态指向本实例 /a2a）；
- `POST /a2a` → JSON-RPC 2.0 分发：message/send（同步返回终态 Task）/
  message/stream（SSE）/ task/get / task/cancel；HTTP 恒 200，
  协议错误以 JSON-RPC error 信封表达（标准码 + data.code 承载 a2a.*）；
- peer 身份经 `X-A2A-Peer-Id` 请求头声明（配置级 peer 身份，复杂认证为非目标），
  未登记 peer → a2a.invalid_request；A2A 关闭时方法调用同样拒绝（发现不受影响）。

message/stream：SSE 帧全部为 `event: message` + JSON-RPC 响应信封 data——
首帧宣告 task_id（受理），过程帧 result 为 status-update（状态/工具进度 + token
增量），末帧 result 为终态 Task（reply text Part）。流式任务登记进
`runtime.a2a_tasks`：调用方从首帧取得 task_id 后可 task/get 轮询与 task/cancel
取消；message/send 为同步路径，执行期间调用方尚无 task_id，注册表在完成时登记。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from agent.a2a.mapper import (
    accepted_frame,
    progress_frame,
    task_frame,
    task_from_response,
)
from agent.a2a.models import A2AMessage, build_agent_card
from agent.a2a.protocol import (
    A2A_ERROR_INTERNAL,
    A2A_ERROR_METHOD_NOT_SUPPORTED,
    A2A_METHOD_MESSAGE_SEND,
    A2A_METHOD_MESSAGE_STREAM,
    A2A_METHOD_TASK_CANCEL,
    A2A_METHOD_TASK_GET,
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_METHOD_NOT_FOUND,
    A2AProtocolError,
    decode_request,
    error_response,
    invalid_request,
)
from agent.response.models import AgentResponse
from agent.runtime import AgentRuntime
from app.deps import get_runtime
from app.sse import sse_frame

logger = logging.getLogger(__name__)

router = APIRouter(tags=["a2a"])

RuntimeDep = Annotated[AgentRuntime, Depends(get_runtime)]

# peer 身份请求头（配置级 peer 身份约定，见模块 docstring）。
PEER_ID_HEADER = "X-A2a-Peer-Id"

# 与 chat 路由同款 SSE 头（流式不压缩、不经反代缓冲；本地常量避免跨路由耦合）。
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# —— 参数提取（路由层校验，全部 fail-fast 为 a2a.invalid_request）——


def _require_peer_id(request: Request) -> str:
    """提取 peer 身份请求头；缺失/为空 → a2a.invalid_request。"""
    peer_id = (request.headers.get(PEER_ID_HEADER) or "").strip()
    if not peer_id:
        raise invalid_request(f"缺少 {PEER_ID_HEADER} 请求头（peer 身份声明）")
    return peer_id


def _require_message_text(params: dict[str, object]) -> str:
    """提取入站任务文本：params.message 按 A2AMessage 校验，text part 拼接须非空。"""
    try:
        message = A2AMessage.model_validate(params.get("message"))
    except ValidationError as exc:
        raise invalid_request("params.message 非法（需 user 角色 + 非空 text part）") from exc
    text = message.text.strip()
    if not text:
        raise invalid_request("params.message 至少需要一个非空 text part")
    if message.role != "user":
        raise invalid_request("params.message.role 必须为 user")
    return text


def _require_task_id(params: dict[str, object]) -> str:
    """提取 task/get / task/cancel 的 params.id。"""
    task_id = params.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise invalid_request("params.id 必须为非空字符串")
    return task_id


# —— method 处理（send 同步 / stream SSE）——


async def _handle_send(
    runtime: AgentRuntime, request_id: str | int | None, params: dict[str, object], peer_id: str
) -> dict[str, object]:
    """message/send：同步执行到终态，返回 Task 帧（完成后登记注册表供 task/get）。"""
    response = await runtime.run_a2a_task(peer_id=peer_id, text=_require_message_text(params))
    task = task_from_response(response.session_id, response)
    runtime.a2a_tasks.record(task)
    return task_frame(request_id, task)


def _handle_stream(
    runtime: AgentRuntime, request_id: str | int | None, params: dict[str, object], peer_id: str
) -> StreamingResponse:
    """message/stream：SSE 下发过程帧 + 终态 Task 帧（任务全程登记注册表，可取消）。"""
    text = _require_message_text(params)
    runtime.resolve_a2a_peer(peer_id)  # fail-fast：流开始前完成 peer 校验（不再产生协议错误）

    async def generate() -> Any:
        task_id: str | None = None
        handle = asyncio.current_task()
        try:
            async for kind, payload in runtime.run_a2a_task_stream(peer_id=peer_id, text=text):
                if kind == "started":
                    task_id = cast(str, payload)
                    runtime.a2a_tasks.register(task_id)
                    if handle is not None:
                        runtime.a2a_tasks.attach_handle(task_id, handle)
                    runtime.a2a_tasks.mark_working(task_id)
                    yield sse_frame(
                        "message", accepted_frame(request_id, task_id=task_id, session_id=task_id)
                    )
                    continue
                if task_id is None:  # 防御：未受理前不存在过程事件
                    continue
                if kind == "done":
                    response = cast(AgentResponse, payload)
                    task = task_from_response(task_id, response)
                    runtime.a2a_tasks.record(task)
                    yield sse_frame("message", task_frame(request_id, task))
                else:
                    yield sse_frame(
                        "message",
                        progress_frame(
                            request_id,
                            task_id=task_id,
                            session_id=task_id,
                            event=kind,
                            payload=payload,
                        ),
                    )
        except asyncio.CancelledError:
            # 客户端断开 / task/cancel 取消执行句柄：任务置 canceled 后随取消语义终止。
            if task_id is not None:
                runtime.a2a_tasks.mark_canceled(task_id)
            raise
        except A2AProtocolError as exc:
            if task_id is not None:
                runtime.a2a_tasks.mark_failed(task_id, exc.message)
            yield sse_frame("message", error_response(request_id, exc).model_dump(mode="json"))
        except Exception as exc:  # 图运行中断：SSE 内 JSON-RPC error 帧，日志留痕
            logger.exception("A2A 流式任务执行异常: %s", exc)
            if task_id is not None:
                runtime.a2a_tasks.mark_failed(task_id, "服务内部错误")
            internal = A2AProtocolError(
                jsonrpc_code=JSONRPC_INTERNAL_ERROR,
                a2a_code=A2A_ERROR_INTERNAL,
                message="服务内部错误",
            )
            yield sse_frame("message", error_response(request_id, internal).model_dump(mode="json"))

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


# —— 路由（薄）——


@router.get("/.well-known/agent.json")
async def agent_card(request: Request) -> Response:
    """A2A 发现：卡片 url 动态指向本实例 /a2a（多实例部署各自正确）。"""
    card = build_agent_card(base_url=str(request.base_url).rstrip("/"))
    return JSONResponse(card.model_dump(mode="json", by_alias=True))


@router.post("/a2a")
async def a2a_endpoint(request: Request, runtime: RuntimeDep) -> Response:
    """JSON-RPC 2.0 单端点：解码 → method 分发 → 响应/错误信封（HTTP 恒 200）。"""
    try:
        rpc = decode_request(await request.body())
    except A2AProtocolError as exc:
        # 解析错误无法回带 id → id 恒 None（JSON-RPC 规范）。
        return JSONResponse(error_response(None, exc).model_dump(mode="json"))
    try:
        if rpc.method == A2A_METHOD_MESSAGE_SEND:
            return JSONResponse(
                await _handle_send(runtime, rpc.id, rpc.params, _require_peer_id(request))
            )
        if rpc.method == A2A_METHOD_MESSAGE_STREAM:
            return _handle_stream(runtime, rpc.id, rpc.params, _require_peer_id(request))
        if rpc.method == A2A_METHOD_TASK_GET:
            return JSONResponse(
                task_frame(rpc.id, runtime.a2a_tasks.get(_require_task_id(rpc.params)))
            )
        if rpc.method == A2A_METHOD_TASK_CANCEL:
            return JSONResponse(
                task_frame(rpc.id, runtime.a2a_tasks.cancel(_require_task_id(rpc.params)))
            )
        raise A2AProtocolError(
            jsonrpc_code=JSONRPC_METHOD_NOT_FOUND,
            a2a_code=A2A_ERROR_METHOD_NOT_SUPPORTED,
            message=f"method 不支持: {rpc.method}",
        )
    except A2AProtocolError as exc:
        return JSONResponse(error_response(rpc.id, exc).model_dump(mode="json"))
    except Exception as exc:  # 未预期异常兜底：日志留痕，不泄露内部细节
        logger.exception("A2A 端点未预期异常: %s", exc)
        internal = A2AProtocolError(
            jsonrpc_code=JSONRPC_INTERNAL_ERROR, a2a_code=A2A_ERROR_INTERNAL, message="服务内部错误"
        )
        return JSONResponse(error_response(rpc.id, internal).model_dump(mode="json"))


__all__ = ["router"]
