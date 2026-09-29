"""交互路由：文字/语音，非流式 + SSE 流式（api.md §4/§5 前端主链路）。

只做参数校验、会话归属校验（fail-fast）与 SSE 翻译，无业务逻辑——
对话编排全部委托 `AgentRuntime.chat/chat_stream`（组合根，app 层零业务逻辑）。

SSE 事件协议（api.md §5.1）：`session` → `status`/`tool`（过程轨迹，按真实执行
顺序）→ `token`（回答节点 LLM 流式生成期间的增量预览）→ `done`
（`InteractionResult`，含决策码 `phase`，回答全量权威）；图运行中断才发 `error`
帧（业务降级已在图内收敛为 `AgentResponse`，HTTP 仍 200）。token 事件经
`AgentRuntime.chat_stream` 透传，本层不做业务改写。

观测（Phase C/C2）：每条路由把「会话 + 用户 + 本轮计时」绑进日志上下文
（`turn_context`），收尾记 `request.finished`；SSE 路由在流内重绑一次上下文 ——
`StreamingResponse` 的生成器由独立任务驱动，若不复绑，流内日志会丢掉会话维度
（表现为「有 trace_id、没有 session_id」，事件表按会话聚合就查不到这一轮）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, Header, UploadFile
from fastapi.responses import StreamingResponse

from agent.errors import SessionError
from agent.runtime import AgentRuntime
from app.asr import transcribe_audio
from app.deps import get_runtime
from app.errors import INTERNAL_ERROR, APIError
from app.models import InteractionResult, TextInteractionRequest
from app.request_context import (
    CODE_STREAM_ERROR,
    FIELD_SESSION_CREATED,
    FIELD_SOURCE,
    STATUS_COMPLETED,
    STATUS_ERROR,
    bind_resolved_session,
    elapsed_ms,
    finish_turn,
    turn_context,
)
from app.routes.sessions import require_user_id
from app.sse import sse_frame
from shared.logging import log_answer_generated

logger = logging.getLogger(__name__)

router = APIRouter(tags=["interactions"])

RuntimeDep = Annotated[AgentRuntime, Depends(get_runtime)]
UserHeader = Annotated[str | None, Header()]

# —— 错误码常量（asr.* 命名空间）——
ASR_AUDIO_OR_TRANSCRIPT_REQUIRED = "asr.audio_or_transcript_required"  # 400：两者皆缺
ASR_UNSUPPORTED_AUDIO_FORMAT = "asr.unsupported_audio_format"  # 415：非 PCM 格式

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭反向代理缓冲，保证帧实时到达
}


def _validate_voice_upload(audio: UploadFile | None, transcript: str | None) -> None:
    """语音上传参数校验（路由层，不读 body）：audio 与 transcript 至少其一；非 PCM → 415。

    流式语音接口在返回 StreamingResponse **之前**调用（fail-fast 415：
    流开始后不再产生 4xx）；转写本体仍由 `app.asr.transcribe_audio` 承担。
    """
    if transcript and transcript.strip():
        return
    if audio is None:
        raise APIError(
            ASR_AUDIO_OR_TRANSCRIPT_REQUIRED,
            "audio 和 transcript 至少需要提供一个",
            status_code=400,
        )
    filename = audio.filename or "audio"
    content_type = audio.content_type or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix == ".pcm" or content_type in {"audio/pcm", "application/octet-stream"}:
        return
    raise APIError(
        ASR_UNSUPPORTED_AUDIO_FORMAT,
        (
            "当前语音接口支持 16kHz/16bit/单声道 PCM 文件；"
            f"收到文件 {filename}, content_type={content_type}"
        ),
        status_code=415,
    )


async def _resolve_session(runtime: AgentRuntime, user_id: str, session_id: str | None) -> str:
    """发号（缺省自动创建）+ 归属强校验（fail-fast），并把会话绑进日志上下文。

    必须在返回 StreamingResponse 之前完成：流开始后不再产生 4xx。

    WHY 顺带 `bind_resolved_session`：入口中间件早于本函数执行（那时还不知道会话 id），
    绑定后本请求内所有日志/事件都带 `session=`，中间件的 `request.finished` 也能带上
    —— 否则「按会话排查」在入口事件上直接断链。
    """
    sid = session_id or (await runtime.sessions.create(user_id=user_id)).session_id
    await runtime.sessions.resolve(user_id=user_id, session_id=sid)
    bind_resolved_session(sid, user_id=user_id)
    return sid


def _result_fields(result: InteractionResult, *, created: bool) -> dict[str, Any]:
    """交互收尾事件字段（非流式路由）：回答摘要 + 决策码 + 引用/工具计数。

    WHY 记计数而不记内容：`answer.generated` 已按「len + sha256」记了回答体，
    此处只补「这轮用了多少引用/工具」—— 计数是聚合统计的输入，内容不是。
    """
    return {
        FIELD_SOURCE: result.source,
        FIELD_SESSION_CREATED: created,
        "phase": result.phase.value,
        "citations": len(result.citations),
        "tools": len(result.tool_trace),
    }


def _log_answer(result: InteractionResult, *, duration_ms: int) -> None:
    """记 `answer.generated`（回答体只留 len + sha256，禁止原文进日志）。"""
    log_answer_generated(
        logger,
        answer=result.answer,
        finished_reason=result.finished_reason,
        duration_ms=duration_ms,
        phase=result.phase.value,
        citations=len(result.citations),
        tools=len(result.tool_trace),
    )


@router.post("/interactions/text", response_model=InteractionResult)
async def text_interaction(
    body: TextInteractionRequest,
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
) -> InteractionResult:
    """文本交互（非流式）：返回 InteractionResult（联调/调试用）。"""
    user_id = require_user_id(x_user_id)
    session_id = await _resolve_session(runtime, user_id, body.session_id)
    started = time.perf_counter()
    with turn_context(session_id=session_id, user_id=user_id):
        try:
            resp = await runtime.chat(user_id=user_id, session_id=session_id, text=body.text)
        except BaseException as exc:  # 含 CancelledError：收尾事件在异常路径同样要落
            finish_turn(
                status=STATUS_ERROR,
                duration_ms=elapsed_ms(started),
                code=CODE_STREAM_ERROR,
                error=type(exc).__name__,
            )
            raise
        result = InteractionResult.from_response(
            resp, user_id=user_id, source="text", input_text=body.text
        )
        duration = elapsed_ms(started)
        _log_answer(result, duration_ms=duration)
        finish_turn(
            status=STATUS_COMPLETED,
            duration_ms=duration,
            **_result_fields(result, created=body.session_id is None),
        )
    return result


@router.post("/interactions/text/stream")
async def text_stream(
    body: TextInteractionRequest,
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
) -> StreamingResponse:
    """文本交互（SSE 流式，前端主接口）。"""
    user_id = require_user_id(x_user_id)
    session_id = await _resolve_session(runtime, user_id, body.session_id)
    created = body.session_id is None
    return _sse_response(
        runtime=runtime,
        user_id=user_id,
        session_id=session_id,
        source="text",
        created=created,
        body_text=body.text,
    )


def _sse_response(
    *,
    runtime: AgentRuntime,
    user_id: str,
    session_id: str,
    source: Literal["text", "voice"],
    created: bool,
    body_text: str,
    audio: UploadFile | None = None,
    transcript: str | None = None,
) -> StreamingResponse:
    """构造交互 SSE 响应（text / voice 共用同一条流式观测口径）。

    WHY 抽公共构造：两条流式路由的帧序、异常映射与收尾事件完全一致，差别只在
    「输入怎么来」（直读 text vs ASR 转写）。分叉两份实现必然漂移 —— 观测口径
    一旦漂移，事件表里同一类交互的字段就对不上。

    上下文绑定：`StreamingResponse` 的生成器由独立任务驱动，故**在流内重绑**
    `turn_context`（否则流内日志带 trace_id 却不带 session_id/user_id）。
    """

    async def generate() -> object:
        started = time.perf_counter()
        status = STATUS_COMPLETED
        code: str | None = None
        result: InteractionResult | None = None
        with turn_context(session_id=session_id, user_id=user_id):
            yield sse_frame("session", {"session_id": session_id})
            if source == "voice":
                yield sse_frame(
                    "status",
                    {"status": "listening", "tool_name": None, "message": "正在识别语音"},
                )
            try:
                text = await _resolve_input(audio, transcript, body_text=body_text, source=source)
                async for event, payload in runtime.chat_stream(
                    user_id=user_id, session_id=session_id, text=text
                ):
                    if event == "done":
                        # 契约适配（本层唯一业务点）：AgentResponse → InteractionResult。
                        payload = InteractionResult.from_response(
                            payload, user_id=user_id, source=source, input_text=text
                        )
                        result = payload
                    yield sse_frame(event, payload)
            except (APIError, SessionError) as exc:  # ASR 边界错误 / 会话归属错误 → error 帧
                status, code = STATUS_ERROR, exc.code
                yield sse_frame("error", {"code": exc.code, "message": exc.message})
            except Exception as exc:  # 图运行中断：SSE error 帧，日志留痕
                logger.exception("%s_stream 运行异常: %s", source, exc)
                status, code = STATUS_ERROR, CODE_STREAM_ERROR
                yield sse_frame("error", {"code": INTERNAL_ERROR, "message": "服务内部错误"})
            finally:
                duration = elapsed_ms(started)
                if result is not None:
                    _log_answer(result, duration_ms=duration)
                finish_turn(
                    status=status,
                    duration_ms=duration,
                    code=code,
                    **{
                        FIELD_SOURCE: source,
                        FIELD_SESSION_CREATED: created,
                        # 决策码与终态原因（无 result 时为 None，交由事件表的 NULL 表达）。
                        "phase": result.phase.value if result is not None else None,
                        "finished_reason": result.finished_reason if result is not None else None,
                        "citations": len(result.citations) if result is not None else 0,
                        "tools": len(result.tool_trace) if result is not None else 0,
                    },
                )

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


async def _resolve_input(
    audio: UploadFile | None,
    transcript: str | None,
    *,
    body_text: str,
    source: str,
) -> str:
    """取本轮输入文本：text 路由直用请求体；voice 路由走 ASR（transcript 优先）。"""
    if source == "text":
        return body_text
    return await transcribe_audio(audio, transcript=transcript)


@router.post("/interactions/voice", response_model=InteractionResult)
async def voice_interaction(
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
    audio: Annotated[UploadFile | None, File()] = None,
    transcript: Annotated[str | None, Form()] = None,
    session_id: Annotated[str | None, Form()] = None,
) -> InteractionResult:
    """语音交互（非流式，multipart）：ASR 转写 → Agent，返回 InteractionResult。"""
    user_id = require_user_id(x_user_id)
    sid = await _resolve_session(runtime, user_id, session_id)
    started = time.perf_counter()
    with turn_context(session_id=sid, user_id=user_id):
        try:
            text = await transcribe_audio(audio, transcript=transcript)
            resp = await runtime.chat(user_id=user_id, session_id=sid, text=text)
        except (APIError, SessionError) as exc:
            finish_turn(
                status=STATUS_ERROR,
                duration_ms=elapsed_ms(started),
                code=exc.code,
                **{FIELD_SOURCE: "voice", FIELD_SESSION_CREATED: session_id is None},
            )
            raise
        result = InteractionResult.from_response(
            resp, user_id=user_id, source="voice", input_text=text
        )
        duration = elapsed_ms(started)
        _log_answer(result, duration_ms=duration)
        finish_turn(
            status=STATUS_COMPLETED,
            duration_ms=duration,
            **_result_fields(result, created=session_id is None),
        )
    return result


@router.post("/interactions/voice/stream")
async def voice_stream(
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
    audio: Annotated[UploadFile | None, File()] = None,
    transcript: Annotated[str | None, Form()] = None,
    session_id: Annotated[str | None, Form()] = None,
) -> StreamingResponse:
    """语音交互（SSE 流式）：ASR 在流内（流首发 listening 提示「识别中」），
    随后 status/tool/done 事件与文本流式完全一致（共用 `_sse_response`）。"""
    user_id = require_user_id(x_user_id)
    sid = await _resolve_session(runtime, user_id, session_id)
    # 格式前置校验（不读 body）：非 PCM → 415 fail-fast，流开始后不再产生 4xx。
    _validate_voice_upload(audio, transcript)
    return _sse_response(
        runtime=runtime,
        user_id=user_id,
        session_id=sid,
        source="voice",
        created=session_id is None,
        body_text="",
        audio=audio,
        transcript=transcript,
    )
