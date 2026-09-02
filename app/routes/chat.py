"""交互路由：文字/语音，非流式 + SSE 流式（api.md §4/§5 前端主链路）。

只做参数校验、会话归属校验（fail-fast）与 SSE 翻译，无业务逻辑——
对话编排全部委托 `AgentRuntime.chat/chat_stream`（组合根，app 层零业务逻辑）。

SSE 事件协议（api.md §5.1）：`session` → `status`/`tool`（过程轨迹，按真实执行
顺序）→ `done`（`InteractionResult`，含决策码 `phase`）；图运行中断才发 `error`
帧（业务降级已在图内收敛为 `AgentResponse`，HTTP 仍 200）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, UploadFile
from fastapi.responses import StreamingResponse

from agent.errors import SessionError
from agent.runtime import AgentRuntime
from app.asr import transcribe_audio
from app.deps import get_runtime
from app.errors import INTERNAL_ERROR, APIError
from app.models import InteractionResult, TextInteractionRequest
from app.routes.sessions import require_user_id
from app.sse import sse_frame

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
    """发号（缺省自动创建）+ 归属强校验（fail-fast）。

    必须在返回 StreamingResponse 之前完成：流开始后不再产生 4xx。
    """
    sid = session_id or (await runtime.sessions.create(user_id=user_id)).session_id
    await runtime.sessions.resolve(user_id=user_id, session_id=sid)
    return sid


@router.post("/interactions/text", response_model=InteractionResult)
async def text_interaction(
    body: TextInteractionRequest,
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
) -> InteractionResult:
    """文本交互（非流式）：返回 InteractionResult（联调/调试用）。"""
    user_id = require_user_id(x_user_id)
    session_id = await _resolve_session(runtime, user_id, body.session_id)
    resp = await runtime.chat(user_id=user_id, session_id=session_id, text=body.text)
    return InteractionResult.from_response(
        resp, user_id=user_id, source="text", input_text=body.text
    )


@router.post("/interactions/text/stream")
async def text_stream(
    body: TextInteractionRequest,
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
) -> StreamingResponse:
    """文本交互（SSE 流式，前端主接口）。"""
    user_id = require_user_id(x_user_id)
    session_id = await _resolve_session(runtime, user_id, body.session_id)

    async def generate() -> object:
        yield sse_frame("session", {"session_id": session_id})
        try:
            async for event, payload in runtime.chat_stream(
                user_id=user_id, session_id=session_id, text=body.text
            ):
                if event == "done":
                    # 契约适配（本层唯一业务点）：AgentResponse → InteractionResult（追加 phase）。
                    payload = InteractionResult.from_response(
                        payload, user_id=user_id, source="text", input_text=body.text
                    )
                yield sse_frame(event, payload)
        except SessionError as exc:  # 理论上已被 fail-fast 拦下，防御性保留
            yield sse_frame("error", {"code": exc.code, "message": exc.message})
        except Exception as exc:  # 图运行中断：SSE error 帧，日志留痕
            logger.exception("chat_stream 运行异常: %s", exc)
            yield sse_frame("error", {"code": INTERNAL_ERROR, "message": "服务内部错误"})

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


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
    text = await transcribe_audio(audio, transcript=transcript)
    resp = await runtime.chat(user_id=user_id, session_id=sid, text=text)
    return InteractionResult.from_response(resp, user_id=user_id, source="voice", input_text=text)


@router.post("/interactions/voice/stream")
async def voice_stream(
    runtime: RuntimeDep,
    x_user_id: UserHeader = None,
    audio: Annotated[UploadFile | None, File()] = None,
    transcript: Annotated[str | None, Form()] = None,
    session_id: Annotated[str | None, Form()] = None,
) -> StreamingResponse:
    """语音交互（SSE 流式）：ASR 在流内（流首发 listening 提示「识别中」），
    随后 status/tool/done 事件与文本流式完全一致。"""
    user_id = require_user_id(x_user_id)
    sid = await _resolve_session(runtime, user_id, session_id)
    # 格式前置校验（不读 body）：非 PCM → 415 fail-fast，流开始后不再产生 4xx。
    _validate_voice_upload(audio, transcript)

    async def generate() -> object:
        yield sse_frame("session", {"session_id": sid})
        yield sse_frame(
            "status", {"status": "listening", "tool_name": None, "message": "正在识别语音"}
        )
        try:
            text = await transcribe_audio(audio, transcript=transcript)
            async for event, payload in runtime.chat_stream(
                user_id=user_id, session_id=sid, text=text
            ):
                if event == "done":
                    payload = InteractionResult.from_response(
                        payload, user_id=user_id, source="voice", input_text=text
                    )
                yield sse_frame(event, payload)
        except APIError as exc:  # ASR 边界错误（凭据缺失 500 / 听写失败 502 等）→ error 帧
            yield sse_frame("error", {"code": exc.code, "message": exc.message})
        except SessionError as exc:
            yield sse_frame("error", {"code": exc.code, "message": exc.message})
        except Exception as exc:  # 图运行中断：SSE error 帧，日志留痕
            logger.exception("voice_stream 运行异常: %s", exc)
            yield sse_frame("error", {"code": INTERNAL_ERROR, "message": "服务内部错误"})

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)
