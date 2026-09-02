from __future__ import annotations

from fastapi import APIRouter, File, Form, Header, Query, Response, UploadFile

from .agent_adapter import submit_to_agent
from .asr import transcribe_audio
from .logging_service import recent_logs, record_event
from .schemas import (
    InteractionResponse,
    RecentLogsResponse,
    SessionContext,
    SessionListResponse,
    StatusEvent,
    TextInteractionRequest,
)
from .sessions import (
    create_session,
    delete_session,
    get_or_create_session,
    list_sessions,
    require_user_id,
)

router = APIRouter()


@router.post("/sessions", response_model=SessionContext, status_code=201)
async def create_session_endpoint(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> SessionContext:
    user_id = require_user_id(x_user_id)
    return create_session(user_id)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint(
    limit: int = Query(default=20, ge=1, le=100),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> SessionListResponse:
    user_id = require_user_id(x_user_id)
    return SessionListResponse(sessions=list_sessions(user_id, limit=limit))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session_endpoint(
    session_id: str,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> Response:
    user_id = require_user_id(x_user_id)
    delete_session(session_id, user_id)
    return Response(status_code=204)


@router.post("/interactions/text", response_model=InteractionResponse)
async def text_interaction(
    request: TextInteractionRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> InteractionResponse:
    user_id = require_user_id(x_user_id)
    session = get_or_create_session(request.session_id, user_id)
    record_event(
        "interaction.text.received",
        session_id=session.session_id,
        user_id=user_id,
        payload={"text_length": len(request.text)},
    )
    answer = await submit_to_agent(
        text=request.text,
        session_id=session.session_id,
        user_id=user_id,
        source="text",
    )
    record_event(
        "interaction.text.completed",
        session_id=session.session_id,
        user_id=user_id,
        payload={"answer_length": len(answer)},
    )
    return InteractionResponse(
        session_id=session.session_id,
        user_id=user_id,
        source="text",
        input_text=request.text,
        answer=answer,
        status_trace=[
            StatusEvent(status="thinking", message="正在处理文本输入"),
            StatusEvent(status="speaking", message="正在生成回答"),
        ],
    )


@router.post("/interactions/voice", response_model=InteractionResponse)
async def voice_interaction(
    audio: UploadFile | None = File(default=None),
    transcript: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> InteractionResponse:
    user_id = require_user_id(x_user_id)
    session = get_or_create_session(session_id, user_id)
    record_event(
        "interaction.voice.received",
        session_id=session.session_id,
        user_id=user_id,
        payload={
            "filename": audio.filename if audio else None,
            "content_type": audio.content_type if audio else None,
            "has_transcript": bool(transcript and transcript.strip()),
        },
    )
    text = await transcribe_audio(audio, transcript=transcript)
    answer = await submit_to_agent(
        text=text,
        session_id=session.session_id,
        user_id=user_id,
        source="voice",
    )
    record_event(
        "interaction.voice.completed",
        session_id=session.session_id,
        user_id=user_id,
        payload={"text_length": len(text), "answer_length": len(answer)},
    )
    return InteractionResponse(
        session_id=session.session_id,
        user_id=user_id,
        source="voice",
        input_text=text,
        answer=answer,
        status_trace=[
            StatusEvent(status="listening", message="已收到语音输入"),
            StatusEvent(status="thinking", message="正在识别语音"),
            StatusEvent(status="speaking", message="正在生成回答"),
        ],
    )


@router.get("/logs/recent", response_model=RecentLogsResponse)
async def logs_recent(limit: int = Query(default=50, ge=1, le=200)) -> RecentLogsResponse:
    return RecentLogsResponse(logs=recent_logs(limit=limit))
