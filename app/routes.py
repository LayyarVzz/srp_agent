from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, UploadFile

from .agent_adapter import submit_to_agent
from .asr import transcribe_audio
from .logging_service import recent_logs, record_event
from .schemas import InteractionResponse, RecentLogsResponse, TextInteractionRequest

router = APIRouter()


@router.post("/interactions/text", response_model=InteractionResponse)
async def text_interaction(request: TextInteractionRequest) -> InteractionResponse:
    record_event(
        "interaction.text.received",
        session_id=request.session_id,
        user_id=request.user_id,
        payload={"text_length": len(request.text)},
    )
    answer = await submit_to_agent(
        text=request.text,
        session_id=request.session_id,
        user_id=request.user_id,
        source="text",
    )
    record_event(
        "interaction.text.completed",
        session_id=request.session_id,
        user_id=request.user_id,
        payload={"answer_length": len(answer)},
    )
    return InteractionResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        source="text",
        input_text=request.text,
        answer=answer,
    )


@router.post("/interactions/voice", response_model=InteractionResponse)
async def voice_interaction(
    audio: UploadFile = File(...),
    transcript: str | None = Form(default=None),
    session_id: str = Form(...),
    user_id: str | None = Form(default=None),
) -> InteractionResponse:
    record_event(
        "interaction.voice.received",
        session_id=session_id,
        user_id=user_id,
        payload={"filename": audio.filename, "content_type": audio.content_type},
    )
    text = await transcribe_audio(audio, transcript=transcript)
    answer = await submit_to_agent(
        text=text,
        session_id=session_id,
        user_id=user_id,
        source="voice",
    )
    record_event(
        "interaction.voice.completed",
        session_id=session_id,
        user_id=user_id,
        payload={"text_length": len(text), "answer_length": len(answer)},
    )
    return InteractionResponse(
        session_id=session_id,
        user_id=user_id,
        source="voice",
        input_text=text,
        answer=answer,
    )


@router.get("/logs/recent", response_model=RecentLogsResponse)
async def logs_recent(limit: int = Query(default=50, ge=1, le=200)) -> RecentLogsResponse:
    return RecentLogsResponse(logs=recent_logs(limit=limit))
