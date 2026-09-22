"""FastAPI 层交互接口离线测试（fake LLM 注入，ASGITransport 直测）。

覆盖：text 非流式 / SSE 流式（帧序列 + done.phase 适配）、澄清流、会话归属 404、
X-User-Id 401、422 校验、voice transcript 降级路径、非 PCM 415。
"""

from __future__ import annotations

import json
import re
from typing import Any

from httpx import ASGITransport, AsyncClient

from agent.intent.models import Intent, IntentResult
from agent.response.models import ClarifyResult
from tests.conftest import (
    chat_turn_messages,
    fake_structured_message,
    make_fake_tool,
    tool_call_messages,
    understand_message,
)

HEADERS = {"X-User-Id": "demo-user"}
CHAT_URL = "/api/v1/interactions/text"
STREAM_URL = "/api/v1/interactions/text/stream"
VOICE_URL = "/api/v1/interactions/voice"
VOICE_STREAM_URL = "/api/v1/interactions/voice/stream"


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """解析 SSE 帧序列：`event: X` + `data: {...}`，返回 [(event, data), ...]。"""
    frames: list[tuple[str, dict[str, Any]]] = []
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        ev = re.search(r"^event: (.+)$", frame, re.M)
        data = re.search(r"^data: (.+)$", frame, re.M)
        assert ev is not None and data is not None, f"非法 SSE 帧: {frame!r}"
        frames.append((ev.group(1), json.loads(data.group(1))))
    return frames


async def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _create_session(client: AsyncClient, user_id: str = "demo-user") -> str:
    resp = await client.post("/api/v1/sessions", headers={"X-User-Id": user_id})
    assert resp.status_code == 201
    return resp.json()["session_id"]


# —— 非流式文字 ——


async def test_chat_text_ok(api_app_factory: Any) -> None:
    """POST /interactions/text：200，phase=answer，自动创建会话并带回 session_id。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "你好"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["phase"] == "answer"
        assert body["source"] == "text"
        assert body["input_text"] == "你好"
        assert body["answer"] == "你好！"
        assert body["session_id"]
        assert body["status_trace"]  # 轨迹非空（thinking/speaking）
        assert body["finished_reason"] == "completed"
    finally:
        await runtime.aclose()


async def test_chat_text_with_session(api_app_factory: Any) -> None:
    """携带既有会话 id 续聊：不新建会话。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            session_id = await _create_session(client)
            resp = await client.post(
                CHAT_URL, headers=HEADERS, json={"session_id": session_id, "text": "你好"}
            )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id
    finally:
        await runtime.aclose()


async def test_chat_text_session_not_found(api_app_factory: Any) -> None:
    """带不存在的会话 id → 404 session_error.not_found（fail-fast）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(
                CHAT_URL, headers=HEADERS, json={"session_id": "no-such-session", "text": "你好"}
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "session_error.not_found"
    finally:
        await runtime.aclose()


async def test_chat_text_invalid_session_id(api_app_factory: Any) -> None:
    """session_id 含点号等非法字符 → 400 session_error.invalid_id。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(
                CHAT_URL, headers=HEADERS, json={"session_id": "a.b", "text": "你好"}
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "session_error.invalid_id"
    finally:
        await runtime.aclose()


async def test_chat_text_cross_user_not_found(api_app_factory: Any) -> None:
    """跨用户访问他人会话 → 404（归属强校验，防枚举）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            session_id = await _create_session(client, user_id="alice")
            resp = await client.post(
                CHAT_URL,
                headers={"X-User-Id": "bob"},
                json={"session_id": session_id, "text": "你好"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "session_error.not_found"
    finally:
        await runtime.aclose()


async def test_chat_text_missing_user_id(api_app_factory: Any) -> None:
    """缺 X-User-Id → 401 auth.identity_required。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(CHAT_URL, json={"text": "你好"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "auth.identity_required"
    finally:
        await runtime.aclose()


async def test_chat_text_validation_422(api_app_factory: Any) -> None:
    """text 缺失/超长 → 422（FastAPI 默认校验体）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            empty = await client.post(CHAT_URL, headers=HEADERS, json={"text": ""})
            missing = await client.post(CHAT_URL, headers=HEADERS, json={})
            too_long = await client.post(CHAT_URL, headers=HEADERS, json={"text": "x" * 8001})
        assert empty.status_code == 422
        assert missing.status_code == 422
        assert too_long.status_code == 422
    finally:
        await runtime.aclose()


async def test_chat_text_clarify_phase(api_app_factory: Any) -> None:
    """意图低置信 → done 载荷 phase=clarify + clarification.options（前端唯一判定码）。"""
    messages = [
        fake_structured_message(IntentResult(intent=Intent.CHAT, confidence=0.3, reason="模糊")),
        # v6.0 T2：该输入（7 字、非寒暄）通过确定性门控 → 意图分类后必有一次查询理解调用。
        understand_message(),
        fake_structured_message(
            ClarifyResult(question="你是想查询 A 还是 B？", options=["查 A", "查 B"])
        ),
    ]
    app, runtime = await api_app_factory(messages)
    try:
        async with await _client(app) as client:
            resp = await client.post(CHAT_URL, headers=HEADERS, json={"text": "就是那个你懂的"})
        assert resp.status_code == 200  # 反问仍是 HTTP 200
        body = resp.json()
        assert body["phase"] == "clarify"
        assert body["finished_reason"] == "needs_clarification"
        assert body["clarification"]["options"] == ["查 A", "查 B"]
        assert body["answer"] == "你是想查询 A 还是 B？"
    finally:
        await runtime.aclose()


# —— 流式文字（SSE）——


async def test_chat_text_stream_frames(api_app_factory: Any) -> None:
    """SSE 帧序列：session 最先、done 最后且含 phase；status 按序在中间。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(STREAM_URL, headers=HEADERS, json={"text": "你好"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = parse_sse(resp.text)
        assert frames[0][0] == "session"
        assert frames[0][1]["session_id"]
        assert frames[-1][0] == "done"
        done = frames[-1][1]
        assert done["phase"] == "answer"
        assert done["answer"] == "你好！"
        assert done["ok"] is True
        statuses = [ev for ev, _ in frames if ev == "status"]
        assert statuses  # 至少一条 status（thinking/speaking）
    finally:
        await runtime.aclose()


async def test_chat_text_stream_emits_answer_tokens(api_app_factory: Any) -> None:
    """SSE 流式：token 帧携带回答增量（拼接 == done.answer），done 仍在末位。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好 世界！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(STREAM_URL, headers=HEADERS, json={"text": "你好"})
        assert resp.status_code == 200
        frames = parse_sse(resp.text)
        tokens = [data["delta"] for ev, data in frames if ev == "token"]
        assert tokens  # 最终回答存在 token 增量
        assert "".join(tokens) == "你好 世界！"
        assert frames[-1][0] == "done"
        assert frames[-1][1]["answer"] == "你好 世界！"
    finally:
        await runtime.aclose()


async def test_chat_stream_tool_answer_emits_tokens(api_app_factory: Any) -> None:
    """工具循环（工具执行后模型直接作答）也走 token 级流式（SSE token 帧）。"""
    # tool_call_messages 已按 T2 口径在意图消息后带上查询理解消息（TOOL_USE 必过门控）。
    messages = tool_call_messages(
        [[{"name": "calc", "args": {}, "id": "call_1"}]], "现在是 14:30。"
    )
    app, runtime = await api_app_factory(messages, tools=[make_fake_tool("calc", content="14:30")])
    try:
        async with await _client(app) as client:
            resp = await client.post(STREAM_URL, headers=HEADERS, json={"text": "现在几点了"})
        assert resp.status_code == 200
        frames = parse_sse(resp.text)
        # 工具调用过程帧仍在（tool 事件），终答随后以 token 增量下发。
        tool_frames = [ev for ev, _ in frames if ev == "tool"]
        assert tool_frames
        tokens = [data["delta"] for ev, data in frames if ev == "token"]
        assert tokens
        assert "".join(tokens) == "现在是 14:30。"
        assert frames[-1][0] == "done"
        assert frames[-1][1]["answer"] == "现在是 14:30。"
        assert frames[-1][1]["finished_reason"] == "completed"
    finally:
        await runtime.aclose()


async def test_chat_text_stream_clarify_done(api_app_factory: Any) -> None:
    """澄清流：done 载荷 phase=clarify（SSE 场景）。"""
    messages = [
        fake_structured_message(IntentResult(intent=Intent.CHAT, confidence=0.2, reason="模糊")),
        # v6.0 T2：「帮我看看」（4 字、非寒暄）过门控 → 意图分类后补一条查询理解。
        understand_message(),
        fake_structured_message(ClarifyResult(question="你想查什么？", options=["天气", "时间"])),
    ]
    app, runtime = await api_app_factory(messages)
    try:
        async with await _client(app) as client:
            resp = await client.post(STREAM_URL, headers=HEADERS, json={"text": "帮我看看"})
        assert resp.status_code == 200
        frames = parse_sse(resp.text)
        assert frames[-1][0] == "done"
        assert frames[-1][1]["phase"] == "clarify"
        assert frames[-1][1]["clarification"]["options"] == ["天气", "时间"]
    finally:
        await runtime.aclose()


async def test_chat_stream_session_not_found_fail_fast(api_app_factory: Any) -> None:
    """流式接口：会话不存在在流开始前 fail-fast（404，而非流内 error 帧）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(
                STREAM_URL,
                headers=HEADERS,
                json={"session_id": "no-such-session", "text": "你好"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "session_error.not_found"
    finally:
        await runtime.aclose()


# —— 语音交互（transcript 降级路径 + 格式校验）——


async def test_voice_transcript_non_stream(api_app_factory: Any) -> None:
    """语音非流式：提供 transcript 跳过 ASR（无麦克风降级路径）。"""
    # v6.0 T2：transcript「现在几点了」过确定性门控 → 意图消息后带一条查询理解。
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！", understand=True))
    try:
        async with await _client(app) as client:
            resp = await client.post(
                VOICE_URL,
                headers=HEADERS,
                data={"transcript": "现在几点了"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "answer"
        assert body["source"] == "voice"
        assert body["input_text"] == "现在几点了"
        assert body["answer"] == "你好！"
    finally:
        await runtime.aclose()


async def test_voice_stream_transcript(api_app_factory: Any) -> None:
    """语音流式：transcript 直传 → SSE 完整事件序列（session→status→token→done）。"""
    # v6.0 T2：transcript「现在几点了」过确定性门控 → 意图消息后带一条查询理解。
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！", understand=True))
    try:
        async with await _client(app) as client:
            resp = await client.post(
                VOICE_STREAM_URL, headers=HEADERS, data={"transcript": "现在几点了"}
            )
        assert resp.status_code == 200
        frames = parse_sse(resp.text)
        assert frames[0][0] == "session"
        # 流首应有「正在识别语音」的 listening 状态（ASR 提示）
        assert ("status", "listening") in [(ev, d.get("status")) for ev, d in frames]
        assert frames[-1][0] == "done"
        assert frames[-1][1]["source"] == "voice"
        assert frames[-1][1]["phase"] == "answer"
        # token 增量拼接 == done.answer（语音链路与文本链路共用 chat_stream）
        tokens = [data["delta"] for ev, data in frames if ev == "token"]
        assert tokens
        assert "".join(tokens) == frames[-1][1]["answer"]
    finally:
        await runtime.aclose()


async def test_voice_unsupported_format_415(api_app_factory: Any) -> None:
    """非 PCM 音频 → 415 asr.unsupported_audio_format（非流式）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(
                VOICE_URL,
                headers=HEADERS,
                files={"audio": ("speech.wav", b"RIFF....", "audio/wav")},
            )
        assert resp.status_code == 415
        assert resp.json()["error"]["code"] == "asr.unsupported_audio_format"
    finally:
        await runtime.aclose()


async def test_voice_stream_unsupported_format_415(api_app_factory: Any) -> None:
    """语音流式：非 PCM 在流开始前 415 fail-fast（不读 body）。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(
                VOICE_STREAM_URL,
                headers=HEADERS,
                files={"audio": ("speech.wav", b"RIFF....", "audio/wav")},
            )
        assert resp.status_code == 415
        assert resp.json()["error"]["code"] == "asr.unsupported_audio_format"
    finally:
        await runtime.aclose()


async def test_voice_missing_audio_and_transcript(api_app_factory: Any) -> None:
    """audio 与 transcript 皆缺 → 400 asr.audio_or_transcript_required。"""
    app, runtime = await api_app_factory(chat_turn_messages(Intent.CHAT, "你好！"))
    try:
        async with await _client(app) as client:
            resp = await client.post(VOICE_URL, headers=HEADERS)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "asr.audio_or_transcript_required"
    finally:
        await runtime.aclose()
