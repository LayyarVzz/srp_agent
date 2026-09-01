"""讯飞语音听写（IAT）客户端：16kHz/16bit/单声道 PCM → 文本。

凭据（APP_ID / API_KEY / API_SECRET）与端点统一由 `settings.py` 管理
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

import websockets
from websockets.exceptions import InvalidStatus

from app.errors import APIError
from settings import get_settings

# —— 错误码常量（asr.* 命名空间）——
ASR_MISSING_CREDENTIALS = "asr.missing_credentials"  # 500：未配置讯飞凭据
ASR_FILE_NOT_FOUND = "asr.file_not_found"  # 400：音频文件不存在
ASR_EMPTY_AUDIO = "asr.empty_audio"  # 400：PCM 文件为空
ASR_EMPTY_RESULT = "asr.empty_result"  # 502：听写未返回文本
ASR_XFYUN_IAT_HANDSHAKE = "asr.xfyun_iat_handshake"  # 502：握手失败
ASR_XFYUN_IAT_ERROR = "asr.xfyun_iat_error"  # 502：听写服务错误

PCM_CHUNK_SIZE = 1280
PCM_CHUNK_INTERVAL_SECONDS = 0.04


def build_iat_url(
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
    host_url: str | None = None,
) -> str:
    """构造讯飞 IAT 鉴权 URL（HMAC-SHA256 签名）。

    凭据缺省时从 `get_settings()` 读取（.env 的 XF_IAT_*）；显式传入用于测试注入。
    """
    settings = get_settings()
    api_key = api_key or settings.xf_iat_api_key.get_secret_value()
    api_secret = api_secret or settings.xf_iat_api_secret.get_secret_value()
    if not api_key or not api_secret:
        raise APIError(
            ASR_MISSING_CREDENTIALS,
            "未配置 XF_IAT_API_KEY 或 XF_IAT_API_SECRET",
            status_code=500,
        )
    host_url = host_url or settings.xf_iat_url

    parsed = urlparse(host_url)
    date = format_datetime(datetime.now(UTC), usegmt=True)
    signature_origin = f"host: {parsed.netloc}\ndate: {date}\nGET {parsed.path} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(
            api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
    query = urlencode(
        {
            "authorization": authorization,
            "date": date,
            "host": parsed.netloc,
        }
    )
    return f"{host_url}?{query}"


async def transcribe_pcm_file(path: str | Path) -> str:
    """识别 16kHz/16bit/单声道 PCM 文件，返回转写文本。"""
    pcm_path = Path(path)
    if not pcm_path.exists():
        raise APIError(ASR_FILE_NOT_FOUND, f"音频文件不存在: {pcm_path}", status_code=400)

    settings = get_settings()
    app_id = settings.xf_iat_app_id.get_secret_value()
    if not app_id:
        raise APIError(ASR_MISSING_CREDENTIALS, "未配置 XF_IAT_APP_ID", status_code=500)

    results: list[str] = []
    try:
        async with websockets.connect(build_iat_url(), ping_interval=None) as websocket:
            receiver = asyncio.create_task(_receive_results(websocket, results))
            try:
                await _send_pcm_chunks(websocket, pcm_path, app_id=app_id)
                await asyncio.wait_for(receiver, timeout=20)
            except asyncio.TimeoutError:
                receiver.cancel()
            finally:
                if receiver.done():
                    receiver.result()
    except InvalidStatus as exc:
        body = getattr(exc.response, "body", b"") if exc.response else b""
        detail = body.decode("utf-8", errors="ignore") if body else str(exc)
        status_code = exc.response.status_code if exc.response else "unknown"
        raise APIError(
            f"{ASR_XFYUN_IAT_HANDSHAKE}.{status_code}",
            f"讯飞语音听写握手失败: {detail}",
            status_code=502,
        ) from exc

    text = "".join(results)
    if not text:
        raise APIError(
            ASR_EMPTY_RESULT,
            "讯飞语音听写未返回识别文本，请检查音频格式是否为 16kHz/16bit/单声道 PCM",
            status_code=502,
        )
    return text


async def _send_pcm_chunks(
    websocket: websockets.ClientConnection,
    pcm_path: Path,
    *,
    app_id: str,
) -> None:
    with pcm_path.open("rb") as audio_file:
        first_chunk = audio_file.read(PCM_CHUNK_SIZE)
        if not first_chunk:
            raise APIError(ASR_EMPTY_AUDIO, "PCM 音频文件为空", status_code=400)

        await websocket.send(
            json.dumps(
                {
                    "common": {"app_id": app_id},
                    "business": {
                        "language": "zh_cn",
                        "domain": "iat",
                        "accent": "mandarin",
                        "vad_eos": 5000,
                    },
                    "data": {
                        "status": 0,
                        "format": "audio/L16;rate=16000",
                        "encoding": "raw",
                        "audio": base64.b64encode(first_chunk).decode("utf-8"),
                    },
                }
            )
        )
        await asyncio.sleep(PCM_CHUNK_INTERVAL_SECONDS)

        while chunk := audio_file.read(PCM_CHUNK_SIZE):
            await websocket.send(
                json.dumps(
                    {
                        "data": {
                            "status": 1,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    }
                )
            )
            await asyncio.sleep(PCM_CHUNK_INTERVAL_SECONDS)

    await websocket.send(
        json.dumps(
            {
                "data": {
                    "status": 2,
                    "format": "audio/L16;rate=16000",
                    "encoding": "raw",
                    "audio": "",
                }
            }
        )
    )


async def _receive_results(
    websocket: websockets.ClientConnection,
    results: list[str],
) -> None:
    async for raw_message in websocket:
        message = json.loads(raw_message)
        code = int(message.get("code", 0))
        if code != 0:
            raise APIError(
                f"{ASR_XFYUN_IAT_ERROR}.{code}",
                f"讯飞语音听写错误: {message.get('message') or raw_message}",
                status_code=502,
            )

        data = message.get("data") or {}
        result = data.get("result") or {}
        for ws in result.get("ws", []):
            candidates = ws.get("cw") or []
            if candidates:
                results.append(str(candidates[0].get("w", "")))

        if data.get("status") == 2:
            return
