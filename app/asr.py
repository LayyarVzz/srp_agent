"""ASR 转写入口：音频上传格式校验 + 转写（讯飞 IAT）。

实现保持 feature/agent-voice-interaction 分支原码（格式校验内联于
`transcribe_audio`）；仅错误类型统一为 `APIError`、配置统一由 settings 管理。
流式接口的前置 415 校验由路由层承担（见 app/routes/chat.py）。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from app.errors import APIError
from app.xfyun_iat import transcribe_pcm_file


async def transcribe_audio(audio: UploadFile | None, transcript: str | None = None) -> str:
    """识别上传音频，返回标准化文本。"""

    if transcript and transcript.strip():
        return transcript.strip()
    if audio is None:
        raise APIError(
            code="asr.audio_or_transcript_required",
            message="audio 和 transcript 至少需要提供一个",
            status_code=400,
        )

    filename = audio.filename or "audio"
    content_type = audio.content_type or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix == ".pcm" or content_type in {"audio/pcm", "application/octet-stream"}:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as temp_audio:
            temp_audio.write(await audio.read())
            temp_path = temp_audio.name
        try:
            return await transcribe_pcm_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    raise APIError(
        code="asr.unsupported_audio_format",
        message=(
            "当前语音接口支持 16kHz/16bit/单声道 PCM 文件；"
            f"收到文件 {filename}, content_type={content_type}"
        ),
        status_code=415,
    )
