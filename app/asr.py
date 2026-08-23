from __future__ import annotations

import os
from pathlib import Path


from fastapi import UploadFile

from .errors import InteractionError
from .xfyun_iat import transcribe_pcm_file


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def transcribe_audio(audio: UploadFile, transcript: str | None = None) -> str:
    """识别上传音频，返回标准化文本。"""

    if transcript and transcript.strip():
        return transcript.strip()

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

    raise InteractionError(
        code="asr.unsupported_audio_format",
        message=(
            "当前语音接口支持 16kHz/16bit/单声道 PCM 文件；"
            f"收到文件 {filename}, content_type={content_type}"
        ),
        status_code=415,
    )
