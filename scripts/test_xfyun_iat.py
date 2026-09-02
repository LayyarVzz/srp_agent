from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.errors import APIError  # noqa: E402
from app.xfyun_iat import transcribe_pcm_file  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test Xunfei IAT streaming ASR.")
    parser.add_argument("audio", help="16kHz, 16bit, mono pcm_s16le file")
    args = parser.parse_args()

    try:
        text = await transcribe_pcm_file(args.audio)
    except APIError as exc:
        print(f"ASR failed: {exc.code} - {exc.message}")
        raise SystemExit(1) from exc
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
