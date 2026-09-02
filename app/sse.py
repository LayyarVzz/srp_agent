"""SSE 帧编码

只做「事件 → 帧」的编码，不做任何业务判断；Pydantic 模型（StatusEvent /
ToolCallRecord / InteractionResult）自动 dump 为 JSON，保证中文可读
（ensure_ascii=False）。
"""

from __future__ import annotations

import json

from pydantic import BaseModel


def sse_frame(event: str, data: object) -> str:
    """编码一条 SSE 帧：`event:` + `data:`，空行结尾（前端按 `\n\n` 切帧）。"""
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
