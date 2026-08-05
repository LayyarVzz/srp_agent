"""current_datetime 工具：返回系统本地时区的当前时间（ISO 8601）与星期。"""

from __future__ import annotations

from datetime import datetime


async def current_datetime() -> str:
    """返回当前时间（ISO 8601，秒级精度）与星期几（英文），使用系统本地时区。

    示例输出：`2026-08-05T14:30:00+08:00 | Wednesday`。
    """
    now = datetime.now().astimezone()
    return f"{now.isoformat(timespec='seconds')} | {now.strftime('%A')}"
