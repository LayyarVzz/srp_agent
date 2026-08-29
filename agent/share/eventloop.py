"""asyncio 事件循环策略工具（Windows 适配）。

psycopg 异步驱动依赖 Selector 语义的 socket 接口，与 Windows 默认的
ProactorEventLoop 不兼容（"Psycopg cannot use the 'ProactorEventLoop'"）。
凡入口层启动事件循环（demo / pytest / 未来 FastAPI）必须先调用
`ensure_selector_event_loop()` 切换到 SelectorEventLoop，且需在 asyncio.run()
之前、任何初始化自身事件循环的库导入之前调用。
"""

from __future__ import annotations

import asyncio
import sys


def ensure_selector_event_loop() -> None:
    """Windows 下将默认事件循环策略切为 SelectorEventLoop（仅 win32 生效）。

    非 Windows 平台为空操作；Linux/macOS 默认即 Selector 语义，无需干预。
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
