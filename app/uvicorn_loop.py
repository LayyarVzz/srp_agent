"""uvicorn 自定义 loop factory（Windows Selector 适配）。

uvicorn 0.36+ 的 `uvicorn.loops.asyncio.asyncio_loop_factory` 在 Windows 上
**硬编码**返回 `ProactorEventLoop`（loops/asyncio.py:10，绕过事件循环策略），
而 psycopg 异步驱动不兼容 Proactor（`Psycopg cannot use the 'ProactorEventLoop'`）。

接线差异（uvicorn/config.py:537 `get_loop_factory`）：
- 内置 `loop` 值（"auto"/"asyncio"…）：uvicorn 先调用工厂得到 loop **类**，
  `asyncio.run` 再实例化一次；
- 自定义模块路径：`get_loop_factory` 把本函数**原样**交给 `asyncio.run`，
  由 `Runner` 直接调用一次——故本工厂必须产出 loop **实例**。
"""

from __future__ import annotations

import asyncio


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """返回 SelectorEventLoop 实例（Windows 下规避 uvicorn 硬编码 Proactor）。

    WHY SelectorEventLoop：与 `agent/share/eventloop.py` 的
    `ensure_selector_event_loop()` 目标一致——psycopg/Postgres 异步连接仅兼容
    Selector 语义；`asyncio.run` 会以无参方式调用本工厂并直接使用返回值。
    """
    return asyncio.SelectorEventLoop()
