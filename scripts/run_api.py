"""API 服务启动入口（替代裸 `uvicorn main:app`，Windows 必用）。

Windows 必读：uvicorn 惰性导入 app 模块（`config.load()` 在事件循环创建**之后**
才执行），`app/main.py` 模块导入期的 `ensure_selector_event_loop()` 对 uvicorn
无效——psycopg 异步连接会报 `Psycopg cannot use the 'ProactorEventLoop'`。
本脚本在 `uvicorn.run`（创建事件循环）**之前**切换 SelectorEventLoop 策略，
故必须经本入口启动（integration.md §7.5 / §9.3）。

dev 单 worker：内存态/PG checkpointer + stdio MCP 子进程（integration.md §9.4）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接 `python scripts/run_api.py` 时脚本目录为 cwd，需显式挂项目根（与其它 scripts 一致）。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from agent.share.eventloop import ensure_selector_event_loop  # noqa: E402
from settings import get_settings  # noqa: E402

ensure_selector_event_loop()


def main() -> None:
    """按 settings 的 host/port 启动 FastAPI 服务（根 `main:app` re-export）。

    `loop` 显式指定 `app.uvicorn_loop:selector_loop_factory`：uvicorn 0.36+ 在
    Windows 硬编码 ProactorEventLoop（绕过策略），psycopg 异步不兼容，必须经
    自定义 loop 工厂切回 Selector（见 app/uvicorn_loop.py 的 WHY）。
    """
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        loop="app.uvicorn_loop:selector_loop_factory",
        reload=False,
    )


if __name__ == "__main__":
    main()
