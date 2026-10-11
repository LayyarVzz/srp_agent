"""一次性迁移脚本：langgraph `store` 表旧会话行 → 独立 `sessions` 表（SQLAlchemy）。

背景：会话元数据曾与长期记忆共存于 langgraph `store` 表，经 namespace
`(user_id, "sessions")` 隔离；现改为独立 `sessions` 表承载（见 agent/session/）。
本脚本读出所有 `prefix = {user_id}.sessions` 的行，写入 `sessions` 表后从 store 删除。

幂等性：按 session_id 冲突跳过（ON CONFLICT DO NOTHING），已删行再跑无匹配；
dev 阶段数据量小、可反复执行，也可选择直接重置（本脚本优于重置：保留既有会话）。
仅针对 Postgres（store 表只存在于 PostgresStore）；未配置 DATABASE_URL 时拒绝执行。

用法（仓库根目录）：
    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/srp_agent \
        uv run python -m scripts.migrate_sessions
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from agent.session import build_session_repository
from agent.share.eventloop import ensure_selector_event_loop
from settings import configure_logging, get_settings

logger = logging.getLogger(__name__)

# store 表 prefix = ".".join(namespace)；会话 namespace 恒为 (user_id, "sessions")。
SESSIONS_PREFIX_SUFFIX = ".sessions"


def _to_async_dsn(dsn: str) -> str:
    """postgresql:// → postgresql+psycopg://（复用 psycopg[binary] 异步驱动）。"""
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+psycopg://", 1)
    raise ValueError(f"不支持的 DSN 前缀: {dsn.split('://', 1)[0]!r}")


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    if not database_url:
        logger.error("未配置 DATABASE_URL：迁移只针对 Postgres store 表的旧会话，请先配置后重试")
        return

    # 同一 Postgres 实例双通道：store 表走原生 SQL（langgraph 自有表），sessions 走仓库。
    engine = create_async_engine(_to_async_dsn(database_url))
    repo = build_session_repository(database_url)
    try:
        await repo.setup()
        async with engine.begin() as conn:
            rows = (
                await conn.execute(
                    text("SELECT prefix, key, value, expires_at FROM store WHERE prefix LIKE :pat"),
                    {"pat": f"%{SESSIONS_PREFIX_SUFFIX}"},
                )
            ).all()

        if not rows:
            logger.info("store 表无旧会话行（已迁移或本无数据），退出")
            return

        migrated = 0
        async with engine.begin() as conn:
            for row in rows:
                prefix, key, value, expires_at = row
                if not prefix.endswith(SESSIONS_PREFIX_SUFFIX):
                    logger.warning("跳过不匹配 prefix: %s", prefix)
                    continue
                # user_id 不允许含点号（label 校验），后缀切片即归属；value 为 jsonb。
                user_id = prefix[: -len(SESSIONS_PREFIX_SUFFIX)]
                payload = json.loads(value) if isinstance(value, str) else value
                session_id = key or payload.get("session_id")
                if not session_id:
                    logger.warning("跳过缺 session_id 的脏行 prefix=%s", prefix)
                    continue
                created_at = datetime.fromisoformat(payload["created_at"])
                # 幂等写入：session_id 已存在（重复执行）则跳过。
                await conn.execute(
                    text(
                        "INSERT INTO sessions (session_id, user_id, created_at, expires_at) "
                        "VALUES (:sid, :uid, :created, :expires) "
                        "ON CONFLICT (session_id) DO NOTHING"
                    ),
                    {
                        "sid": session_id,
                        "uid": user_id,
                        "created": created_at,
                        "expires": expires_at,
                    },
                )
                # 迁移成功（含已存在）后清理 store 旧行，保证再次执行无匹配。
                await conn.execute(
                    text("DELETE FROM store WHERE prefix = :prefix AND key = :key"),
                    {"prefix": prefix, "key": key},
                )
                migrated += 1
        logger.info("已迁移 %d 条会话元数据 → sessions 表（并清理 store 旧行）", migrated)
    finally:
        await repo.aclose()
        await engine.dispose()


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
