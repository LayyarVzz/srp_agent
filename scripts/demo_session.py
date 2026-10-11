"""S1 会话管理验收脚本：不依赖 LLM / 工具，直连 SessionManager 演示完整 CRUD 与归属校验。

演示内容：
  1. 发号：为同一 user 创建多个会话（session_id 由 SessionManager 生成，调用方不指定）
  2. 归属校验：正确 owner 可 resolve；非 owner / 不存在的 session → session_error.not_found
     （命名空间按 user 隔离，resolve 不区分「不存在」与「存在但不属于你」，防跨用户枚举）
  3. 列会话：按 created_at 降序（前端「历史会话」列表的基础能力）
  4. 删除：delete 后 resolve 必然 not_found（delete 先 resolve 确认归属再删）
  5. 用户隔离：user B 的会话列表/解析完全独立于 user A
  6. 存储后端：会话元数据落在独立 `sessions` 表（SQLAlchemy）——dev=SQLite memory、
     prod=Postgres；TTL 由 `expires_at` 承载（配置 SESSION_TTL_MINUTES 后过期即不可见）
  7. 元数据 vs 消息：会话身份写在 `sessions` 表，消息本体由 checkpointer 承载
     （thread_id == session_id），本脚本不跑对话因此 `checkpoints` 表无新增——
     点明两者分工。

用法（仓库根目录）：
    uv run python -m scripts.demo_session                 # dev：SQLite memory（零配置）
    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/srp_agent \
        uv run python -m scripts.demo_session             # prod：Postgres `sessions` 表
    # 生产路径可在 DataGrip / psql 查证：\\d sessions 见独立表与 (user_id, created_at) 索引；
    #   SELECT * FROM store WHERE prefix LIKE '%.sessions' 应为空（旧 store 会话已迁移/不在）。
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from agent.core.config import AgentFrameworkConfig
from agent.errors import (
    SESSION_ERROR_INVALID_ID,
    SESSION_ERROR_NOT_FOUND,
    SessionError,
)
from agent.session import build_session_backend
from agent.share.eventloop import ensure_selector_event_loop
from settings import configure_logging, get_settings

logger = logging.getLogger(__name__)

DEMO_USER_A = "demo-user-a"
DEMO_USER_B = "demo-user-b"
# 固定非法 session_id，演示 invalid_id 护栏（非空、不含点号）。
BAD_SESSION_ID = "bad.session.id"


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    cfg = AgentFrameworkConfig.get_default()

    # 会话模块自装：配置了 DATABASE_URL 自动切 Postgres，无则降级 SQLite memory。
    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    session_backend = await build_session_backend(database_url, config=cfg.session)
    logger.info(
        "会话仓库: %s（TTL=%s 分钟）",
        type(session_backend.repository).__name__,
        cfg.session.ttl_minutes,
    )
    sessions = session_backend.manager

    try:
        # —— 0. 复位：清空该用户历史会话，保证 Postgres 跨运行幂等（顺便演示 list+delete）——
        for uid in (DEMO_USER_A, DEMO_USER_B):
            for old in await sessions.list(user_id=uid):
                await sessions.delete(user_id=uid, session_id=old.session_id)
        logger.info("== 0. 已复位 demo 用户的历史会话 ==")

        # —— 1. 发号：user A 建 3 个会话 ——
        logger.info("== 1. 发号：user=%s 创建 3 个会话 ==", DEMO_USER_A)
        session_ids: list[str] = []
        for _ in range(3):
            ctx = await sessions.create(user_id=DEMO_USER_A)
            session_ids.append(ctx.session_id)
            logger.info(
                "  已创建 session=%s… created_at=%s",
                ctx.session_id[:8],
                ctx.created_at.isoformat(),
            )

        # —— 2. 归属校验：owner 可 resolve ——
        logger.info("== 2. 归属校验（owner resolve 通过；非 owner → not_found）==")
        resolved = await sessions.resolve(user_id=DEMO_USER_A, session_id=session_ids[0])
        logger.info(
            "  ✓ owner resolve 通过：session=%s… → user=%s",
            resolved.session_id[:8],
            resolved.user_id,
        )

        # 非 owner：user B 拿着 A 的 session_id 解析 → not_found（不泄露存在性）。
        try:
            await sessions.resolve(user_id=DEMO_USER_B, session_id=session_ids[0])
            raise AssertionError("B 不应能 resolve A 的会话")
        except SessionError as exc:
            assert exc.code == SESSION_ERROR_NOT_FOUND
            logger.info("  ✓ B 解析 A 的会话 → %s（符合预期）", exc.code)

        # 同一 owner 解析不存在的 session → 同样 not_found（不区分存在性）。
        try:
            await sessions.resolve(user_id=DEMO_USER_A, session_id=uuid.uuid4().hex)
            raise AssertionError("不存在的 session 不应 resolve 成功")
        except SessionError as exc:
            assert exc.code == SESSION_ERROR_NOT_FOUND
            logger.info("  ✓ A 解析不存在的 session → %s（符合预期）", exc.code)

        # 非法 id（含点号）→ invalid_id 护栏。
        try:
            await sessions.resolve(user_id=DEMO_USER_A, session_id=BAD_SESSION_ID)
            raise AssertionError("非法 session_id 不应通过")
        except SessionError as exc:
            assert exc.code == SESSION_ERROR_INVALID_ID
            logger.info("  ✓ 非法 id %r → %s（符合预期）", BAD_SESSION_ID, exc.code)

        # —— 3. 列会话：按 created_at 降序 ——
        logger.info("== 3. 列会话（created_at 降序）==")
        listed = await sessions.list(user_id=DEMO_USER_A)
        # 只断言时间序：同微秒创建的会话 created_at 相等，SQL 对相等值顺序不定，
        # 精确逆序断言在此会 flaky（断言集合 + 降序时间戳即验证契约）。
        assert {c.session_id for c in listed} == set(session_ids)
        created = [c.created_at for c in listed]
        assert created == sorted(created, reverse=True)
        for i, c in enumerate(listed):
            logger.info("  [%d] %s… %s", i, c.session_id[:8], c.created_at.isoformat())

        # —— 4. 删除：delete 后 resolve → not_found ——
        logger.info("== 4. 删除会话 ==")
        await sessions.delete(user_id=DEMO_USER_A, session_id=session_ids[1])
        logger.info(
            "  已删除 session=%s…（元数据；checkpointer 线程清理列为开放项）", session_ids[1][:8]
        )
        try:
            await sessions.resolve(user_id=DEMO_USER_A, session_id=session_ids[1])
            raise AssertionError("已删除的会话不应 resolve 成功")
        except SessionError as exc:
            assert exc.code == SESSION_ERROR_NOT_FOUND
            logger.info("  ✓ 删除后 resolve → %s（符合预期）", exc.code)

        # —— 5. 用户隔离：B 的会话与 A 完全独立 ——
        logger.info("== 5. 用户隔离 ==")
        await sessions.create(user_id=DEMO_USER_B)
        listed_b = await sessions.list(user_id=DEMO_USER_B)
        listed_a = await sessions.list(user_id=DEMO_USER_A)
        assert len(listed_b) == 1 and len(listed_a) == 2  # B=1，A 删掉 1 个剩 2
        assert all(c.user_id == DEMO_USER_B for c in listed_b)
        assert all(c.user_id == DEMO_USER_A for c in listed_a)
        logger.info("  ✓ A 有 %d 个会话、B 有 %d 个会话，互不可见", len(listed_a), len(listed_b))

        # —— 6. 收尾总结 ——
        logger.info("== 6. 结论 ==")
        logger.info(
            "会话元数据落在独立 sessions 表，共 %d 条；消息本体由 checkpointer 承载。",
            len(listed_a) + len(listed_b),
        )
        logger.info("演示完成：create / resolve(归属校验) / list / delete / 用户隔离 全部通过。")
    finally:
        await session_backend.aclose()


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
