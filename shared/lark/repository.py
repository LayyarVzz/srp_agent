"""飞书绑定记录的持久化 + 并发安全刷新（SQLAlchemy，dev=SQLite memory / prod=Postgres）。

镜像 `agent/session/repository.py` 形态（高频、点查、带 TTL 的数据属 SQLAlchemy 表，
不属 langgraph `BaseStore`），但**多一层职责**：令牌加密与刷新并发控制 —— 二者都是
「存储语义不可分割的一部分」（加密决定落库什么、乐观锁决定并发下什么被写入）。

**并发刷新（§7 强制要求，实测依据**：`refresh_token` 刷新后**轮换**、旧值立即失效**）**：
多 api worker / 多副本并发刷新同一用户会互相作废，后到者拿旧值必然失败，还可能把
已刷好的新 token 覆盖成失败态。实现按 §7 的六步：

```
① 读行（含 version / expires_at / 旧 refresh_token）
② 未临近过期 → 直接用现有 UAT
③ 临近过期    → 调刷新（网络 IO，**不持锁**）
④ 条件更新：WHERE user_id=? AND version=?  → version=version+1
⑤ 影响行数 = 0 → 别的 worker 已刷新成功 → **重读**，直接用它的新 UAT（不重试刷新）
⑥ 刷新返回 invalid_grant → **重读**该行：
     行已变化 → 用新 token 重试一次（并发竞争，非真过期）
     行未变化 → 判定绑定失效 → tool_error.lark_unbound（引导重新绑定）
```

**fail-closed**：未配置 `lark_token_key`（cipher 为 None）时本仓库拒绝一切写入与
读取，绑定功能整体关闭并记日志 —— 绝不落明文 token。

**列类型**：token 列必须 `Text`（UAT/refresh 实测 8093/8254 字符，加密后更长；
`String(n)` 会截断且难排查）。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import DateTime, Integer, Text, delete, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from shared.lark.errors import (
    BINDING_DISABLED_REASON,
    LarkBoundError,
    LarkCliError,
    LarkUnboundError,
)
from shared.lark.models import (
    BINDING_STATUS_ACTIVE,
    BINDING_STATUS_INVALID,
    LarkBinding,
    LarkDeviceFlow,
    LarkTokenSet,
)
from shared.lark.token_cipher import LarkTokenCipher

logger = logging.getLogger(__name__)

# refresh_token 空闲上限（实测 604800s = 7 天）；响应缺该字段时按此兜底。
DEFAULT_REFRESH_TTL_S = 604800
# access_token 兜底寿命（实测 7200s）；响应缺 expires_in 时按此兜底。
DEFAULT_ACCESS_TTL_S = 7200

# 刷新函数签名：`async (refresh_token) -> LarkTokenSet`。
# WHY 在 Protocol 之前定义：契约里引用它作参数类型，且在 `from __future__ import
# annotations` 之外仍需是可求值的模块级名字（运行期不要靠字符串前向引用）。
RefreshFn = Callable[[str], Awaitable[LarkTokenSet]]


class LarkBindingRepository(Protocol):
    """绑定存储契约（结构型协议；测试可注入假实现）。

    契约覆盖**消费方真正用到的全部方法**（而非最小子集）：绑定工具需要
    `peek_device_flow` / `decrypt_access_token` / `delete_binding`，
    凭据解析需要 `resolve_access_token` —— 缺一条就得在消费侧写 `type: ignore`，
    等于把契约漏洞藏进调用点。
    """

    enabled: bool

    async def setup(self) -> None:
        """幂等建表（装配期调用一次）。"""

    async def aclose(self) -> None:
        """关闭连接池。"""

    async def get_binding(self, user_id: str) -> LarkBinding | None:
        """按 user_id 取绑定（无则 None）。"""

    async def save_device_flow(self, flow: LarkDeviceFlow) -> None:
        """登记设备码待定态（upsert）。"""

    async def get_device_flow(self, *, user_id: str, device_code: str) -> LarkDeviceFlow | None:
        """取待定态（排除已过期）；未命中/过期返回 None。"""

    async def peek_device_flow(self, user_id: str) -> LarkDeviceFlow | None:
        """取该用户当前待定态（无需 device_code；排除已过期）。

        WHY 与 `get_device_flow` 并存：「完成绑定」的调用方只知道 user_id
        （设备码在服务端留存），而「换码」必须校验 device_code 匹配以防串码。
        """
        ...

    async def drop_device_flow(self, *, user_id: str) -> None:
        """清除该用户的待定态（换码成功或重新发起时）。"""

    async def save_binding(
        self,
        *,
        user_id: str,
        tokens: LarkTokenSet,
        open_id: str | None = None,
        user_name: str | None = None,
        now: datetime | None = None,
    ) -> LarkBinding:
        """写入/覆盖绑定（加密落库，version 归零）。"""

    async def delete_binding(self, user_id: str) -> bool:
        """删除绑定，返回是否真的删掉了（解绑）。"""

    async def resolve_access_token(
        self,
        user_id: str,
        *,
        refresh: RefreshFn,
        now: datetime | None = None,
    ) -> str:
        """取该用户当前可用的 UAT（临近过期则按乐观锁语义刷新）。"""

    async def decrypt_access_token(self, binding: LarkBinding) -> str:
        """解密某条绑定的 UAT（仅供解绑时的远端撤销使用；不参与取用路径）。"""
        ...


class _Base(DeclarativeBase):
    """绑定域表所属 declarative 基类（独立于 sessions 表元数据）。"""


class LarkBindingRow(_Base):
    """`lark_bindings` 表行模型：归属 + 身份 + 加密令牌 + 生命周期 + 乐观锁。

    PK=user_id（一个用户至多一条绑定）；token 列**密文**（Text）；`version` 承载
    乐观锁（并发刷新抢更新用条件更新 + 影响行数判定）。
    """

    __tablename__ = "lark_bindings"

    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    open_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    access_token_ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token_ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    refresh_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default=BINDING_STATUS_ACTIVE)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class LarkDeviceFlowRow(_Base):
    """`lark_device_flows` 表行模型：待定设备码（带 TTL）。

    WHY 落库而非进程内存：多 worker / 多副本部署下，发起绑定的 worker 与
    完成绑定的 worker 可能不是同一个（§7 明确要求评估该场景）。
    """

    __tablename__ = "lark_device_flows"

    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    device_code: Mapped[str] = mapped_column(Text, nullable=False)
    user_code: Mapped[str] = mapped_column(Text, nullable=False)
    verification_uri: Mapped[str] = mapped_column(Text, nullable=False)
    verification_uri_complete: Mapped[str] = mapped_column(Text, nullable=False)
    flow_id: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    interval_s: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SQLAlchemyLarkBindingRepository:
    """绑定记录的 SQLAlchemy 实现（同一实现服务 dev/prod）。

    `cipher` 为 None 即 **fail-closed**：绑定功能关闭，任何读写都拒绝
    （`LarkBoundError(BINDING_DISABLED_REASON)`），绝不以明文形式落库。
    """

    def __init__(
        self,
        cipher: LarkTokenCipher | None,
        database_url: str | None = None,
        *,
        engine: AsyncEngine | None = None,
        refresh_skew_s: int = 300,
    ) -> None:
        """构造仓库。显式传 engine 覆盖 DSN（测试隔离用）；`refresh_skew_s` 为提前刷新量。"""
        self._cipher = cipher
        self._refresh_skew_s = refresh_skew_s
        self._engine = engine or _build_engine(database_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    # —— 装配 ——

    async def setup(self) -> None:
        """幂等建表（缺表才建），装配期调用一次。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def aclose(self) -> None:
        """关闭连接池。"""
        await self._engine.dispose()

    @property
    def enabled(self) -> bool:
        """绑定功能是否启用（密钥已配置）。"""
        return self._cipher is not None

    # —— 设备码待定态 ——

    async def save_device_flow(self, flow: LarkDeviceFlow) -> None:
        """登记/覆盖待定态（同一用户重新发起即覆盖旧 flow）。"""
        self._require_enabled()
        async with self._session_factory() as session:
            await session.merge(_flow_to_row(flow))
            await session.commit()

    async def get_device_flow(self, *, user_id: str, device_code: str) -> LarkDeviceFlow | None:
        """取待定态；`device_code` 须匹配（防用他人/过期码换绑）+ 过期视同不存在。"""
        self._require_enabled()
        async with self._session_factory() as session:
            row = await session.get(LarkDeviceFlowRow, user_id)
        if row is None or row.device_code != device_code or _expired(row.expires_at):
            return None
        return _flow_from_row(row)

    async def drop_device_flow(self, *, user_id: str) -> None:
        """清除待定态（换码成功或重新发起时）。"""
        self._require_enabled()
        async with self._session_factory() as session:
            await session.execute(
                delete(LarkDeviceFlowRow).where(LarkDeviceFlowRow.user_id == user_id)
            )
            await session.commit()

    async def peek_device_flow(self, user_id: str) -> LarkDeviceFlow | None:
        """取该用户当前待定态（无需 device_code；已过期视同不存在）。"""
        self._require_enabled()
        async with self._session_factory() as session:
            row = await session.get(LarkDeviceFlowRow, user_id)
        if row is None or _expired(row.expires_at):
            return None
        return _flow_from_row(row)

    # —— 绑定读写 ——

    async def get_binding(self, user_id: str) -> LarkBinding | None:
        """按 user_id 取绑定（含密文与 version，**不解密** —— 解密由取用处显式进行）。"""
        self._require_enabled()
        async with self._session_factory() as session:
            row = await session.get(LarkBindingRow, user_id)
        return _binding_from_row(row) if row is not None else None

    async def save_binding(
        self,
        *,
        user_id: str,
        tokens: LarkTokenSet,
        open_id: str | None = None,
        user_name: str | None = None,
        now: datetime | None = None,
    ) -> LarkBinding:
        """写入绑定（明文 → 密文后落库）；已存在则整体覆盖且 **version 归零**。

        WHY version 归零：换绑是「新的生命周期」——旧令牌已作废，任何并发的
        刷新竞争结果都不该被保留（它在竞态里用的旧 refresh_token 已失效）。
        """
        self._require_enabled()
        issued = now or datetime.now(UTC)
        binding = LarkBinding(
            user_id=user_id,
            open_id=open_id,
            user_name=user_name,
            access_token_ciphertext=self._encrypt(tokens.access_token),
            refresh_token_ciphertext=self._encrypt(tokens.refresh_token),
            expires_at=issued + timedelta(seconds=tokens.expires_in or DEFAULT_ACCESS_TTL_S),
            refresh_expires_at=issued
            + timedelta(
                seconds=tokens.refresh_expires_in
                if tokens.refresh_expires_in is not None
                else DEFAULT_REFRESH_TTL_S
            ),
            status=BINDING_STATUS_ACTIVE,
            version=0,
            updated_at=issued,
        )
        async with self._session_factory() as session:
            await session.merge(_binding_to_row(binding))
            await session.commit()
        return binding

    async def delete_binding(self, user_id: str) -> bool:
        """删除绑定，返回是否真的删掉了（对未绑定用户为 False，不报错）。"""
        self._require_enabled()
        async with self._session_factory() as session:
            row = await session.get(LarkBindingRow, user_id)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def decrypt_access_token(self, binding: LarkBinding) -> str:
        """解密某条绑定的 UAT。

        WHY 独立成公开方法而非暴露 `_decrypt`：解绑需要用它调远端撤销，
        而消费侧访问私有成员会把「加密边界」变成隐式约定（分层与可测性都受损）。
        """
        self._require_enabled()
        return self._decrypt(binding.access_token_ciphertext)

    # —— 令牌取用（含并发安全刷新，§7 六步）——

    async def resolve_access_token(
        self,
        user_id: str,
        *,
        refresh: RefreshFn,
        now: datetime | None = None,
    ) -> str:
        """取该用户当前可用的 `user_access_token`（临近过期则先刷新）。

        `refresh` 是注入的刷新函数（`async (refresh_token) -> LarkTokenSet`）——
        WHY 注入而非内联调 OAuth 客户端：本模块只负责**存储与并发语义**，
        网络调用属上层组装，便于离线测试完整覆盖六步竞态。
        失败语义：绑定缺失/失效/竞态后仍拿不到 → `LarkUnboundError`（引导重新绑定）。
        """
        moment = now or datetime.now(UTC)
        binding = await self.get_binding(user_id)
        if binding is None:
            raise LarkUnboundError(user_id)
        if not binding.is_active():
            raise LarkUnboundError(user_id, detail="绑定已失效")

        # ② 未临近过期 → 直接用现有的（不刷新，省一次网络往返与轮换风险）。
        if not self._needs_refresh(binding, moment):
            return self._decrypt(binding.access_token_ciphertext)

        # ③ 临近过期 → 刷新（网络 IO，不持锁）。
        old_refresh = self._decrypt(binding.refresh_token_ciphertext)
        try:
            tokens = await refresh(old_refresh)
        except LarkCliError:
            # ⑥ 刷新失败 → 重读判定「并发竞争」还是「真过期」。
            return await self._resolve_after_refresh_failure(user_id, binding.version, moment)

        # ④ 条件更新（乐观锁）；⑤ 影响行数为 0 → 别人已刷新成功 → 重读用新值。
        updated = await self._conditional_update(
            user_id, expected_version=binding.version, tokens=tokens, now=moment
        )
        if not updated:
            logger.info("飞书令牌刷新竞态：其他 worker 已刷新 user=%s，改用其结果", user_id)
            return await self._read_fresh_token(user_id, moment)
        return tokens.access_token

    async def _resolve_after_refresh_failure(
        self, user_id: str, expected_version: int, now: datetime
    ) -> str:
        """刷新失败后的竞态判定（§7 第 ⑥ 步）：重读该行。

        行已变化（version 不同）→ 并发竞争：别人已刷新成功，**直接用新 token**；
        行未变化 → 刷新是真失败（refresh_token 真的过期/被撤销）→ 标记失效并引导重绑。
        """
        fresh = await self.get_binding(user_id)
        if fresh is not None and fresh.version != expected_version and fresh.is_active():
            logger.info("飞书刷新失败实为并发竞争（version 已前进）user=%s，采用新令牌", user_id)
            if not self._needs_refresh(fresh, now):
                return self._decrypt(fresh.access_token_ciphertext)
        # 行未变化（或仍未临近过期判定不成立）→ 判定绑定失效，引导重新绑定。
        await self._mark_invalid(user_id)
        raise LarkUnboundError(user_id, detail="刷新凭证已失效")

    async def _read_fresh_token(self, user_id: str, now: datetime) -> str:
        """竞态失败方重读：若新值仍临近过期（极端时序），视为失效要求重绑。"""
        fresh = await self.get_binding(user_id)
        if fresh is None or not fresh.is_active():
            raise LarkUnboundError(user_id)
        if self._needs_refresh(fresh, now):
            # 重读到的仍是临过期值：说明并发方刷新的结果也没落地，放弃本轮。
            raise LarkUnboundError(user_id, detail="令牌刷新未生效")
        return self._decrypt(fresh.access_token_ciphertext)

    async def _conditional_update(
        self,
        user_id: str,
        *,
        expected_version: int,
        tokens: LarkTokenSet,
        now: datetime,
    ) -> bool:
        """乐观锁条件更新：`WHERE user_id=? AND version=?` → version+1。返回是否抢到。"""
        async with self._session_factory() as session:
            stmt = (
                update(LarkBindingRow)
                .where(
                    LarkBindingRow.user_id == user_id,
                    LarkBindingRow.version == expected_version,
                )
                .values(
                    access_token_ciphertext=self._encrypt(tokens.access_token),
                    refresh_token_ciphertext=self._encrypt(tokens.refresh_token),
                    expires_at=now + timedelta(seconds=tokens.expires_in or DEFAULT_ACCESS_TTL_S),
                    refresh_expires_at=now
                    + timedelta(
                        seconds=tokens.refresh_expires_in
                        if tokens.refresh_expires_in is not None
                        else DEFAULT_REFRESH_TTL_S
                    ),
                    status=BINDING_STATUS_ACTIVE,
                    version=expected_version + 1,
                    updated_at=now,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
            return bool(result.rowcount)

    async def _mark_invalid(self, user_id: str) -> None:
        """标记绑定失效（保留行与密文，便于诊断「为什么掉绑」；取用侧一律判未绑定）。"""
        async with self._session_factory() as session:
            await session.execute(
                update(LarkBindingRow)
                .where(LarkBindingRow.user_id == user_id)
                .values(status=BINDING_STATUS_INVALID, updated_at=datetime.now(UTC))
            )
            await session.commit()

    # —— 内部辅助 ——

    def _needs_refresh(self, binding: LarkBinding, now: datetime) -> bool:
        """是否临近过期需刷新（提前量 `refresh_skew_s`，配置项 `lark_token_refresh_skew_s`）。"""
        return _to_utc(binding.expires_at) <= now + timedelta(seconds=self._refresh_skew_s)

    def _encrypt(self, value: str) -> str:
        """加密（空值保持空串：未带 offline_access 时 refresh_token 可能缺失）。"""
        if not value:
            return ""
        return self._cipher_required().encrypt(value)

    def _decrypt(self, ciphertext: str) -> str:
        """解密（空密文 → 空串，不抛错）。"""
        if not ciphertext:
            return ""
        return self._cipher_required().decrypt(ciphertext)

    def _cipher_required(self) -> LarkTokenCipher:
        """取加密器；未配置即 fail-closed 抛错。

        WHY 不用 `assert`：安全守卫不得依赖断言（`python -O` 会整体剥离），
        否则「未配置密钥」会退化为「以 None 调用加密」或更糟的明文路径。
        """
        if self._cipher is None:
            raise LarkBoundError(BINDING_DISABLED_REASON)
        return self._cipher

    def _require_enabled(self) -> None:
        """fail-closed 守卫：密钥未配置时拒绝一切读写（绝不落明文）。"""
        if self._cipher is None:
            raise LarkBoundError(BINDING_DISABLED_REASON)


# 刷新函数签名：`async (refresh_token) -> LarkTokenSet`（定义在文件上部，此处不再重复）。


def _binding_to_row(binding: LarkBinding) -> LarkBindingRow:
    """域模型 → 行模型。"""
    return LarkBindingRow(
        user_id=binding.user_id,
        open_id=binding.open_id,
        user_name=binding.user_name,
        access_token_ciphertext=binding.access_token_ciphertext,
        refresh_token_ciphertext=binding.refresh_token_ciphertext,
        expires_at=binding.expires_at,
        refresh_expires_at=binding.refresh_expires_at,
        status=binding.status,
        version=binding.version,
        updated_at=binding.updated_at,
    )


def _binding_from_row(row: LarkBindingRow) -> LarkBinding:
    """行模型 → 域模型（时间统一规整为 aware UTC）。"""
    return LarkBinding(
        user_id=row.user_id,
        open_id=row.open_id,
        user_name=row.user_name,
        access_token_ciphertext=row.access_token_ciphertext,
        refresh_token_ciphertext=row.refresh_token_ciphertext,
        expires_at=_to_utc(row.expires_at),
        refresh_expires_at=_to_utc(row.refresh_expires_at) if row.refresh_expires_at else None,
        status=row.status,
        version=row.version,
        updated_at=_to_utc(row.updated_at),
    )


def _flow_to_row(flow: LarkDeviceFlow) -> LarkDeviceFlowRow:
    """待定态 → 行模型。"""
    return LarkDeviceFlowRow(
        user_id=flow.user_id,
        device_code=flow.device_code,
        user_code=flow.user_code,
        verification_uri=flow.verification_uri,
        verification_uri_complete=flow.verification_uri_complete,
        flow_id=flow.flow_id,
        expires_at=flow.expires_at,
        interval_s=flow.interval_s,
        created_at=flow.created_at,
    )


def _flow_from_row(row: LarkDeviceFlowRow) -> LarkDeviceFlow:
    """行模型 → 待定态。"""
    return LarkDeviceFlow(
        user_id=row.user_id,
        device_code=row.device_code,
        user_code=row.user_code,
        verification_uri=row.verification_uri,
        verification_uri_complete=row.verification_uri_complete,
        flow_id=row.flow_id,
        expires_at=_to_utc(row.expires_at),
        interval_s=row.interval_s,
        created_at=_to_utc(row.created_at),
    )


def _to_utc(value: datetime) -> datetime:
    """时区规整：naive 视为 UTC，aware 统一转 UTC（SQLite 读回可能丢 tz）。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _expired(value: datetime | None) -> bool:
    """过期判断（None 视为永不过期）。"""
    if value is None:
        return False
    return _to_utc(value) <= datetime.now(UTC)


def _build_engine(database_url: str | None) -> AsyncEngine:
    """按 DSN 有无构造异步 engine：无 → SQLite memory；有 → Postgres（psycopg 异步驱动）。

    WHY StaticPool + check_same_thread=False：`:memory:` 与连接一一对应，换连接即
    「换库」；固定单连接保证 dev 单进程语义一致（与 sessions 表同款先例）。
    """
    if database_url is None:
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    url = database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return create_async_engine(url)


def build_lark_binding_repository(
    cipher: LarkTokenCipher | None,
    database_url: str | None = None,
    *,
    refresh_skew_s: int = 300,
) -> SQLAlchemyLarkBindingRepository:
    """统一构造绑定仓库（镜像 `build_session_repository` 的 DSN 裁决）。"""
    return SQLAlchemyLarkBindingRepository(
        cipher,
        database_url,
        refresh_skew_s=refresh_skew_s,
    )


__all__ = [
    "DEFAULT_ACCESS_TTL_S",
    "DEFAULT_REFRESH_TTL_S",
    "LarkBindingRepository",
    "RefreshFn",
    "SQLAlchemyLarkBindingRepository",
    "build_lark_binding_repository",
]
