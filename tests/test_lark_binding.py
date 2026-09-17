"""飞书绑定域测试：设备码 OAuth（离线 fake 响应）+ 令牌加密 + 并发安全刷新。

全部离线（`httpx.MockTransport` + 内存 SQLite），零网络、零 lark-cli、零真实凭据。

覆盖 dev-version5.1.md 的关键验收点：
- **V51-M3**：设备码闭环（发起 → pending → slow_down → 成功换码）；
- **§7 最小 scope**：必须显式声明且含 `offline_access`（否则无 refresh_token）；
- **§4.3 端点**：认证族 `accounts.*` 发起、业务族 `open.*` 换码/刷新（域名不可混用）；
- **§7 并发刷新六步**：乐观锁抢更新 / 竞态重读 / `invalid_grant` 两种含义的区分；
- **加密与 fail-closed**：密文落库、无密钥即关闭。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from shared.lark import (
    BINDING_DISABLED_REASON,
    BINDING_STATUS_INVALID,
    LARK_MINIMAL_SCOPES,
    LARK_UNBOUND_PREFIX,
    FernetTokenCipher,
    LarkBoundError,
    LarkCliError,
    LarkDeviceFlow,
    LarkOAuthClient,
    LarkTokenSet,
    LarkUnboundError,
    build_lark_binding_repository,
    build_token_cipher,
)
from shared.lark.models import (
    DEVICE_FLOW_DONE,
    DEVICE_FLOW_EXPIRED,
    DEVICE_FLOW_PENDING,
    DEVICE_FLOW_SLOW_DOWN,
)
from shared.lark.oauth import (
    ERR_AUTHORIZATION_PENDING,
    ERR_SLOW_DOWN,
    PATH_DEVICE_AUTHORIZATION,
    PATH_TOKEN_ENDPOINT,
    PATH_USER_INFO,
)

# 假凭据（测试豁免 S105/S106 已在 pyproject per-file-ignores 声明）。
APP_ID = "cli_test_app"
APP_SECRET = "test-app-secret"
TOKEN_KEY = "unit-test-token-key"


def _client(handler: Any) -> LarkOAuthClient:
    """构造打桩 OAuth 客户端（httpx.MockTransport：拦截全部请求，零网络）。"""
    return LarkOAuthClient(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        accounts_base_url="https://accounts.feishu.cn",
        open_base_url="https://open.feishu.cn",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _json_response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


# —— 设备码发起 ——


async def test_start_device_flow_declares_minimal_scope_with_offline_access() -> None:
    """发起设备码必须**显式**声明最小 scope 且含 `offline_access`（§7 最易漏的一条）。

    WHY 断言请求体而不只看返回值：不传 scope 时飞书按应用**全量权限**授予
    （实测 4933 字符含高危权限）；漏 offline_access 则无 refresh_token →
    UAT 2h 后强制重绑。二者都是「请求参数错、运行期才暴露」的坑。
    """
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["host"] = request.url.host
        seen["body"] = json.loads(request.content)
        return _json_response(
            {
                "device_code": "d" * 100,
                "user_code": "KR2E-FZQP",
                "verification_uri": "https://accounts.feishu.cn/oauth/v1/device/verify",
                "flow_id": "flow-123",
                "expires_in": 600,
                "interval": 5,
            }
        )

    flow = await _client(handler).start_device_flow(user_id="user-a")
    scopes = seen["body"]["scope"].split()
    assert "offline_access" in scopes
    assert set(scopes) == set(LARK_MINIMAL_SCOPES)
    # 端点必须落在认证族域名（accounts），且是实测路径。
    assert seen["host"] == "accounts.feishu.cn"
    assert seen["path"] == PATH_DEVICE_AUTHORIZATION
    assert seen["body"]["client_id"] == APP_ID
    # 验证页链接必须带 flow_id（实测：旧形态 /page/cli?user_code= 已作废）。
    assert "flow_id=flow-123" in flow.verification_uri_complete
    assert "user_code=KR2E-FZQP" in flow.verification_uri_complete
    assert flow.expires_at > flow.created_at


async def test_start_device_flow_rejects_response_without_device_code() -> None:
    """响应缺 device_code/user_code → 明确报错（不可带着空码进入轮询）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"user_code": "X"})

    with pytest.raises(LarkCliError):
        await _client(handler).start_device_flow(user_id="user-a")


# —— 换码轮询（状态机）——


async def test_exchange_pending_then_slow_down_then_success() -> None:
    """换码状态机：authorization_pending → slow_down → 成功（实测错误码 20094/20095）。"""
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == PATH_TOKEN_ENDPOINT
        body = json.loads(request.content)
        assert body["grant_type"] == "authorization_code"
        calls.append(body["code"])
        if len(calls) == 1:
            return _json_response(
                {"code": ERR_AUTHORIZATION_PENDING, "error": "authorization_pending"}
            )
        if len(calls) == 2:
            # slow_down 实测以 4xx 返回，故客户端需允许读取错误响应体。
            return _json_response({"code": ERR_SLOW_DOWN, "error": "slow_down"}, status=400)
        return _json_response(
            {
                "access_token": "uat-abc",
                "refresh_token": "rt-abc",
                "expires_in": 7200,
                "refresh_token_expires_in": 604800,
            }
        )

    client = _client(handler)
    flow = LarkDeviceFlow(
        user_id="user-a",
        device_code="d" * 100,
        user_code="KR2E-FZQP",
        verification_uri="https://accounts.feishu.cn/oauth/v1/device/verify",
        verification_uri_complete="https://accounts.feishu.cn/oauth/v1/device/verify?flow_id=f",
        flow_id="f",
        expires_at=datetime.now(UTC) + timedelta(seconds=600),
        interval_s=5.0,
        created_at=datetime.now(UTC),
    )
    state1, tokens1 = await client.exchange_device_code(flow)
    state2, tokens2 = await client.exchange_device_code(flow)
    state3, tokens3 = await client.exchange_device_code(flow)
    assert (state1, tokens1) == (DEVICE_FLOW_PENDING, None)
    assert (state2, tokens2) == (DEVICE_FLOW_SLOW_DOWN, None)
    assert state3 == DEVICE_FLOW_DONE
    assert tokens3 is not None
    assert tokens3.access_token == "uat-abc"
    assert tokens3.refresh_token == "rt-abc"
    # slow_down 后间隔上调（RFC 8628），且有上限。
    assert client.next_interval(5.0) == 10.0
    assert client.next_interval(9999.0) <= 60.0


async def test_exchange_expired_or_invalid_grant_requires_restart() -> None:
    """expired_token / invalid_grant → 必须重新发起（不可继续轮询）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"code": 20037, "error": "expired_token"}, status=400)

    flow = LarkDeviceFlow(
        user_id="user-a",
        device_code="d",
        user_code="u",
        verification_uri="v",
        verification_uri_complete="vc",
        flow_id="f",
        expires_at=datetime.now(UTC) + timedelta(seconds=600),
        created_at=datetime.now(UTC),
    )
    state, tokens = await _client(handler).exchange_device_code(flow)
    assert state == DEVICE_FLOW_EXPIRED
    assert tokens is None


async def test_refresh_requires_rotated_refresh_token() -> None:
    """刷新响应**必须**含新 refresh_token（轮换语义下缺了就等于下次刷不了）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["grant_type"] == "refresh_token"
        return _json_response({"access_token": "uat-new", "expires_in": 7200})

    with pytest.raises(LarkCliError, match="refresh_token"):
        await _client(handler).refresh("rt-old")


async def test_fetch_user_info_sends_bearer_token() -> None:
    """用户信息经 Bearer 头取（用于确认「绑的是谁」，不参与鉴权判定）。"""
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["path"] = request.url.path
        return _json_response({"code": 0, "data": {"open_id": "ou_123", "name": "张三"}})

    info = await _client(handler).fetch_user_info("uat-x")
    assert seen["auth"] == "Bearer uat-x"
    assert seen["path"] == PATH_USER_INFO
    assert info.open_id == "ou_123"
    assert info.name == "张三"


async def test_revoke_failure_does_not_raise() -> None:
    """解绑的远端撤销失败**不得**抛错（本地清除才是主语义，用户必须解得掉绑）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    assert await _client(handler).revoke("uat-x") is False


async def test_request_error_is_wrapped_without_leaking_body() -> None:
    """HTTP 错误归一化为 `LarkCliError`，且**不回显响应体**（可能含 token）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"access_token": "leaked-token-value"})

    with pytest.raises(LarkCliError) as excinfo:
        await _client(handler).fetch_user_info("uat-x")
    assert "leaked-token-value" not in str(excinfo.value)
    assert "401" in str(excinfo.value)


# —— 令牌加密 ——


def test_cipher_roundtrip_and_plaintext_absence() -> None:
    """加解密往返正确，且密文不含明文（落库的是密文）。"""
    cipher = build_token_cipher(TOKEN_KEY)
    assert cipher is not None
    token = "u-abcdefghijklmnop" * 300  # 8000+ 字符（贴近实测 UAT 长度）
    sealed = cipher.encrypt(token)
    assert token not in sealed
    assert cipher.decrypt(sealed) == token


def test_cipher_rejects_wrong_key_without_leaking_ciphertext() -> None:
    """换密钥后解密失败 → 报错且不回显密文（避免 oracle / 日志污染）。"""
    sealed = build_token_cipher(TOKEN_KEY).encrypt("secret-uat")  # type: ignore[union-attr]
    other = build_token_cipher("another-key")
    with pytest.raises(LarkCliError) as excinfo:
        other.decrypt(sealed)  # type: ignore[union-attr]
    assert sealed not in str(excinfo.value)
    assert "重新绑定" in str(excinfo.value)


def test_cipher_is_deterministic_across_instances() -> None:
    """同一口令在不同实例/副本派生出相同密钥（多副本共享 `lark_token_key` 才能互解）。"""
    assert FernetTokenCipher(TOKEN_KEY).decrypt(FernetTokenCipher(TOKEN_KEY).encrypt("x")) == "x"


def test_build_token_cipher_is_fail_closed_without_key() -> None:
    """未配置密钥 → 返回 None（fail-closed 判据），不构造可用的加密器。"""
    assert build_token_cipher(None) is None
    assert build_token_cipher("") is None
    assert build_token_cipher("   ") is None


def test_token_set_repr_hides_secrets() -> None:
    """`LarkTokenSet` 的 repr 不得包含令牌（防异常回显/调试日志泄露）。"""
    tokens = LarkTokenSet(access_token="uat-secret", refresh_token="rt-secret", expires_in=7200)
    text = repr(tokens)
    assert "uat-secret" not in text
    assert "rt-secret" not in text


# —— 仓库：加密落库 + fail-closed ——


async def test_repository_stores_ciphertext_only() -> None:
    """落库的是密文：明文 token 不出现在数据库列值里。"""
    repo = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repo.setup()
    try:
        await repo.save_binding(
            user_id="user-a",
            tokens=LarkTokenSet(
                access_token="uat-plain", refresh_token="rt-plain", expires_in=7200
            ),
        )
        binding = await repo.get_binding("user-a")
        assert binding is not None
        assert binding.access_token_ciphertext != "uat-plain"
        assert "uat-plain" not in binding.access_token_ciphertext
        assert "rt-plain" not in binding.refresh_token_ciphertext
        assert binding.status == "active"
    finally:
        await repo.aclose()


async def test_repository_fail_closed_without_key() -> None:
    """无密钥 → 一切读写拒绝（`LarkBoundError`），绝不落明文（fail-closed）。"""
    repo = build_lark_binding_repository(None)
    await repo.setup()
    try:
        assert repo.enabled is False
        with pytest.raises(LarkBoundError) as excinfo:
            await repo.get_binding("user-a")
        assert BINDING_DISABLED_REASON in str(excinfo.value)
        with pytest.raises(LarkBoundError):
            await repo.save_binding(
                user_id="user-a",
                tokens=LarkTokenSet(access_token="x", expires_in=1),
            )
    finally:
        await repo.aclose()


async def test_resolve_unbound_user_raises_routable_unbound_error() -> None:
    """未绑定用户取令牌 → `LarkUnboundError`，消息带可路由前缀（引导绑定而非降级）。"""
    repo = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repo.setup()

    async def _no_refresh(_: str) -> LarkTokenSet:  # pragma: no cover - 不应被调用
        raise AssertionError("未绑定用户不应触发刷新")

    try:
        with pytest.raises(LarkUnboundError) as excinfo:
            await repo.resolve_access_token("ghost", refresh=_no_refresh)
        assert str(excinfo.value).startswith(LARK_UNBOUND_PREFIX)
    finally:
        await repo.aclose()


# —— 刷新路径与并发（§7 六步）——


async def _repo_with_binding(*, expires_at: datetime, version: int = 0) -> tuple[Any, LarkTokenSet]:
    """建库并写入一条绑定（可指定过期时刻/版本），返回 (repo, 明文令牌)。"""
    repo = build_lark_binding_repository(build_token_cipher(TOKEN_KEY), refresh_skew_s=300)
    await repo.setup()
    tokens = LarkTokenSet(
        access_token="uat-old", refresh_token="rt-old", expires_in=7200, refresh_expires_in=604800
    )
    now = datetime.now(UTC)
    await repo.save_binding(user_id="user-a", tokens=tokens, now=now)
    if expires_at != now + timedelta(seconds=7200) or version:
        # 直接改写过期时刻/版本，构造「临近过期」等测试前提。
        from sqlalchemy import update

        from shared.lark.repository import LarkBindingRow

        async with repo._session_factory() as session:
            await session.execute(
                update(LarkBindingRow)
                .where(LarkBindingRow.user_id == "user-a")
                .values(expires_at=expires_at, version=version)
            )
            await session.commit()
    return repo, tokens


async def test_no_refresh_when_token_is_fresh() -> None:
    """未临近过期 → 直接复用现有 UAT（不刷新，省一次往返与轮换风险）。"""
    repo, _ = await _repo_with_binding(expires_at=datetime.now(UTC) + timedelta(seconds=7000))
    called: list[str] = []

    async def _refresh(rt: str) -> LarkTokenSet:  # pragma: no cover - 不应被调用
        called.append(rt)
        raise AssertionError("fresh token 不应触发刷新")

    try:
        assert await repo.resolve_access_token("user-a", refresh=_refresh) == "uat-old"
        assert called == []
    finally:
        await repo.aclose()


async def test_refresh_when_near_expiry_updates_and_bumps_version() -> None:
    """临近过期（skew 内）→ 刷新并条件更新，version +1（乐观锁生效）。"""
    repo, _ = await _repo_with_binding(expires_at=datetime.now(UTC) + timedelta(seconds=100))

    async def _refresh(rt: str) -> LarkTokenSet:
        assert rt == "rt-old"  # 用的是库里的旧 refresh_token
        return LarkTokenSet(
            access_token="uat-new",
            refresh_token="rt-new",
            expires_in=7200,
            refresh_expires_in=604800,
        )

    try:
        assert await repo.resolve_access_token("user-a", refresh=_refresh) == "uat-new"
        binding = await repo.get_binding("user-a")
        assert binding is not None
        assert binding.version == 1
    finally:
        await repo.aclose()


async def test_concurrent_refresh_loser_reads_winner_result() -> None:
    """并发刷新竞态：条件更新失败方**重读**胜者的新 UAT，不重试刷新（§7 第 ⑤ 步）。

    WHY 这是本设计最关键的并发断言：`refresh_token` 轮换后旧值立即失效，
    败方若重试或覆盖写，会把胜者已刷好的令牌作废 → 用户莫名掉绑。
    """
    repo, _ = await _repo_with_binding(expires_at=datetime.now(UTC) + timedelta(seconds=100))
    refresh_calls: list[str] = []

    async def _refresh(_rt: str) -> LarkTokenSet:
        refresh_calls.append("called")
        # 模拟「我刷的过程中，另一个 worker 已经刷完并抢到了条件更新」：
        # 抢先写入新令牌并推进 version → 本方的条件更新必然影响 0 行。
        from sqlalchemy import update

        from shared.lark.repository import LarkBindingRow

        async with repo._session_factory() as session:
            await session.execute(
                update(LarkBindingRow)
                .where(LarkBindingRow.user_id == "user-a")
                .values(
                    access_token_ciphertext=repo._encrypt("uat-winner"),
                    refresh_token_ciphertext=repo._encrypt("rt-winner"),
                    expires_at=datetime.now(UTC) + timedelta(seconds=7200),
                    version=1,
                )
            )
            await session.commit()
        return LarkTokenSet(
            access_token="uat-loser",
            refresh_token="rt-loser",
            expires_in=7200,
            refresh_expires_in=604800,
        )

    try:
        token = await repo.resolve_access_token("user-a", refresh=_refresh)
        # 采用胜者结果，且本方刷新只发生一次（不重试）。
        assert token == "uat-winner"
        assert refresh_calls == ["called"]
        binding = await repo.get_binding("user-a")
        assert binding is not None and binding.version == 1
    finally:
        await repo.aclose()


async def test_refresh_failure_with_advanced_version_uses_new_token() -> None:
    """刷新报错但 version 已前进 → 判定为**并发竞争**，采用别人的新令牌（§7 第 ⑥ 步）。

    WHY 必须区分：`invalid_grant` 在刷新语境下有「并发竞争 / 真过期」两种含义；
    直接判失效会让并发下的用户被误判掉绑（要求重新扫码）。
    """
    repo, _ = await _repo_with_binding(expires_at=datetime.now(UTC) + timedelta(seconds=100))

    async def _refresh(_rt: str) -> LarkTokenSet:
        # 模拟其他 worker 已刷好（version 前进）后，本方拿旧 refresh_token 必然失败。
        from sqlalchemy import update

        from shared.lark.repository import LarkBindingRow

        async with repo._session_factory() as session:
            await session.execute(
                update(LarkBindingRow)
                .where(LarkBindingRow.user_id == "user-a")
                .values(
                    access_token_ciphertext=repo._encrypt("uat-winner"),
                    refresh_token_ciphertext=repo._encrypt("rt-winner"),
                    expires_at=datetime.now(UTC) + timedelta(seconds=7200),
                    version=5,
                )
            )
            await session.commit()
        raise LarkCliError("invalid_grant")

    try:
        assert await repo.resolve_access_token("user-a", refresh=_refresh) == "uat-winner"
        binding = await repo.get_binding("user-a")
        assert binding is not None and binding.status == "active"
    finally:
        await repo.aclose()


async def test_refresh_failure_without_change_marks_invalid_and_unbound() -> None:
    """刷新失败且行**未变化** → 判定真失效：标记 invalid 并抛未绑定（引导重绑）。"""
    repo, _ = await _repo_with_binding(expires_at=datetime.now(UTC) + timedelta(seconds=100))

    async def _refresh(_rt: str) -> LarkTokenSet:
        raise LarkCliError("invalid_grant")

    try:
        with pytest.raises(LarkUnboundError) as excinfo:
            await repo.resolve_access_token("user-a", refresh=_refresh)
        assert str(excinfo.value).startswith(LARK_UNBOUND_PREFIX)
        binding = await repo.get_binding("user-a")
        assert binding is not None
        assert binding.status == BINDING_STATUS_INVALID

        # 失效后再取 → 直接判未绑定（不再尝试刷新）。
        async def _never(_rt: str) -> LarkTokenSet:  # pragma: no cover
            raise AssertionError("失效绑定不应再刷新")

        with pytest.raises(LarkUnboundError):
            await repo.resolve_access_token("user-a", refresh=_never)
    finally:
        await repo.aclose()


# —— 设备码待定态（落库，多 worker 可见）——


async def test_device_flow_roundtrip_and_ttl_expiry() -> None:
    """待定态落库可跨 worker 读取；过期视同不存在；`device_code` 必须匹配。"""
    repo = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repo.setup()
    now = datetime.now(UTC)
    flow = LarkDeviceFlow(
        user_id="user-a",
        device_code="d" * 100,
        user_code="KR2E-FZQP",
        verification_uri="https://accounts.feishu.cn/oauth/v1/device/verify",
        verification_uri_complete="https://accounts.feishu.cn/oauth/v1/device/verify?flow_id=f",
        flow_id="f",
        expires_at=now + timedelta(seconds=600),
        interval_s=5.0,
        created_at=now,
    )
    try:
        await repo.save_device_flow(flow)
        got = await repo.get_device_flow(user_id="user-a", device_code=flow.device_code)
        assert got is not None and got.user_code == "KR2E-FZQP"
        # 码不匹配（用别人的码换绑）→ 未命中
        assert await repo.get_device_flow(user_id="user-a", device_code="other") is None
        # 过期 → 未命中
        expired = flow.model_copy(
            update={"expires_at": now - timedelta(seconds=1), "device_code": "e" * 100}
        )
        await repo.save_device_flow(expired)
        assert await repo.get_device_flow(user_id="user-a", device_code="e" * 100) is None
        # 清除
        await repo.drop_device_flow(user_id="user-a")
        assert await repo.get_device_flow(user_id="user-a", device_code=flow.device_code) is None
    finally:
        await repo.aclose()


async def test_delete_binding_reports_whether_removed() -> None:
    """解绑：存在则删除返回 True；重复解绑返回 False（不报错）。"""
    repo = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repo.setup()
    try:
        await repo.save_binding(
            user_id="user-a",
            tokens=LarkTokenSet(access_token="uat", refresh_token="rt", expires_in=7200),
        )
        assert await repo.delete_binding("user-a") is True
        assert await repo.delete_binding("user-a") is False
        assert await repo.get_binding("user-a") is None
    finally:
        await repo.aclose()


async def test_bindings_are_isolated_per_user() -> None:
    """多用户绑定互不可见：取 A 的绑定拿不到 B 的令牌（V51-M1 隔离要求）。"""
    repo = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repo.setup()
    try:
        await repo.save_binding(
            user_id="user-a",
            tokens=LarkTokenSet(access_token="uat-a", refresh_token="rt-a", expires_in=7200),
        )
        await repo.save_binding(
            user_id="user-b",
            tokens=LarkTokenSet(access_token="uat-b", refresh_token="rt-b", expires_in=7200),
        )

        async def _never(_rt: str) -> LarkTokenSet:  # pragma: no cover
            raise AssertionError("fresh binding 不应刷新")

        assert await repo.resolve_access_token("user-a", refresh=_never) == "uat-a"
        assert await repo.resolve_access_token("user-b", refresh=_never) == "uat-b"
        await repo.delete_binding("user-a")
        assert await repo.get_binding("user-a") is None
        assert await repo.get_binding("user-b") is not None
    finally:
        await repo.aclose()
