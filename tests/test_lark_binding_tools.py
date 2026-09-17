"""飞书绑定工具测试（4 工具闭环 + 按用户隔离 + 未绑定引导）。

全部离线：OAuth 用 `httpx.MockTransport` 打桩、绑定存储用内存 SQLite、CLI 用 fake runner。
经**真实 FastMCP 客户端**调用（`Client(server.mcp)`），故同时验证工具注册与
`_lark_scope` 作用域形参在 MCP 线上的可传递性（v5.1 §5.4）。

覆盖验收点：
- **V51-M3**：设备码闭环（发起 → 完成授权 → 落库）；
- **V51-M1**：A/B 两用户各自绑定互不可见（同一服务实例、不同作用域）；
- **§6.3**：未绑定 → `tool_error.lark_unbound` 前缀（图侧引导而非降级）；
- **fail-closed**：未配置密钥/应用时绑定工具返回确定性说明，不抛未捕获异常。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastmcp import Client

from services.lark_mcp import server
from services.lark_mcp.binding import LarkBindingService
from services.lark_mcp.credentials import (
    BindingCredentialProvider,
    ConfigCredentialProvider,
    build_credential_provider,
)
from services.lark_mcp.models import LarkCliCredentials
from shared.lark import (
    LARK_UNBOUND_PREFIX,
    LarkDeviceFlow,
    LarkOAuthClient,
    LarkTokenSet,
    build_lark_binding_repository,
    build_token_cipher,
)
from shared.lark.oauth import (
    ERR_AUTHORIZATION_PENDING,
    PATH_DEVICE_AUTHORIZATION,
    PATH_TOKEN_ENDPOINT,
)

APP_ID = "cli_bind_test"
APP_SECRET = "bind-test-secret"
TOKEN_KEY = "bind-test-token-key"


def _oauth(handler: Any) -> LarkOAuthClient:
    """打桩 OAuth 客户端（MockTransport：零网络）。"""
    return LarkOAuthClient(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        accounts_base_url="https://accounts.feishu.cn",
        open_base_url="https://open.feishu.cn",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _device_auth_payload(device_code: str = "d" * 100) -> dict[str, Any]:
    return {
        "device_code": device_code,
        "user_code": "KR2E-FZQP",
        "verification_uri": "https://accounts.feishu.cn/oauth/v1/device/verify",
        "flow_id": "flow-1",
        "expires_in": 600,
        "interval": 5,
    }


async def _install_binding_service(handler: Any) -> Any:
    """装配真实绑定服务（真仓库 + 打桩 OAuth）并挂到 server（返回仓库供断言）。"""
    repository = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repository.setup()
    service = LarkBindingService(repository=repository, oauth=_oauth(handler), app_id=APP_ID)
    server.binding_service = service
    return repository


async def _call_tool(name: str, args: dict[str, Any] | None = None) -> str:
    """经真实 MCP 客户端调用工具，返回文本内容。"""
    async with Client(server.mcp) as client:
        result = await client.call_tool(name, args or {})
    return _text_of(result)


def _text_of(result: Any) -> str:
    """提取工具返回文本（FastMCP 结果为 content 列表）。"""
    blocks = getattr(result, "content", None) or []
    if blocks:
        return getattr(blocks[0], "text", str(blocks[0]))
    return str(result)


# —— 工具面登记 ——


async def test_binding_tools_are_registered_with_scope_param_hidden_via_prefix() -> None:
    """4 个绑定工具已注册，且作用域形参以 `_lark_scope` 命名（供拦截器注入）。"""
    async with Client(server.mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    for name in ("lark_bind_start", "lark_bind_complete", "lark_bind_status", "lark_unbind"):
        assert name in tools, name
        properties = set((tools[name].inputSchema or {}).get("properties", {}))
        assert "_lark_scope" in properties, f"{name} 缺少作用域形参"


# —— 绑定闭环 ——


async def test_bind_start_returns_verification_link_and_stores_pending_flow() -> None:
    """发起绑定：返回带 flow_id 的验证链接，并把待定态落库（供后续完成）。"""
    seen_body: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json=_device_auth_payload())

    repository = await _install_binding_service(handler)
    try:
        text = await _call_tool("lark_bind_start", {"_lark_scope": "user-a"})
        assert "https://accounts.feishu.cn/oauth/v1/device/verify" in text
        assert "flow_id=flow-1" in text
        assert "KR2E-FZQP" in text  # 用户码便于人工核对
        # scope 必须显式且含 offline_access（否则无 refresh_token，2h 后强制重绑）。
        scopes = seen_body["scope"].split()
        assert "offline_access" in scopes
        # 待定态已落库（换码阶段需要它）。
        pending = await repository.peek_device_flow("user-a")
        assert pending is not None and pending.device_code == "d" * 100
        # 不同用户互不可见：B 没有待定态。
        assert await repository.peek_device_flow("user-b") is None
    finally:
        await repository.aclose()


async def test_bind_complete_pending_then_success_persists_encrypted_binding() -> None:
    """完成绑定：先「待授权」提示，授权后一次调用即落库（密文）并回报账号。"""
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == PATH_DEVICE_AUTHORIZATION:
            return httpx.Response(200, json=_device_auth_payload())
        if request.url.path == PATH_TOKEN_ENDPOINT:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    400, json={"code": ERR_AUTHORIZATION_PENDING, "error": "authorization_pending"}
                )
            return httpx.Response(
                200,
                json={
                    "access_token": "uat-final",
                    "refresh_token": "rt-final",
                    "expires_in": 7200,
                    "refresh_token_expires_in": 604800,
                },
            )
        # user_info（v1，Bearer）
        return httpx.Response(200, json={"code": 0, "data": {"open_id": "ou_a", "name": "张三"}})

    repository = await _install_binding_service(handler)
    try:
        await _call_tool("lark_bind_start", {"_lark_scope": "user-a"})
        # 用户还没授权 → 引导继续（不是错误）
        pending_text = await _call_tool("lark_bind_complete", {"_lark_scope": "user-a"})
        assert "还没有检测到授权" in pending_text

        done_text = await _call_tool("lark_bind_complete", {"_lark_scope": "user-a"})
        assert "绑定成功" in done_text
        assert "张三" in done_text
        # 输出绝不含 token（安全硬约束）。
        assert "uat-final" not in done_text
        assert "rt-final" not in done_text

        binding = await repository.get_binding("user-a")
        assert binding is not None and binding.open_id == "ou_a"
        # 落库为密文（明文不得出现）。
        assert "uat-final" not in binding.access_token_ciphertext
        assert "rt-final" not in binding.refresh_token_ciphertext
        # 待定态已清理（绑定完成后不应残留）。
        assert await repository.peek_device_flow("user-a") is None
    finally:
        await repository.aclose()


async def test_bind_complete_without_pending_flow_is_deterministic() -> None:
    """没有待定流程时完成绑定 → 明确提示重新发起（不静默失败）。"""

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("无待定态时不应发起任何 HTTP 请求")

    repository = await _install_binding_service(handler)
    try:
        text = await _call_tool("lark_bind_complete", {"_lark_scope": "user-a"})
        assert "没有进行中的授权流程" in text
    finally:
        await repository.aclose()


async def test_bind_complete_expired_device_code_asks_restart() -> None:
    """设备码在飞书侧已过期（换码返回 expired_token）→ 明确提示**重新发起**。

    WHY 走这条路而非「本地 expires_at 过期」：本地过期时 `peek_device_flow` 直接
    返回 None（视同不存在），路径与「没发起过」相同；而**服务端判定过期**才是
    需要清理待定态并引导重新发起的那条分支（实测 code 20037 / expired_token）。
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == PATH_TOKEN_ENDPOINT:
            return httpx.Response(400, json={"code": 20037, "error": "expired_token"})
        return httpx.Response(200, json=_device_auth_payload())

    repository = await _install_binding_service(handler)
    try:
        await _call_tool("lark_bind_start", {"_lark_scope": "user-a"})
        assert await repository.peek_device_flow("user-a") is not None
        text = await _call_tool("lark_bind_complete", {"_lark_scope": "user-a"})
        assert "已失效" in text
        assert "重新发起" in text
        # 过期待定态应被清理，避免用户反复触发同一个失效码。
        assert await repository.peek_device_flow("user-a") is None
    finally:
        await repository.aclose()


async def test_bind_complete_local_expired_flow_is_same_as_absent() -> None:
    """本地已过期的待定态视同不存在（不拿过期码去飞书换绑）。"""

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        if request.url.path == PATH_DEVICE_AUTHORIZATION:
            return httpx.Response(200, json=_device_auth_payload())
        raise AssertionError("过期待定态不应触发换码请求")

    repository = await _install_binding_service(handler)
    try:
        await _call_tool("lark_bind_start", {"_lark_scope": "user-a"})
        # 把待定态改成已过期（本地 TTL 判定）
        from sqlalchemy import update

        from shared.lark.repository import LarkDeviceFlowRow

        async with repository._session_factory() as session:
            await session.execute(
                update(LarkDeviceFlowRow).values(
                    expires_at=datetime.now(UTC) - timedelta(seconds=1)
                )
            )
            await session.commit()
        text = await _call_tool("lark_bind_complete", {"_lark_scope": "user-a"})
        assert "没有进行中的授权流程" in text
    finally:
        await repository.aclose()


# —— 状态查询 ——


async def test_bind_status_reports_unbound_then_bound_account() -> None:
    """状态查询：未绑定给引导；绑定后给出账号名与剩余时间，且**不含 token**。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == PATH_DEVICE_AUTHORIZATION:
            return httpx.Response(200, json=_device_auth_payload())
        if request.url.path == PATH_TOKEN_ENDPOINT:
            return httpx.Response(
                200,
                json={
                    "access_token": "uat-secret-value",
                    "refresh_token": "rt-secret-value",
                    "expires_in": 7200,
                    "refresh_token_expires_in": 604800,
                },
            )
        return httpx.Response(200, json={"code": 0, "data": {"open_id": "ou_a", "name": "李四"}})

    repository = await _install_binding_service(handler)
    try:
        unbound = await _call_tool("lark_bind_status", {"_lark_scope": "user-a"})
        assert "还没有绑定" in unbound

        await _call_tool("lark_bind_start", {"_lark_scope": "user-a"})
        await _call_tool("lark_bind_complete", {"_lark_scope": "user-a"})

        bound = await _call_tool("lark_bind_status", {"_lark_scope": "user-a"})
        assert "李四" in bound
        assert "剩余约" in bound
        assert "uat-secret-value" not in bound
        assert "rt-secret-value" not in bound
    finally:
        await repository.aclose()


async def test_bind_status_reports_invalid_binding() -> None:
    """绑定被标记失效 → 状态查询引导重新绑定（不谎报「已绑定」）。"""

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("状态查询不应发起 HTTP 请求")

    repository = await _install_binding_service(handler)
    try:
        await repository.save_binding(
            user_id="user-a",
            tokens=LarkTokenSet(access_token="uat", refresh_token="rt", expires_in=7200),
        )
        from sqlalchemy import update

        from shared.lark.models import BINDING_STATUS_INVALID
        from shared.lark.repository import LarkBindingRow

        async with repository._session_factory() as session:
            await session.execute(
                update(LarkBindingRow)
                .where(LarkBindingRow.user_id == "user-a")
                .values(status=BINDING_STATUS_INVALID)
            )
            await session.commit()
        text = await _call_tool("lark_bind_status", {"_lark_scope": "user-a"})
        assert "已失效" in text
        assert "重新绑定" in text
    finally:
        await repository.aclose()


# —— 解绑 ——


async def test_unbind_clears_local_binding_even_if_revoke_fails() -> None:
    """解绑：远端撤销失败也要清掉本地记录（用户必须解得掉绑）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/revoke"):
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"code": 0, "data": {"open_id": "ou_a", "name": "A"}})

    repository = await _install_binding_service(handler)
    try:
        await repository.save_binding(
            user_id="user-a",
            tokens=LarkTokenSet(access_token="uat", refresh_token="rt", expires_in=7200),
        )
        text = await _call_tool("lark_unbind", {"_lark_scope": "user-a"})
        assert "已解除" in text
        assert await repository.get_binding("user-a") is None
        # 重复解绑 → 明确说无需解绑（幂等）
        again = await _call_tool("lark_unbind", {"_lark_scope": "user-a"})
        assert "没有绑定" in again
    finally:
        await repository.aclose()


# —— 多用户隔离（V51-M1 核心）——


async def test_two_users_bindings_are_isolated() -> None:
    """A/B 两用户各自绑定：互不可见、互不串号（同一服务实例、仅作用域不同）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == PATH_DEVICE_AUTHORIZATION:
            return httpx.Response(200, json=_device_auth_payload())
        if request.url.path == PATH_TOKEN_ENDPOINT:
            return httpx.Response(
                200,
                json={
                    "access_token": "uat",
                    "refresh_token": "rt",
                    "expires_in": 7200,
                    "refresh_token_expires_in": 604800,
                },
            )
        return httpx.Response(200, json={"code": 0, "data": {"open_id": "ou", "name": "X"}})

    repository = await _install_binding_service(handler)
    try:
        for user in ("user-a", "user-b"):
            await _call_tool("lark_bind_start", {"_lark_scope": user})
            await _call_tool("lark_bind_complete", {"_lark_scope": user})
        assert await repository.get_binding("user-a") is not None
        assert await repository.get_binding("user-b") is not None
        # 解绑 A 不影响 B（隔离的强断言）
        await _call_tool("lark_unbind", {"_lark_scope": "user-a"})
        assert await repository.get_binding("user-a") is None
        assert await repository.get_binding("user-b") is not None
    finally:
        await repository.aclose()


# —— 作用域缺失 / 绑定未启用 ——


async def test_missing_scope_is_rejected() -> None:
    """缺作用域 → 工具报错（说明注入链路断裂），不猜身份。"""

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("不应发起请求")

    repository = await _install_binding_service(handler)
    try:
        with pytest.raises(Exception, match="_lark_scope"):
            await _call_tool("lark_bind_status", {})
    finally:
        await repository.aclose()


async def test_binding_tools_report_disabled_when_service_absent() -> None:
    """绑定未启用（无应用凭据/密钥）→ 工具明确报「未启用」，而非静默失败或误判未绑定。

    WHY 断言原因关键词而非仅断言「抛错」：绑定未启用与用户未绑定是**两种不同的处置**
    （前者是部署问题、后者要引导用户去绑定）。缺了这句文案，agent 侧只能当成普通
    工具执行失败并降级，用户永远看不到「服务端没配好」这一事实。
    """
    original = server.binding_service
    server.binding_service = None
    try:
        with pytest.raises(Exception) as excinfo:
            await _call_tool("lark_bind_status", {"_lark_scope": "user-a"})
        assert "未启用" in str(excinfo.value)
        # 不得把「部署未启用」误报成「你没绑定」（两者要求用户做的事完全不同）。
        assert LARK_UNBOUND_PREFIX not in str(excinfo.value)
    finally:
        server.binding_service = original


# —— 凭据提供者装配（绑定态 vs 配置态）——


async def test_binding_credential_provider_requires_matching_scope() -> None:
    """绑定态凭据解析：无绑定 → 未绑定引导；有绑定 → 返回**该用户自己**的 UAT。"""

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("解析凭据不应发起 HTTP 请求")

    repository = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repository.setup()
    try:
        provider = BindingCredentialProvider(
            repository=repository, oauth=_oauth(handler), app_id=APP_ID, app_secret=APP_SECRET
        )
        with pytest.raises(Exception) as excinfo:
            await provider.resolve("user-a")
        assert LARK_UNBOUND_PREFIX in str(excinfo.value)

        await repository.save_binding(
            user_id="user-a",
            tokens=LarkTokenSet(access_token="uat-a", refresh_token="rt-a", expires_in=7200),
        )
        creds = await provider.resolve("user-a")
        assert isinstance(creds, LarkCliCredentials)
        assert creds.user_access_token == "uat-a"
        assert creds.app_id == APP_ID
        # 另一个用户仍未绑定 → 不得借用 A 的令牌
        with pytest.raises(Exception) as excinfo2:
            await provider.resolve("user-b")
        assert LARK_UNBOUND_PREFIX in str(excinfo2.value)
    finally:
        await repository.aclose()


def test_build_credential_provider_prefers_binding_when_available() -> None:
    """装配裁决：绑定可用 → 按用户隔离；不可用 → 退回配置态单用户（零回归）。"""
    from services.lark_mcp.config import LarkMCPRuntimeSettings

    settings = LarkMCPRuntimeSettings(
        lark_app_id=APP_ID,
        lark_app_secret=APP_SECRET,  # type: ignore[arg-type]
        lark_token_key=TOKEN_KEY,  # type: ignore[arg-type]
        lark_binding_enabled=True,
    )
    repository = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    provider = build_credential_provider(
        settings, repository=repository, oauth=_oauth(_never_called)
    )
    assert isinstance(provider, BindingCredentialProvider)

    # 密钥缺失 → fail-closed 关闭绑定，退回配置态（多用户隔离不可用）
    disabled_repo = build_lark_binding_repository(None)
    fallback = build_credential_provider(
        settings, repository=disabled_repo, oauth=_oauth(_never_called)
    )
    assert isinstance(fallback, ConfigCredentialProvider)

    # 绑定总开关关闭 → 同样退回配置态
    off = LarkMCPRuntimeSettings(
        lark_app_id=APP_ID,
        lark_app_secret=APP_SECRET,  # type: ignore[arg-type]
        lark_token_key=TOKEN_KEY,  # type: ignore[arg-type]
        lark_binding_enabled=False,
    )
    assert isinstance(
        build_credential_provider(off, repository=repository, oauth=_oauth(_never_called)),
        ConfigCredentialProvider,
    )


async def _never_called(request: httpx.Request) -> httpx.Response:  # pragma: no cover
    raise AssertionError("本用例不应发起 HTTP 请求")


# —— 设备码待定态与作用域一致性 ——


async def test_pending_flow_is_scoped_to_its_owner() -> None:
    """A 的待定态不得被 B 用来完成绑定（防串号）。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == PATH_DEVICE_AUTHORIZATION:
            return httpx.Response(200, json=_device_auth_payload())
        return httpx.Response(
            200,
            json={
                "access_token": "uat",
                "refresh_token": "rt",
                "expires_in": 7200,
                "refresh_token_expires_in": 604800,
            },
        )

    repository = await _install_binding_service(handler)
    try:
        await _call_tool("lark_bind_start", {"_lark_scope": "user-a"})
        # B 从未发起 → 完成绑定应提示无流程（而不是替 A 完成）
        text = await _call_tool("lark_bind_complete", {"_lark_scope": "user-b"})
        assert "没有进行中的授权流程" in text
        assert await repository.get_binding("user-b") is None
        assert await repository.peek_device_flow("user-a") is not None
    finally:
        await repository.aclose()


def test_device_flow_model_defaults_are_sane() -> None:
    """待定态模型的默认间隔与 TTL 与实测一致（interval 5s / 10 分钟）。"""
    now = datetime.now(UTC)
    flow = LarkDeviceFlow(
        user_id="u",
        device_code="d" * 100,
        user_code="KR2E-FZQP",
        verification_uri="v",
        verification_uri_complete="vc",
        flow_id="f",
        expires_at=now + timedelta(seconds=600),
        created_at=now,
    )
    assert flow.interval_s == 5.0
    assert (flow.expires_at - flow.created_at).total_seconds() == 600
