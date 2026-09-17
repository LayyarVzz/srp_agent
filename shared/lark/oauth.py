"""飞书 OAuth 客户端（设备码授权 + 换码 + 刷新，RFC 8628）。

**端点实测固化（§4.3 / §14.4，勿改回 v1）**：认证族与业务族**分属两个域名** ——

| 用途 | 端点 |
|---|---|
| 发起设备码 | `POST {accounts_base}/oauth/v1/device_authorization` |
| 验证页 | `{accounts_base}/oauth/v1/device/verify?flow_id=&user_code=` |
| 换码 / 刷新 | `POST {open_base}/open-apis/authen/v2/oauth/token` |
| 用户信息 | `GET {open_base}/open-apis/authen/v1/user_info` |

v1 的 `/open-apis/authen/v1/...` 只认 `code`（走不了 device flow）；`accounts.*` 是
唯一能发起设备码的域。**两处域名混用会静默失败**，故以常量集中声明。

**scope 必须显式声明**（§7 实测，最易漏且后果严重）：
不传 scope → 飞书按应用**全量已申请权限**授予（实测 4933 字符，含 `base:app:create`
等高危权限）；不传 `offline_access` → 响应**不含 refresh_token**，UAT 2h 后强制重绑。
故 `LARK_MINIMAL_SCOPES` 显式列出「5 个工具所需 + offline_access」。

**轮询语义**（RFC 8628 + 实测错误码）：`authorization_pending`(20094) 继续、
`slow_down`(20095) 降频、`expired_token`/`invalid_grant` 需重新发起、成功一次性拿齐
`access_token` + `refresh_token` + `expires_in`。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from shared.lark.errors import LarkCliError
from shared.lark.models import (
    DEVICE_FLOW_DONE,
    DEVICE_FLOW_EXPIRED,
    DEVICE_FLOW_PENDING,
    DEVICE_FLOW_SLOW_DOWN,
    LarkDeviceFlow,
    LarkTokenSet,
    LarkUserInfo,
)

logger = logging.getLogger(__name__)

# —— 认证族端点（accounts 域）——
PATH_DEVICE_AUTHORIZATION = "/oauth/v1/device_authorization"
PATH_DEVICE_VERIFY = "/oauth/v1/device/verify"

# —— 业务族端点（open 域）——
# 命名说明：换码与刷新**共用同一端点**（靠 grant_type 区分），故名为 ENDPOINT 而非 TOKEN。
PATH_TOKEN_ENDPOINT = "/open-apis/authen/v2/oauth/token"
PATH_USER_INFO = "/open-apis/authen/v1/user_info"

# 最小 scope（§7 实测结论）：5 个工具所需 + offline_access（换 refresh_token 的必要条件）。
LARK_MINIMAL_SCOPES: tuple[str, ...] = (
    "offline_access",  # 必须：否则无 refresh_token，UAT 2h 后强制重绑
    "calendar:calendar",  # lark_calendar_get_agenda
    "task:task",  # lark_task_create
    "im:message",  # lark_im_send_message
    "contact:user.base:readonly",  # lark_contact_resolve_name
    "docx:document:readonly",  # lark_docs_read
)

# 实测错误码：授权待定 / 降频（§7 表格）。
ERR_AUTHORIZATION_PENDING = 20094
ERR_SLOW_DOWN = 20095

# 轮询间隔的上下界（响应 interval 实测默认 5s；slow_down 后按 RFC 8628 每次 +5s）。
_DEFAULT_INTERVAL_S = 5.0
_SLOW_DOWN_STEP_S = 5.0
_MAX_INTERVAL_S = 60.0


class LarkOAuthClient:
    """飞书 OAuth 客户端（自建应用作纯客户端，无 bot 身份 —— §4.4 D5.1.4）。"""

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        accounts_base_url: str,
        open_base_url: str,
        timeout_s: float = 15.0,
        scopes: tuple[str, ...] = LARK_MINIMAL_SCOPES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """构造客户端。`client` 可注入（测试用 `httpx.MockTransport`，零网络）。"""
        self._app_id = app_id
        self._app_secret = app_secret
        self._accounts_base = accounts_base_url.rstrip("/")
        self._open_base = open_base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._scopes = scopes
        self._client = client

    # —— 设备码发起 ——

    async def start_device_flow(
        self, *, user_id: str, now: datetime | None = None
    ) -> LarkDeviceFlow:
        """发起设备码授权，返回待定态（含可直接展示/转二维码的验证链接）。

        `scope` 显式声明为最小集合（含 `offline_access`）—— 漏了它就没有
        refresh_token，用户每 2 小时需重绑（§7）。
        """
        issued_at = now or datetime.now(UTC)
        payload = {
            "client_id": self._app_id,
            "scope": " ".join(self._scopes),
        }
        data = await self._post(f"{self._accounts_base}{PATH_DEVICE_AUTHORIZATION}", payload)
        device_code = str(data.get("device_code") or "")
        user_code = str(data.get("user_code") or "")
        flow_id = str(data.get("flow_id") or "")
        if not device_code or not user_code:
            raise LarkCliError("飞书设备码响应缺少 device_code/user_code，无法发起绑定")
        verification_uri = str(
            data.get("verification_uri") or f"{self._accounts_base}{PATH_DEVICE_VERIFY}"
        )
        # WHY 自行拼 verification_uri_complete：实测验证页**必须带 flow_id**
        # （旧形态 /page/cli?user_code= 已作废）；响应里的 verification_uri 不含
        # 查询串，直接给用户会打开一个空验证页。
        complete = (
            str(data.get("verification_uri_complete") or "")
            or f"{verification_uri}?flow_id={flow_id}&user_code={user_code}"
        )
        expires_in = int(data.get("expires_in") or 600)  # 实测 600s
        interval = float(data.get("interval") or _DEFAULT_INTERVAL_S)  # 实测默认 5s
        return LarkDeviceFlow(
            user_id=user_id,
            device_code=device_code,
            user_code=user_code,
            verification_uri=verification_uri,
            verification_uri_complete=complete,
            flow_id=flow_id,
            expires_at=issued_at + timedelta(seconds=expires_in),
            interval_s=interval,
            created_at=issued_at,
        )

    # —— 换码 / 刷新 ——

    async def exchange_device_code(self, flow: LarkDeviceFlow) -> tuple[str, LarkTokenSet] | None:
        """轮询一次换码。

        返回 `(处置码, 令牌集合)`：处置码为 `done`/`pending`/`slow_down`/`expired`
        （集中常量，见 `shared.lark.models`）；`done` 时令牌集合非空，其余为 None。

        WHY 返回处置码而非抛错：轮询是**正常控制流**（用户还没点授权），
        以异常表达会让调用方靠 except 做流程分支，且掩盖真正的失败。
        """
        data = await self._post(
            f"{self._open_base}{PATH_TOKEN_ENDPOINT}",
            {
                "grant_type": "authorization_code",
                "client_id": self._app_id,
                "client_secret": self._app_secret,
                "code": flow.device_code,
            },
            allow_error_body=True,
        )
        code = _error_code(data)
        if code in (ERR_AUTHORIZATION_PENDING,):
            return DEVICE_FLOW_PENDING, None
        if code in (ERR_SLOW_DOWN,):
            return DEVICE_FLOW_SLOW_DOWN, None
        if code is not None:
            # expired_token / invalid_grant 等 → 必须重新发起（§7）。
            logger.info("飞书设备码换码未成功（code=%s），需重新发起绑定", code)
            return DEVICE_FLOW_EXPIRED, None
        return DEVICE_FLOW_DONE, _token_set(data)

    async def refresh(self, refresh_token: str) -> LarkTokenSet:
        """用 refresh_token 换新令牌。

        ⚠️ **refresh_token 会轮换**（实测）：成功后返回新值、**旧值立即失效**。
        故调用方必须按 §7 的乐观锁流程处理并发（条件更新 + 竞态重读），
        且 `invalid_grant` 存在「并发竞争 / 真过期」两种含义，**不可**直接判失效。
        """
        data = await self._post(
            f"{self._open_base}{PATH_TOKEN_ENDPOINT}",
            {
                "grant_type": "refresh_token",
                "client_id": self._app_id,
                "client_secret": self._app_secret,
                "refresh_token": refresh_token,
            },
        )
        return _token_set(data, require_refresh=True)

    # —— 用户信息 ——

    async def fetch_user_info(self, access_token: str) -> LarkUserInfo:
        """取绑定用户基本信息（仅用于确认「绑的是谁」，不参与鉴权）。"""
        url = f"{self._open_base}{PATH_USER_INFO}"
        data = await self._request("GET", url, headers={"Authorization": f"Bearer {access_token}"})
        # v1 user_info 返回 {code, msg, data:{open_id, name, ...}}；兼容扁平形态。
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        open_id = str(payload.get("open_id") or "")
        if not open_id:
            raise LarkCliError("飞书用户信息响应缺少 open_id")
        return LarkUserInfo(open_id=open_id, name=_optional_str(payload.get("name")))

    # —— 解绑 ——

    async def revoke(self, access_token: str) -> bool:
        """撤销授权（尽力而为）。返回是否成功撤销。

        WHY 不抛错：解绑的**本地清除是主语义** —— 远端撤销失败（网络/已过期）
        不应让用户「解不掉绑」；失败仅记日志，本地记录照清。
        """
        url = f"{self._open_base}/open-apis/authen/v1/revoke"
        try:
            data = await self._request(
                "POST",
                url,
                json_body={"token": access_token},
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except LarkCliError as exc:
            logger.warning("飞书令牌远端撤销失败（本地绑定仍将清除）：%s", exc)
            return False
        return _error_code(data) is None

    # —— 内部 HTTP ——

    async def _post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        allow_error_body: bool = False,
    ) -> dict[str, Any]:
        """POST JSON。`allow_error_body` 用于换码轮询：4xx 响应体含 `code` 需读取。"""
        return await self._request(
            "POST", url, json_body=payload, allow_error_body=allow_error_body
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_error_body: bool = False,
    ) -> dict[str, Any]:
        """统一发请求并解析 JSON。

        WHY `allow_error_body`：换码轮询的 `authorization_pending` 会以 4xx 返回，
        业务上属正常控制流，需拿到响应体里的 `code` 才能分流（§7）。
        """
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout_s)
        try:
            resp = await client.request(method, url, json=json_body, headers=headers)
            if resp.status_code >= 400 and not allow_error_body:
                # WHY 不回显响应体：可能含 token；只给状态码与路径。
                raise LarkCliError(
                    f"飞书 OAuth 请求失败：{method} {_safe_path(url)} → {resp.status_code}"
                )
            try:
                data = resp.json()
            except ValueError as exc:
                raise LarkCliError(f"飞书 OAuth 响应非 JSON：{method} {_safe_path(url)}") from exc
            if not isinstance(data, dict):
                raise LarkCliError(f"飞书 OAuth 响应结构异常：{method} {_safe_path(url)}")
            return data
        except httpx.HTTPError as exc:
            raise LarkCliError(
                f"飞书 OAuth 请求异常：{_safe_path(url)}（{type(exc).__name__}）"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    def next_interval(self, current_s: float) -> float:
        """`slow_down` 后的新轮询间隔（RFC 8628：每次上调，设上限防失控）。"""
        return min(current_s + _SLOW_DOWN_STEP_S, _MAX_INTERVAL_S)


def _error_code(data: dict[str, Any]) -> int | None:
    """提取业务错误码；无错误返回 None。

    v2 token 端点成功时无 `code`；失败时为 `{"code": 20094, "error": "...", ...}`。
    兼容 `error` 字符串形态（RFC 8628 标准字段）。
    """
    raw = data.get("code")
    if isinstance(raw, int) and raw != 0:
        return raw
    if isinstance(raw, str) and raw.isdigit() and int(raw) != 0:
        return int(raw)
    if isinstance(data.get("error"), str) and data["error"]:
        # 标准形态 `{"error":"authorization_pending"}` → 映射到实测数字码口径。
        mapping = {
            "authorization_pending": ERR_AUTHORIZATION_PENDING,
            "slow_down": ERR_SLOW_DOWN,
        }
        return mapping.get(str(data["error"]))
    return None


def _token_set(data: dict[str, Any], *, require_refresh: bool = False) -> LarkTokenSet:
    """从响应体提取令牌集合；缺 access_token 视为协议错误。

    `require_refresh`：刷新路径**必须**拿到新 refresh_token（轮换语义下没有它
    就等于下次刷不了），缺失即报错而非静默沿用旧值。
    """
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise LarkCliError("飞书令牌响应缺少 access_token")
    refresh_token = str(payload.get("refresh_token") or "")
    if require_refresh and not refresh_token:
        raise LarkCliError("飞书刷新响应缺少 refresh_token（无法续期，请重新绑定）")
    expires_in = int(payload.get("expires_in") or 7200)
    raw_refresh_expires = payload.get("refresh_token_expires_in")
    refresh_expires_in = int(raw_refresh_expires) if raw_refresh_expires else None
    return LarkTokenSet(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        refresh_expires_in=refresh_expires_in,
    )


def _optional_str(value: Any) -> str | None:
    """可选字符串规整（空串视同缺省，避免把 "" 当姓名展示）。"""
    text = str(value).strip() if value is not None else ""
    return text or None


def _safe_path(url: str) -> str:
    """只保留 path 供日志/异常使用（不带查询串，避免泄露参数）。"""
    return url.split("?", 1)[0].split("://", 1)[-1]


__all__ = [
    "ERR_AUTHORIZATION_PENDING",
    "ERR_SLOW_DOWN",
    "LARK_MINIMAL_SCOPES",
    "PATH_DEVICE_AUTHORIZATION",
    "PATH_DEVICE_VERIFY",
    "PATH_TOKEN_ENDPOINT",
    "PATH_USER_INFO",
    "LarkOAuthClient",
]
