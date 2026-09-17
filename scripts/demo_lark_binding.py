"""v5.1 飞书绑定验收脚本：多用户「各自绑定、互不可见」全链路演示（默认离线可跑）。

演示内容（对应 dev-version5.1.md 的 V51-M1 / M2 / M3）：
  1. **M3 设备码闭环**：发起 → 拿到验证链接 → 完成授权 → 令牌**加密落库**；
  2. **M1 身份隔离**：A / B 两人各自绑定，工具调用注入**各自的** UAT（互不串号）；
  3. **互不可见反例**：只绑定 A 时 B 调工具 → `tool_error.lark_unbound`（**绝不借用** A 的令牌）；
  4. **解密可回**：库内为密文（打印密文前缀），可解回原值证明是加密而非丢弃；
  5. **解绑隔离**：解绑 A 不影响 B；
  6. **确定性引导**：未绑定的工具消息前缀即 `tool_error.lark_unbound:`（图侧走绑定引导，
     而非 fallback 降级）—— 直接复用图侧同一判定函数验证。

**默认离线模式**（零网络、零真实飞书应用、零 lark-cli）：OAuth 用 `httpx.MockTransport`
打桩、CLI 子进程用替身捕获。这样任何人都能复跑并看到完整结论。

真实模式（`--live`）：把打桩层换成真实飞书应用 —— 会打印真实验证链接，
需要你用飞书扫码授权（**不自动化**，因为授权必须由人完成）：

    LARK_APP_ID=cli_xxx LARK_APP_SECRET=xxx LARK_TOKEN_KEY=$(openssl rand -hex 32) \
        uv run python -m scripts.demo_lark_binding --live

生产落库（Postgres）：加 `DATABASE_URL=postgresql://user:pass@host:5432/db`
—— 表 `lark_bindings`（密文令牌 + 乐观锁 version）与 `lark_device_flows`（设备码待定态）。

用法（仓库根目录）：
    uv run python -m scripts.demo_lark_binding            # 离线全链路（默认）
    uv run python -m scripts.demo_lark_binding --live     # 真实飞书（需人工扫码）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

import httpx

from agent.share.eventloop import ensure_selector_event_loop
from agent.tools.lark_scope import is_lark_unbound_message
from services.lark_mcp.binding import LarkBindingService
from services.lark_mcp.config import LarkMCPRuntimeSettings
from services.lark_mcp.credentials import BindingCredentialProvider
from shared.lark import (
    LarkOAuthClient,
    build_lark_binding_repository,
    build_token_cipher,
)
from shared.lark.errors import LarkUnboundError
from shared.lark.oauth import PATH_DEVICE_AUTHORIZATION, PATH_TOKEN_ENDPOINT

logger = logging.getLogger(__name__)

USER_A = "demo-user-a"
USER_B = "demo-user-b"

# 打桩用的假令牌（离线模式；真实模式由飞书返回）。
# S105/S106 不适用：下列是**演示用占位值**（离线打桩的假凭据，非真实密钥），
# 文件末尾已为该演示脚本声明 per-file-ignore（见 pyproject.toml）。
FAKE_UAT = {USER_A: "uat-demo-user-A", USER_B: "uat-demo-user-B"}
FAKE_RT = {USER_A: "rt-demo-user-A", USER_B: "rt-demo-user-B"}
FAKE_NAME = {USER_A: "张三", USER_B: "李四"}
FAKE_OPEN_ID = {USER_A: "ou_demo_a", USER_B: "ou_demo_b"}

# 演示用 Fernet 派生源（离线模式专用；真实模式必须用 LARK_TOKEN_KEY）。
DEMO_TOKEN_KEY_SOURCE = "demo-token-key-32bytes-minimum!!"
# 演示用应用凭据（离线打桩；真实模式从环境读取 LARK_APP_ID / LARK_APP_SECRET）。
DEMO_APP_ID = "cli_demo"
DEMO_APP_SECRET_SOURCE = "demo-app-secret"


class OfflineCliStub:
    """离线模式下的 lark-cli 替身：记录每次调用注入的 UAT（身份证据），不发子进程。"""

    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        self.calls = []

    async def run(self, args: list[str], *, credentials: Any) -> dict[str, Any]:
        self.calls.append(
            {"args": list(args), "app_id": credentials.app_id, "uat": credentials.user_access_token}
        )
        return {"ok": True, "command": " ".join(args[:2]), "identity": "user"}


def _offline_oauth() -> LarkOAuthClient:
    """打桩飞书认证族端点（零网络）：设备码 → 换码 → 用户信息，按 grant_type 分流。"""
    exchange_count = {"n": 0}
    order = [USER_A, USER_B]

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == PATH_DEVICE_AUTHORIZATION:
            return httpx.Response(
                200,
                json={
                    "device_code": "d" * 64,
                    "user_code": "KR2E-FZQP",
                    "verification_uri": "https://accounts.feishu.cn/oauth/v1/device/verify",
                    "flow_id": "demo-flow",
                    "expires_in": 600,
                    "interval": 5,
                },
            )
        if path == PATH_TOKEN_ENDPOINT:
            body = json.loads(request.content)
            # 必须按 grant_type 分流：换码请求体里设备码的字段名是 `code`，
            # 按 device_code 判定会静默落到刷新分支（真实踩过的坑）。
            if body.get("grant_type") == "authorization_code":
                user = order[min(exchange_count["n"], len(order) - 1)]
                exchange_count["n"] += 1
                return httpx.Response(
                    200,
                    json={
                        "access_token": FAKE_UAT[user],
                        "refresh_token": FAKE_RT[user],
                        "expires_in": 7200,
                        "refresh_token_expires_in": 604800,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "access_token": "uat-after-refresh",
                    "refresh_token": "rt-after-refresh",
                    "expires_in": 7200,
                    "refresh_token_expires_in": 604800,
                },
            )
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        user = next((u for u, t in FAKE_UAT.items() if t == token), USER_A)
        return httpx.Response(
            200,
            json={"code": 0, "data": {"open_id": FAKE_OPEN_ID[user], "name": FAKE_NAME[user]}},
        )

    return LarkOAuthClient(
        app_id=DEMO_APP_ID,
        app_secret=DEMO_APP_SECRET_SOURCE,
        accounts_base_url="https://accounts.feishu.cn",
        open_base_url="https://open.feishu.cn",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _demo_binding(
    *,
    repository: Any,
    service: LarkBindingService,
    provider: BindingCredentialProvider,
    cli: Any,
) -> None:
    """离线全链路演示主体（真实模式复用同一段逻辑，只换掉打桩层）。"""
    # —— 1. M3：A 的设备码闭环 ——
    logger.info("== 1. M3 设备码闭环：user=%s 发起绑定 ==", USER_A)
    start_text = await service.start(USER_A)
    for line in start_text.splitlines():
        logger.info("  %s", line)

    logger.info("  用户完成授权后推进绑定（可重复调用：未授权则返回「待授权」）…")
    done_text = await service.complete(USER_A)
    logger.info("  %s", done_text)
    assert "绑定成功" in done_text, "绑定未成功，后续演示无意义"

    binding = await repository.get_binding(USER_A)
    assert binding is not None
    logger.info(
        "  落库检查：open_id=%s 姓名=%s status=%s version=%s",
        binding.open_id,
        binding.user_name,
        binding.status,
        binding.version,
    )
    # —— 2. 加密落库（密文可见、明文不可见）——
    cipher_prefix = binding.access_token_ciphertext[:24]
    logger.info("== 2. 令牌加密落库 ==")
    logger.info(
        "  access_token 密文前缀: %s…（长度 %d）",
        cipher_prefix,
        len(binding.access_token_ciphertext),
    )
    assert FAKE_UAT[USER_A] not in binding.access_token_ciphertext
    assert FAKE_RT[USER_A] not in binding.refresh_token_ciphertext
    decrypted = await repository.decrypt_access_token(binding)
    assert decrypted == FAKE_UAT[USER_A]
    logger.info("  ✓ 明文零出现，且可解回原值（证明确为加密而非丢弃）")

    # —— 3. 工具调用：A 的 UAT 注入子进程 env ——
    logger.info("== 3. M1 身份注入：A 调用工具 ==")
    creds_a = await provider.resolve(USER_A)
    await cli.run(["calendar", "+agenda"], credentials=creds_a)
    logger.info("  ✓ 注入 UAT=%s（app_id=%s）", creds_a.user_access_token, creds_a.app_id)
    assert creds_a.user_access_token == FAKE_UAT[USER_A]

    # —— 4. 互不可见反例：B 未绑定 → 确定性拒绝，不借用 A ——
    logger.info("== 4. M1 互不可见：user=%s 未绑定就调用工具 ==", USER_B)
    try:
        await provider.resolve(USER_B)
        raise AssertionError("B 未绑定不应解析成功")
    except LarkUnboundError as exc:
        message = str(exc)
        logger.info("  ✓ 确定性拒绝：%s", message.splitlines()[0][:100])
        assert is_lark_unbound_message(message), "未绑定消息必须带 tool_error.lark_unbound 前缀"
        logger.info(
            "  ✓ 图侧判定函数识别为未绑定（%s）→ 路由到绑定引导而非 fallback 降级",
            "is_lark_unbound_message=True",
        )
    # B 解析失败**不得产生任何子进程调用**；若产生了，说明凭据解析回落到别人的令牌。
    assert [c["uat"] for c in cli.calls] == [FAKE_UAT[USER_A]]
    logger.info("  ✓ B 的失败未产生额外调用，仅 A 的 1 次调用（零借用）")

    # —— 5. B 绑定后：各用各的 ——
    logger.info("== 5. B 也绑定 → A/B 各用各的令牌 ==")
    await service.start(USER_B)
    done_b = await service.complete(USER_B)
    assert "绑定成功" in done_b
    creds_b = await provider.resolve(USER_B)
    await cli.run(["calendar", "+agenda"], credentials=creds_b)
    tokens = [c["uat"] for c in cli.calls]
    logger.info("  子进程注入的 UAT 序列：%s", tokens)
    assert tokens == [FAKE_UAT[USER_A], FAKE_UAT[USER_B]], "A/B 令牌串号！"
    logger.info("  ✓ A 用 %s、B 用 %s —— 互不串号", FAKE_UAT[USER_A], FAKE_UAT[USER_B])

    # —— 6. 解绑隔离 ——
    logger.info("== 6. 解绑隔离：解绑 A 不影响 B ==")
    logger.info("  %s", await service.unbind(USER_A))
    assert await repository.get_binding(USER_A) is None
    assert await repository.get_binding(USER_B) is not None
    logger.info("  ✓ A 已解绑（记录清除），B 仍绑定（记录保留）")
    logger.info("  %s", await service.status(USER_A))
    logger.info("  %s", await service.status(USER_B))


async def _main_offline(settings: LarkMCPRuntimeSettings) -> None:
    """离线全链路（默认）：打桩 OAuth + 替身 CLI，零网络零真实应用。"""
    cipher = build_token_cipher(DEMO_TOKEN_KEY_SOURCE)
    assert cipher is not None
    repository = build_lark_binding_repository(cipher)
    await repository.setup()
    oauth = _offline_oauth()
    service = LarkBindingService(repository=repository, oauth=oauth, app_id=DEMO_APP_ID)
    provider = BindingCredentialProvider(
        repository=repository,
        oauth=oauth,
        app_id=DEMO_APP_ID,
        app_secret=DEMO_APP_SECRET_SOURCE,
    )
    try:
        await _demo_binding(
            repository=repository, service=service, provider=provider, cli=OfflineCliStub()
        )
        logger.info("== 结论 ==")
        logger.info(
            "离线演示完成：设备码闭环 / 加密落库 / 按用户注入 / 互不可见 / 解绑隔离 全部通过。"
        )
        logger.info("真实飞书验证：uv run python -m scripts.demo_lark_binding --live")
    finally:
        await repository.aclose()


async def _main_live(settings: LarkMCPRuntimeSettings) -> None:
    """真实飞书模式：用真实应用发起设备码授权，打印链接等待人工扫码。"""
    app_id = settings.lark_app_id or ""
    app_secret = settings.lark_app_secret.get_secret_value()
    token_key = settings.lark_token_key.get_secret_value()
    if not (app_id and app_secret and token_key):
        raise SystemExit(
            "真实模式需要 LARK_APP_ID / LARK_APP_SECRET / LARK_TOKEN_KEY（令牌加密密钥）。\n"
            '可先生成密钥：python -c "import secrets;print(secrets.token_hex(32))"'
        )
    cipher = build_token_cipher(token_key)
    assert cipher is not None
    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    repository = build_lark_binding_repository(cipher, database_url)
    await repository.setup()
    oauth = LarkOAuthClient(
        app_id=app_id,
        app_secret=app_secret,
        accounts_base_url=settings.lark_accounts_base_url,
        open_base_url=settings.lark_open_base_url,
        timeout_s=settings.lark_oauth_timeout_s,
    )
    service = LarkBindingService(repository=repository, oauth=oauth, app_id=app_id)
    try:
        logger.info("== 真实模式：为 %s 发起设备码授权 ==", USER_A)
        logger.info("%s", await service.start(USER_A))
        logger.info("请在浏览器打开上面的链接用飞书扫码授权；授权完成后回到本终端按回车继续。")
        await asyncio.to_thread(input, "授权完成后按回车…")
        logger.info("%s", await service.complete(USER_A))
        logger.info("== 绑定状态 ==")
        logger.info("%s", await service.status(USER_A))
        logger.info("== 真实环境验活：用该用户身份跑一次 calendar +agenda ==")
        from services.lark_mcp.cli import LarkCliRunner

        runner = LarkCliRunner(
            command=settings.lark_cli_command, timeout_s=settings.lark_cli_timeout_s
        )
        if runner.command is None:
            logger.warning("未探测到 lark-cli，跳过真实调用验活")
            return
        provider = BindingCredentialProvider(
            repository=repository, oauth=oauth, app_id=app_id, app_secret=app_secret
        )
        credentials = await provider.resolve(USER_A)
        payload = await runner.run(["calendar", "+agenda"], credentials=credentials)
        logger.info("✓ lark-cli 以绑定用户身份执行成功：%s", str(payload)[:200])
    finally:
        await repository.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="飞书多用户绑定验收演示（v5.1）")
    parser.add_argument("--live", action="store_true", help="使用真实飞书应用（需人工扫码授权）")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True
    )
    settings = LarkMCPRuntimeSettings()
    if args.live:
        await _main_live(settings)
    else:
        await _main_offline(settings)


if __name__ == "__main__":
    # Windows 下 psycopg 异步需 SelectorEventLoop，须在 asyncio.run 之前。
    ensure_selector_event_loop()
    asyncio.run(main())
