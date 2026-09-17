"""作用域 → 飞书凭据的解析层（v5.1，dev-version5.1.md §6.2/§6.3）。

WHY 独立模块：工具函数只关心「拿到一份该用户的凭据」，解析策略（数据库绑定 /
刷新 / 未绑定判定 / 配置兜底）集中在此，便于装配替换与离线测试注入 fake。

两条实现（装配层按「绑定功能是否可用」二选一）：
- `ConfigCredentialProvider`：配置态单用户凭据（无库 / 未启用绑定的本地形态，零回归）；
- `BindingCredentialProvider`：按 `_lark_scope`（= 调用方 `user_id`）查各自的加密
  绑定记录并解析可用 UAT（必要时刷新）—— **多用户互不可见**的生产形态。

本模块**只依赖 services 侧与 shared 侧**，不 import agent（分层约束）。
"""

from __future__ import annotations

import logging
from typing import Protocol

from services.lark_mcp.config import LarkMCPRuntimeSettings
from services.lark_mcp.models import LarkCliCredentials
from shared.lark.errors import LarkUnboundError
from shared.lark.oauth import LarkOAuthClient
from shared.lark.repository import LarkBindingRepository

logger = logging.getLogger(__name__)


class LarkCredentialProvider(Protocol):
    """作用域 → 凭据契约（服务侧默认实现见 `build_credential_provider`）。"""

    async def resolve(self, scope: str) -> LarkCliCredentials:
        """解析该作用域的可用凭据；未绑定 / 刷新失败 → `LarkUnboundError`。"""
        ...


class ConfigCredentialProvider:
    """配置态凭据提供者：从 `LARKSUITE_CLI_*` 环境（或 .env）取「单用户」凭据。

    WHY 保留这条路径（v5.1 §11 零回归）：无数据库 / 未启用绑定时（本地 demo、
    单机验证、既有 v5.0 用法），Agent 仍可在 .env 里配一份 UAT 直接跑通工具面；
    此时作用域**不影响**凭据（配置即身份），但多用户隔离能力明确不启用。

    多用户生产形态下本类不应被装配 —— 装配层在「绑定功能可用」时改用
    `BindingCredentialProvider`（按 `_lark_scope` 查各自的加密绑定记录）。
    """

    def __init__(self, settings: LarkMCPRuntimeSettings) -> None:
        self._settings = settings

    def available(self) -> bool:
        """配置里是否有可用的 app 凭据（决定本提供者能否兜底）。"""
        return bool(
            self._settings.lark_app_id and self._settings.lark_app_secret.get_secret_value()
        )

    async def resolve(self, scope: str) -> LarkCliCredentials:
        """返回配置态凭据；配置缺失 → `LarkUnboundError`（引导绑定）。"""
        uat = self._settings.lark_user_access_token.get_secret_value()
        if not (self.available() and uat):
            raise LarkUnboundError(
                "未启用飞书绑定且未配置 LARK_USER_ACCESS_TOKEN（配置态单用户凭据）",
            )
        return LarkCliCredentials(
            app_id=self._settings.lark_app_id or "",
            app_secret=self._settings.lark_app_secret.get_secret_value(),
            user_access_token=uat,
        )


class BindingCredentialProvider:
    """绑定态凭据提供者：按作用域（`user_id`）解析**该用户自己**的 UAT。

    **身份隔离的核心实现**：作用域是唯一身份来源，凭据只从该用户的绑定记录取；
    查不到绑定 / 绑定失效 / 刷新失败 → `LarkUnboundError`（图侧走绑定引导），
    **绝不**回落到「别的用户的 UAT」或配置态共享凭据 —— 任何形式的凭据共享都越界
    （CLAUDE.md「飞书身份与作用域」硬约束）。

    刷新经 `LarkOAuthClient.refresh` 注入给仓库（本身不做网络 IO，六步并发语义
    在 `shared.lark.repository` 内实现）。
    """

    def __init__(
        self,
        *,
        repository: LarkBindingRepository,
        oauth: LarkOAuthClient,
        app_id: str,
        app_secret: str,
    ) -> None:
        self._repository = repository
        self._oauth = oauth
        self._app_id = app_id
        self._app_secret = app_secret

    async def resolve(self, scope: str) -> LarkCliCredentials:
        """取该作用域的可用 UAT；未绑定 → `LarkUnboundError`（带绑定引导话术）。"""
        token = await self._repository.resolve_access_token(  # type: ignore[attr-defined]
            scope, refresh=self._oauth.refresh
        )
        return LarkCliCredentials(
            app_id=self._app_id,
            app_secret=self._app_secret,
            user_access_token=token,
        )


def build_credential_provider(
    settings: LarkMCPRuntimeSettings,
    *,
    repository: LarkBindingRepository | None,
    oauth: LarkOAuthClient | None,
) -> LarkCredentialProvider:
    """按装配条件选择凭据提供者（绑定优先，配置态兜底）。

    WHY 绑定优先：只要绑定功能可用（有库 + 有密钥 + 有应用凭据），就必须按用户隔离；
    仅当绑定不可用时才退回配置态单用户（本地 demo / 未配置密钥的零回归路径）。
    """
    binding_ready = (
        settings.lark_binding_enabled
        and repository is not None
        and oauth is not None
        and bool(settings.lark_app_id)
        and bool(settings.lark_app_secret.get_secret_value())
        and getattr(repository, "enabled", True)
    )
    if binding_ready:
        logger.info("飞书凭据按用户绑定解析（多用户隔离已启用）")
        return BindingCredentialProvider(
            repository=repository,  # type: ignore[arg-type]
            oauth=oauth,  # type: ignore[arg-type]
            app_id=settings.lark_app_id or "",
            app_secret=settings.lark_app_secret.get_secret_value(),
        )
    logger.warning(
        "飞书绑定功能未启用（%s），退回配置态单用户凭据（多用户隔离不可用）",
        "缺少数据库或加密密钥或应用凭据"
        if not settings.lark_binding_enabled or repository is None or oauth is None
        else "lark_token_key 未配置",
    )
    return ConfigCredentialProvider(settings)


__all__ = [
    "BindingCredentialProvider",
    "ConfigCredentialProvider",
    "LarkCredentialProvider",
    "build_credential_provider",
]
