"""作用域 → 飞书凭据的解析层（v5.1，dev-version5.1.md §6.2/§6.3）。

WHY 独立模块：工具函数只关心「拿到一份该用户的凭据」，解析策略（数据库绑定 /
刷新 / 未绑定判定 / 配置兜底）集中在此，便于装配替换与离线测试注入 fake。

本模块**只依赖 services 侧与 shared 侧**，不 import agent（分层约束）。
"""

from __future__ import annotations

import logging
from typing import Protocol

from services.lark_mcp.config import LarkMCPRuntimeSettings
from services.lark_mcp.models import LarkCliCredentials
from shared.lark.errors import LarkUnboundError

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


__all__ = ["ConfigCredentialProvider", "LarkCredentialProvider"]
