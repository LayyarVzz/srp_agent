"""lark_mcp 服务侧数据模型（v5.1）。

`LarkCliCredentials` 是**子进程凭证载体**（dev-version5.1.md §10）：
每次工具调用按作用域解析出一份，注入 lark-cli 子进程 env（env 凭证链优先于钥匙串），
**不落盘、不共享、不进日志**。纯 user 模式下 `user_access_token` 是唯一执行身份。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LarkCliCredentials(BaseModel):
    """单次 lark-cli 调用的凭证集合（一名用户的凭据，不可复用给他人）。

    字段即 §4.1 的 env 注入集合：`app_id`/`app_secret` 供 CLI 刷新与标识，
    `user_access_token` 是**唯一执行身份**。`repr=False` 的三项在 pydantic
    输出中隐藏（防日志/异常栈意外泄露；`SecretStr` 会在模型边界暴露
    `get_secret_value`，此处以最小可用形状收敛）。
    """

    app_id: str = Field(description="共享应用 App ID（仅用于授权与刷新）")
    app_secret: str = Field(repr=False, description="共享应用 App Secret（不落日志）")
    user_access_token: str = Field(repr=False, description="该用户的 user_access_token")

    def is_complete(self) -> bool:
        """是否具备执行一次调用的最小充分集（缺 UAT = 未绑定，§6.3 最小阻断判据）。"""
        return bool(self.app_id and self.app_secret and self.user_access_token)


__all__ = ["LarkCliCredentials"]
