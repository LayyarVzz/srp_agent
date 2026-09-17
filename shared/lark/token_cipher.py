"""飞书令牌加密（Fernet，v5.1 §7「加密」与 CLAUDE.md 安全约束）。

**硬约束**：`user_access_token` / `refresh_token` 必须加密落库，**绝不落明文**；
`lark_token_key` 未配置时绑定功能 **fail-closed 关闭**（拒绝启用并记日志）。

WHY 派生而非直接用配置值当密钥：Fernet 要求 32 字节 urlsafe-base64 密钥，
而运维自然会给「一串口令」。这里以 SHA-256 摘要 + urlsafe-base64 归一化任意长度
口令 → 稳定得到合法密钥（同口令跨进程/跨副本一致，容器与 api 需共享同一
`lark_token_key` 才能互相解密 —— 见 §9 配置变更）。

WHY 不引入 KDF（bcrypt/argon2）：本密钥是**服务端保管的应用密钥**，不是用户口令，
没有「离线爆破低熵口令」的威胁模型；加盐派生反而会让同一口令在不同副本产生
不同密钥而无法互解。故用确定性摘要，并在文档中明确「请使用高熵随机串」。
"""

from __future__ import annotations

import base64
import hashlib
import logging
from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken

from shared.lark.errors import LarkCliError

logger = logging.getLogger(__name__)


class LarkTokenCipher(Protocol):
    """令牌加解密契约（结构型协议；测试可注入确定性的假实现）。"""

    def encrypt(self, plaintext: str) -> str:
        """明文 → 密文（可安全落库/落日志）。"""

    def decrypt(self, ciphertext: str) -> str:
        """密文 → 明文；密文损坏或密钥不符时抛 `LarkCliError`。"""


class FernetTokenCipher:
    """基于 Fernet 的令牌加解密实现（对称认证加密：密文被篡改即解密失败）。"""

    def __init__(self, key: str) -> None:
        """按配置口令构造；或直接传入合法 Fernet 密钥（32 字节 base64）。"""
        self._fernet = Fernet(_derive_fernet_key(key))

    def encrypt(self, plaintext: str) -> str:
        """加密并编码为 ASCII 字符串（Fernet token 本身即 urlsafe base64）。"""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """解密；失败统一抛 `LarkCliError`（不泄露底层异常细节，避免 oracle 风险）。"""
        try:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError) as exc:
            # WHY 不回显密文：密文本身虽不可直接读，但回显会进日志/前端，无收益有风险。
            raise LarkCliError("飞书令牌解密失败（密钥不符或密文损坏），请重新绑定") from exc


def _derive_fernet_key(key: str) -> bytes:
    """把任意长度的配置口令归一化为 Fernet 合法密钥（32 字节 urlsafe-base64）。

    已是合法 Fernet 密钥（44 字符 base64 且解出 32 字节）则原样使用
    —— 允许运维直接注入 `Fernet.generate_key()` 的产物，避免二次摘要导致的
    「以为换密钥实际没换」类误操作。
    """
    raw = key.encode("utf-8")
    if len(raw) == 44:
        try:
            if len(base64.urlsafe_b64decode(raw)) == 32:
                return raw
        except (ValueError, TypeError):
            pass  # 不是合法 base64 → 走摘要派生
    digest = hashlib.sha256(raw).digest()
    return base64.urlsafe_b64encode(digest)


def build_token_cipher(key: str | None) -> FernetTokenCipher | None:
    """按 `lark_token_key` 构造加密器；未配置返回 None（**fail-closed 判据**）。

    WHY 返回 None 而非抛错：装配期据此关闭绑定功能并记日志（§7/§13），
    让「未配置密钥」表现为「绑定不可用、其余功能照常」，而非整个服务起不来。
    """
    if not key or not key.strip():
        logger.warning("未配置 lark_token_key：飞书绑定功能已关闭（fail-closed，绝不落明文 token）")
        return None
    return FernetTokenCipher(key)


__all__ = [
    "FernetTokenCipher",
    "LarkTokenCipher",
    "build_token_cipher",
]
