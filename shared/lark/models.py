"""飞书绑定域模型（跨边界一律 Pydantic，v5.1 §7 / §10）。

WHY 独立于 `services/lark_mcp/models.py`：绑定记录是**数据**（存储 + 生命周期），
`LarkCliCredentials` 是**子进程凭证载体**（一次性注入 env）—— 前者按 `user_id`
长期落库、后者用完即弃，混在一处会让「什么能落库」的边界模糊。

WHY 放根级 `shared/lark/` 而非设计稿 §10 写的 `agent/lark/`（**记录在案的偏离**）：
消费方是 `services/lark_mcp`（绑定工具 + 凭据解析）与 `agent` 两侧。按 CLAUDE.md
「上层依赖下层接口」，`services` **不得** import `agent`；若放 `agent/lark/`，
服务侧要么违反分层、要么被迫重复实现。`shared/` 正是「≥2 消费方」的宿主
（与 `shared/embeddings.py` 同理）。

**安全不变量**：本模块**不含任何明文 token 字段** —— 落库的 token 只以密文形态
出现在 `LarkBinding` 的 `*_ciphertext` 字段中，明文仅在内存里短暂存在
（见 `token_cipher.py` 与仓库的加解密路径）。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# —— 绑定状态（status 列取值，集中声明禁散落字面量）——
BINDING_STATUS_ACTIVE = "active"  # 绑定有效
# 刷新失败且确认非并发竞争 → 需用户重新扫码（§7 第 ⑥ 步）
BINDING_STATUS_INVALID = "invalid"

# —— 设备码授权轮询的处置（RFC 8628 实测语义，§7 表格）——
DEVICE_FLOW_PENDING = "pending"  # authorization_pending(20094)：继续轮询
DEVICE_FLOW_SLOW_DOWN = "slow_down"  # slow_down(20095)：降频后继续
DEVICE_FLOW_EXPIRED = "expired"  # expired_token/invalid_grant：需重新发起
# 用户在授权页主动拒绝（access_denied）：与「过期」区分 —— 话术不同（不是链接失效，
# 是用户没同意），但处置一致（都需重新发起）。
DEVICE_FLOW_DENIED = "denied"
DEVICE_FLOW_DONE = "done"  # 成功换码

# 实测设备码寿命 600s（§7：`expires_in` 实测 600s = 10 分钟）。
DEVICE_FLOW_TTL_S = 600


class LarkBinding(BaseModel):
    """一条飞书账号绑定记录（`lark_bindings` 表的内存投影）。

    `user_id` 是绑定归属（与记忆/会话同一口径，天然按用户隔离、互不可见）；
    `open_id` / `user_name` 仅用于向用户确认「绑的是谁」，不参与鉴权判定。

    `version` 是**乐观锁**版本号：刷新并发时以「条件更新 + 影响行数」判定胜负
    （§7 强制要求），旧值立即失效的 `refresh_token` 轮换语义靠它避免互相作废。
    """

    user_id: str
    open_id: str | None = None
    user_name: str | None = None
    # 密文（Fernet）。绝不为明文；列类型 Text（UAT/refresh 实测 8000+ 字符）。
    access_token_ciphertext: str = ""
    refresh_token_ciphertext: str = ""
    # access_token 过期时刻（无需提前量；提前量在读取时以 skew 计算）。
    expires_at: datetime
    # refresh_token 过期时刻（实测 7 天空闲上限；每次刷新重置）。
    refresh_expires_at: datetime | None = None
    status: str = BINDING_STATUS_ACTIVE
    version: int = 0
    updated_at: datetime

    def is_active(self) -> bool:
        """绑定是否可用于取用（失效状态一律走「请重新绑定」）。"""
        return self.status == BINDING_STATUS_ACTIVE


class LarkDeviceFlow(BaseModel):
    """设备码授权的待定态（绑定发起后、用户完成授权前）。

    `device_code` 实测 100 字符、`user_code` 形如 `KR2E-FZQP`（9 字符）；
    `flow_id` 是**验证页必需参数**（实测旧形态 `/page/cli?user_code=` 已作废）。
    带 `expires_at`（实测 600s）：过期后必须重新发起，不可无限轮询。
    """

    user_id: str
    device_code: str
    user_code: str
    verification_uri: str
    # 含 flow_id + user_code 的完整验证链接（可直接转二维码 / 点击跳转）。
    verification_uri_complete: str
    flow_id: str
    expires_at: datetime
    # 轮询间隔（秒）：由响应 `interval` 给出（实测默认 5s），`slow_down` 时上调。
    interval_s: float = Field(default=5.0, gt=0)
    created_at: datetime


class LarkTokenSet(BaseModel):
    """一次 OAuth 换码/刷新得到的令牌集合（明文，**仅内存短暂存在**）。

    WHY 用 `repr=False`（Pydantic `Field(repr=False)`）：该对象可能进入异常回显 /
    调试日志，默认 repr 会把 token 打进日志 —— 与「token 不写日志」硬约束冲突。
    """

    access_token: str = Field(repr=False)
    refresh_token: str = Field(default="", repr=False)
    expires_in: int  # access_token 寿命（实测 7200s）
    refresh_expires_in: int | None = None  # 仅带 offline_access 时返回（实测 604800s）


class LarkUserInfo(BaseModel):
    """飞书用户基本信息（绑定确认用；不含任何令牌）。"""

    open_id: str
    name: str | None = None


__all__ = [
    "BINDING_STATUS_ACTIVE",
    "BINDING_STATUS_INVALID",
    "DEVICE_FLOW_DENIED",
    "DEVICE_FLOW_DONE",
    "DEVICE_FLOW_EXPIRED",
    "DEVICE_FLOW_PENDING",
    "DEVICE_FLOW_SLOW_DOWN",
    "DEVICE_FLOW_TTL_S",
    "LarkBinding",
    "LarkDeviceFlow",
    "LarkTokenSet",
    "LarkUserInfo",
]
