"""飞书账号绑定工具（对话内设备码授权，v5.1 §7）。

四个工具（均带 `_lark_scope` 作用域形参，由 agent 侧拦截器注入调用方身份）：

| 工具 | 语义 |
|---|---|
| `lark_bind_start` | 发起设备码授权 → 返回可打开的验证链接（10 分钟内有效） |
| `lark_bind_complete` | **轮询一次**换码 → 拿到 UAT + refresh_token → 加密落库 |
| `lark_bind_status` | 已绑定？绑的谁？token 是否临近过期？ |
| `lark_unbind` | 撤销（尽力而为）+ 清除本地绑定 |

WHY 把 `lark_bind_complete` 设计成「单次轮询」而非「阻塞等到用户授权为止」：
MCP 工具调用有超时（`mcp_timeout_s`，默认 10s）而用户扫码通常要几十秒到几分钟，
阻塞式等待必然超时失败。故设计为**可重复调用的状态推进**：未授权返回「待授权」
提示（Agent 引导用户完成后再试一次），已授权则一步完成绑定落库。

**身份语义**：绑定操作的作用域同样来自 `_lark_scope`（注入，LLM 不可见）——
「谁在绑定」不由模型决定，A 用户无法代 B 用户完成绑定。

**安全**：token 全程只在内存短暂存在（`LarkTokenSet`），落库前即加密；
工具输出**绝不含**任何 token（只含 open_id / 姓名 / 有效期）。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from shared.lark.errors import (
    LarkBoundError,
    LarkCliError,
)
from shared.lark.models import (
    DEVICE_FLOW_DENIED,
    DEVICE_FLOW_EXPIRED,
    DEVICE_FLOW_PENDING,
    DEVICE_FLOW_SLOW_DOWN,
    LarkDeviceFlow,
)
from shared.lark.oauth import LarkOAuthClient
from shared.lark.repository import LarkBindingRepository

logger = logging.getLogger(__name__)

# 未授权时的引导话术（Agent 据此让用户先完成授权再重试；非错误、属正常控制流）。
_PENDING_HINT = (
    "还没有检测到授权。请在浏览器打开上面的链接完成授权（用飞书扫码或登录确认），"
    "然后告诉我「我已完成授权」，我会继续完成绑定。"
)
_EXPIRED_HINT = "授权链接已失效（有效期为 10 分钟），请重新发起绑定：对我说「帮我绑定飞书」。"
# 用户主动拒绝（access_denied）：不是链接失效，话术必须区分（否则等于把用户的动作说成系统故障）。
_DENIED_HINT = (
    "你在授权页取消了授权（或未确认），本次绑定未完成。"
    "需要时对我说「帮我绑定飞书」，我会重新生成授权链接。"
)


class LarkBindingService:
    """绑定链路编排（仓库 + OAuth 客户端 + 应用凭据）。

    WHY 独立于工具函数：工具层只做参数校验与结果序列化，编排逻辑（发起/推进/查询/
    解绑）集中在此，便于离线测试直接构造（注入 fake 仓库与打桩 HTTP 客户端）。
    """

    def __init__(
        self,
        *,
        repository: LarkBindingRepository,
        oauth: LarkOAuthClient,
        app_id: str,
    ) -> None:
        self._repository = repository
        self._oauth = oauth
        self._app_id = app_id

    async def start(self, user_id: str) -> str:
        """发起设备码授权，返回人类可读的验证链接说明。"""
        if not self._app_id:
            raise LarkBoundError("未配置飞书应用 app_id，无法发起绑定")
        flow = await self._oauth.start_device_flow(user_id=user_id)
        await self._repository.save_device_flow(flow)
        logger.info("已发起飞书绑定设备码流程 user=%s（10 分钟内有效）", user_id)
        return (
            f"已生成飞书授权链接（有效期 10 分钟，用户码 {flow.user_code}）：\n"
            f"{flow.verification_uri_complete}\n"
            "请在浏览器中打开上面的链接，用飞书扫码或登录确认授权。"
            "完成授权后告诉我「我已完成授权」，我会继续完成绑定。"
        )

    async def complete(self, user_id: str) -> str:
        """推进绑定：轮询一次换码；成功则加密落库并回报绑定的账号。"""
        # 先取待定态：需用户身份 + 设备码同时匹配（防用他人/过期码换绑）。
        pending = await self._repository.peek_device_flow(user_id)
        if pending is None:
            return (
                "没有进行中的授权流程（可能已过期或未发起）。"
                "请先对我说「帮我绑定飞书」，重新获取授权链接。"
            )
        state, tokens = await self._oauth.exchange_device_code(pending)
        if state in (DEVICE_FLOW_PENDING, DEVICE_FLOW_SLOW_DOWN):
            if state == DEVICE_FLOW_SLOW_DOWN:
                # 降频：实测按 client 维度限频，故每用户独立退避；对用户仍表现为「待授权」。
                logger.info("飞书设备码轮询触发 slow_down user=%s，本次间隔上调", user_id)
            return _PENDING_HINT
        if state == DEVICE_FLOW_DENIED:
            # 用户拒绝 ≠ 过期：待定态同样作废（同一设备码不能再换码），但话术不同。
            await self._repository.drop_device_flow(user_id=user_id)
            return _DENIED_HINT
        if state == DEVICE_FLOW_EXPIRED or tokens is None:
            # 仅在**确认过期**时丢弃待定态；未识别错误由 exchange 抛 LarkCliError 上抛
            # （不在此处吞成「过期」），保证用户仍可用原链接重试（见 oauth.exchange_device_code）。
            await self._repository.drop_device_flow(user_id=user_id)
            return _EXPIRED_HINT

        # 成功：尽力取用户信息（确认「绑的是谁」），失败不阻断绑定本身。
        open_id: str | None = None
        user_name: str | None = None
        try:
            info = await self._oauth.fetch_user_info(tokens.access_token)
            open_id, user_name = info.open_id, info.name
        except LarkCliError as exc:
            logger.warning("绑定成功但取用户信息失败（不影响绑定）：%s", exc)

        await self._repository.save_binding(
            user_id=user_id,
            tokens=tokens,
            open_id=open_id,
            user_name=user_name,
        )
        await self._repository.drop_device_flow(user_id=user_id)
        who = user_name or open_id or "你的飞书账号"
        logger.info("飞书绑定完成 user=%s open_id=%s", user_id, open_id)
        return (
            f"绑定成功：已关联飞书账号「{who}」。"
            "此后你的飞书日程、任务、消息等操作都只作用于这个账号，其他人看不到。"
        )

    async def status(self, user_id: str) -> str:
        """查询绑定状态（已绑定？绑的谁？token 是否临近过期？）。"""
        binding = await self._repository.get_binding(user_id)
        if binding is None:
            return _unbound_status_message(await self._repository.peek_device_flow(user_id))
        if not binding.is_active():
            return (
                "你的飞书绑定已失效（授权被撤销或长期未使用）。请重新绑定：对我说「帮我绑定飞书」。"
            )
        now = datetime.now(UTC)
        who = binding.user_name or binding.open_id or "未知账号"
        remaining = (binding.expires_at - now).total_seconds()
        refresh_left = (
            (binding.refresh_expires_at - now).total_seconds()
            if binding.refresh_expires_at is not None
            else None
        )
        parts = [f"已绑定飞书账号「{who}」。"]
        # WHY 只报剩余时间不报 token：token 属敏感信息，绝不进工具输出/前端。
        parts.append(
            f"访问令牌剩余约 {max(int(remaining), 0) // 60} 分钟（临近过期时会自动续期）。"
        )
        if refresh_left is not None and refresh_left <= 0:
            parts.append("长期未使用已超过 7 天，授权即将失效，建议重新绑定。")
        elif refresh_left is not None and refresh_left < 86400:
            parts.append("距离需要重新授权已不足 1 天（长期未使用会掉绑），建议尽快使用一次。")
        return " ".join(parts)

    async def unbind(self, user_id: str) -> str:
        """解绑：尽力撤销远端授权 + 清除本地记录（本地清除是主语义）。"""
        binding = await self._repository.get_binding(user_id)
        if binding is None:
            return "你当前没有绑定飞书账号，无需解绑。"
        # WHY 单独 try：解密失败（密钥轮换）不应让用户「解不掉绑」。
        try:
            token = await self._repository.decrypt_access_token(binding)
            await self._oauth.revoke(token)
        except (LarkCliError, LarkBoundError) as exc:
            logger.warning("解绑时远端撤销失败（本地仍清除）：%s", exc)
        removed = await self._repository.delete_binding(user_id)
        await self._repository.drop_device_flow(user_id=user_id)
        logger.info("飞书解绑完成 user=%s（本地记录已清除=%s）", user_id, removed)
        return "已解除飞书绑定，本地保存的授权信息已删除。需要时可以重新绑定。"


def _unbound_status_message(pending: LarkDeviceFlow | None) -> str:
    """未绑定时的状态说明（有待定授权流程则提示先完成授权）。"""
    if pending is not None:
        return (
            "你还没有完成飞书绑定，且有一个进行中的授权流程。请完成授权后告诉我「我已完成授权」。"
        )
    return (
        "你还没有绑定飞书账号。对助理说「帮我绑定飞书」即可获取授权链接，"
        "在浏览器打开并扫码确认即可完成绑定。"
    )


__all__ = ["LarkBindingService"]
