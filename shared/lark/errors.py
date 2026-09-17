"""飞书绑定域错误类型与可路由错误码（v5.1，dev-version5.1.md §6.3）。

WHY 独立于 services/cli.py：错误码 `tool_error.lark_unbound` 的**消费方在 agent 侧**
（图需据此路由到绑定引导而非 `fallback_chat`），异常类型的**产出方在 services 侧**
（MCP 工具执行）——两侧共用的常量/类型只能落在根级 shared 层（分层约束）。
"""

from __future__ import annotations

# 可路由错误码：飞书未绑定 / 刷新失败 → 绑定引导（澄清原语），而非 fallback_chat。
TOOL_ERROR_LARK_UNBOUND = "tool_error.lark_unbound"

# 错误消息前缀（机器可读锚点）：跨 MCP 线后异常会退化成文本，
# agent 侧只能靠该常量前缀确定性识别 `lark_unbound`（见 agent/core/graph.py）。
LARK_UNBOUND_PREFIX = f"{TOOL_ERROR_LARK_UNBOUND}: "

# 面向用户的引导话术主体（不含绑定入口链接；链接由图侧从 ToolMessage 中提取补全）。
LARK_UNBOUND_GUIDE = (
    "该操作需要你的飞书账号授权，但你还没有绑定飞书（或授权已失效）。"
    "请先完成绑定：对助理说「帮我绑定飞书」，我会给你一条授权链接，"
    "在浏览器里打开并扫码确认即可；绑定后每个人的飞书操作都只作用于自己的账号。"
)

# fail-closed 关闭原因（未配置 lark_token_key）：绑定功能整体不可用（§7「加密」）。
# WHY 用常量：该文案会同时出现在日志与工具错误里，集中声明保证口径一致、可被测试断言。
BINDING_DISABLED_REASON = "飞书绑定功能未启用（服务端未配置 lark_token_key，已按 fail-closed 关闭）"


class LarkCliError(RuntimeError):
    """lark-cli 调用失败（非零退出 / 输出解析失败 / 超时 / ok=false）。

    由 FastMCP 统一转为工具错误 → Agent 侧 ToolNode(status="error")
    → `tool_error.execution`（可路由降级）；参数缺失语义由工具 schema 层保留。
    """


class LarkBoundError(RuntimeError):
    """飞书绑定域基础异常（OAuth 交互 / 绑定存储 / 令牌解密失败）。

    与 `LarkCliError` 区分：本类不表示「CLI 调用失败」，而是绑定链路自身的失败
    （设备码过期、刷新失败、fail-closed 未配置密钥等）。
    """


class LarkUnboundError(LarkBoundError):
    """用户未绑定飞书（或缺 `user_access_token`）→ 语义 = 引导绑定。

    WHY 独立异常而非复用 `LarkCliError`：裸 `LarkCliError` 会被归一成
    `tool_error.execution` → `fallback_chat`，把「你还没绑定」变成「我答不上来」。
    本异常携带 `tool_error.lark_unbound` 语义，图侧据此走引导路径（§6.3）。

    `verification_uri_complete` 非空时随异常上抛给用户（绑定入口）：由
    服务侧在「未绑定时需要发起绑定」的场景填入（`lark_bind_status` 等）。

    `detail` 可选补充「为什么未绑定」的可读原因（如「绑定已失效」/「刷新凭证已失效」）
    —— 缺省时按 user_id 生成兜底文案；`detail` 只影响人类可读文本，
    **不影响**前缀锚点（agent 侧识别仍只依赖 `LARK_UNBOUND_PREFIX`）。
    """

    def __init__(
        self,
        message: str,
        *,
        detail: str | None = None,
        verification_uri_complete: str | None = None,
    ) -> None:
        text = f"{message}（{detail}）" if detail else message
        super().__init__(unbound_message(text, verification_uri_complete))
        self.detail = text
        self.verification_uri_complete = verification_uri_complete


def unbound_message(detail: str, verification_uri_complete: str | None = None) -> str:
    """构造未绑定错误消息：常量前缀（机器可读）+ 人类可读引导 + 绑定入口。

    WHY 拼接而非结构化承载：ToolMessage 内容跨 MCP 线只能是文本；前缀是
    agent 侧识别 `lark_unbound` 的唯一确定性依据，链接尾巴供引导话术直接复用。
    """
    text = f"{LARK_UNBOUND_PREFIX}{detail} {LARK_UNBOUND_GUIDE}"
    if verification_uri_complete:
        text = f"{text} 绑定入口：{verification_uri_complete}"
    return text


__all__ = [
    "BINDING_DISABLED_REASON",
    "LARK_UNBOUND_GUIDE",
    "LARK_UNBOUND_PREFIX",
    "TOOL_ERROR_LARK_UNBOUND",
    "LarkBoundError",
    "LarkCliError",
    "LarkUnboundError",
    "unbound_message",
]
