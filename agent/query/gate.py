"""查询理解的确定性预门控（v6.0 T2，dev-version6.0.md §2.3 D3）。

**门控只做便宜判定**（字符串结构 + 长度 + 词表 + 已知意图），不做语义判断 ——
语义判断是 LLM 的职责，放在这里会变成第二个（且不可解释的）判据。

WHY 必须存在：`understand_query` 在每轮主链路上（记忆召回要用它的产物），
若无门控就是「每轮无条件一次 LLM 调用」= 全量对话变慢。
门控命中 → 直接构造 `retrieval_needed=False` 的常量结果，**一次 LLM 都不调**。

风险取向与 T1 一致（宁漏不误丢，§0.2）：门控只能拦「明显不需要检索」的输入，
判断不了就放行 —— 误拦会让改写不生效，比多花一次调用更糟。
"""

from __future__ import annotations

import re

from agent.core.config import QueryUnderstandingConfig
from agent.intent.models import Intent

# 寒暄/确认词表：**精确命中**（整句归一后相等）才算，避免误伤「你好，我们公司年假怎么算」。
# 只收「明确无信息需求」的短句：问候、道谢、确认、告别。
_SMALL_TALK_PHRASES = frozenset(
    {
        "你好",
        "您好",
        "hi",
        "hello",
        "嗨",
        "在吗",
        "在么",
        "早上好",
        "中午好",
        "晚上好",
        "晚安",
        "谢谢",
        "谢谢你",
        "多谢",
        "感谢",
        "thanks",
        "thank you",
        "好的",
        "好",
        "嗯",
        "嗯嗯",
        "ok",
        "okay",
        "收到",
        "明白了",
        "知道了",
        "了解",
        "再见",
        "拜拜",
        "bye",
    }
)

# 可判别字符：出现任一即视为「可能携带检索意图」（数字/拉丁字母/CJK 表意文字）。
# 纯标点、纯空白、纯 emoji、纯语气词（啊/呀/哦/哈）不含这些 → 无可检索内容。
_MEANINGFUL_CHAR_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


def normalize_greeting(text: str) -> str:
    """归一化用于寒暄词表匹配：去空白、去尾随标点、小写。"""
    stripped = text.strip().lower()
    return stripped.rstrip("。！？!?.,，、~～… 　")


def is_trivial_input(text: str, *, min_query_chars: int) -> bool:
    """结构性平凡输入：长度不足、无可判别字符、或整句命中寒暄词表。

    注意 `min_query_chars` 判定的是**原始长度**（含标点的极短输入如「?」）；
    中文无空格，「那它的上限呢」6 字不会被误判。
    """
    raw = text.strip()
    if not raw:
        return True
    if len(raw) < min_query_chars:
        return True
    if not _MEANINGFUL_CHAR_RE.search(raw):
        return True
    return normalize_greeting(raw) in _SMALL_TALK_PHRASES


def should_understand(
    text: str,
    *,
    settings: QueryUnderstandingConfig,
    intent: Intent | None = None,
) -> bool:
    """是否值得调用 LLM 做查询理解。

    `intent` 是**可选上下文**（由上游 `classify_intent` 提供，此时必然可得）。
    注意它**不用于「短即闲聊」式跳过**：长度无法区分「好的呀」（闲聊）与
    「帮我看看」（请求），按长度猜会把请求误跳过、让改写与检索凭空失效 ——
    不值得为省一次调用去猜。意图当前仅作将来更精确判据的入口保留。
    """
    if not settings.enabled:
        return False
    return not is_trivial_input(text, min_query_chars=settings.min_query_chars)
