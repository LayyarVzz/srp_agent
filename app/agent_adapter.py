from __future__ import annotations

from datetime import UTC, datetime


async def submit_to_agent(
    *,
    text: str,
    session_id: str,
    user_id: str | None,
    source: str,
) -> str:
    """提交标准化输入到 Agent。

    当前接口层先保持可运行的最小适配：如果后续 Agent 暴露统一 run_agent
    入口，可在这里替换为真实调用，不需要改路由和 ASR 逻辑。
    """

    try:
        from agent.core.graph import run_agent  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return _fallback_answer(text=text, session_id=session_id, source=source)

    result = await run_agent(
        {
            "input": text,
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "source": source,
        }
    )
    if isinstance(result, dict):
        return str(result.get("final_answer") or result.get("response") or result)
    return str(result)


def _fallback_answer(*, text: str, session_id: str, source: str) -> str:
    now = datetime.now(UTC).isoformat()
    return (
        "Agent framework is not connected yet. "
        f"Received {source} input for session {session_id} at {now}: {text}"
    )
