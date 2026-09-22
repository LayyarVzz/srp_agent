"""查询理解：检索前改写与多路召回融合（v6.0 T2）。

四类改写（主改写 / 子查询 / 同义 / HyDE 假设文档）由一次结构化 LLM 调用产出，
经确定性预门控（`gate`）决定是否调用；产物供给长期记忆语义召回与 RAG 工具检索。
融合为纯函数（`fusion`），记忆召回与 RAG 补检共用。

WHY 用模块级 `__getattr__` 惰性导出（PEP 562）而非模块顶层 re-export：
`agent.memory.adapter` 需要 `agent.query.fusion`，而在包顶层导入 `gate` / `rewriter`
会连带导入 `agent.core.config`（以及 `agent.core.__init__` → `graph` → `agent.memory`），
形成 `memory → query → core → memory` 的循环导入。惰性导出让「只用到 `fusion`」的
调用方不会付出整包的导入链，同时保留 `from agent.query import QueryRewriter` 的使用手感。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅供类型检查器解析符号，运行时不导入（避免上述循环）。
    # F401：这些名字是**公开 re-export**（经 `__getattr__` 在运行时惰性解析），
    # 不是未使用导入；`__all__` 已声明它们。
    from agent.query.fusion import (  # noqa: F401
        DEFAULT_RRF_K,
        RankedHit,
        dedup_key,
        reciprocal_rank_fusion,
    )
    from agent.query.gate import (  # noqa: F401
        is_trivial_input,
        normalize_greeting,
        should_understand,
    )
    from agent.query.models import (  # noqa: F401
        QueryUnderstanding,
        QueryUnderstandingResult,
        QueryVariant,
        RankedQuery,
    )
    from agent.query.rewriter import (  # noqa: F401
        REWRITE_PROMPT,
        QueryRewriter,
        StructuredInvoker,
    )

# 导出名 → 定义所在子模块（惰性解析，见模块 docstring）。
_LAZY_EXPORTS: dict[str, str] = {
    "DEFAULT_RRF_K": "agent.query.fusion",
    "RankedHit": "agent.query.fusion",
    "dedup_key": "agent.query.fusion",
    "reciprocal_rank_fusion": "agent.query.fusion",
    "is_trivial_input": "agent.query.gate",
    "normalize_greeting": "agent.query.gate",
    "should_understand": "agent.query.gate",
    "QueryUnderstanding": "agent.query.models",
    "QueryUnderstandingResult": "agent.query.models",
    "QueryVariant": "agent.query.models",
    "RankedQuery": "agent.query.models",
    "REWRITE_PROMPT": "agent.query.rewriter",
    "QueryRewriter": "agent.query.rewriter",
    "StructuredInvoker": "agent.query.rewriter",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """按需从子模块解析导出名（首次访问后写入模块命名空间，后续零开销）。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让 `dir(agent.query)` 反映惰性导出（保持可发现性）。"""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
