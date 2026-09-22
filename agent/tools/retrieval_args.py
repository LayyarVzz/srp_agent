"""RAG 检索实参确定性改写 + 召回不足惰性补检（v6.0 T2，dev-version6.0.md §6）。

**与 `LarkScopeInterceptor` 同一机制**（`langchain-mcp-adapters` 的 `ToolCallInterceptor`）：
在客户端校验之后、直接上 MCP 线之前改写 `request.args`。区别在于本拦截器做两件事：

1. **确定性覆盖检索实参**（§6.2）：把模型填的 `query` 归一为图内查询理解的 `main_query`；
   有假设文档且 RAG 侧声明了入参时补 `use_hypothetical=True`。
   WHY 覆盖而不靠提示词：正确性不该依赖模型是否听话 —— 模型仍可自由调用工具、仍可自己填
   query，只是该参数被归一（`override_model_query=False` 可退回 v5.1 原样透传）。
2. **召回不足时惰性补检 + RRF 融合**（§6.3）：首轮结果为空或最高分过低时，用子查询/同义
   变体再检索并融合，**重组为仍然合法的 `SearchKnowledgeResponse`** ——
   响应适配层、护栏、citations 全链路无需感知。

**RAG 侧零改动**：融合在 agent 侧完成；HyDE 仅依赖一个**可选**入参，按可用性探测传入
（RAG 未声明 `use_hypothetical` 时不带该键，自动降级为不启用）。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from agent.core.config import RetrievalConfig
from agent.query.fusion import RankedHit, dedup_key, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# RAG MCP 服务名（与 `services/rag_mcp/client_config.py::RAG_MCP_SERVER_NAME` 同值）。
# WHY 此处不 import 服务模块：agent 不允许依赖 services 内部实现（上层依赖下层接口）；
# 装配层把连接以该名登记，拦截器按名放行非目标服务。
RAG_SERVER_NAME = "rag"

# 检索查询实参名与 HyDE 开关实参名（RAG 工具契约；禁止散落字符串字面量）。
RAG_QUERY_ARG = "query"
RAG_HYPOTHETICAL_ARG = "use_hypothetical"

# 去重时取内容前缀的字符数（多路返回同一片段时长度可能不同，前缀足够稳定归并）。
_DEDUP_CONTENT_PREFIX = 64


def _runtime_state(request: Any) -> dict[str, Any]:
    """取图状态快照（图内执行才有；图外调用返回空 dict → 拦截器整体放行）。

    `dispatch_tool` 把 `retrieval_query` 等键随输入透传给 `ToolNode`，
    `_extract_state` 对普通 dict 原样返回，故这里即图状态视图。
    """
    runtime = getattr(request, "runtime", None)
    state = getattr(runtime, "state", None)
    return state if isinstance(state, dict) else {}


def _declared_args(request: Any) -> set[str] | None:
    """该工具在**服务端**声明的形参名集合；取不到时返回 None（未知 → 保守不新增参数）。

    WHY 依次探多个来源：`ToolCallInterceptor` 的 request 字段随 adapters 版本演进而
    `tool` 挂 `BaseTool`、也可能直接带 `tool_call_schema` 字典 —— 逐个探测比绑定单一
    形状更耐用；全都取不到时**不新增参数**（宁可 HyDE 不启用，也不发一个 RAG 可能不认的键）。
    """
    candidates: list[Any] = []
    tool = getattr(request, "tool", None)
    if tool is not None:
        candidates.append(getattr(tool, "args_schema", None))
        candidates.append(getattr(tool, "tool_call_schema", None))
    candidates.append(getattr(request, "tool_call_schema", None))
    for schema in candidates:
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(properties, dict):
            return set(properties)
    return None


def _parse_hits(content: Any) -> list[dict[str, Any]]:
    """解析 RAG 返回的 chunks（兼容 ToolMessage 文本与已解析结构；失败返回空列表）。"""
    payload = getattr(content, "content", content)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return []
    if not isinstance(payload, dict):
        return []
    chunks = payload.get("chunks")
    return [c for c in chunks if isinstance(c, dict)] if isinstance(chunks, list) else []


def _max_score(chunks: Sequence[dict[str, Any]]) -> float:
    """本路结果的最高相似度（缺分数按 0 计，防脏数据抛错）。"""
    scores: list[float] = []
    for chunk in chunks:
        value = chunk.get("score")
        if isinstance(value, int | float):
            scores.append(float(value))
    return max(scores, default=0.0)


def _to_hits(chunks: Sequence[dict[str, Any]]) -> list[RankedHit[dict[str, Any]]]:
    """把 chunks 转成可融合的 `RankedHit`（去重键 = source.id + 内容前缀）。"""
    hits: list[RankedHit[dict[str, Any]]] = []
    for chunk in chunks:
        source = chunk.get("source") if isinstance(chunk.get("source"), dict) else {}
        source_id = str(source.get("id") or "")
        content = str(chunk.get("content") or "")
        score = chunk.get("score")
        hits.append(
            RankedHit(
                key=dedup_key(source_id, content, content_prefix=_DEDUP_CONTENT_PREFIX),
                item=chunk,
                score=float(score) if isinstance(score, int | float) else None,
            )
        )
    return hits


class RetrievalQueryInterceptor:
    """仅作用于 rag_mcp 服务：实参归一 + 召回不足惰性补检。"""

    def __init__(self, *, config: RetrievalConfig, server_name: str = RAG_SERVER_NAME) -> None:
        self._config = config
        self._server_name = server_name

    async def __call__(self, request: Any, handler: Any) -> Any:
        if getattr(request, "server_name", None) != self._server_name:
            return await handler(request)
        state = _runtime_state(request)
        main_query = state.get("retrieval_query")
        if self._config.override_model_query and isinstance(main_query, str) and main_query.strip():
            request = self._with_rewritten_args(request, main_query=main_query.strip(), state=state)
        if not self._config.multi_query_enabled:
            return await handler(request)
        return await self._call_with_supplement(request, handler, main_query=main_query)

    def _with_rewritten_args(self, request: Any, *, main_query: str, state: dict[str, Any]) -> Any:
        """覆盖 `query`，并按可用性补 `use_hypothetical`（RAG 未声明则不新增参数）。"""
        args = dict(getattr(request, "args", None) or {})
        model_query = args.get(RAG_QUERY_ARG)
        if isinstance(model_query, str) and model_query.strip() != main_query:
            logger.info("检索实参已归一：模型填 %s → 改写 %s", model_query, main_query)
        args[RAG_QUERY_ARG] = main_query

        understanding = state.get("query_understanding")
        hypothetical = getattr(understanding, "hypothetical_answer", None)
        if hypothetical and self._supports_hypothetical(request):
            args[RAG_HYPOTHETICAL_ARG] = True
            logger.info("HyDE 假设文档随检索传入（仅作嵌入输入，不进引用与生成 prompt）")
        return request.override(args=args)

    def _supports_hypothetical(self, request: Any) -> bool:
        """RAG 工具是否声明了 `use_hypothetical`（取不到声明时视为不支持，宁可不启用）。"""
        declared = _declared_args(request)
        return declared is not None and RAG_HYPOTHETICAL_ARG in declared

    async def _call_with_supplement(self, request: Any, handler: Any, *, main_query: Any) -> Any:
        """首轮检索；结果空/弱且变体池非空 → 补检并 RRF 融合（受预算与上限约束）。"""
        started = time.monotonic()
        response = await handler(request)
        if not isinstance(main_query, str) or not main_query.strip():
            return response
        variants = self._variants(request)
        if not variants:
            return response
        first = _parse_hits(response)
        if first and _max_score(first) >= self._config.weak_score_threshold:
            return response

        merged: list[list[RankedHit[dict[str, Any]]]] = [_to_hits(first)]
        used = 0
        for variant in variants:
            if used >= self._config.max_variants - 1:
                break
            if time.monotonic() - started > self._config.multi_query_budget_s:
                logger.info(
                    "多查询补检超预算（%.1fs），停止补检", self._config.multi_query_budget_s
                )
                break
            extra = await handler(self._supplement_request(request, variant))
            used += 1
            chunks = _parse_hits(extra)
            merged.append(_to_hits(chunks))
            if chunks and _max_score(chunks) >= self._config.weak_score_threshold:
                break  # 已有足够好的结果，不再为边际收益付延迟

        fused = reciprocal_rank_fusion(merged, k=self._config.rrf_k)
        if not fused:
            return response
        logger.info(
            "召回不足触发补检：首轮 %d 条（最高分 %.2f）→ 补检 %d 路 → 融合 %d 条",
            len(first),
            _max_score(first),
            used,
            len(fused),
        )
        return self._rebuild_response(response, main_query=main_query, fused=fused)

    def _variants(self, request: Any) -> list[str]:
        """补检变体池 = 查询理解产出的子查询 + 同义（不含主改写 —— 它已是首轮查询）。"""
        state = _runtime_state(request)
        understanding = state.get("query_understanding")
        pool = getattr(understanding, "ranked_queries", None)
        if not callable(pool):
            return []
        ranked = pool(max_variants=self._config.max_variants)
        return [item.query for item in ranked[1:]]

    def _supplement_request(self, request: Any, query: str) -> Any:
        """构造一次补检请求：只换 query，其余实参（含 scope/开关）保持不变。"""
        args = dict(getattr(request, "args", None) or {})
        args[RAG_QUERY_ARG] = query
        return request.override(args=args)

    def _rebuild_response(
        self, response: Any, *, main_query: str, fused: Sequence[RankedHit[dict[str, Any]]]
    ) -> Any:
        """把融合结果重组为**原返回类型**（保结构，护栏与引用链路零改动）。"""
        payload = getattr(response, "content", response)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                return response
        if not isinstance(payload, dict):
            return response
        rebuilt = {
            **payload,
            # 顶层 query 回填主改写：调用方看到的是「用哪个查询得到的结果」，可解释。
            RAG_QUERY_ARG: main_query,
            "chunks": [hit.item for hit in fused],
        }
        content = json.dumps(rebuilt, ensure_ascii=False)
        if hasattr(response, "content") and hasattr(response, "model_copy"):
            return response.model_copy(update={"content": content})
        return content


__all__ = [
    "RAG_HYPOTHETICAL_ARG",
    "RAG_QUERY_ARG",
    "RAG_SERVER_NAME",
    "RetrievalQueryInterceptor",
]
