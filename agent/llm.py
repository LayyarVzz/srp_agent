"""LLM 统一访问点（LLMService）。

Provider 无关：经 langchain-openai 的 ChatOpenAI 包装任意 OpenAI 兼容端点
（DeepSeek / Qwen DashScope / 本地 vLLM 等），禁止直接调用第三方 SDK。
消费合并后的 `LLMConfig`（见 `agent/core/config.py`），自身不读取环境变量。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from agent.core.config import LLMConfig
from agent.errors import (
    LLM_ERROR_AUTH,
    LLM_ERROR_CONSTRUCTION,
    LLM_ERROR_REQUEST,
    LLMError,
)

logger = logging.getLogger(__name__)

_S = TypeVar("_S", bound=BaseModel)


def _repair_truncated_json(text: str) -> str | None:
    """把被截断的 JSON 参数补全为可解析字符串（尽力而为，纯函数）。

    策略：从「最后一个可能完整闭合的位置」（字符串收尾引号或右括号）向前尝试；
    若截断发生在字符串值内部（回声循环的常态），EOF 处补收尾引号也是候选 ——
    保住该字段已写出的部分。越靠后的候选保住的字段越多（宁多勿丢），
    结构不完整的候选在 JSON 校验时自然失败被跳过。
    解析用 `strict=False`（与 provider 侧参数解析同口径，容忍字符串内裸控制字符）。
    """
    if not text or not text.strip():
        return None
    candidates: list[int] = []
    stack: list[str] = []
    in_string = escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                candidates.append(i)
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]" and stack:
                stack.pop()
                candidates.append(i)
    if in_string:
        # 字符串被截断未闭合：EOF 处补收尾引号也是候选（第一个尝试，保住字段已写出的部分）。
        candidates.append(len(text) - 1)
    for cut in reversed(candidates):
        candidate = _balanced_json(text[: cut + 1])
        if candidate is not None:
            return candidate
    return None


def _balanced_json(prefix: str) -> str | None:
    """把 JSON 前缀补全为可解析字符串；失败 → None。

    三步：① 字符串内的裸控制字符（回声循环常见的未转义换行等）转义为合法转义序列
    —— pydantic 的 JSON 校验是严格模式，raw 控制字符会让重验失败；
    ② 未终止的字符串补收尾引号（悬空反斜杠会转义补上的引号，奇数个则去掉一个）；
    ③ 配平未闭合括号。
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = escape = False
    for ch in prefix:
        if in_string:
            if escape:
                escape = False
                out.append(ch)
            elif ch == "\\":
                escape = True
                out.append(ch)
            elif ch == '"':
                in_string = False
                out.append(ch)
            elif ord(ch) < 0x20:
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, f"\\u{ord(ch):04x}"))
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]" and stack:
                stack.pop()
    if in_string:
        if escape:
            out = out[:-1]  # 末尾是悬空反斜杠：去掉它，避免转义补上的收尾引号
        out.append('"')
    out.extend("}" if c == "{" else "]" for c in reversed(stack))
    candidate = "".join(out)
    try:
        json.loads(candidate, strict=False)
        return candidate
    except ValueError:
        return None


def _salvage_truncated_args[S: BaseModel](
    raw: BaseMessage | None, schema: type[S]
) -> S | None:
    """从被截断的 tool_call 参数中抢救结构化输出（宁漏不误丢的解析侧兜底）。

    触发场景（实测）：模型在某个字符串字段写完后滑入回声循环（连发换行/重复句），
    直至 max_tokens 截断 —— `arguments` JSON 结构不完整（字符串未闭合、括号未配平），
    langchain 把该 tool call 归入 `invalid_tool_calls`，结构化解析**整体返回 None**，
    已完整产出的字段（如 content/category）随之静默丢失。

    修复：对原始 arguments 串做 `_repair_truncated_json` 后用 schema 重验；
    重验通过 → 返回实例（缺失字段由 schema 默认值兜底）；无原始消息 / 无可修复参数 /
    修复后校验失败 → None（与现状同语义，由调用方守卫）。
    """
    if not isinstance(raw, BaseMessage):
        return None
    for invalid in getattr(raw, "invalid_tool_calls", None) or []:
        args_text = invalid.get("args") if isinstance(invalid, dict) else None
        if not args_text or not isinstance(args_text, str):
            continue
        repaired = _repair_truncated_json(args_text)
        if repaired is None:
            continue
        try:
            return schema.model_validate_json(repaired)
        except ValidationError:
            continue
    return None


def merge_ai_message_chunks(chunks: Sequence[AIMessageChunk]) -> AIMessage:
    """把 provider 流式的 AIMessageChunk 序列聚合成等价单条 AIMessage。

    WHY 不能把 chunk 直接入图状态：路由/工具适配以 `isinstance(msg, AIMessage)`
    与 `.tool_calls` 判定（AIMessageChunk 不是 AIMessage 子类），须还原成与
    `ainvoke_tools` 返回同构的 AIMessage。content / tool_call 的跨块合并复用
    langchain 的 AIMessageChunk `+` 语义（content 拼接、tool_call args JSON
    片段按 index 合并）；空流防御性返回空 AIMessage。
    """
    if not chunks:
        return AIMessage(content="")
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    return AIMessage(
        content=merged.content,
        tool_calls=list(merged.tool_calls or []),
        additional_kwargs=dict(merged.additional_kwargs or {}),
        id=merged.id,
    )


class LLMService:
    """Provider 无关的 LLM 统一访问点：构造、结构化输出、异步调用。

    WHY 依赖注入：`chat_model` 允许测试注入离线 fake 模型，且允许调用方
    （如未来 A2A 子智能体）传入自定义模型而不改动本类。
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self._config = config
        self._chat_model = chat_model

    @property
    def chat_model(self) -> BaseChatModel:
        """返回底层聊天模型，未注入时按配置懒构造真客户端。

        WHY 统一关闭思考：懒构造即带 `structured_extra_body`（provider 无关，
        qwen/deepseek 两口径同时下发），text/tool/structured 三条路径共用同一模型，
        不再存在「普通对话保留思考 / 结构化关闭思考」的双模型切换。
        """
        if self._chat_model is None:
            self._chat_model = self._build_chat_model(extra_body=self._config.structured_extra_body)
        return self._chat_model

    def _build_chat_model(self, *, extra_body: Mapping[str, object] | None = None) -> ChatOpenAI:
        cfg = self._config
        api_key = cfg.api_key.get_secret_value()
        if not api_key:
            raise LLMError(LLM_ERROR_AUTH, "未配置 LLM_API_KEY（或 LLM_API_KEY 为空）")
        try:
            # base_url/model 经 effective_* 解析（显式覆盖优先，否则回退预设）。
            # extra_body 统一携带关闭思考的 body（见 structured_extra_body），
            # 所有 LLM 路径（text/tool/structured）共用。
            return ChatOpenAI(
                model=cfg.effective_model,
                api_key=api_key,
                base_url=cfg.effective_base_url,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.request_timeout,
                max_retries=cfg.max_retries,
                extra_body=extra_body,
            )
        except Exception as exc:
            logger.exception("LLM 客户端构造失败")
            raise LLMError(LLM_ERROR_CONSTRUCTION, f"LLM 客户端构造失败: {exc}") from exc

    def structured_model(
        self,
        schema: type[_S],
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] | None = None,
        include_raw: bool = False,
    ) -> Runnable[..., Any]:
        """把 schema 绑定为工具调用，返回可解析出 schema 实例的 Runnable。

        WHY 默认 method=function_calling：
        - langchain-openai >= 0.3 默认 method 是 json_schema（OpenAI Structured Output API），
          DeepSeek 明确拒绝（response_format.type json_schema is unavailable）；
        - DashScope 兼容模式同样走 tool-calling；
        - function_calling 是 DeepSeek / Qwen / 任意兼容端点最广的公共能力。
        不要传 strict=True（DeepSeek 不支持强制 JSON Schema）。

        `include_raw=True`：返回 `{"raw": AIMessage, "parsed": _S | None,
        "parsing_error": ...}` —— 截断修复路径需要原始消息里的 invalid_tool_calls
        （仅解析后的形态拿不到原始参数，无法抢救被截断的输出）。
        """
        effective_method = method or self._config.structured_method
        # chat_model 已在懒构造时统一关闭思考（structured_extra_body），直接复用即可，
        # 无需再构造带 extra_body 的专用模型。
        return self.chat_model.with_structured_output(
            schema, method=effective_method, include_raw=include_raw
        )

    def tool_model(self, tools: Sequence[BaseTool]) -> Runnable:
        """把工具列表绑定到模型，返回可产出 `AIMessage.tool_calls` 的 Runnable。

        WHY 复用 `chat_model`：懒构造时已统一关闭思考（`structured_extra_body`，
        qwen/deepseek 两口径同时下发），`bind_tools` 可直接使用，无需再构造专用模型。
        """
        return self.chat_model.bind_tools(tools)

    async def astream_tools(
        self, tools: Sequence[BaseTool], prompt: str | Sequence
    ) -> AsyncIterator[AIMessageChunk]:
        """带工具绑定的流式调用：逐块产出 AIMessageChunk（content / tool_call 增量）。

        WHY 供 call_model 流式消费：工具循环里模型的 content-only 直答即本轮
        最终回答，需在生成的同时把 content 增量实时外发（token 帧）；调用方以
        `merge_ai_message_chunks` 还原等价 AIMessage 后沿用原路由判定。
        """
        try:
            async for chunk in self.tool_model(tools).astream(prompt):
                yield chunk
        except Exception as exc:
            logger.warning("工具选择流式调用失败: %s", exc)
            raise LLMError(LLM_ERROR_REQUEST, f"LLM 工具选择调用失败: {exc}") from exc

    async def ainvoke_tools(self, tools: Sequence[BaseTool], prompt: str | Sequence) -> AIMessage:
        """一次工具选择/作答调用；任何失败归一化为 LLMError(llm_error.request)。

        返回带 `tool_calls`（模型选择工具）或不带 `tool_calls`（模型直接作答）的
        AIMessage，由图内 `route_tool_choice` 据此分流。

        WHY 聚合 astream_tools：与 ainvoke_text/astream_text 同模式，工具选择
        与作答共用同一条流式 provider 路径，避免两条实现行为分叉。
        """
        chunks: list[AIMessageChunk] = []
        try:
            async for chunk in self.astream_tools(tools, prompt):
                chunks.append(chunk)
        except LLMError:
            raise
        resp = merge_ai_message_chunks(chunks)
        if not isinstance(resp, AIMessage):
            raise LLMError(LLM_ERROR_REQUEST, "工具选择未返回 AIMessage")
        return resp

    async def ainvoke_structured(
        self,
        schema: type[_S],
        prompt: str | Sequence,
        *,
        config: RunnableConfig | None = None,
    ) -> _S | None:
        """一次结构化输出调用；任何失败归一化为 LLMError(llm_error.request)。

        返回 None 的语义：模型未产出有效结构化产出（无工具调用，或参数被截断
        且修复失败）——调用方（意图/改写/抽取/判重等）一律按「无产出」守卫。

        截断修复：模型在字符串字段滑入回声循环直至 max_tokens 截断时，tool call
        参数 JSON 不完整，langchain 把它归入 invalid_tool_calls、解析整体返回 None，
        已完整产出的字段会静默丢失。此处对原始参数做括号配平修复后重验 schema
        （`_salvage_truncated_args`），救回即返回实例（带 warning 可观测）；救不回
        才返回 None。解析成功的主路径与修复前的行为逐字一致。

        `config` 透传给底层结构化 Runnable（默认 None，与不传逐字同路，现有调用方零改动）：
        图内调用靠 langchain 回调上下文继承观测 handler；带外路径（记忆抽取/判重）的
        task 上下文继承自 `chat_stream` 调用方而非图运行时，回调**不继承**，
        必须经此显式传 config 才能挂上回调被 Langfuse 跟踪（O3）。
        """
        try:
            bundle = await self.structured_model(schema, include_raw=True).ainvoke(
                prompt, config=config
            )
        except Exception as exc:
            logger.warning("结构化输出调用失败: %s", exc)
            raise LLMError(LLM_ERROR_REQUEST, f"LLM 结构化输出调用失败: {exc}") from exc
        if not isinstance(bundle, dict):
            return None
        parsed = bundle.get("parsed")
        if parsed is not None:
            return parsed
        salvaged = _salvage_truncated_args(bundle.get("raw"), schema)
        if salvaged is not None:
            logger.warning(
                "结构化输出参数不完整（疑似 max_tokens 截断/回声循环），已修复解析：%s",
                schema.__name__,
            )
            return salvaged
        return None

    async def astream_text(self, prompt: str | Sequence) -> AsyncIterator[str]:
        """流式文本补全：逐增量产出回答文本片段（provider 真流式）。

        仅用于「最终回答」类文本生成：图内回答节点（generate_answer /
        fallback_chat）边消费增量边累积完整回答，同时经 langgraph custom 通道
        把增量实时外发为 SSE token 帧；错误统一归一为 LLMError(llm_error.request)。

        WHY 必须对模型调用真正 astream()：`ainvoke` 走非流式端点，回调只会在
        调用结束时补发整条消息，拿不到 token 级增量（langgraph messages/custom
        通道均依赖真实流式事件）。
        """
        try:
            async for chunk in self.chat_model.astream(prompt):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:
            logger.warning("文本生成失败: %s", exc)
            raise LLMError(LLM_ERROR_REQUEST, f"LLM 文本生成失败: {exc}") from exc

    async def ainvoke_text(self, prompt: str | Sequence) -> str:
        """普通文本补全便利方法（聚合 astream_text 的完整文本）。

        WHY 单一调用路径：ainvoke 与 astream 共用同一流式实现，避免两条
        provider 路径行为分叉；P1 generate_answer / fallback_chat 已切到
        astream_text，本方法供其余文本补全消费方使用。
        """
        parts: list[str] = []
        async for part in self.astream_text(prompt):
            parts.append(part)
        return "".join(parts)
