"""LLM 统一访问点（LLMService）。

Provider 无关：经 langchain-openai 的 ChatOpenAI 包装任意 OpenAI 兼容端点
（DeepSeek / Qwen DashScope / 本地 vLLM 等），禁止直接调用第三方 SDK。
消费合并后的 `LLMConfig`（见 `agent/core/config.py`），自身不读取环境变量。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.core.config import LLMConfig
from agent.errors import (
    LLM_ERROR_AUTH,
    LLM_ERROR_CONSTRUCTION,
    LLM_ERROR_REQUEST,
    LLMError,
)

logger = logging.getLogger(__name__)

_S = TypeVar("_S", bound=BaseModel)


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
        # 记录是否注入自定义模型：`chat_model` 属性懒缓存会改写 `_chat_model`，
        # 不能再用「_chat_model is None」判断「未注入」，否则结构化输出会被缓存到的
        # 普通模型污染（见 structured_model）。
        self._chat_model_injected = chat_model is not None

    @property
    def chat_model(self) -> BaseChatModel:
        """返回底层聊天模型，未注入时按配置懒构造真客户端。"""
        if self._chat_model is None:
            self._chat_model = self._build_chat_model()
        return self._chat_model

    def _build_chat_model(self, *, extra_body: Mapping[str, object] | None = None) -> ChatOpenAI:
        cfg = self._config
        api_key = cfg.api_key.get_secret_value()
        if not api_key:
            raise LLMError(LLM_ERROR_AUTH, "未配置 LLM_API_KEY（或 LLM_API_KEY 为空）")
        try:
            # base_url/model 经 effective_* 解析（显式覆盖优先，否则回退预设）。
            # extra_body 供结构化输出关闭 DeepSeek V4 思考模式（见 structured_extra_body）。
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
    ) -> Runnable[..., _S]:
        """把 schema 绑定为工具调用，返回可解析出 schema 实例的 Runnable。

        WHY 默认 method=function_calling：
        - langchain-openai >= 0.3 默认 method 是 json_schema（OpenAI Structured Output API），
          DeepSeek 明确拒绝（response_format.type json_schema is unavailable）；
        - DashScope 兼容模式同样走 tool-calling；
        - function_calling 是 DeepSeek / Qwen / 任意兼容端点最广的公共能力。
        不要传 strict=True（DeepSeek 不支持强制 JSON Schema）。
        """
        effective_method = method or self._config.structured_method
        extra_body = self._config.structured_extra_body
        if extra_body is not None and not self._chat_model_injected:
            # DeepSeek V4 思考模式拒绝显式 tool_choice，无法在思考模式下强制 schema 工具；
            # 未注入自定义模型时，构造带 extra_body（thinking=disabled）的专用模型再绑定，
            # 普通对话（ainvoke_text）仍走 self.chat_model、保留思考模式。
            model: BaseChatModel = self._build_chat_model(extra_body=extra_body)
        else:
            model = self.chat_model
        return model.with_structured_output(schema, method=effective_method)

    async def ainvoke_structured(self, schema: type[_S], prompt: str | Sequence) -> _S:
        """一次结构化输出调用；任何失败归一化为 LLMError(llm_error.request)。"""
        try:
            return await self.structured_model(schema).ainvoke(prompt)
        except Exception as exc:
            logger.warning("结构化输出调用失败: %s", exc)
            raise LLMError(LLM_ERROR_REQUEST, f"LLM 结构化输出调用失败: {exc}") from exc

    async def ainvoke_text(self, prompt: str | Sequence) -> str:
        """普通文本补全便利方法（P1 generate_answer 复用）。"""
        try:
            resp = await self.chat_model.ainvoke(prompt)
            return str(resp.content or "")
        except Exception as exc:
            logger.warning("文本生成失败: %s", exc)
            raise LLMError(LLM_ERROR_REQUEST, f"LLM 文本生成失败: {exc}") from exc
