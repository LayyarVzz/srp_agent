"""根级共享 Embedding 底座（RAG 与长期记忆共用，禁直连 SDK）。

提供 provider 无关的统一 Embedding 构造工厂 `EmbeddingsFactory` 与配置模型
`EmbeddingConfig`：语义召回开关、模型/维度对齐（与 RAG 复用同一 `EMBEDDING_*` 配置）、
降级信号由 `is_available` 承载。`agent/` 与 `services/` 均经
`from shared import EmbeddingConfig, EmbeddingsFactory` 引用同一份代码与配置契约。

职责边界：本模块只做「构造 / 判定可用 / 转译 langgraph Store index 配置」。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr, model_validator


class EmbeddingConfig(BaseModel):
    """语义召回开关 + 模型对齐配置。默认全关（enabled=False），未配时自动降级。

    环境项（EMBEDDING_* / 密钥）由装配层经 `from_runtime` 注入，本模型自身不读环境变量。
    """

    enabled: bool = False  # 语义召回总开关；未配模型/无 key 时自动降级
    provider: Literal["openai", "openai_compatible"] = "openai_compatible"
    model: str = ""  # 与 RAG 同一模型名（EMBEDDING_MODEL）
    dims: int = Field(default=0, ge=0)  # 与模型一致；enabled 时必须 > 0
    base_url: str | None = None  # OpenAI 兼容端点；缺省用官方 OpenAI
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    pass_dimensions: bool = True
    # langchain-openai 默认 True 会对文本做 tiktoken 分块，并把 input 编码成 token ID
    # 整数数组发送；OpenAI 官方端点接受该形状，但千问 compatible-mode 只收字符串数组
    # （否则 400 "input must be an array of strings"）。置 False 直发原文（绕过 tokenization）。
    # 代价：不做超长文本分块，超模型上限会报错（内容已受 max_recall_chars 护栏约束）。
    check_embedding_ctx_length: bool = False

    @model_validator(mode="after")
    def _validate_enabled_dims(self) -> EmbeddingConfig:
        """enabled 且 dims<=0 视为误配：启动即抛错，避免建表静默失败。"""
        if self.enabled and self.dims <= 0:
            raise ValueError("embedding enabled 时必须配置 dims > 0（与所选模型一致）")
        return self

    @classmethod
    def from_runtime(
        cls,
        *,
        enabled: bool = False,
        provider: Literal["openai", "openai_compatible"] = "openai_compatible",
        model: str = "",
        dims: int = 0,
        base_url: str | None = None,
        api_key: SecretStr | str = "",
        behavior: EmbeddingConfig | None = None,
    ) -> EmbeddingConfig:
        """由运行选择 + 可选行为覆盖合并出完整配置（装配层调用，镜像 LLMConfig.from_runtime）。"""
        behavior_fields = behavior.model_dump() if behavior else {}
        key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        return cls(
            enabled=enabled,
            provider=provider,
            model=model,
            dims=dims,
            base_url=base_url,
            api_key=key,
            **behavior_fields,
        )


class EmbeddingsFactory:
    """Provider 无关的统一 Embedding 构造工厂。"""

    def __init__(self, config: EmbeddingConfig, *, embedder: Embeddings | None = None) -> None:
        self._config = config
        self._embedder = embedder  # 测试/离线注入，跳过真实构造（镜像 LLMService.chat_model）

    def is_available(self) -> bool:
        """语义检索可用信号：enabled 且 dims>0 且 model 非空。"""
        return self._config.enabled and self._config.dims > 0 and bool(self._config.model)

    def build(self) -> Embeddings:
        """构造 Embeddings 实例。注入 embedder 时原样返回，否则懒构造 OpenAIEmbeddings。"""
        if self._embedder is not None:
            return self._embedder
        from langchain_openai import OpenAIEmbeddings  # 懒 import：降级路径零 import 成本

        kwargs: dict[str, Any] = {"model": self._config.model}
        if self._config.pass_dimensions:
            # 显式传 dimensions：让兼容端点（千问等）输出维度对齐 pgvector 列维度。
            kwargs["dimensions"] = self._config.dims
        # 千问 compatible-mode 只收字符串数组，禁 tiktoken token ID 数组（见字段注释）。
        kwargs["check_embedding_ctx_length"] = self._config.check_embedding_ctx_length
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        if self._config.api_key.get_secret_value():
            kwargs["api_key"] = self._config.api_key.get_secret_value()
        return OpenAIEmbeddings(**kwargs)

    def build_index_config(self) -> dict[str, Any] | None:
        """转译为 langgraph Store 的 index 配置；不可用时返回 None。"""
        if not self.is_available():
            return None
        return {
            "dims": self._config.dims,
            "embed": self.build(),
            "fields": ["content"],
            "distance_type": "cosine",
            "ann_index_config": {"index_type": "hnsw"},
        }
