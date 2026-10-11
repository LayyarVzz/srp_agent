"""shared/embeddings.py —— EmbeddingConfig / EmbeddingsFactory 单测（离线，无网络）。

覆盖 P3-1 验收：is_available 降级信号、build_index_config 形状（对齐 P1.md §2.2）、
enabled 且 dims<=0 启动抛错、from_runtime 注入。全部注入 `tests.fakes.FakeEmbeddings`
离线跑通，不触网。
"""

from __future__ import annotations

import pytest

from shared.embeddings import EmbeddingConfig, EmbeddingsFactory
from tests.fakes import FakeEmbeddings


def test_embedding_config_default_disabled() -> None:
    """默认配置全关：语义召回不启用、不抛降级错。"""
    cfg = EmbeddingConfig()
    assert cfg.enabled is False
    assert cfg.dims == 0
    assert cfg.model == ""


def test_embedding_config_enabled_with_zero_dims_raises() -> None:
    """enabled 且 dims<=0 → 启动即抛错（防误配静默，P3-1 护栏）。"""
    with pytest.raises(ValueError, match="dims"):
        EmbeddingConfig(enabled=True, model="text-embedding-3-small", dims=0)


def test_is_available_gating() -> None:
    """可用信号 = enabled 且 dims>0 且 model 非空（D3 降级判定）。"""
    assert EmbeddingsFactory(EmbeddingConfig()).is_available() is False
    assert (
        EmbeddingsFactory(EmbeddingConfig(enabled=True, model="m", dims=8)).is_available() is True
    )
    assert (
        EmbeddingsFactory(EmbeddingConfig(enabled=True, model="", dims=8)).is_available() is False
    )


def test_build_index_config_shape_with_injected_embedder() -> None:
    """注入 fake embedder：index 配置形状与 P1.md §2.2 对齐（fields/distance/hnsw）。"""
    fake = FakeEmbeddings(dims=8)
    factory = EmbeddingsFactory(
        EmbeddingConfig(enabled=True, model="test-model", dims=8), embedder=fake
    )
    idx = factory.build_index_config()
    assert idx is not None
    assert idx["dims"] == 8
    assert idx["fields"] == ["content"]
    assert idx["distance_type"] == "cosine"
    assert idx["ann_index_config"] == {"index_type": "hnsw"}
    assert idx["embed"] is fake


def test_build_index_config_none_when_unavailable() -> None:
    """未启用（默认配置）→ build_index_config 返回 None（降级信号，零回归）。"""
    assert EmbeddingsFactory(EmbeddingConfig()).build_index_config() is None


def test_build_returns_injected_embedder() -> None:
    """注入 embedder 时 build() 原样返回（跳过真实构造，离线可测）。"""
    fake = FakeEmbeddings(dims=4)
    factory = EmbeddingsFactory(EmbeddingConfig(enabled=True, model="m", dims=4), embedder=fake)
    assert factory.build() is fake


def test_from_runtime_injection() -> None:
    """from_runtime：settings 运行值注入（镜像 LLMConfig.from_runtime 套路）。"""
    cfg = EmbeddingConfig.from_runtime(
        enabled=True,
        model="text-embedding-3-small",
        dims=1536,
        base_url="http://localhost:11434/v1",
        api_key="sk-test",
    )
    assert cfg.enabled is True
    assert cfg.model == "text-embedding-3-small"
    assert cfg.dims == 1536
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.api_key.get_secret_value() == "sk-test"


def test_build_passes_dimensions_when_configured(monkeypatch) -> None:
    """千问场景：base_url + api_key + dimensions 透传（OpenAIEmbeddings 构造不触网）。

    monkeypatch 注入假 OPENAI_API_KEY：openai SDK 构造即校验凭据（_enforce_credentials），
    有 key 则构造成功且不发网络请求，可离线断言配置透传（dimensions 对齐 pgvector 列维度）。
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    factory = EmbeddingsFactory(
        EmbeddingConfig(
            enabled=True,
            model="qwen3.7-text-embedding",
            dims=1024,
            base_url="https://ws.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
    )
    emb = factory.build()
    assert emb.dimensions == 1024
    assert emb.model == "qwen3.7-text-embedding"


def test_build_skips_dimensions_when_disabled(monkeypatch) -> None:
    """pass_dimensions=False：不传 dimensions，走模型默认输出维度。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    factory = EmbeddingsFactory(
        EmbeddingConfig(
            enabled=True, model="text-embedding-ada-002", dims=1536, pass_dimensions=False
        )
    )
    assert factory.build().dimensions is None


def test_build_check_embedding_ctx_length(monkeypatch) -> None:
    """千问兼容端点必须直发字符串数组：默认 False（禁 tiktoken token ID 编码）。

    langchain-openai 默认 True 会把 input 编码成 token ID 整数数组，OpenAI 官方接受
    但千问 compatible-mode 只收字符串数组（400）。默认 False 走直发路径；可显式开启。
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    factory = EmbeddingsFactory(
        EmbeddingConfig(enabled=True, model="qwen3.7-text-embedding", dims=1024)
    )
    assert factory.build().check_embedding_ctx_length is False
    # 官方 OpenAI 长文本长度安全场景可显式开启。
    factory_on = EmbeddingsFactory(
        EmbeddingConfig(
            enabled=True,
            model="text-embedding-3-small",
            dims=1536,
            check_embedding_ctx_length=True,
        )
    )
    assert factory_on.build().check_embedding_ctx_length is True
