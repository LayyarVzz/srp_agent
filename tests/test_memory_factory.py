"""agent/memory/factory.py —— 记忆后端工厂单测（离线，不依赖数据库）。

覆盖：统一入口 `build_memory_backends` 的 in-memory 降级 / 显式 postgres 无 DSN
快速失败；MemoryBackends.aclose 对 dev 路径是安全 no-op。会话元数据由
`agent/session` 自装（其单测见 tests/test_session.py 的 build_session_backend）。
"""

from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from agent.core.config import AgentFrameworkConfig
from agent.memory import MemoryBackends, build_memory_backends, store_has_embeddings
from shared.embeddings import EmbeddingConfig
from tests.fakes import FakeEmbeddings


async def test_build_memory_backends_default_in_memory() -> None:
    """无 DSN → 统一入口降级 InMemoryStore，checkpointer=None；只含记忆后端、不再带会话。"""
    cfg = AgentFrameworkConfig.get_default()
    backends = await build_memory_backends(cfg, database_url=None)
    assert isinstance(backends.store, InMemoryStore)
    assert backends.checkpointer is None


async def test_build_memory_backends_explicit_postgres_without_dsn_fails() -> None:
    """显式 store_type=postgres 但无 DSN → 快速失败，错误信息含 DATABASE_URL。"""
    cfg = AgentFrameworkConfig.get_default()
    cfg.memory.store_type = "postgres"
    with pytest.raises(ValueError, match="DATABASE_URL"):
        await build_memory_backends(cfg, database_url=None)


async def test_memory_backends_aclose_dev_path_noop() -> None:
    """dev 路径（无 postgres CM 句柄）aclose 安全 no-op，不抛错。"""
    backends = MemoryBackends(store=InMemoryStore(), checkpointer=None)
    await backends.aclose()


async def test_build_memory_backends_index_config_sets_embeddings() -> None:
    """配 embedding（enabled+dims+model）→ store 装配后带 embeddings（语义底座生效）。

    注入 FakeEmbeddings（build_memory_backends 的 embedder 测试缝）离线跑通：
    OpenAIEmbeddings 构造即强制校验凭据（openai SDK `_enforce_credentials`），
    无 key 会抛 OpenAIError，故经注入缝替换真实构造。
    """
    cfg = AgentFrameworkConfig.get_default()
    cfg.memory.embedding = EmbeddingConfig(enabled=True, model="test-model", dims=8)
    backends = await build_memory_backends(cfg, database_url=None, embedder=FakeEmbeddings(dims=8))
    assert isinstance(backends.store, InMemoryStore)
    assert backends.store.embeddings is not None
    assert store_has_embeddings(backends.store) is True


async def test_build_memory_backends_embedding_disabled_embeddings_none() -> None:
    """默认配置（embedding 未启用）→ index_config=None → store.embeddings 为 None（零回归）。"""
    cfg = AgentFrameworkConfig.get_default()
    backends = await build_memory_backends(cfg, database_url=None)
    assert isinstance(backends.store, InMemoryStore)
    assert backends.store.embeddings is None
    assert store_has_embeddings(backends.store) is False


def test_store_has_embeddings() -> None:
    """store_has_embeddings：配 index 的 InMemoryStore → True；未配 → False。"""
    assert store_has_embeddings(InMemoryStore()) is False
    assert (
        store_has_embeddings(
            InMemoryStore(index={"dims": 8, "embed": FakeEmbeddings(8), "fields": ["content"]})
        )
        is True
    )
