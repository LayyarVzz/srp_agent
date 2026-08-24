"""公共 Embedding 工厂：构造 RAG / Memory 共享的 OpenAI-compatible 向量客户端。

本模块只承载「校验显式参数 → 构造客户端 → 生成文本向量」的最小能力。
RAG 与 Memory 均依赖 embedding，避免两个服务互相依赖；索引、检索、缓存、provider 分派、
本地模型与异步客户端均不属于本模块边界。
"""

from __future__ import annotations

from openai import OpenAI


class EmbeddingClient:
    """OpenAI-compatible 文本向量客户端。

    调用方只依赖 `embed_documents` / `embed_query` 两个稳定方法；运行环境配置由
    装配层读取后显式传入，公共层不读取 `settings.py`。
    """

    def __init__(self, *, api_key: str, base_url: str | None, model: str | None) -> None:
        # WHY 快速失败：Embedding 是 RAG / Memory 的共同基础能力，缺配置应在构造期暴露。
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY未配置")
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL未配置")
        if not model:
            raise ValueError("EMBEDDING_MODEL未配置")

        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量；空输入保持既有行为，直接返回空列表。"""
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """生成单条查询向量，复用批量接口以保持行为一致。"""
        return self.embed_documents([text])[0]


def build_embedding_client(
    *,
    api_key: str,
    base_url: str | None,
    model: str | None,
) -> EmbeddingClient:
    """统一 Embedding 构造入口。

    WHY 公共层不读取运行环境：`settings.py` 当前会导入 Agent 配置，若 embedding
    依赖 settings，会让独立 RAG MCP 服务经公共模块间接绑定 Agent。
    """
    return EmbeddingClient(api_key=api_key, base_url=base_url, model=model)
