"""测试共享 fake 设施（离线确定性，无 numpy 依赖）。

P1.md §9 规格：langgraph store 的 `embeddings` 契约是「接受 str 返回定维 list[float]」；
numpy 未安装时 `langchain_core.embeddings.FakeEmbeddings` 会 NameError，故自制纯 Python
定维哈希实现（`embed_query` / `embed_documents` / `aembed_*` 走基类默认实现）。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """无 numpy 的定维哈希 embedding（定维 + 归一化，供确定性测试断言）。"""

    def __init__(self, dims: int = 8) -> None:
        self._dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        h = sum((i + 1) * ord(c) for i, c in enumerate(text)) or 0
        vec = [(h * (i + 1)) % 97 / 97.0 for i in range(self._dims)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]
