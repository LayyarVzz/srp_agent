"""Embedding client smoke test."""

from __future__ import annotations

from embedding import build_embedding_client
from settings import get_settings


def _validate_vector(vector: list[float], name: str) -> None:
    if not vector:
        raise RuntimeError(f"{name}为空")
    if not all(isinstance(value, float) for value in vector):
        raise RuntimeError(f"{name}不是float向量")


def main() -> None:
    """Call the OpenAI-compatible embedding API and validate vector shape."""
    settings = get_settings()
    client = build_embedding_client(
        api_key=settings.embedding_api_key.get_secret_value(),
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )
    documents = [
        "员工累计工作年限已满 10 年不满 20 年的，每年享有 10 天带薪年休假。",
        "病假应按公司要求补充医院证明或其他有效证明材料。",
    ]
    query = "10年年假"

    document_vectors = client.embed_documents(documents)
    query_vector = client.embed_query(query)

    if len(document_vectors) != len(documents):
        raise RuntimeError("文档向量数量与输入文档数量不一致")

    for index, vector in enumerate(document_vectors):
        _validate_vector(vector, f"document_vectors[{index}]")

    _validate_vector(query_vector, "query_vector")

    document_dimension = len(document_vectors[0])
    if any(len(vector) != document_dimension for vector in document_vectors):
        raise RuntimeError("多条文档向量维度不一致")
    if len(query_vector) != document_dimension:
        raise RuntimeError("文档向量与Query向量维度不一致")

    print("EmbeddingClient验证通过")
    print(f"文档数量: {len(document_vectors)}")
    print(f"向量维度: {document_dimension}")


if __name__ == "__main__":
    main()
