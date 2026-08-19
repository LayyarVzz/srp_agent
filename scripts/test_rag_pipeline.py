"""RAGPipeline模块级回归验证脚本。"""

from __future__ import annotations

from services.rag_mcp.rag.builder import KnowledgeBuilder
from services.rag_mcp.rag.pipeline import RAGPipeline
from services.rag_mcp.schemas import KnowledgeChunk, KnowledgeSource, SearchKnowledgeResponse

KNOWLEDGE_FILE_PATH = "services/rag_mcp/knowledge/data/employee_policy.txt"


def _build_pipeline() -> RAGPipeline:
    source = KnowledgeSource(
        id="employee_policy",
        title="员工制度",
        url=None,
    )
    chunks = KnowledgeBuilder(KNOWLEDGE_FILE_PATH, source).build()
    return RAGPipeline(chunks)


def _normalize(text: str) -> str:
    """移除空白字符，便于验证当前知识原文中的制度片段。"""
    return "".join(text.split())


def _contains_all(chunk: KnowledgeChunk, expected_parts: list[str]) -> bool:
    normalized_content = _normalize(chunk.content)
    return all(_normalize(part) in normalized_content for part in expected_parts)


def _chunk_id(chunk: KnowledgeChunk) -> str:
    return f"{chunk.source.id}:{chunk.metadata.get('chunk_index')}"


def _validate_response(response: SearchKnowledgeResponse, query: str) -> None:
    if response.query != query:
        raise RuntimeError("query字段与输入不一致")
    if not isinstance(response.chunks, list):
        raise RuntimeError("chunks字段不是列表")
    if not response.chunks:
        raise RuntimeError("chunks为空")

    for chunk in response.chunks:
        if not isinstance(chunk, KnowledgeChunk):
            raise RuntimeError("chunk不是KnowledgeChunk")
        if not chunk.content:
            raise RuntimeError("chunk.content为空")
        if not chunk.source or not chunk.source.id or not chunk.source.title:
            raise RuntimeError("chunk.source无效")
        if not isinstance(chunk.score, float):
            raise RuntimeError("chunk.score不是float")

    scores = [chunk.score for chunk in response.chunks]
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("召回结果未按BM25 score降序排序")


def _run_case(
    pipeline: RAGPipeline,
    *,
    index: int,
    query: str,
    expected_top1: str | None = None,
    expected_in_top_k: set[str] | None = None,
    expected_parts: list[str] | None = None,
) -> None:
    chunks = pipeline.search(query=query, top_k=5)
    response = SearchKnowledgeResponse(query=query, chunks=chunks)
    _validate_response(response=response, query=query)

    print(f"测试{index}：{query}")
    print(f"召回数量：{len(response.chunks)}")
    print("召回结果：")
    for chunk_index, chunk in enumerate(response.chunks, start=1):
        print(f"[{chunk_index}] chunk_id: {_chunk_id(chunk)}")
        print(f"[{chunk_index}] content: {chunk.content}")
        print(f"    source: {chunk.source.model_dump()}")
        print(f"    metadata: {chunk.metadata}")
        print(f"    score: {chunk.score}")

    retrieved_ids = [_chunk_id(chunk) for chunk in response.chunks]

    if expected_top1 is not None and retrieved_ids[0] != expected_top1:
        raise RuntimeError(f"Top1不是预期Chunk：expected={expected_top1}, actual={retrieved_ids[0]}")

    if expected_in_top_k is not None and expected_in_top_k.isdisjoint(retrieved_ids):
        raise RuntimeError(f"TopK中未找到预期Chunk：expected_any={sorted(expected_in_top_k)}")

    if expected_parts is not None and not any(
        _contains_all(chunk, expected_parts) for chunk in response.chunks
    ):
        raise RuntimeError("召回结果中未找到目标制度内容")

    print("验证结果：通过")
    print()


def _validate_top_k(pipeline: RAGPipeline) -> None:
    query = "员工工作10年有多少天年假？"
    chunks = pipeline.search(query=query, top_k=2)
    response = SearchKnowledgeResponse(query=query, chunks=chunks)
    _validate_response(response=response, query=query)
    if len(response.chunks) > 2:
        raise RuntimeError("top_k=2时召回数量超过2")
    scores = [chunk.score for chunk in response.chunks]
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("top_k=2结果未按BM25 score降序排序")

    print("top_k验证：通过")
    print(f"top_k=2召回数量：{len(response.chunks)}")
    for chunk_index, chunk in enumerate(response.chunks, start=1):
        print(f"[{chunk_index}] chunk_id: {_chunk_id(chunk)}")
        print(f"    score: {chunk.score}")
    print()


def main() -> None:
    """直接调用RAGPipeline，验证BM25已经进入Pipeline。"""
    pipeline = _build_pipeline()

    cases = [
        {
            "query": "员工工作10年有多少天年假？",
            "expected_top1": "employee_policy:7",
            "expected_parts": ["10 年不满 20 年", "10 天"],
        },
        {
            "query": "员工工作20年有多少天年假？",
            "expected_in_top_k": {"employee_policy:8"},
            "expected_parts": ["20 年及以上", "15 天"],
        },
        {
            "query": "员工工作不满1年有年假吗？",
            "expected_top1": "employee_policy:5",
            "expected_parts": ["不满 1 年", "不享受带薪年休假"],
        },
        {
            "query": "员工请假需要提交什么？",
            "expected_in_top_k": {
                "employee_policy:11",
                "employee_policy:12",
                "employee_policy:13",
                "employee_policy:14",
                "employee_policy:15",
            },
        },
        {
            "query": "病假需要什么证明？",
            "expected_top1": "employee_policy:14",
            "expected_parts": ["病假", "医院证明", "有效证明材料"],
        },
        {
            "query": "员工婚假有多少天？",
        },
    ]

    for index, case in enumerate(cases, start=1):
        _run_case(
            pipeline,
            index=index,
            **case,
        )

    _validate_top_k(pipeline)
    print("RAGPipeline回归验证通过")


if __name__ == "__main__":
    main()
