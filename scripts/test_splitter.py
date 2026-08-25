"""Splitter回归验证脚本。"""

from __future__ import annotations

from services.rag_mcp.knowledge.loader import load_text_file
from services.rag_mcp.rag.models import DocumentChunk
from services.rag_mcp.rag.splitter import split_markdown, split_text
from services.rag_mcp.schemas import KnowledgeSource


def _build_source() -> KnowledgeSource:
    return KnowledgeSource(
        id="employee_policy",
        title="员工制度",
        url=None,
    )


def main() -> None:
    """读取员工制度文本并打印DocumentChunk切分结果。"""
    text = load_text_file("services/rag_mcp/knowledge/data/employee_policy.txt")
    source = _build_source()
    chunks = split_text(text, source)
    expected_contents = [
        "员工制度\n\n一、年假规定\n\n公司员工可按照本制度申请带薪年休假。员工年假应结合本人累计工作年限、当年度出勤情况以及公司业务安排统筹使用。",
        "员工申请年假前，应确认本人剩余年假额度。年假原则上应在当年度内使用完毕；确因工作安排无法休完的，可按照公司当年度人力资源政策处理。\n\n二、工作年限对应年假天数",
        "员工累计工作年限不满 1 年的，原则上不享受带薪年休假。\n\n员工累计工作年限已满 1 年不满 10 年的，每年享有 5 天带薪年休假。",
        "员工累计工作年限已满 10 年不满 20 年的，每年享有 10 天带薪年休假。\n\n员工累计工作年限已满 20 年及以上的，每年享有 15 天带薪年休假。",
        "新入职员工当年度年假天数可根据入职日期折算，具体折算方式以人力资源部门确认为准。\n\n三、请假流程",
        "员工申请年假、事假、病假或其他假期时，应提前在公司考勤系统提交请假申请。\n\n请假申请应填写请假类型、开始时间、结束时间、请假原因以及必要的证明材料。",
        "直属主管负责审批员工请假申请。连续请假超过 3 个工作日的，应根据部门管理要求提交更高层级审批。",
        "病假应按公司要求补充医院证明或其他有效证明材料。未按流程提交申请或未获得审批而缺勤的，可能按旷工处理。",
        "员工休假结束后，应按时返岗。如需延长假期，应在原假期结束前重新提交申请并获得审批。",
    ]

    if not chunks:
        raise RuntimeError("Splitter未返回任何chunk")
    if len(chunks) != len(expected_contents):
        raise RuntimeError("DocumentChunk数量与80字符Paragraph Packing结果不一致")

    print("原始文本类型:")
    print(type(text))
    print("切分结果类型:")
    print(type(chunks))
    print("chunk数量:")
    print(len(chunks))
    print("chunk内容:")
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, DocumentChunk):
            raise RuntimeError("chunk不是DocumentChunk")
        if chunk.content != expected_contents[chunk_index]:
            raise RuntimeError("chunk.content与80字符Paragraph Packing结果不一致")
        if len(chunk.content) > 80:
            raise RuntimeError("chunk.content长度超过80")
        if chunk.source != source:
            raise RuntimeError("chunk.source与传入KnowledgeSource不一致")
        if chunk.metadata.get("chunk_index") != chunk_index:
            raise RuntimeError("chunk_index未按原始切分顺序递增")
        if chunk.chunk_id != f"{source.id}:{chunk_index}":
            raise RuntimeError("chunk_id不符合当前MVP规则")

        print(f"{chunk_index + 1}. content: {chunk.content}")
        print(f"   chunk_id: {chunk.chunk_id}")
        print(f"   source: {chunk.source.model_dump()}")
        print(f"   metadata: {chunk.metadata}")

    fallback_text = (
        "短段落。\n\n"
        "第一句用于句子级切分。第二句也用于句子级切分。"
        "这是一个超过八十个字符且没有句末标点的超长句子"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        "继续补足长度以触发字符硬切"
    )
    fallback_chunks = split_text(fallback_text, source)
    if not all(len(chunk.content) <= 80 for chunk in fallback_chunks):
        raise RuntimeError("Sentence/Character Fallback后仍存在超过80字符的chunk")
    if not any("第一句用于句子级切分。" in chunk.content for chunk in fallback_chunks):
        raise RuntimeError("Sentence Fallback未保留中文句子")
    if not any("ABCDEFGHIJKLMNOPQRSTUVWXYZ" in chunk.content for chunk in fallback_chunks):
        raise RuntimeError("Character Fallback未保留长句内容")
    for chunk_index, chunk in enumerate(fallback_chunks):
        if chunk.metadata != {"chunk_index": chunk_index}:
            raise RuntimeError("fallback chunk metadata不符合当前MVP规则")
        if chunk.chunk_id != f"{source.id}:{chunk_index}":
            raise RuntimeError("fallback chunk_id不符合当前MVP规则")

    markdown_text = """# 员工制度
## 请假管理
### 病假
病假应提交医院证明材料。

员工申请病假时应提前在系统提交申请并补充证明材料主管审批通过后员工应按时返岗并完成销假确认必要时还应继续补充有效证明材料以便人力资源部门留存
## 年假管理
年假应在当年度内使用。
"""
    markdown_chunks = split_markdown(markdown_text, source)
    if not markdown_chunks:
        raise RuntimeError("Markdown Splitter未返回任何chunk")
    if any(chunk.content.lstrip().startswith("#") for chunk in markdown_chunks):
        raise RuntimeError("Markdown标题被单独或原样生成为chunk")
    if not all(len(chunk.content) <= 80 for chunk in markdown_chunks):
        raise RuntimeError("Markdown正文切分后存在超过80字符的chunk")

    actual_heading_paths = [chunk.metadata.get("heading_path") for chunk in markdown_chunks]
    sick_leave_path = ["员工制度", "请假管理", "病假"]
    annual_leave_path = ["员工制度", "年假管理"]
    if actual_heading_paths[0] != sick_leave_path:
        raise RuntimeError("Markdown三级heading_path不符合预期")
    if actual_heading_paths[-1] != annual_leave_path:
        raise RuntimeError("Markdown浅层标题出现时未丢弃更深层级")
    if sum(heading_path == sick_leave_path for heading_path in actual_heading_paths) < 2:
        raise RuntimeError("Markdown正文未按80字符规则切分为多个chunk")
    if markdown_chunks[0].content != "病假应提交医院证明材料。":
        raise RuntimeError("Markdown标题未与后续正文按section自然绑定")
    for chunk_index, chunk in enumerate(markdown_chunks):
        expected_metadata = {
            "chunk_index": chunk_index,
            "heading_path": actual_heading_paths[chunk_index],
        }
        if chunk.metadata != expected_metadata:
            raise RuntimeError("Markdown chunk metadata不符合当前规则")
        if chunk.chunk_id != f"{source.id}:{chunk_index}":
            raise RuntimeError("Markdown chunk_id不符合当前MVP规则")

    print("Splitter DocumentChunk验证通过")


if __name__ == "__main__":
    main()
