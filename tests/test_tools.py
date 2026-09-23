"""agent/tools —— 响应/适配模型单测（离线）。

覆盖：错误码常量、ToolError/ToolResult/ToolCallRecord 默认值与 model_dump 往返
（checkpointer msgpack 兼容前提）。工具执行契约已迁移到 LangChain ToolNode，
不再测试自研注册表/工具选择模型。
"""

from __future__ import annotations

from agent.tools.models import (
    TOOL_ERROR_EXECUTION,
    TOOL_ERROR_MISSING_ARGUMENT,
    TOOL_ERROR_UNKNOWN_TOOL,
    ToolCallRecord,
    ToolError,
    ToolResult,
)


def test_tool_error_codes_constants() -> None:
    """错误码集中声明（CLAUDE.md：禁止散落字符串字面量）。"""
    assert TOOL_ERROR_EXECUTION == "tool_error.execution"
    assert TOOL_ERROR_UNKNOWN_TOOL == "tool_error.unknown_tool"
    assert TOOL_ERROR_MISSING_ARGUMENT == "tool_error.missing_argument"


def test_tool_error_defaults_and_roundtrip() -> None:
    err = ToolError(code=TOOL_ERROR_EXECUTION, message="boom")
    assert err.retryable is False
    assert err.model_dump() == {
        "code": "tool_error.execution",
        "message": "boom",
        "retryable": False,
    }
    assert ToolError.model_validate(err.model_dump()) == err


def test_tool_result_defaults() -> None:
    """缺省值契约：耗时未测量时为 0（真实执行路径由图的派发计时写入，见 test_log_events）。"""
    result = ToolResult(tool_name="calc", ok=True, data={"result": 1})
    assert result.citations == []
    assert result.duration_ms == 0
    assert result.error is None
    assert result.model_dump()["ok"] is True


def test_tool_result_error_roundtrip() -> None:
    result = ToolResult(
        tool_name="calc",
        ok=False,
        error=ToolError(code=TOOL_ERROR_UNKNOWN_TOOL, message="unknown"),
    )
    dumped = result.model_dump()
    assert dumped["ok"] is False
    assert dumped["error"]["code"] == "tool_error.unknown_tool"
    assert ToolResult.model_validate(dumped) == result


def test_tool_call_record_defaults() -> None:
    record = ToolCallRecord(tool_name="calc", arguments={"expression": "1+1"}, status="ok")
    assert record.result is None
    assert record.model_dump()["status"] == "ok"


def test_tool_call_record_status_accepts_error() -> None:
    record = ToolCallRecord(
        tool_name="calc",
        arguments={},
        status="error",
        result=ToolResult(
            tool_name="calc",
            ok=False,
            error=ToolError(code=TOOL_ERROR_EXECUTION, message="boom"),
        ),
    )
    assert record.model_dump()["result"]["error"]["code"] == "tool_error.execution"
