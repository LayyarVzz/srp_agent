"""agent/a2a/protocol.py 单测：JSON-RPC 2.0 编解码与错误码。"""

from __future__ import annotations

import json

import pytest

from agent.a2a.protocol import (
    A2A_ERROR_INVALID_REQUEST,
    A2A_ERROR_TASK_NOT_FOUND,
    A2A_METHOD_MESSAGE_SEND,
    A2A_METHOD_MESSAGE_STREAM,
    A2A_METHOD_TASK_CANCEL,
    A2A_METHOD_TASK_GET,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_PARSE_ERROR,
    SUPPORTED_METHODS,
    A2AProtocolError,
    decode_request,
    error_response,
    result_response,
)


def test_supported_methods() -> None:
    """核心子集四个 method（能力矩阵内）。"""
    assert SUPPORTED_METHODS == {
        A2A_METHOD_MESSAGE_SEND,
        A2A_METHOD_MESSAGE_STREAM,
        A2A_METHOD_TASK_GET,
        A2A_METHOD_TASK_CANCEL,
    }


def test_decode_request_ok() -> None:
    """合法请求：字段齐全解码成功（id 字符串/整数均可）。"""
    rpc = decode_request(
        json.dumps({"jsonrpc": "2.0", "id": "req-1", "method": "task/get", "params": {"id": "t1"}})
    )
    assert rpc.id == "req-1"
    assert rpc.method == "task/get"
    assert rpc.params == {"id": "t1"}
    rpc2 = decode_request(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "task/get"}))
    assert rpc2.id == 7
    assert rpc2.params == {}  # 缺省 params → 空对象


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        ("not json", JSONRPC_PARSE_ERROR),  # 非法 JSON
        ('["array"]', JSONRPC_INVALID_REQUEST),  # 非对象
        ('{"jsonrpc": "1.0", "id": 1, "method": "task/get"}', JSONRPC_INVALID_REQUEST),
        ('{"jsonrpc": "2.0", "id": 1}', JSONRPC_INVALID_REQUEST),  # 缺 method
        ('{"jsonrpc": "2.0", "id": 1, "method": ""}', JSONRPC_INVALID_REQUEST),
        ('{"jsonrpc": "2.0", "id": 1, "method": "m", "params": [1]}', JSONRPC_INVALID_REQUEST),
        ('{"jsonrpc": "2.0", "id": 1.5, "method": "m"}', JSONRPC_INVALID_REQUEST),  # id 非法
    ],
)
def test_decode_request_rejects(raw: str, expected_code: int) -> None:
    """结构非法请求统一抛 A2AProtocolError（标准码 + a2a.invalid_request）。"""
    with pytest.raises(A2AProtocolError) as exc_info:
        decode_request(raw)
    assert exc_info.value.jsonrpc_code == expected_code
    assert exc_info.value.a2a_code == A2A_ERROR_INVALID_REQUEST


def test_result_response_envelope() -> None:
    """成功响应信封：jsonrpc/id/result 三字段。"""
    resp = result_response("req-1", {"id": "t1"})
    assert resp.model_dump(mode="json") == {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {"id": "t1"},
        "error": None,
    }


def test_error_response_carries_a2a_code() -> None:
    """错误响应：标准码入 code，a2a.* 内部码入 data.code（两层并存）。"""
    exc = A2AProtocolError(
        jsonrpc_code=-32001, a2a_code=A2A_ERROR_TASK_NOT_FOUND, message="任务不存在: t1"
    )
    resp = error_response("req-1", exc)
    data = resp.model_dump(mode="json")
    assert data["id"] == "req-1"
    assert data["result"] is None
    assert data["error"]["code"] == -32001
    assert data["error"]["data"] == {"code": "a2a.task_not_found"}


def test_error_response_without_id() -> None:
    """解析错误回带 id=None（请求无法解析出 id 的规范行为）。"""
    exc = A2AProtocolError(
        jsonrpc_code=JSONRPC_PARSE_ERROR, a2a_code=A2A_ERROR_INVALID_REQUEST, message="bad"
    )
    assert error_response(None, exc).model_dump(mode="json")["id"] is None
