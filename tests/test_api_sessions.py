"""FastAPI 层会话接口离线测试（httpx ASGITransport，不触发 lifespan）。

覆盖：healthz、创建/列表/删除会话、X-User-Id 401、跨用户归属 404。
"""

from __future__ import annotations

from typing import Any

from httpx import ASGITransport, AsyncClient

HEADERS = {"X-User-Id": "demo-user"}


async def _client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_healthz(api_app_factory: Any) -> None:
    """GET /healthz：存活探针 + llm_configured 字段。"""
    app, runtime = await api_app_factory([])
    try:
        async with await _client(app) as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["app"] == "srp-agent"
        assert "llm_configured" in body
    finally:
        await runtime.aclose()


async def test_create_session(api_app_factory: Any) -> None:
    """POST /api/v1/sessions：201 + ok 信封 + 服务端发号。"""
    app, runtime = await api_app_factory([])
    try:
        async with await _client(app) as client:
            resp = await client.post("/api/v1/sessions", headers=HEADERS)
        assert resp.status_code == 201
        body = resp.json()
        assert body["ok"] is True
        assert body["user_id"] == "demo-user"
        assert body["session_id"]
        assert body["created_at"]
    finally:
        await runtime.aclose()


async def test_create_session_missing_user_id(api_app_factory: Any) -> None:
    """缺 X-User-Id → 401 auth.identity_required（错误信封）。"""
    app, runtime = await api_app_factory([])
    try:
        async with await _client(app) as client:
            resp = await client.post("/api/v1/sessions")
        assert resp.status_code == 401
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "auth.identity_required"
        assert body["error"]["path"] == "/api/v1/sessions"
    finally:
        await runtime.aclose()


async def test_list_sessions(api_app_factory: Any) -> None:
    """GET /api/v1/sessions：200 + 两条会话都在 + 按 created_at 降序（同微秒允许相等）。"""
    app, runtime = await api_app_factory([])
    try:
        async with await _client(app) as client:
            created = await client.post("/api/v1/sessions", headers=HEADERS)
            first_id = created.json()["session_id"]
            second = await client.post("/api/v1/sessions", headers=HEADERS)
            second_id = second.json()["session_id"]
            resp = await client.get("/api/v1/sessions", headers=HEADERS)
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 2
        # 降序契约：前项不早于后项（两次 POST 可能落在同一微秒，此时允许相等）。
        assert sessions[0]["created_at"] >= sessions[1]["created_at"]
        assert {s["session_id"] for s in sessions} == {first_id, second_id}
        assert {s["user_id"] for s in sessions} == {"demo-user"}
    finally:
        await runtime.aclose()


async def test_list_sessions_isolated_by_user(api_app_factory: Any) -> None:
    """会话按 user 隔离：他人创建的会话不出现在自己列表。"""
    app, runtime = await api_app_factory([])
    try:
        async with await _client(app) as client:
            await client.post("/api/v1/sessions", headers={"X-User-Id": "alice"})
            resp = await client.get("/api/v1/sessions", headers={"X-User-Id": "bob"})
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []
    finally:
        await runtime.aclose()


async def test_delete_session(api_app_factory: Any) -> None:
    """DELETE /api/v1/sessions/{id}：204；删除后归属校验 → 404。"""
    app, runtime = await api_app_factory([])
    try:
        async with await _client(app) as client:
            created = await client.post("/api/v1/sessions", headers=HEADERS)
            session_id = created.json()["session_id"]
            resp = await client.delete(f"/api/v1/sessions/{session_id}", headers=HEADERS)
            assert resp.status_code == 204
            again = await client.delete(f"/api/v1/sessions/{session_id}", headers=HEADERS)
        assert again.status_code == 404
        assert again.json()["error"]["code"] == "session_error.not_found"
    finally:
        await runtime.aclose()


async def test_delete_session_cross_user_not_found(api_app_factory: Any) -> None:
    """跨用户删除他人会话 → 404 session_error.not_found（防枚举）。"""
    app, runtime = await api_app_factory([])
    try:
        async with await _client(app) as client:
            created = await client.post("/api/v1/sessions", headers={"X-User-Id": "alice"})
            session_id = created.json()["session_id"]
            resp = await client.delete(
                f"/api/v1/sessions/{session_id}", headers={"X-User-Id": "bob"}
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "session_error.not_found"
    finally:
        await runtime.aclose()
