"""v5.1 飞书绑定**多用户全链路离线验收**（V51-M1 / M2 / M3）。

这是本里程碑的验收测试：把「**每个 Agent 用户可以绑定各自的飞书账号、互不可见**」
这条产品承诺，用一条**端到端**路径钉死 —— 而不是只测各个零件：

    设备码授权（打桩 HTTP）
        → 绑定落库（真仓库 + 真 Fernet 加密）
        → 作用域拦截器按 user_id 注入 `_lark_scope`（真拦截器）
        → 凭据提供者按作用域取**该用户**的 UAT（真解析 + 真刷新语义）
        → lark-cli 子进程 env（捕获注入实参，零真实 CLI / 零网络）

关键断言（M1 的实质）：
1. A/B 两用户各自绑定后，工具调用注入的 `LARKSUITE_CLI_USER_ACCESS_TOKEN`
   **分别是各自的 token**（互不串号）；
2. **互不可见**：B 未绑定时 A 的 token 绝不会被 B 用到（不是「共享一份」）；
3. 绑定完成后**不重新绑定即不可换身份**（解绑 A 不影响 B）；
4. 落库全程密文（库内明文 token 零出现）；
5. 未绑定用户的工具调用 → `tool_error.lark_unbound`（图侧走绑定引导）。

WHY 打桩子进程而不拉真实 CLI：Windows 测试循环（conftest 统一 SelectorEventLoop，
psycopg 需要）不支持 asyncio 子进程；捕获 `create_subprocess_exec` 的 argv/env
同样覆盖「身份注入」这一验收点，真实 CLI 端到端见 `scripts/demo_lark_binding.py`
与真机验收（dev-version5.1.md §14）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import Client
from langchain_core.messages import AIMessage

from agent.tools.lark_scope import LARK_SCOPE_ARG, LarkScopeInterceptor
from services.lark_mcp import server
from services.lark_mcp.binding import LarkBindingService
from services.lark_mcp.cli import (
    ENV_APP_ID,
    ENV_DEFAULT_AS,
    ENV_STRICT_MODE,
    ENV_USER_ACCESS_TOKEN,
    LarkCliRunner,
)
from services.lark_mcp.credentials import BindingCredentialProvider
from shared.lark import (
    LARK_UNBOUND_PREFIX,
    LarkOAuthClient,
    build_lark_binding_repository,
    build_token_cipher,
)
from shared.lark.oauth import PATH_DEVICE_AUTHORIZATION, PATH_TOKEN_ENDPOINT

APP_ID = "cli_acceptance_app"
APP_SECRET = "acceptance-secret"
TOKEN_KEY = "acceptance-token-key"

# 两个用户的「飞书身份」与令牌（互不相同，用于串号检测）。
USER_A, USER_B = "user-a", "user-b"
UAT_A, UAT_B = "uat-for-user-A-only", "uat-for-user-B-only"
RT_A, RT_B = "rt-for-user-A-only", "rt-for-user-B-only"
OPEN_ID_A, NAME_A = "ou_aaaa", "张三"
OPEN_ID_B, NAME_B = "ou_bbbb", "李四"

# 工具调用用到的目标账号（由 fake CLI 回显，用于确认「查到的是谁」）。
_DEFAULT_DEVICE_CODE = "d" * 100


class ProcRecorder:
    """捕获 `create_subprocess_exec`（argv/env）的替身，回放预设 CLI 输出。

    env 是本次验收的核心证据：工具调用注入的 UAT 就是**执行身份**，
    故必须逐次留证（哪个作用域 → 哪个 token）。
    """

    def __init__(self, *, stdout: bytes | None = None, returncode: int = 0) -> None:
        self.argvs: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self._stdout = stdout if stdout is not None else json.dumps({"ok": True}).encode()
        self._returncode = returncode

    async def __call__(self, *argv: Any, **kwargs: Any) -> Any:
        self.argvs.append([str(a) for a in argv])
        self.envs.append(kwargs["env"])
        return _FakeProc(stdout=self._stdout, returncode=self._returncode)

    def tokens(self) -> list[str]:
        """按调用顺序取出各次子进程注入的 UAT（身份证据序列）。"""
        return [env.get(ENV_USER_ACCESS_TOKEN, "") for env in self.envs]


class _FakeProc:
    """最小子进程替身（communicate/kill/wait 契约与 asyncio 对齐）。"""

    def __init__(self, *, stdout: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._returncode = returncode

    @property
    def returncode(self) -> int:
        return self._returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""

    def kill(self) -> None:  # pragma: no cover - 正常路径不触发
        return None

    async def wait(self) -> int:  # pragma: no cover
        return self._returncode


def _oauth_handler() -> Any:
    """打桩飞书 OAuth 端点：设备码发起 / 换码 / 用户信息（零网络）。

    **换码的令牌分配**：MockTransport 拿不到服务侧的 `user_id`（真实链路里
    device_code 就是被换绑者的凭据，作用域不出现在 HTTP 线上），故按**换码调用序**
    依次发放 `(UAT_A, RT_A)`、`(UAT_B, RT_B)`；本文件每个用例内的绑定顺序恒为
    A→B，因此该分配与 `USER_x ↔ UAT_x` 一一对应（用户信息端点亦按此表反查）。

    用户信息端点按 `Authorization: Bearer <uat>` 区分「绑的是谁」——这让绑定回执
    里的账号名能反过来证明「哪个 token 绑到了谁」（串号检测的关键证据）。
    """
    names = {UAT_A: NAME_A, UAT_B: NAME_B}
    open_ids = {NAME_A: OPEN_ID_A, NAME_B: OPEN_ID_B}
    sequence = [(UAT_A, RT_A), (UAT_B, RT_B)]
    issued = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == PATH_DEVICE_AUTHORIZATION:
            return httpx.Response(200, json=_device_payload())
        if path == PATH_TOKEN_ENDPOINT:
            body = json.loads(request.content)
            # 按 grant_type 分流（**不要**按 device_code 字段判定：换码请求体里
            # 设备码的字段名是 `code`，按 device_code 判会静默落到刷新分支）。
            if body.get("grant_type") == "authorization_code":  # 换码
                uat, rt = sequence[min(issued["n"], len(sequence) - 1)]
                issued["n"] += 1
                return httpx.Response(
                    200,
                    json={
                        "access_token": uat,
                        "refresh_token": rt,
                        "expires_in": 7200,
                        "refresh_token_expires_in": 604800,
                    },
                )
            # 刷新（本用例不触发；保留以证端点完整 + 说明刷新同样按序发放）。
            return httpx.Response(
                200,
                json={
                    "access_token": sequence[-1][0],
                    "refresh_token": "rt-refreshed",
                    "expires_in": 7200,
                    "refresh_token_expires_in": 604800,
                },
            )
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        name = names.get(token, "未知")
        return httpx.Response(
            200, json={"code": 0, "data": {"open_id": open_ids[name], "name": name}}
        )

    return handler


def _device_payload(device_code: str = _DEFAULT_DEVICE_CODE) -> dict[str, Any]:
    return {
        "device_code": device_code,
        "user_code": "KR2E-FZQP",
        "verification_uri": "https://accounts.feishu.cn/oauth/v1/device/verify",
        "flow_id": "flow-acceptance",
        "expires_in": 600,
        "interval": 5,
    }


async def _wire_acceptance_stack() -> Any:
    """装配验收用的**真实**链路（绑定服务 + 凭据提供者 + runner），返回仓库。"""
    repository = build_lark_binding_repository(build_token_cipher(TOKEN_KEY))
    await repository.setup()
    oauth = LarkOAuthClient(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        accounts_base_url="https://accounts.feishu.cn",
        open_base_url="https://open.feishu.cn",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_oauth_handler())),
    )
    server.binding_service = LarkBindingService(repository=repository, oauth=oauth, app_id=APP_ID)
    server.credential_provider = BindingCredentialProvider(
        repository=repository, oauth=oauth, app_id=APP_ID, app_secret=APP_SECRET
    )
    return repository


@pytest.fixture
def patch_exec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProcRecorder:
    """打桩子进程执行 + 提供一个「存在」的假命令路径（构造器校验存在性）。"""
    stub = tmp_path / "lark-cli.exe"
    stub.write_bytes(b"")
    recorder = ProcRecorder()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)
    # runner 为模块级单例：替换为指向假命令的实例（执行由 recorder 拦截，不会真跑）。
    server.runner = LarkCliRunner(command=str(stub), timeout_s=10.0, max_output_chars=10_000)
    return recorder


async def _call(name: str, args: dict[str, Any]) -> Any:
    """经真实 MCP 客户端调用工具。"""
    async with Client(server.mcp) as client:
        return await client.call_tool(name, args)


def _text(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    return getattr(blocks[0], "text", "") if blocks else ""


# —— 验收 1：两用户各自绑定（M3 设备码闭环）——


async def test_two_users_bind_their_own_accounts(patch_exec: ProcRecorder) -> None:
    """M3：A/B 各自走完设备码闭环，绑到**各自的**飞书账号（回执含各自账号名）。"""
    repository = await _wire_acceptance_stack()
    try:
        for user in (USER_A, USER_B):
            await _call("lark_bind_start", {LARK_SCOPE_ARG: user})
            text = _text(await _call("lark_bind_complete", {LARK_SCOPE_ARG: user}))
            assert "绑定成功" in text

        binding_a = await repository.get_binding(USER_A)
        binding_b = await repository.get_binding(USER_B)
        assert binding_a is not None and binding_b is not None
        # 绑到的是各自的账号（open_id 不同 → 未被串号覆盖）
        assert {binding_a.open_id, binding_b.open_id} == {OPEN_ID_A, OPEN_ID_B}
        assert binding_a.open_id != binding_b.open_id
    finally:
        await repository.aclose()


async def test_binding_tokens_are_encrypted_at_rest(patch_exec: ProcRecorder) -> None:
    """安全硬约束：库内**零明文** token（UAT / refresh 均密文，且能解回原值）。"""
    repository = await _wire_acceptance_stack()
    try:
        await _call("lark_bind_start", {LARK_SCOPE_ARG: USER_A})
        await _call("lark_bind_complete", {LARK_SCOPE_ARG: USER_A})
        binding = await repository.get_binding(USER_A)
        assert binding is not None
        for plaintext in (UAT_A, RT_A):
            assert plaintext not in binding.access_token_ciphertext
            assert plaintext not in binding.refresh_token_ciphertext
        # 可解回原值（证明确实是加密而非丢弃）
        assert await repository.decrypt_access_token(binding) == UAT_A
    finally:
        await repository.aclose()


# —— 验收 2：互不可见 / 互不串号（M1 核心）——


async def test_each_user_tool_call_injects_its_own_token(patch_exec: ProcRecorder) -> None:
    """**M1 核心**：同一服务实例下，A/B 的工具调用分别注入各自的 UAT（互不串号）。

    这是「互不可见」的**唯一可信证据**：不看文档、不看配置，只看子进程实际拿到的
    执行身份 —— 身份来自作用域（`user_id`），而非任何共享配置。
    """
    repository = await _wire_acceptance_stack()
    try:
        for user in (USER_A, USER_B):
            await _call("lark_bind_start", {LARK_SCOPE_ARG: user})
            await _call("lark_bind_complete", {LARK_SCOPE_ARG: user})

        await _call("lark_calendar_get_agenda", {LARK_SCOPE_ARG: USER_A})
        await _call("lark_calendar_get_agenda", {LARK_SCOPE_ARG: USER_B})
        await _call("lark_docs_read", {LARK_SCOPE_ARG: USER_A, "doc": "doccn123"})

        assert patch_exec.tokens() == [UAT_A, UAT_B, UAT_A]
        # 每次调用的 app_id 一致（应用是共享 OAuth 客户端），但身份是各自的人。
        assert all(env.get(ENV_APP_ID) == APP_ID for env in patch_exec.envs)
        assert all(env.get(ENV_STRICT_MODE) == "user" for env in patch_exec.envs)
        assert all(env.get(ENV_DEFAULT_AS) == "user" for env in patch_exec.envs)
    finally:
        await repository.aclose()


async def test_unbound_user_never_borrows_another_users_token(patch_exec: ProcRecorder) -> None:
    """**互不可见的关键反例**：只有 A 绑定，B 调用工具 → 未绑定，**绝不**用 A 的 token。"""
    repository = await _wire_acceptance_stack()
    try:
        await _call("lark_bind_start", {LARK_SCOPE_ARG: USER_A})
        await _call("lark_bind_complete", {LARK_SCOPE_ARG: USER_A})

        with pytest.raises(Exception) as excinfo:
            await _call("lark_calendar_get_agenda", {LARK_SCOPE_ARG: USER_B})
        message = str(excinfo.value)
        assert LARK_UNBOUND_PREFIX in message
        # 未绑定文案会进模型上下文并呈现给用户 → 内部 user_id 绝不出现（信息泄露）。
        assert USER_B not in message, f"未绑定消息泄露了内部 user_id：{message[:120]}"
        # B 的失败没有产生任何子进程（A 的 token 没被借用）
        assert patch_exec.envs == []

        # A 仍可正常使用（未受 B 的失败影响）
        await _call("lark_calendar_get_agenda", {LARK_SCOPE_ARG: USER_A})
        assert patch_exec.tokens() == [UAT_A]
    finally:
        await repository.aclose()


async def test_unbind_only_affects_that_user(patch_exec: ProcRecorder) -> None:
    """解绑的隔离性：解绑 A 后 A 未绑定、B 不受影响（各自独立生命周期）。"""
    repository = await _wire_acceptance_stack()
    try:
        for user in (USER_A, USER_B):
            await _call("lark_bind_start", {LARK_SCOPE_ARG: user})
            await _call("lark_bind_complete", {LARK_SCOPE_ARG: user})

        assert "已解除" in _text(await _call("lark_unbind", {LARK_SCOPE_ARG: USER_A}))
        with pytest.raises(Exception) as excinfo:
            await _call("lark_calendar_get_agenda", {LARK_SCOPE_ARG: USER_A})
        assert LARK_UNBOUND_PREFIX in str(excinfo.value)

        await _call("lark_calendar_get_agenda", {LARK_SCOPE_ARG: USER_B})
        assert patch_exec.tokens() == [UAT_B]  # B 照常可用，且用的是 B 自己
    finally:
        await repository.aclose()


# —— 验收 3：作用域拦截器接入真链路（身份由状态决定，非会话缓存）——


def _intercept_request(user_id: str | None, *, args: dict[str, Any] | None = None) -> Any:
    """构造真实 `MCPToolCallRequest`（拦截器线上形态）。"""
    from langchain_mcp_adapters.interceptors import MCPToolCallRequest

    state: dict[str, Any] = {"messages": [AIMessage(content="hi")]}
    if user_id is not None:
        state["user_id"] = user_id
    return MCPToolCallRequest(
        name="lark_calendar_get_agenda",
        args=dict(args or {}),
        server_name="lark_mcp",
        runtime=type("R", (), {"state": state})(),
    )


class _ArgsRecorder:
    """记录拦截器实际放行的 args（= 真正上 MCP 线的载荷）。"""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def __call__(self, request: Any) -> Any:
        self.seen.append(dict(request.args))
        return request


async def test_interceptor_overrides_model_supplied_scope() -> None:
    """模型即便在 args 里瞎填身份，也被图状态 `user_id` **无条件覆盖**（越权防线）。"""
    from agent.tools.lark_scope import UNKNOWN_SCOPE

    interceptor = LarkScopeInterceptor()
    for model_value in ({}, {LARK_SCOPE_ARG: "user-evil"}, {LARK_SCOPE_ARG: None}):
        recorder = _ArgsRecorder()
        await interceptor(_intercept_request(USER_A, args=model_value), recorder)
        assert recorder.seen[0][LARK_SCOPE_ARG] == USER_A, model_value

    # 状态无 user_id → 哨兵（服务侧必然视为未绑定），绝不留空/None 造成歧义调用
    recorder = _ArgsRecorder()
    await interceptor(_intercept_request(None), recorder)
    assert recorder.seen[0][LARK_SCOPE_ARG] == UNKNOWN_SCOPE


async def test_interceptor_reads_scope_per_call_not_per_session() -> None:
    """作用域**逐次调用**从状态读取：同一「会话」内不同用户互不串号。"""
    interceptor = LarkScopeInterceptor()
    for user in (USER_A, USER_B, USER_A):
        recorder = _ArgsRecorder()
        await interceptor(_intercept_request(user), recorder)
        assert recorder.seen[0][LARK_SCOPE_ARG] == user


async def test_interceptor_ignores_non_lark_servers() -> None:
    """非飞书服务不被注入作用域（拦截器不得污染其他 MCP 工具的参数）。"""
    from langchain_mcp_adapters.interceptors import MCPToolCallRequest

    interceptor = LarkScopeInterceptor()
    recorder = _ArgsRecorder()
    request = MCPToolCallRequest(
        name="calculate",
        args={},
        server_name="tools_mcp",
        runtime=type("R", (), {"state": {"user_id": USER_A}})(),
    )
    await interceptor(request, recorder)
    assert LARK_SCOPE_ARG not in recorder.seen[0]


def test_scope_is_hidden_from_model_view_but_not_from_execution() -> None:
    """LLM 不可见：模型侧 schema 无 `_lark_scope`，执行侧原对象仍保留该形参。"""
    from langchain_core.tools import StructuredTool

    from agent.tools.lark_scope import visible_tools

    async def _run(**kwargs: Any) -> str:  # pragma: no cover - 仅取 schema
        return "ok"

    tool = StructuredTool.from_function(
        coroutine=_run, name="lark_docs_read", description="读文档", infer_schema=False
    )
    tool.args_schema = {
        "title": "lark_docs_read",
        "type": "object",
        "properties": {"doc": {"type": "string"}, LARK_SCOPE_ARG: {"type": "string"}},
    }
    visible = visible_tools([tool])
    assert LARK_SCOPE_ARG not in visible[0].tool_call_schema.get("properties", {})
    assert "doc" in visible[0].tool_call_schema.get("properties", {})
    # 执行侧对象未被改写（拦截器注入的实参仍能被服务端接受）
    assert LARK_SCOPE_ARG in tool.tool_call_schema.get("properties", {})


# —— 验收 4：并发调用不串号 ——


async def test_concurrent_tool_calls_for_different_users_do_not_cross(
    patch_exec: ProcRecorder,
) -> None:
    """并发调用不同用户的工具 → 各自的 token 各归其位（并发下不串号）。"""
    repository = await _wire_acceptance_stack()
    try:
        for user in (USER_A, USER_B):
            await _call("lark_bind_start", {LARK_SCOPE_ARG: user})
            await _call("lark_bind_complete", {LARK_SCOPE_ARG: user})

        # 交错并发（验证无共享可变身份状态）
        scopes = [USER_A, USER_B, USER_A, USER_B, USER_A]
        await asyncio.gather(
            *(_call("lark_calendar_get_agenda", {LARK_SCOPE_ARG: s}) for s in scopes)
        )
        assert sorted(patch_exec.tokens()) == sorted([UAT_A, UAT_B, UAT_A, UAT_B, UAT_A])
    finally:
        await repository.aclose()
