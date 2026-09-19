"""services/lark_mcp/cli.py 单测：LarkCliRunner 子进程契约（假进程，零真实 CLI、零网络）。

WHY 不拉真实桩进程：测试循环（conftest 统一 SelectorEventLoop，psycopg 需要）在
Windows 下不支持 asyncio 子进程；改以 monkeypatch 捕获 create_subprocess_exec 的
argv/env 并模拟 stdout/stderr/returncode，同样覆盖「--as user/--format 追加」
「env 白名单 + 按用户凭据注入（注入 + 隔离**双断言**）」「JSON 收敛」「超时杀进程」
「未绑定确定性拒绝」契约。真实 CLI 端到端见 demo 与真机验收。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from services.lark_mcp.cli import (
    ENV_APP_ID,
    ENV_DEFAULT_AS,
    ENV_STRICT_MODE,
    ENV_USER_ACCESS_TOKEN,
    LarkCliError,
    LarkCliRunner,
    _build_exec_args,
    build_lark_cli_env,
    resolve_lark_cli_command,
)
from services.lark_mcp.models import LarkCliCredentials
from shared.lark.errors import LarkUnboundError

CREDS = LarkCliCredentials(
    app_id="cli_test_app",
    app_secret="secret-should-never-enter-env",
    user_access_token="u-user-a-token",
)


class FakeProc:
    """模拟 asyncio 子进程：记录 kill 调用，communicate 按预设延迟返回结果。"""

    def __init__(
        self,
        *,
        stdout: bytes = b'{"ok": true, "data": "stub"}',
        stderr: bytes = b"",
        returncode: int = 0,
        delay_s: float = 0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._delay_s = delay_s
        self.killed = False

    @property
    def returncode(self) -> int:
        return self._returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self._returncode


class ProcRecorder:
    """捕获 create_subprocess_exec 调用（argv/env），按序回放预设进程。

    WHY async __call__：create_subprocess_exec 是协程函数（须 await），替身需同构。
    """

    def __init__(self, procs: list[FakeProc] | None = None) -> None:
        self.procs = procs if procs is not None else [FakeProc()]
        self.argvs: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    async def __call__(self, *argv: Any, **kwargs: Any) -> FakeProc:
        self.argvs.append([str(a) for a in argv])
        self.envs.append(kwargs["env"])
        return self.procs.pop(0)


@pytest.fixture
def patch_exec(monkeypatch: pytest.MonkeyPatch) -> ProcRecorder:
    recorder = ProcRecorder()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)
    return recorder


@pytest.fixture
def fake_cli(tmp_path: Path) -> str:
    """存在的假命令路径（构造器会校验存在性；执行由 patch_exec 拦截，不会真跑）。"""
    stub = tmp_path / "lark-cli.exe"
    stub.write_bytes(b"")
    return str(stub)


async def test_run_success_appends_user_identity_and_format(
    patch_exec: ProcRecorder, fake_cli: str
) -> None:
    """成功路径：返回解析后的 JSON 载荷，argv 末尾统一追加 --as user --format json。

    v5.1：身份**恒为 user**（无 bot 身份，D5.1.4）——原 `--as bot` 断言按 §4.4 修正。
    """
    runner = LarkCliRunner(command=fake_cli, timeout_s=10.0)
    payload = await runner.run(["im", "+messages-send", "--text", "hi"], credentials=CREDS)
    assert payload == {"ok": True, "data": "stub"}
    assert patch_exec.argvs == [
        [
            fake_cli,
            "im",
            "+messages-send",
            "--text",
            "hi",
            "--as",
            "user",
            "--format",
            "json",
        ]
    ]


async def test_run_injects_credentials_and_keeps_isolation(
    patch_exec: ProcRecorder, monkeypatch: pytest.MonkeyPatch, fake_cli: str
) -> None:
    """注入 + 隔离**双断言**（v5.1 §6.1）：

    - 注入：该用户的 UAT 与应用 ID、以及两个身份策略项出现在子进程 env；
    - 隔离：App Secret **不进 env**（只参与服务端换 token/刷新），业务密钥（LLM_API_KEY /
      XF_IAT_API_SECRET）不透传，钥匙串访问路径（USERPROFILE/APPDATA/HOMEDRIVE）被剔除
      —— 这组 env 即容器内可用的最小充分集。
    """
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("XF_IAT_API_SECRET", "top-secret")
    monkeypatch.setenv("USERPROFILE", r"C:\Users\someone")
    monkeypatch.setenv("APPDATA", r"C:\Users\someone\AppData\Roaming")
    monkeypatch.setenv("HOMEDRIVE", "C:")
    runner = LarkCliRunner(command=fake_cli, timeout_s=10.0)
    await runner.run(["calendar", "+agenda"], credentials=CREDS)
    env = patch_exec.envs[0]
    # 注入断言
    assert env[ENV_USER_ACCESS_TOKEN] == "u-user-a-token"
    assert env[ENV_APP_ID] == "cli_test_app"
    assert env[ENV_STRICT_MODE] == "user"  # CLI 侧禁止 bot 身份
    assert env[ENV_DEFAULT_AS] == "user"
    # 隔离断言
    assert "LARKSUITE_CLI_APP_SECRET" not in env
    assert "secret-should-never-enter-env" not in env.values()
    assert "LLM_API_KEY" not in env
    assert "XF_IAT_API_SECRET" not in env
    for keyring_path in ("USERPROFILE", "APPDATA", "HOMEDRIVE"):
        assert keyring_path not in env, keyring_path
    assert env["FASTMCP_CHECK_FOR_UPDATES"] == "off"


async def test_run_without_credentials_raises_unbound(
    patch_exec: ProcRecorder, fake_cli: str
) -> None:
    """凭据缺 UAT（未绑定）→ LarkUnboundError（引导绑定），且**不启动子进程**。

    纯 user 模式下不存在 bot 兜底：不得退化为匿名/应用身份调用（§6.3）。
    """
    runner = LarkCliRunner(command=fake_cli, timeout_s=10.0)
    incomplete = LarkCliCredentials(app_id="a", app_secret="b", user_access_token="")
    with pytest.raises(LarkUnboundError):
        await runner.run(["im", "+messages-send"], credentials=incomplete)
    assert patch_exec.argvs == []


async def test_nonzero_exit_raises_with_stderr_summary(
    monkeypatch: pytest.MonkeyPatch, fake_cli: str
) -> None:
    """非零退出 → LarkCliError，stderr JSON 错误对象的 message 进摘要。"""
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        ProcRecorder(
            [
                FakeProc(
                    stdout=b"",
                    stderr=b'{"type":"api","code":99991672,"message":"token invalid"}',
                    returncode=1,
                )
            ]
        ),
    )
    runner = LarkCliRunner(command=fake_cli, timeout_s=10.0)
    with pytest.raises(LarkCliError, match="token invalid"):
        await runner.run(["im", "+messages-send"], credentials=CREDS)


async def test_invalid_json_output_raises(patch_exec: ProcRecorder, fake_cli: str) -> None:
    """stdout 非法 JSON（exit 0）→ LarkCliError。"""
    patch_exec.procs = [FakeProc(stdout=b"not-json-at-all")]
    runner = LarkCliRunner(command=fake_cli, timeout_s=10.0)
    with pytest.raises(LarkCliError, match="JSON"):
        await runner.run(["calendar", "+agenda"], credentials=CREDS)


async def test_ok_false_payload_raises(patch_exec: ProcRecorder, fake_cli: str) -> None:
    """ok!=true（exit 0）→ LarkCliError（成功判定以 ok==true 为准，不猜退出码语义）。"""
    patch_exec.procs = [FakeProc(stdout=b'{"ok": false, "error": {"message": "denied"}}')]
    runner = LarkCliRunner(command=fake_cli, timeout_s=10.0)
    with pytest.raises(LarkCliError, match="denied"):
        await runner.run(["task", "+create"], credentials=CREDS)


async def test_timeout_kills_process(monkeypatch: pytest.MonkeyPatch, fake_cli: str) -> None:
    """communicate 超过 timeout_s → 进程被 kill 并抛 LarkCliError（超时）。"""
    proc = FakeProc(delay_s=2.0)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", ProcRecorder([proc]))
    runner = LarkCliRunner(command=fake_cli, timeout_s=0.2)
    with pytest.raises(LarkCliError, match="超时"):
        await runner.run(["docs", "+fetch"], credentials=CREDS)
    assert proc.killed is True


def test_build_exec_args_wraps_cmd_shim() -> None:
    """.cmd/.bat 垫片（npm shim）CreateProcess 不能直接执行，须经 cmd.exe 中转。"""
    argv = _build_exec_args("C:/x/lark-cli.cmd", ["im", "+messages-send"])
    assert argv == ["cmd.exe", "/c", "C:/x/lark-cli.cmd", "im", "+messages-send"]
    argv2 = _build_exec_args("C:/x/lark-cli.exe", ["im", "+messages-send"])
    assert argv2 == ["C:/x/lark-cli.exe", "im", "+messages-send"]


def test_env_allowlist_keys() -> None:
    """白名单构造：仅收敛系统项（不读 .env），并关闭 fastmcp 更新检查。"""
    env = build_lark_cli_env()
    assert env["FASTMCP_CHECK_FOR_UPDATES"] == "off"
    assert env[ENV_STRICT_MODE] == "user"
    assert env[ENV_DEFAULT_AS] == "user"
    # 未传凭据时不注入 LARKSUITE_CLI_APP_ID / UAT（凭据永不隐式取自 os.environ）。
    assert ENV_APP_ID not in env
    assert ENV_USER_ACCESS_TOKEN not in env
    for forbidden in ("LLM_API_KEY", "EMBEDDING_API_KEY", "XF_IAT_API_SECRET"):
        assert forbidden not in env


def test_resolve_command_override_absolute(tmp_path) -> None:
    """LARK_CLI_COMMAND 显式覆盖（绝对路径存在）优先于 PATH 探测。"""
    stub = tmp_path / "lark-cli.exe"
    stub.write_bytes(b"")
    assert resolve_lark_cli_command(str(stub)) == str(stub)


def test_resolve_command_override_relative_to_repo_root() -> None:
    """相对路径覆盖按仓库根解析（项目级安装场景）。"""
    resolved = resolve_lark_cli_command("pyproject.toml")
    assert resolved is not None
    assert Path(resolved).is_absolute()
    assert resolved.endswith("pyproject.toml")


def test_resolve_command_override_missing_returns_none(tmp_path) -> None:
    """覆盖路径不存在 → None（调用方门控跳过登记，零回归）。"""
    assert resolve_lark_cli_command(str(tmp_path / "nope.exe")) is None


def test_resolve_command_falls_back_to_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """无覆盖 → shutil.which("lark-cli") 探测；探测不到返回 None。"""
    monkeypatch.setattr(
        "services.lark_mcp.cli.shutil.which",
        lambda name: "/usr/bin/lark-cli" if name == "lark-cli" else None,
    )
    assert resolve_lark_cli_command(None) == "/usr/bin/lark-cli"
    monkeypatch.setattr("services.lark_mcp.cli.shutil.which", lambda name: None)
    assert resolve_lark_cli_command(None) is None


async def test_runner_without_command_raises_on_run(fake_cli: str) -> None:
    """启动时未解析到命令（路径不存在）→ 调用时抛 LarkCliError（门控已拦截，运行期兜底）。"""
    runner = LarkCliRunner(command="/nope/lark-cli.exe", timeout_s=5.0)
    with pytest.raises(LarkCliError, match="不可用"):
        await runner.run(["im", "+messages-send"], credentials=CREDS)
