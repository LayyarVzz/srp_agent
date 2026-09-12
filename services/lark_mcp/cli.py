"""lark-cli 子进程调用层：命令探测 + JSON 收敛执行器。

T4 飞书工具面（dev-version5.0 §3）：lark-cli 无原生 MCP 模式，CLAUDE.md 硬约束
「工具一律经 MCP 客户端接入」→ 本服务以子进程调用 lark-cli 并收敛其 JSON 输出。

安全纪律（§3.3）：
- lark-cli 的 token 存于 CLI 自身凭据库（OS 钥匙串），本进程不接触明文 token；
- 子进程 env 白名单最小化，不继承无关密钥（如 LLM_API_KEY）；
- 日志只记命令组名与结果摘要，禁止记录消息体明文敏感字段。

输出契约（lark-cli 1.0.95）：`--format json`（默认）；成功 = stdout 含 `"ok": true`
且退出码 0；错误进 stderr 且非零退出码（错误对象含 type/code/message/hint）。
lark-cli 1.0.95 虽支持 `--jq` 过滤，但 JSON 解析收敛在 Python 侧更利于截断与
结构化判错，故不依赖 `--jq`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Windows 下 npm shim 是 .cmd/.bat（CreateProcess 不能直接执行），须经 cmd.exe 中转。
_SCRIPT_EXTS = (".cmd", ".bat")

# 子进程 env 白名单：仅保留 CLI 运行与 OS 钥匙串访问所需系统项，
# 不透传业务密钥（LLM_API_KEY / EMBEDDING_API_KEY / XF_IAT_* 等一律隔离）。
_ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "TEMP",
    "TMP",
    "COMPUTERNAME",
    "OS",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
)


class LarkCliError(RuntimeError):
    """lark-cli 调用失败（非零退出 / 输出解析失败 / 超时 / ok=false）。

    由 FastMCP 统一转为工具错误 → Agent 侧 ToolNode(status="error")
    → `tool_error.execution`（可路由降级）；参数缺失语义由工具 schema 层保留。
    """


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_lark_cli_env() -> dict[str, str]:
    """按白名单构造子进程环境变量（安全纪律：env 最小化，不继承无关密钥）。"""
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    env["FASTMCP_CHECK_FOR_UPDATES"] = "off"
    return env


def resolve_lark_cli_command(override: str | None = None) -> str | None:
    """探测可执行的 lark-cli 命令路径；未探测到返回 None（调用方门控跳过）。

    解析顺序（dev-version5.0 §2.1）：`LARK_CLI_COMMAND` 显式覆盖 > PATH 探测
    （`shutil.which("lark-cli")`，Windows 下经 PATHEXT 命中 lark-cli.exe/.cmd）。
    覆盖值支持相对路径（相对仓库根解析，项目级安装场景
    `.tools/lark/node_modules/@larksuite/cli/bin/lark-cli.exe`）。
    """
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = _get_repo_root() / candidate
        if not candidate.exists():
            logger.warning("LARK_CLI_COMMAND 指向的路径不存在：%s", candidate)
            return None
        return str(candidate)
    found = shutil.which("lark-cli")
    if found is None:
        logger.info("PATH 中未探测到 lark-cli（可用 LARK_CLI_COMMAND 显式指定）")
    return found


def _build_exec_args(command: str, args: list[str]) -> list[str]:
    """构造 CreateProcess 可直接执行的 argv；脚本 shim（.cmd/.bat）经 cmd.exe 中转。"""
    if command.lower().endswith(_SCRIPT_EXTS):
        return ["cmd.exe", "/c", command, *args]
    return [command, *args]


class LarkCliRunner:
    """lark-cli 子进程执行器：每次工具调用拉起一次 CLI，收敛 stdout JSON。

    WHY 每次调用独立子进程：CLI 无服务端模式，进程级开销（数十 ms）远小于
    飞书 API 网络耗时；且天然隔离状态、崩溃互不影响。
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        timeout_s: float = 30.0,
        max_output_chars: int = 10_000,
    ) -> None:
        # 启动时解析一次；None 不视为致命——门控装配已保证注册时命令存在，
        # 运行期丢失（如 PATH 变更）在调用时报错并走既有降级路径。
        self._command = resolve_lark_cli_command(command)
        self._timeout_s = timeout_s
        self._max_output_chars = max_output_chars

    @property
    def command(self) -> str | None:
        return self._command

    @property
    def max_output_chars(self) -> int:
        return self._max_output_chars

    async def run(self, args: list[str], *, as_user: bool = False) -> dict[str, Any]:
        """执行一条 lark-cli 命令并返回解析后的 JSON 载荷。

        args 只含域级命令组与业务参数（不含 `--as` / `--format`，由本层统一追加）；
        `as_user=False` 默认 bot 身份（dev-version5.0 §3.3），contact 域等仅支持
        user 身份的命令由工具函数显式传 `as_user=True`。
        """
        if self._command is None:
            raise LarkCliError(
                "lark-cli 命令不可用（启动时未探测到，请检查安装或 LARK_CLI_COMMAND）"
            )
        argv = _build_exec_args(
            self._command, [*args, "--as", "user" if as_user else "bot", "--format", "json"]
        )
        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=build_lark_cli_env(),
            )
        except OSError as exc:
            raise LarkCliError(f"lark-cli 进程启动失败：{exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise LarkCliError(f"lark-cli 调用超时（>{self._timeout_s:.0f}s）") from None
        elapsed = time.perf_counter() - started
        # 日志纪律：只记命令组名（前两段）与结果摘要，不记业务参数（消息体等）。
        command_group = " ".join(args[:2]) if len(args) >= 2 else (args[0] if args else "<empty>")
        if proc.returncode != 0:
            logger.warning(
                "lark-cli %s 失败（exit=%s，%.2fs）", command_group, proc.returncode, elapsed
            )
            raise LarkCliError(_summarize_failure(command_group, stderr))
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("lark-cli %s 输出解析失败（%.2fs）", command_group, elapsed)
            raise LarkCliError(f"lark-cli 输出不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            logger.warning("lark-cli %s 返回 ok!=true（%.2fs）", command_group, elapsed)
            raise LarkCliError(_summarize_payload_failure(command_group, payload))
        logger.info("lark-cli %s 成功（%.2fs）", command_group, elapsed)
        return payload


def _truncate(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…[截断]"


def _summarize_failure(command_group: str, stderr: bytes) -> str:
    """把 CLI 失败摘要收敛为单行消息（stderr 可能是 JSON 错误对象或纯文本）。"""
    raw = stderr.decode("utf-8", errors="replace").strip()
    if raw:
        try:
            err = json.loads(raw)
        except json.JSONDecodeError:
            return f"lark-cli {command_group} 失败：{_truncate(raw)}"
        if isinstance(err, dict):
            detail = err.get("message") or err.get("code") or raw
            return f"lark-cli {command_group} 失败：{_truncate(str(detail))}"
    return f"lark-cli {command_group} 失败（无错误输出）"


def _summarize_payload_failure(command_group: str, payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("message") or json.dumps(
            payload, ensure_ascii=False
        )
        return f"lark-cli {command_group} 返回失败：{_truncate(str(detail))}"
    return f"lark-cli {command_group} 返回了非预期的输出结构"


__all__ = [
    "LarkCliError",
    "LarkCliRunner",
    "build_lark_cli_env",
    "resolve_lark_cli_command",
]
