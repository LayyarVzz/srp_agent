"""lark-cli 子进程调用层：命令探测 + JSON 收敛执行器 + 按用户凭据注入。

T4 飞书工具面（dev-version5.0 §3）：lark-cli 无原生 MCP 模式，CLAUDE.md 硬约束
「工具一律经 MCP 客户端接入」→ 本服务以子进程调用 lark-cli 并收敛其 JSON 输出。

v5.1（dev-version5.1.md §4/§6）改造要点：
- **按调用注入凭据**：每次工具调用 = 一个新子进程，凭据 = **一个具体的人**的 UAT，
  经 env 注入（`LARKSUITE_CLI_USER_ACCESS_TOKEN`）。官方原文的凭证链优先级为
  「自定义 Provider → env → 钥匙串」，env 命中即跳过钥匙串 → 容器内无钥匙串也可用。
- **无 bot 身份**（D5.1.4）：argv 恒 `--as user`，`as_user` 开关已删除；
  并以 `LARKSUITE_CLI_STRICT_MODE=user` / `LARKSUITE_CLI_DEFAULT_AS=user` 在 CLI 侧兜底。

安全纪律（§3.3 / v5.1 §6.1）：
- 子进程 env 白名单最小化，**不透传业务密钥**（LLM_API_KEY / XF_IAT_* 等一律隔离）；
- 只额外注入 `credentials` 里**显式声明**的 `LARKSUITE_CLI_*` 键（类型化对象，非 os.environ 透传）；
- App Secret 仅参与服务端换 token/刷新，**不进子进程 env**（本层只注入 app_id + UAT）；
- 日志只记命令组名与结果摘要，禁止记录消息体明文与任何 token。

输出契约（lark-cli 1.0.95）：`--format json`（默认）；成功 = stdout 含 `"ok": true`
且退出码 0；错误进 stderr 且非零退出码（错误对象含 type/code/message/hint）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Protocol

from services.lark_mcp.models import LarkCliCredentials
from shared.lark.errors import LarkCliError, LarkUnboundError

logger = logging.getLogger(__name__)

# Windows 下 npm shim 是 .cmd/.bat（CreateProcess 不能直接执行），须经 cmd.exe 中转。
_SCRIPT_EXTS = (".cmd", ".bat")

# 子进程 env 白名单：仅保留 CLI 运行所需系统项，
# 不透传业务密钥（LLM_API_KEY / EMBEDDING_API_KEY / XF_IAT_* 等一律隔离）。
#
# WHY 不含 USERPROFILE / APPDATA / HOMEDRIVE（v5.1 §6.1 实测基线）：剔除钥匙串访问
# 路径后子进程读不到 CLI 本地凭据库，**环境变量凭证链仍正常命中并成功执行**
# —— 这组 env 即「容器内可用的最小充分集」，也保证身份不会悄悄回落到本机既有登录。
# 例外：`LARKSUITE_CLI_CONFIG_DIR` 若由部署显式给出则透传（路径非凭据；供容器固化配置目录）。
_ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "COMPUTERNAME",
    "OS",
    "LANG",
    "LC_ALL",
    "TERM",
    "LARKSUITE_CLI_CONFIG_DIR",
)

# —— LARKSUITE_CLI_* 键名常量（集中声明，禁止散落字符串字面量）——
# S105 不适用：下列常量是**环境变量名**（CLI 读取的键名由官方二进制固化），非密钥值。
ENV_APP_ID = "LARKSUITE_CLI_APP_ID"
ENV_APP_SECRET = "LARKSUITE_CLI_APP_SECRET"  # noqa: S105
ENV_USER_ACCESS_TOKEN = "LARKSUITE_CLI_USER_ACCESS_TOKEN"  # noqa: S105
ENV_STRICT_MODE = "LARKSUITE_CLI_STRICT_MODE"
ENV_DEFAULT_AS = "LARKSUITE_CLI_DEFAULT_AS"

# CLI 侧身份策略（§4.4）：只允许 user 身份、默认 user —— 部署期固化，随容器重建保持。
STRICT_MODE_USER = "user"
DEFAULT_AS_USER = "user"


class LarkCredentialProvider(Protocol):
    """作用域 → 凭据的解析契约（服务侧可注入实现；测试注入 fake）。

    实现负责：按 `scope`（调用方 `user_id`）查绑定、必要时刷新并落库；
    未绑定 / 刷新失败一律抛 `LarkUnboundError`（§6.3 最小阻断判据）。
    """

    async def resolve(self, scope: str) -> LarkCliCredentials:
        """解析该作用域的可用凭据；未绑定 → `LarkUnboundError`。"""
        ...


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_lark_cli_env(*, credentials: LarkCliCredentials | None = None) -> dict[str, str]:
    """按白名单构造子进程环境变量（安全纪律：env 最小化，不继承无关密钥）。

    `credentials` 非空时额外注入该用户的 `LARKSUITE_CLI_*` 键（**唯一执行身份**）；
    无论是否注入凭据，两个身份策略项恒定注入（§4.4）。

    WHY 白名单仍是白名单：`credentials` 是**类型化对象**，只有 app_id 与 UAT
    会进入 env（app_secret 不注入——它只参与服务端换 token/刷新），
    既保证「不继承无关密钥」的既有断言，又杜绝 os.environ 透传。
    """
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    env["FASTMCP_CHECK_FOR_UPDATES"] = "off"
    env[ENV_STRICT_MODE] = STRICT_MODE_USER
    env[ENV_DEFAULT_AS] = DEFAULT_AS_USER
    if credentials is not None:
        env[ENV_APP_ID] = credentials.app_id
        env[ENV_USER_ACCESS_TOKEN] = credentials.user_access_token
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
    飞书 API 网络耗时；且天然隔离状态（**每用户的凭据互不残留**）、崩溃互不影响。
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

    async def run(
        self,
        args: list[str],
        *,
        credentials: LarkCliCredentials,
    ) -> dict[str, Any]:
        """执行一条 lark-cli 命令并返回解析后的 JSON 载荷。

        args 只含域级命令组与业务参数（不含 `--as` / `--format`，由本层统一追加）；
        身份恒为 user（`--as user`），凭据经 `credentials` 注入子进程 env。

        WHY `credentials` 必填：纯 user 模式下不存在 bot 兜底，缺少该用户的 UAT
        就没有任何合法执行身份 —— 调用方必须在入口处确定性拒绝（§6.3），
        本层以必填形参把这条纪律固定在类型上（而非运行期判空）。
        """
        if not credentials.is_complete():
            # 防御性兜底：调用方（服务侧）已按作用域解析并保证完整，
            # 走到这里说明装配错误 —— 按未绑定语义抛出，绝不退化为匿名调用。
            raise LarkUnboundError("飞书凭据不完整（缺少 user_access_token）")
        if self._command is None:
            raise LarkCliError(
                "lark-cli 命令不可用（启动时未探测到，请检查安装或 LARK_CLI_COMMAND）"
            )
        argv = _build_exec_args(
            self._command, [*args, "--as", STRICT_MODE_USER, "--format", "json"]
        )
        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=build_lark_cli_env(credentials=credentials),
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
        detail = (
            payload.get("error")
            or payload.get("message")
            or json.dumps(payload, ensure_ascii=False)
        )
        return f"lark-cli {command_group} 返回失败：{_truncate(str(detail))}"
    return f"lark-cli {command_group} 返回了非预期的输出结构"


__all__ = [
    "DEFAULT_AS_USER",
    "ENV_APP_ID",
    "ENV_APP_SECRET",
    "ENV_DEFAULT_AS",
    "ENV_STRICT_MODE",
    "ENV_USER_ACCESS_TOKEN",
    "STRICT_MODE_USER",
    "LarkCliError",
    "LarkCliRunner",
    "LarkCredentialProvider",
    "LarkUnboundError",
    "build_lark_cli_env",
    "resolve_lark_cli_command",
]
