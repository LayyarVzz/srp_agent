# lark_mcp — 飞书工具面（T4，dev-version5.0 §3）

把飞书官方 CLI（`lark-cli`）封装为 5 个最简 MCP 工具，Agent 经 `MultiServerMCPClient`
接入（图零改动）。lark-cli 无原生 MCP 模式，本服务即「自建 FastMCP + 子进程调 CLI」
的合规接入层（决策 D5.1）。

## 工具集（最精简 5 个）

| 工具 | CLI 命令面 | 读/写 | 说明 |
|---|---|---|---|
| `lark_im_send_message` | `im +messages-send` | **写** | 收件人 `chat_id`/`user_id` 二选一 + `text`；`as_user` 默认 bot 身份 |
| `lark_calendar_get_agenda` | `calendar +agenda` | 读 | 默认今天，`start`/`end` ISO 8601；查"我的日程"需 `as_user=true` |
| `lark_task_create` | `task +create` | **写** | `summary` 必填，`due` 支持 ISO 8601 / `date:YYYY-MM-DD` / `+2d` |
| `lark_docs_read` | `docs +fetch` | 读 | `doc` 传 URL 或 token，返回 markdown（超长截断） |
| `lark_contact_resolve_name` | `contact +search-user` | 读 | 姓名/邮箱 → `open_id` + `p2p_chat_id`；**固定 user 身份**（接口限制） |

> 5 个命令风险级别均为 read/write（无 high-risk-write），无需 `--yes`；
> 未来接入 high-risk-write 命令时须由服务层显式传 `--yes` 并在 description 声明风险。

## 安装（项目级，不污染全局）

npm 包只是「下载器 + 垫片」，postinstall 会把平台二进制下载到包内 `bin/`：

```bash
npm install --prefix .tools/lark @larksuite/cli@1.0.95
# 二进制：.tools/lark/node_modules/@larksuite/cli/bin/lark-cli.exe
```

`.tools/` 已在 `.gitignore`（不入库；卸载 = 删目录）。`.env` 配置：

```bash
LARK_CLI_COMMAND=.tools/lark/node_modules/@larksuite/cli/bin/lark-cli.exe
```

> 不要使用官方 `npx @larksuite/cli@latest install` 向导：其第一步强制
> `npm install -g`（全局污染 PATH），还会全局安装 AI Skills（本项目用不上）。

## 一次性鉴权（凭据存 OS 钥匙串，服务进程不接触 token）

用项目内二进制直接执行（Windows/Git Bash 下）：

```bash
LARK=.tools/lark/node_modules/@larksuite/cli/bin/lark-cli.exe

# 1) 绑定飞书开放平台应用（bot 身份底座；需要 App ID / App Secret）
$LARK config init --new      # 输出授权 URL → 浏览器完成 → 自动退出

# 2) 登录用户身份（device flow；im/contact 等需要 user_access_token）
$LARK auth login --domain im,calendar,task,docs,contact --recommend

# 3) 自检
$LARK auth check             # exit 0 = ok
$LARK auth status
```

身份语义：默认 bot（应用 tenant token）；`as_user=true` / contact 搜索走登录用户的
user token。**user 身份未登录时相关工具会报错**（`auth login` 后恢复）。

## 运行与装配

- Agent 侧门控登记（`agent/runtime.py::_build_mcp_servers`）：`cfg.lark.enabled`
  **且** `LARK_CLI_ENABLED` **且** 命令探测成功（`resolve_lark_cli_command`）才注册
  `lark_mcp` 服务；否则跳过并记日志——**无 CLI 环境零回归**。
- 独立运行：`uv run python -m services.lark_mcp`（stdio；配
  `MCP_TRANSPORT=streamable-http` 切 HTTP，默认端口 8101）。
- 演示：`uv run python -m scripts.demo_lark`。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `LARK_CLI_ENABLED` | `true` | Agent 侧登记总开关（settings.py） |
| `LARK_CLI_COMMAND` | 空 | 显式指定 CLI 路径（相对路径按仓库根解析）；空则探测 PATH |
| `LARK_CLI_TIMEOUT_S` | `30` | 单次 CLI 子进程超时 |
| `LARK_OUTPUT_MAX_CHARS` | `10000` | 工具输出长度上限 |
| `MCP_TRANSPORT` 等 | `stdio` | 与 tools_mcp 同名的传输/地址配置（HTTP 模式端口 8101） |

## 排障

- **找不到命令**：`.env` 配 `LARK_CLI_COMMAND` 指向 `lark-cli.exe` 绝对/相对路径；
  Windows 下不要指向 npm 的 `.cmd` 垫片（服务层会经 `cmd.exe` 中转，但直接指 `.exe` 最稳）。
- **`--as user` 报鉴权错误**：先 `auth login`；token 过期重新 `auth login` 即可。
- **bot 收不到/发不出消息**：确认应用（bot）在目标会话中，且开通了
  `im:message` 相关 scope；日程/任务/文档同理按域开通 scope。
- **日志纪律**：服务日志只记命令组名与结果摘要，不记录消息体明文。
