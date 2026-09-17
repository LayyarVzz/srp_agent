# lark_mcp — 飞书工具面 + 账号绑定（dev-version5.0 §3 / dev-version5.1 §4-§8）

把飞书官方 CLI（`lark-cli`）封装为 MCP 工具，Agent 经 `MultiServerMCPClient` 接入
（图零改动）。lark-cli 无原生 MCP 模式，本服务即「自建 FastMCP + 子进程调 CLI」
的合规接入层（决策 D5.1）。

**v5.1 身份模型（重要变更）**：飞书身份**按 Agent 用户隔离、互不可见**。

- **无 bot 身份**（D5.1.4）：argv 恒 `--as user`，原 `as_user` 开关**已删除**
  —— 身份不再是「LLM 可猜的开关」，而是「谁在调用」；
- 应用**仅作 OAuth 客户端**：`app_id`/`app_secret` 只参与设备码授权、换码与刷新；
- 每个用户经**对话内设备码授权**绑定各自的飞书账号，`user_access_token` 加密落库；
- 工具的 `_lark_scope` 由 agent 侧拦截器从图状态 `user_id` **无条件注入**，
  且已从模型可见的工具 schema 剥离 → **LLM 看不到、也填不了**。

## 工具集

### 业务工具（5 个，v5.0 原有）

| 工具 | CLI 命令面 | 读/写 | 说明 |
|---|---|---|---|
| `lark_im_send_message` | `im +messages-send` | **写** | 收件人 `chat_id`/`user_id` 二选一 + `text`；以**绑定用户本人**身份发送 |
| `lark_calendar_get_agenda` | `calendar +agenda` | 读 | 默认今天，`start`/`end` ISO 8601；查**该用户自己的**日程 |
| `lark_task_create` | `task +create` | **写** | `summary` 必填，`due` 支持 ISO 8601 / `date:YYYY-MM-DD` / `+2d` |
| `lark_docs_read` | `docs +fetch` | 读 | `doc` 传 URL 或 token，返回 markdown（超长截断） |
| `lark_contact_resolve_name` | `contact +search-user` | 读 | 姓名/邮箱 → `open_id` + `p2p_chat_id`；可见范围即该用户本人的范围 |

> 5 个命令风险级别均为 read/write（无 high-risk-write），无需 `--yes`；
> 未来接入 high-risk-write 命令时须由服务层显式传 `--yes` 并在 description 声明风险。

### 绑定工具（4 个，v5.1 新增）

| 工具 | 说明 |
|---|---|
| `lark_bind_start` | 发起设备码授权 → 返回可打开的验证链接（10 分钟内有效） |
| `lark_bind_complete` | **轮询一次**换码 → 拿到 UAT + refresh_token → 加密落库并回报账号名 |
| `lark_bind_status` | 已绑定？绑的谁？令牌剩余有效期？（**绝不回传 token**） |
| `lark_unbind` | 尽力撤销远端授权 + 清除本地绑定（高风险写：仅在用户明确要求时调用） |

**为什么 `lark_bind_complete` 是「单次轮询」而不是「阻塞等到授权为止」**：MCP 工具调用
有超时（`MCP_TIMEOUT_S`，默认 10s），而用户扫码常要几十秒到几分钟，阻塞式等待必然超时。
故设计为**可重复调用的状态推进**：未授权 → 返回「待授权」引导（正常控制流，非错误），
用户说「我已完成授权」→ 再调一次即完成绑定。

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

## 配置（v5.1 按用户绑定）

```bash
# 飞书开放平台应用（仅作 OAuth 客户端；工具调用恒为 user 身份，无 bot）
LARK_APP_ID=cli_xxxxxxxxxxxx
LARK_APP_SECRET=xxxxxxxxxxxxxxxx

# 令牌加密密钥（Fernet 派生源）——**必须配置**，否则绑定功能 fail-closed 关闭
# 生成：python -c "import secrets;print(secrets.token_hex(32))"
LARK_TOKEN_KEY=xxxxxxxxxxxxxxxx

# 绑定记录所在库（不配 → SQLite memory，仅本地演示；生产必须 Postgres）
DATABASE_URL=postgresql://user:pass@host:5432/srp_agent

# 品牌域名（默认飞书；Lark 国际版改成 larksuite.com 两组）
LARK_ACCOUNTS_BASE_URL=https://accounts.feishu.cn
LARK_OPEN_BASE_URL=https://open.feishu.cn
```

### 应用侧需要开通的 scope

代码里显式声明的最小集（`shared/lark/oauth.py::LARK_MINIMAL_SCOPES`）：

```
offline_access                 # 必须！否则拿不到 refresh_token，UAT 2h 后强制重绑
calendar:calendar
task:task
im:message
contact:user.base:readonly
docx:document:readonly
```

### 配置态单用户模式（零回归路径）

不配 `LARK_TOKEN_KEY`（或 `LARK_BINDING_ENABLED=false`）时，绑定功能**整体关闭**，
服务退回「配置态单用户凭据」：从 `LARK_USER_ACCESS_TOKEN` + `LARK_APP_ID/SECRET` 读取
一份固定凭据（本地 demo / 单机验证用）。此时**多用户隔离不可用**，装配日志会明确警告。

## 运行与装配

- Agent 侧门控登记（`agent/runtime.py::_build_mcp_servers`）：`cfg.lark.enabled`
  **且** `LARK_CLI_ENABLED` **且** 命令探测成功（`resolve_lark_cli_command`）才注册
  `lark_mcp` 服务；否则跳过并记日志——**无 CLI 环境零回归**。
- 服务侧装配（`services/lark_mcp/server.py::_build_binding_pieces`）：有应用凭据 +
  `lark_token_key` → 构造 `BindingCredentialProvider`（按用户隔离）；否则 →
  `ConfigCredentialProvider`（配置态单用户）。
- 独立运行：`uv run python -m services.lark_mcp`（stdio；配
  `MCP_TRANSPORT=streamable-http` 切 HTTP，默认端口 8101）。
- 演示：
  - `uv run python -m scripts.demo_lark` —— 业务工具面；
  - `uv run python -m scripts.demo_lark_binding` —— **多用户绑定全链路**（默认离线，
    零网络零真实应用；`--live` 用真实飞书应用并人工扫码）。

## 数据模型

两张表（`shared/lark/repository.py`，SQLAlchemy；dev=SQLite memory / prod=Postgres）：

- `lark_bindings`：`user_id`(PK) → `open_id` / `user_name` / **加密的** UAT + refresh_token
  / `expires_at` / `refresh_expires_at` / `status` / `version`（乐观锁）；
- `lark_device_flows`：`user_id`(PK) → `device_code` / `user_code` / `flow_id` /
  `verification_uri_complete` / `expires_at` / `interval_s`（设备码待定态）。

**令牌安全**：列类型用 `Text`（实测 UAT 8093 字符、refresh 8254 字符，加密后更长）；
密钥缺失 → fail-closed；token 不写日志、不进 `ToolMessage`、不回传前端。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `LARK_CLI_ENABLED` | `true` | Agent 侧登记总开关（settings.py） |
| `LARK_CLI_COMMAND` | 空 | 显式指定 CLI 路径（相对路径按仓库根解析）；空则探测 PATH |
| `LARK_CLI_TIMEOUT_S` | `30` | 单次 CLI 子进程超时 |
| `LARK_OUTPUT_MAX_CHARS` | `10000` | 工具输出长度上限 |
| `LARK_APP_ID` / `LARK_APP_SECRET` | 空 | 飞书应用（OAuth 客户端） |
| `LARK_TOKEN_KEY` | 空 | 令牌加密密钥；**空 → 绑定 fail-closed 关闭** |
| `LARK_BINDING_ENABLED` | `true` | 绑定总开关（密钥缺失时仍自动关闭） |
| `LARK_USER_ACCESS_TOKEN` | 空 | 配置态单用户凭据（仅无绑定模式） |
| `LARK_ACCOUNTS_BASE_URL` | `https://accounts.feishu.cn` | 认证族域名 |
| `LARK_OPEN_BASE_URL` | `https://open.feishu.cn` | 业务族域名 |
| `LARK_TOKEN_REFRESH_SKEW_S` | `300` | 提前刷新窗口（距过期 < 阈值即刷新） |
| `LARK_OAUTH_TIMEOUT_S` | `15` | OAuth HTTP 调用超时 |
| `LARK_DEVICE_FLOW_POLL_MAX_S` | `600` | 设备码轮询总上限 |
| `MCP_TRANSPORT` 等 | `stdio` | 与 tools_mcp 同名的传输/地址配置（HTTP 模式端口 8101） |

## OAuth 端点（实测固化，勿改回 v1）

认证族与业务族**分属两个域名**：

| 用途 | 端点 |
|---|---|
| 设备码发起 | `POST {accounts_base}/oauth/v1/device_authorization` |
| 用户验证页 | `{accounts_base}/oauth/v1/device/verify?flow_id=&user_code=` |
| 换码 / 刷新 | `POST {open_base}/open-apis/authen/v2/oauth/token` |
| 用户信息 | `GET {open_base}/open-apis/authen/v1/user_info` |

v1 端点只认 `code`，走不了 device flow；`scope` 必须显式且含 `offline_access`。

## 排障

- **找不到命令**：`.env` 配 `LARK_CLI_COMMAND` 指向 `lark-cli.exe` 绝对/相对路径；
  Windows 下不要指向 npm 的 `.cmd` 垫片（服务层会经 `cmd.exe` 中转，但直接指 `.exe` 最稳）。
- **工具报 `tool_error.lark_unbound`**：该用户尚未绑定（或授权失效）。属**预期引导**，
  Agent 会走「确定性绑定引导」让用户先绑定——不是故障。
- **绑定工具报「绑定功能未启用」**：服务端缺 `LARK_APP_ID` / `LARK_APP_SECRET` /
  `LARK_TOKEN_KEY` 之一。这是部署问题，与「用户没绑定」是两回事（处置不同）。
- **授权链接过期**：设备码有效期 10 分钟；过期后重新发起（`lark_bind_start`）。
- **刷新失败 / 掉绑**：`refresh_token` 有效期为最后使用后 7 天（`refresh_token_expires_in=604800`）；
  长期不使用会掉绑，重新绑定即可。并发刷新经乐观锁（`version` + 条件更新 + 竞态重读）保证
  不串号（`refresh_token` **会轮换**，旧值立即失效）。
- **日志纪律**：服务日志只记命令组名与结果摘要，不记录消息体明文与任何 token。