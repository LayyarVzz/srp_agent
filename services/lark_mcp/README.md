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

- Agent 侧门控登记（`agent/runtime.py::_build_lark_connection`），**按传输方式分两路**：
  - **stdio（本机默认）**：`cfg.lark.enabled` **且** `LARK_CLI_ENABLED` **且** 命令探测成功
    （`resolve_lark_cli_command`）才注册 `lark_mcp` 服务；否则跳过并记日志
    ——**无 CLI 环境零回归**；
  - **streamable-http（容器部署）**：由 `LARK_MCP_TRANSPORT=streamable-http` 决定，
    **不做本地命令探测** —— lark-cli 住在 lark_mcp 容器内，api 侧探测的是本机 PATH，
    与远端容器能力无关（探测必失败，且失败会静默丢掉整个飞书工具面）。
- 服务侧装配（`services/lark_mcp/server.py::_build_binding_pieces`）：有应用凭据 +
  `lark_token_key` → 构造 `BindingCredentialProvider`（按用户隔离）；否则 →
  `ConfigCredentialProvider`（配置态单用户）。
- 独立运行：`uv run python -m services.lark_mcp`（stdio；配
  `MCP_TRANSPORT=streamable-http` 切 HTTP，默认端口 8101）。
- 演示：
  - `uv run python -m scripts.demo_lark` —— 业务工具面；
  - `uv run python -m scripts.demo_lark_binding` —— **多用户绑定全链路**（默认离线，
    零网络零真实应用；`--live` 用真实飞书应用并人工扫码）。

## 容器化（独立 `lark_mcp` 容器）

以 streamable-http 运行在 8101，api 经 `LARK_MCP_TRANSPORT/HOST/PORT` 连过来
（决策与差异见 `docs/plan-docker-observability.md` §2.1）：

```bash
# 仓库根目录；凭据从宿主 .env 注入（LARK_APP_ID / LARK_APP_SECRET / LARK_TOKEN_KEY）
docker compose up -d --build lark_mcp
docker compose logs -f lark_mcp
curl -s http://127.0.0.1:8101/health        # {"status":"ok"}
docker compose exec lark_mcp lark-cli --version   # 1.0.95
```

**镜像形态（`Dockerfile.lark_mcp`）**：node 阶段只用于取 `@larksuite/cli` 的**独立
二进制**（npm 包只是下载器 + 垫片），运行期是**纯 python-slim，镜像内无 node**。

> 构建期提示：npm 包的 postinstall 用 `spawnSync curl` 下载平台二进制，而
> `node:*-slim` 基底**不含 curl** → 构建阶段必须先 `apt-get install curl`
> （实测报错 `Failed to install lark-cli: spawnSync curl ENOENT`，是缺工具不是网络问题）。
> 该二进制实测为**静态链接**（`ldd` 报 not a dynamic executable），故不依赖基底 libc 版本。

### 三个容器化陷阱（踩过，勿改）

1. **`LARKSUITE_CLI_CONFIG_DIR` 必须显式给出**（镜像内已固化 `/tmp/lark-cli`）：
   `services/lark_mcp/cli.py` 的 `_ENV_ALLOWLIST` 刻意剔掉 `HOME`/`USERPROFILE`/`APPDATA`
   （设计意图：身份不得回落本机既有登录），容器内若无可写 config 目录，部分命令会失败。
2. **`LARK_DATABASE_URL` 必须指向编排内的 postgres**：缺 DSN 时绑定域退化为
   `sqlite+aiosqlite:///:memory:` ——「看着成功、实际无效」。HTTP 形态下虽然进程常驻，
   容器重建 / 多副本仍各库各的；**且 `lark_bind_start`（写待定态）与 `lark_bind_complete`
   （读回）是两次独立工具调用**，内存库会让对话内绑定闭环静默断链（表现为误导性的
   「还没检测到授权」）。compose 已注入
   `postgresql://postgres:postgres@postgres:5432/srp_agent`（与记忆/会话同库）。
3. **`LARK_TOKEN_KEY` 必须与 api 容器同值**：密文由 lark_mcp 写入
   （`shared/lark/repository.py`），api 侧要解密取用；不同值 = 绑定链路静默失败。

### 容器内联调（不经 compose）

```bash
docker run --rm -p 127.0.0.1:8101:8101 \
  -e MCP_TRANSPORT=streamable-http -e MCP_HOST=0.0.0.0 -e MCP_PORT=8101 \
  -e LARK_CLI_COMMAND=/usr/local/bin/lark-cli \
  -e LARK_DATABASE_URL='sqlite+aiosqlite:///:memory:' \
  srp-agent-lark-mcp:latest
```

> 绑定域的真实闭环（写密文 → 读回）**必须**用 Postgres：内存库只够冒烟探活。

### 镜像瘦身与裁剪边界（Phase B）

本镜像不是「装齐 `uv.lock` 的全部依赖」，而是按**本服务真实 import 面**裁剪
（plan-docker-observability.md §3）：

| 手段 | 效果（本机实测） |
|---|---|
| 多阶段（builder → runtime）+ `--mount=type=cache,target=/root/.cache/uv` + 显式 `UV_CACHE_DIR` | 镜像内**零 uv 缓存残留**（原来每镜像残留 **241 MB**） |
| uv 钉 `0.11.28`（原 `:latest`）+ `UV_PYTHON_DOWNLOADS=never` | 不自下载 CPython；构建可复现 |
| 构建期不编译字节码 + runtime `PYTHONDONTWRITEBYTECODE=1` | `.pyc` 归零（api/tools 各省 **88 MB**；lark/a2a 本就没有） |
| `uv sync --no-install-package …`（见 `Dockerfile.lark_mcp`） | `.venv` **240 MB → 136 MB**；镜像 946 MB → 417 MB |

**裁剪边界（改动本服务代码后必须复核）**：本镜像的第三方 import 面 =
`fastmcp` / `starlette`（服务与 `/health`）+ `pydantic` + `pydantic-settings` +
`sqlalchemy` + `aiosqlite` + **`psycopg[binary]`**（绑定域仓库建表与读写）+
`cryptography`（Fernet）+ `httpx`（OAuth）+ `langchain-mcp-adapters`。

> ⚠️ **`psycopg` 不可裁（实测踩过）**：绑定域仓库把 DSN 改写成
> `postgresql+psycopg://`（`shared/lark/repository.py::_build_engine`），而 SQLAlchemy
> 的方言是**运行期按 URL 惰性导入**的 —— 裁掉后构建期不报错，启动建表阶段才
> `ModuleNotFoundError`。容器内 compose 恒注入 Postgres DSN，故必须保留。
> 同理 `langchain-mcp-adapters` 来自 `client_config.py` 的 `StdioConnection` 导入，也不可裁。

**复核方式（零成本）**：
`docker run --rm srp-agent-lark-mcp:latest /app/.venv/bin/python -c "import services.lark_mcp.server"`。
**本裁剪方式的已知代价**：新增 import 时构建期不报错、运行期才报 —— 因此给本服务加依赖
必须同步更新 Dockerfile 的排除清单。

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
| `LARK_CLI_COMMAND` | 空 | 显式指定 CLI 路径（相对路径按仓库根解析）；空则探测 PATH；容器内为 `/usr/local/bin/lark-cli` |
| `LARK_MCP_TRANSPORT` | `stdio` | **客户端**连接方式（settings.py）；容器部署 = `streamable-http` |
| `LARK_MCP_HOST` / `LARK_MCP_PORT` | `127.0.0.1` / `8101` | **客户端**连接地址；容器部署 = 服务名 `lark_mcp` / `8101` |
| `LARK_DATABASE_URL` | 空 | 绑定域库（**优先于 `DATABASE_URL`**）；容器部署必须指向编排内 Postgres |
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