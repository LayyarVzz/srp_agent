# srp-agent

由虚拟数字人承载的通用个人助理 —— **Agent 后端服务**（意图识别 / 任务编排 / 记忆 / 工具调用 / 语音交互 / 智能体互联）。

前端只与 FastAPI 交互层通信；Agent 是决策中枢，经 MCP 接入工具，经 Langfuse 上报链路观测。

```
Vue 3 + Three.js 前端  ←→  FastAPI 交互服务（SSE）  ←→  LangChain/LangGraph Agent  ←→  MCP 工具服务  ←→  LLM / ASR
```

---

## 1. 已实现功能

### 1.1 编排内核（LangGraph `StateGraph`）

| 能力 | 说明 |
|---|---|
| 意图识别 | 结构化输出 `IntentResult{intent, confidence, reason}`；`CHAT / TOOL_USE / PLAN`；分类失败走确定性关键词兜底 |
| ReAct 工具循环 | `call_model ⇄ dispatch_tool`（`ToolNode`），工具迭代硬上限 3 轮，超限按 `finished_reason=tool_limit` 收尾 |
| 多步规划（PLAN） | Plan-and-Solve：`plan_task → execute_step → plan_step_advance`，失败重规划 ≤1 次；步骤数上限 6、工具预算 12 |
| 并行子代理 | 就绪批 ≥2 步时经 LangGraph `Send` 扇出到预编译子图并行执行后 join（`SubagentConfig` 可关闭，关闭即回到串行零回归） |
| 澄清式追问 | 意图置信度 < 0.5 或工具参数缺失 → 反问问句 + 候选选项（每轮上限 1 次，防循环） |
| 输出护栏 | `validate_output` 对回答做 schema 与「技能指令性」扫描，不满足则降级；工具/RAG 内容一律视为不可信数据 |
| 降级路径 | LLM 不可用、工具失败、护栏不通过 → 确定性兜底话术，HTTP 仍 200（**不用 5xx 表达「答不上来」**） |

### 1.2 记忆

- **长期记忆**：langgraph `Store`（dev = `InMemoryStore` / prod = `PostgresStore` + pgvector），
  统一经 `MemoryStore` 适配层访问；写入为**带外、非阻塞、尽力而为**；抽取用结构化 LLM 输出。
- **语义召回**：配置 `EMBEDDING_*` 后按 pgvector 相似度召回并混合重排；未配置自动降级为 importance 确定性路径。
- **去重与价值判定**：content-hash 精确层 + 语义近似层；写入前判定「是否值得记住」，单轮落库上限 5 条。
- **短期上下文**：checkpointer 承载（`thread_id == session_id`），token/字符预算裁剪 + 滚动摘要 + 会话关键信息。
- **会话元数据**：独立 `sessions` 表（SQLAlchemy），只存 `session_id↔user_id` 映射与 TTL，消息本体在 checkpointer。
- **技能（程序性记忆）**：内置技能为只读资产；提取技能经 `MemoryStore` 存取，带 `pending → approved/rejected` 生命周期，**未审批永不注入 prompt**。

### 1.3 工具面（全部经 MCP 接入，无进程内直连）

| 服务 | 工具 | 门控 |
|---|---|---|
| `tools_mcp` | `current_datetime`、`calculate` | 始终登记 |
| `lark_mcp` | 飞书 IM / 日程 / 任务 / 文档 / 通讯录 5 个工具 + 账号绑定 4 个工具 | `LARK_CLI_ENABLED` 且命令探测成功（HTTP 形态不做本地探测）；飞书身份按 `user_id` 隔离，凭据加密落库 |
| `a2a_mcp` | `a2a_call_agent`（调用远端 A2A 智能体） | `A2A_MCP_ENABLED` 且远端注册表非空（空表即不登记，零回归） |

### 1.4 交互与语音

- **SSE 流式**：`session → status / tool / token → done`，状态事件按真实执行顺序下发，直接驱动数字人动画。
- **token 级流式**：回答逐增量下发（打字机 / TTS 口型），以 `done.answer` 为最终权威。
- **语音**：讯飞 IAT 听写，仅支持 16kHz/16bit/单声道 PCM；未配凭据时可用 `transcript` 直传文本完成降级联调。
- **A2A 智能体互联**：入站 `AgentCard` 发现 + JSON-RPC 2.0 单端点（`message/send`、`message/stream`、`task/get`、`task/cancel`）；
  出站经 `a2a_mcp` 把远端智能体当普通 MCP 工具调用。

语音输入接入讯飞语音听写 IAT，需在 `.env` 中配置（未配置时语音接口返回 `asr.missing_credentials`，
文本链路与 `transcript` 直传降级不受影响）：

```env
XF_IAT_APP_ID=
XF_IAT_API_KEY=
XF_IAT_API_SECRET=
```

音频文件使用 16kHz、16bit、单声道 PCM。可用脚本单独测试语音识别：

```bash
uv run python scripts/test_xfyun_iat.py path/to/audio.pcm
```

---

## 2. A2A 智能体互联（扩展项，仅供脚本级验证）

**定位**：A2A 是**编排层的扩展点举证**，不是主链路的必需能力。

### 2.1 实现边界

| 支持 | 不支持（刻意裁剪） |
|---|---|
| `GET /.well-known/agent.json` AgentCard 发现 | file / function-call Part（仅 text Part） |
| `message/send`（同步到终态） | 多技能协商（`skills` 仅声明 1 项） |
| `message/stream`（SSE 过程帧 + 终态 Task） | 推送 webhook（`pushNotifications: false`） |
| `task/get`、`task/cancel` | 状态迁移历史（`stateTransitionHistory: false`） |
| peer 身份：配置级注册表（`A2A_PEER_MAP`） | 复杂认证（OAuth / mTLS / 卡片签名） |


### 2.2 已知边界

- **协议版本**：`protocolVersion = 0.3.0`
- **验证方式仅脚本级**：无常驻远端智能体、`tests/` 中无用例覆盖；验证完全依赖
  `scripts/demo_a2a.py`（进程内 uvicorn 起 B 实例、强制 `MCP_TRANSPORT=stdio` 自包含，零成本可跑）。
  CI 不覆盖，无 Docker 级端到端用例。
- **任务注册表是进程内的**：无持久化，`task/get` / `task/cancel` 必须打到处理该任务的**同一实例**；
  多副本部署下会随机失效。`message/send` 返回即终态（拿不到 task_id），**只有 `message/stream`
  路径的取消才真正生效**。
- **容器环境下默认不可用**：compose 未注入 `A2A_PEER_MAP` → 入站调用一律 `a2a.invalid_request`
  （仅 AgentCard 发现可用）；`A2A_MCP_AGENTS` 未配置 → 出站工具**不登记**。两项均在
  `AgentRuntime` 装配期读入、**不热加载**，改配置后须 `--force-recreate` 而非 `restart`。
- **peer 身份为配置级约定**：`A2A_MCP_PEER_ID`（默认 `srp-agent`）必须与对端 `A2A_PEER_MAP`
  的登记项**精确匹配**，否则对端返回 `a2a.invalid_request`。peer 的入站身份落匿名命名空间
  `a2a:<peer_id>`，与人类用户的记忆/会话物理隔离。

### 2.3 验证命令

```bash
uv run python -m scripts.demo_a2a     # 双实例互联：发现 → 调用 → 流式 → 整合（需 LLM_API_KEY）
```

---

## 3. 目录结构

```
srp_agent/
├── agent/            # Agent 内核：core（编排图）/ intent / tools / memory / skills
│                     #            session / query / response / errors / a2a / runtime.py
├── app/              # FastAPI 交互服务：routes（sessions/chat/logs/metrics/health）
│                     #            a2a / sse.py / metrics.py / request_context.py
├── services/         # MCP 服务：tools_mcp / lark_mcp / a2a_mcp（+ rag_mcp，本文档外）
├── shared/           # 跨模块基础设施：logging / observability / events_store / embeddings / lark
├── scripts/          # demo 与验收脚本（demo_chat / demo_a2a / demo_plan ...）
├── tests/            # pytest 离线单测（fake LLM，零网络）
├── docs/             # api.md（接口契约）/ deploy.md（部署运维）/ TESTING.md（测试指南）
├── settings.py       # 运行环境配置（env / .env）
├── compose.yaml      # docker compose 编排
└── Dockerfile.*      # api / tools_mcp / lark_mcp / a2a_mcp 镜像
```

依赖管理**仅使用 [uv](https://docs.astral.sh/uv/)**，Python `>=3.12`。

---

## 4. 本地部署

### 4.1 最小路径（全内存，零外部依赖）

```bash
# 1) 安装依赖（uv 自动按 .python-version 建 .venv）
uv sync

# 2) 准备配置，至少填 LLM_API_KEY
cp .env.example .env

# 3) 启动 API（Windows 必须经此入口；见下方说明）
uv run python scripts/run_api.py
```

服务起在 `http://127.0.0.1:8000`：

| 地址 | 用途 |
|---|---|
| `GET /healthz` | 存活探针 |
| `GET /docs` | Swagger UI（OpenAPI 自动生成） |
| `GET /metrics` | 进程内指标快照 |

零配置下：长期记忆 `InMemoryStore`、短期上下文 `MemorySaver`、会话 `SQLite memory`、事件表内存 SQLite
—— **进程重启数据即消失**，适合先跑通链路。

> **Windows 必读**：`psycopg` 异步驱动要求 `SelectorEventLoop`，uvicorn 在 Windows 硬编码 Proactor。
> 故必须用 `scripts/run_api.py`（内部显式指定 loop 工厂），直接 `uvicorn main:app` 会报
> `Psycopg cannot use the 'ProactorEventLoop'`。Linux / 容器内无此问题。

### 4.2 接入 Postgres（持久化 + 语义召回）

```bash
docker run -d --name srp-agent-pg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=srp_agent -p 5432:5432 -v srp_pgdata:/var/lib/postgresql \
  pgvector/pgvector:pg17        # 必须 pgvector 镜像，postgres:latest 不含 vector 扩展
```

`.env` 中配置：

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/srp_agent
EMBEDDING_ENABLED=true              # 可选：启用语义召回/去重
EMBEDDING_MODEL=<与 RAG 同一模型>
EMBEDDING_DIMS=<与模型一致，如 1024>
EMBEDDING_BASE_URL=<OpenAI 兼容端点>
EMBEDDING_API_KEY=<密钥>
```

首次启动自动建表（`checkpoints` / `store` / `sessions` / `interaction_events` / `lark_bindings`），无需手动迁移。

### 4.3 容器化部署（compose）

```bash
docker compose up -d --build     # 首次：构建全部镜像并启动
docker compose up -d             # 之后直接启动
docker compose ps                # 查看状态
docker compose logs -f api       # 跟踪日志
docker compose down              # 停止（数据卷保留）
docker compose down -v           # 停止并删除数据卷（**删库**）
```

| 服务 | 镜像 | 宿主端口 | 说明 |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg17` | 5432 | 记忆 / 会话 / 事件 / 飞书绑定共库 |
| `tools_mcp` | `srp-agent-tools-mcp` | 127.0.0.1:8100 | 纯函数工具 |
| `lark_mcp` | `srp-agent-lark-mcp` | 127.0.0.1:8101 | 飞书工具面 + 绑定（lark-cli 在容器内） |
| `a2a_mcp` | `srp-agent-a2a-mcp` | 127.0.0.1:8102 | 远端智能体协作工具 |
| `api` | `srp-agent-api` | 8000 | FastAPI + Agent 组合根 |

MCP 端口只绑回环（容器间经 compose 网络互访）；`api` 等各 MCP 服务 `healthy` 后才启动，避免工具集静默降级。

**只需在宿主 `.env` 里配这些项**，compose 会注入容器（其余地址项已由 compose 固化为服务名）：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-xxx
LARK_APP_ID= / LARK_APP_SECRET= / LARK_TOKEN_KEY=   # 飞书绑定三件套，缺一即 fail-closed 关闭
A2A_MCP_AGENTS={"peer_agent":"http://host.docker.internal:9000"}   # 远端智能体注册表
LANGFUSE_*                                          # 观测，缺省关闭
LOG_FORMAT=text                                     # text | json
```

> `DATABASE_URL` / `MCP_*` / `LARK_MCP_*` / `A2A_MCP_HOST` 由 compose 写成服务名，**不要**在 `.env` 覆盖。

### 4.4 验收脚本

```bash
uv run pytest                      # ① 离线单测（fake LLM，零网络）
uv run ruff check .                # ② Lint
uv run pytest tests/test_log_events.py -q   # ③ 事件链路口径与日志脱敏
uv run python -m scripts.demo_chat # ④ Agent 全流程（需 LLM_API_KEY）
uv run python -m scripts.demo_session   # 会话管理（无需 LLM）
uv run python -m scripts.demo_plan      # 多步规划 + 并行子代理
uv run python -m scripts.demo_clarify   # 澄清式追问
uv run python -m scripts.demo_a2a       # A2A 双实例互联（进程内自包含）
```

> **测试前请留意**：`tests/` 的用例依赖环境独立。本地 `.env` 中的 `LOG_FORMAT` / `A2A_MCP_AGENTS`
> 等运行期项会影响断言（`tests/conftest.py` 已做隔离，见该文件说明）。


---

## 5. API 层接口

Base URL：`http://<host>:8000`，业务接口统一前缀 `/api/v1`。
**除健康检查、指标、A2A 与日志链路回溯外，均需请求头 `X-User-Id`**（MVP 身份约定，不做真实登录；仅用于会话归属隔离）。

### 5.1 会话管理

| 方法 | 路径 | 说明 | 响应 |
|---|---|---|---|
| `POST` | `/api/v1/sessions` | 创建会话（服务端 uuid4 发号） | `201` `SessionContext` |
| `GET` | `/api/v1/sessions?limit=1..100` | 会话列表（按创建时间降序） | `200` |
| `DELETE` | `/api/v1/sessions/{session_id}` | 删除会话（仅元数据） | `204` |

### 5.2 交互

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/interactions/text` | 文本交互（非流式，联调/调试） |
| `POST` | `/api/v1/interactions/text/stream` | **文本交互（SSE 流式，前端主接口）** |
| `POST` | `/api/v1/interactions/voice` | 语音交互（multipart，非流式） |
| `POST` | `/api/v1/interactions/voice/stream` | 语音交互（SSE 流式） |

请求体（文本）：`{ "session_id": "可选，缺省服务端自动创建", "text": "1..8000 字符" }`
请求体（语音，form-data）：`audio`（16k/16bit/mono PCM）与 `transcript` 二选一 + 可选 `session_id`。

**SSE 事件表**：

| event | 载荷 | 语义 |
|---|---|---|
| `session` | `{session_id}` | 流开始即发 |
| `status` | `{status, tool_name, message}` | 图节点入口逐条下发，**按真实执行顺序**，仅驱动动画 |
| `tool` | `{tool_name, arguments, status, result}` | 工具执行后下发 |
| `token` | `{delta}` | 回答增量（打字机 / TTS），拼接 ≈ `done.answer` |
| `done` | `InteractionResult` | 最后一条，本轮结束，**回答全量权威** |
| `error` | `{code, message}` | 仅图运行中断时出现（业务降级不发此帧） |

**三层状态契约**：

- 传输层 HTTP：`200`（含内部降级）/ `201` / `204` / `400` / `401` / `404` / `415` / `422` / `500`；
- **决策层 `phase`（前端唯一分支依据）**：`answer`（播报）/ `clarify`（渲染选项按钮，不播报）/ `error`；
- 过程层 `status`：`listening / thinking / retrieving / using_tool / planning / delegating / clarifying / speaking / error`；
- 内部层 `finished_reason`：`completed / tool_limit / partial / fallback / error / needs_clarification`。

### 5.3 A2A 智能体互联

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/.well-known/agent.json` | AgentCard 发现（`url` 动态指向本实例 `/a2a`） |
| `POST` | `/a2a` | JSON-RPC 2.0 单端点：`message/send`、`message/stream`(SSE)、`task/get`、`task/cancel` |

- peer 身份经请求头 `X-A2A-Peer-Id` 声明；未登记 peer → `a2a.invalid_request`（peer 注册表见 `A2A_PEER_MAP`）。
- `task_id == session_id`；任务注册表为**进程内**（无持久化），`message/send` 返回即终态，`task/cancel` 仅对流式任务生效。
- HTTP 恒 `200`，协议错误以 JSON-RPC error 信封表达（标准码 + `error.data.code` 承载 `a2a.*`）。

### 5.4 日志与指标

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/v1/logs/recent?limit=1..200` | 本人最近交互事件（时间降序，必带 `X-User-Id`） |
| `GET` | `/api/v1/logs/trace/{trace_id}?limit=1..500` | 按 trace_id 取全链路事件（**升序** = 真实发生顺序） |
| `GET` | `/healthz` | 存活探针 |
| `GET` | `/metrics` | 进程内指标快照（计数 + 耗时分布） |

统一错误信封：

```json
{ "ok": false, "error": { "code": "session_error.not_found", "message": "...", "path": "..." } }
```

---

## 6. 日志观测

### 6.1 全链路 trace

api 入口为每个请求分配/沿用 `trace_id`（`req_` + 16 字节 hex）并**回写响应头 `X-Request-Id`**。
该 id 经 ContextVar 注入**每一条日志**（含第三方库），并在 api 与 3 个 MCP 进程的日志里统一可 grep：

```bash
# 前端拿到的 trace_id 直接查一条链路的事件
curl -H "X-User-Id: demo-user" \
  "http://127.0.0.1:8000/api/v1/logs/trace/req_a3273356c5994378"

# 容器日志按链路聚合
docker compose logs api | grep trace=req_a3273356c5994378
```

text 形态前缀固定为 `[trace=… session=… user=…]`，可直接 `grep`/`awk` 初筛。

### 6.2 形态与开关

| env | 默认 | 说明 |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` 更细 |
| `LOG_FORMAT` | `text` | `text`（人眼可读带关联前缀）\| `json`（一行一条 JSONL，供 Loki/ELK） |
| `LOG_SERVICE` | `api` | 落在每条日志/事件的 `service` 字段 |

容器内**一切走 stdout/stderr，不写文件**：聚合交 `docker compose logs -f`，轮转交 compose 的 `json-file`
driver（`max-size=10m` / `max-file=3`）。4 个进程共用同一 `LOG_FORMAT`，避免混合行格式。

### 6.3 结构化事件

六类事件同时写 stdout 与 `interaction_events` 表（跨进程、跨重启可回溯）：

| event | 触发点 | 关键字段 |
|---|---|---|
| `request.received` | HTTP 入口中间件 | `method` / `path` / `source` |
| `intent.classified` | 意图识别节点 | `intent` / `confidence` |
| `tool.called` | 工具执行 | `tool_name` / `status` / `code` / `duration_ms` / `batch_size` |
| `answer.generated` | 回答生成 | `len` / `sha256` / `phase` / `citations` / `tools` |
| `memory.saved` | 记忆带外写入 | `action` / `kind` / `memory_id` |
| `request.finished` | 入口中间件 / Agent runtime | `status` / `code` / `duration_ms` / `tokens` |

- 落库是**带外的**：`EventStoreSink` 订阅事件后经专用线程批量写库，队列满时丢最旧并计数（`dropped`），**绝不阻塞业务请求**。
- **两条 `request.finished` 是设计如此**（聚合时不要重复计数）：入口中间件那条记 **HTTP 往返**（有 `status_code`），
  Agent runtime 那条记**一轮交互**（`source == "agent"`，带 LLM token 累计）。
- `tool.called.duration_ms` 是**批次级**（`ToolNode` 并行执行同批 `tool_calls`）；子代理工具为整段子图级。

### 6.4 指标

`GET /metrics` 消费**同一条事件总线**（口径与结构化事件同名同源，不引入第二套），返回
请求/回答计数、错误计数、`finished_reason` 分布、按来源的耗时分布（`count / avg / p50 / p95 / max`）、工具调用统计等。
响应含 `service` / `pid` —— **多副本时指标只代表本进程**，不要误读为全局数。

### 6.5 安全口径（硬约束）

- 消息体**只记 `len=` 与 `sha256[:8]`**，从不记原文（提示词、回答、工具参数一律如此）。
- 脱敏在写出前完成：`api_key / authorization / token / user_access_token / refresh_token / secret`
  等键名，加 `sk-` / `Bearer` / JWT / 邮箱 / 手机号正则兜底，统一替换为 `***`。
- 飞书 token **加密落库**（Fernet 派生密钥），不写日志、不进 `ToolMessage`、不回传前端；
  `LARK_TOKEN_KEY` 未配置时绑定功能 fail-closed 关闭，绝不落明文。
- 事件查询按用户隔离：`/logs/recent` 必带 `X-User-Id`（缺失 → `401`）；
  `/logs/trace/{id}` 依赖 trace_id 不可猜。

---

## 7. Langfuse 跟踪

**默认关闭**：未配置 `LANGFUSE_*` 即全程零观测回调（图 config 不带 handler，逐字零回归），
云端不可达时 Agent 照常回答（fail-open）。观测是外挂能力，**不得成为对话链路的前置条件**。

### 7.1 开启

```env
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com     # 区域端点：EU 默认 / US us. / JP jp. / HIPAA hipaa.
LANGFUSE_TRACING_ENVIRONMENT=dev                 # 只允许 [a-z0-9_-]、≤40 字符、不得以 langfuse 开头
LANGFUSE_SAMPLE_RATE=1.0                         # 0~1；越界启动即失败
LANGFUSE_TIMEOUT=5                               # SDK API 请求超时（秒）
```

只有 `api` 进程需要注入 `LANGFUSE_*`（MCP 服务容器的工具 span 由 api 侧回调统一产出）。

### 7.2 设计要点

| 要点 | 实现 |
|---|---|
| 装配点唯一 | `shared/observability.py` 是**全项目唯一**写观测 env、构造 SDK 客户端与掩码的地方；业务代码不读 env |
| 接入方式 | 只经 langchain **回调**（每请求新建 handler），不在图节点内手动 `start_span` —— 不给对话链路引入同步网络点 |
| 归因三键 | `langfuse_session_id`（= `session_id`）、`langfuse_user_id`、`langfuse_trace_name`，由 SDK 在根 run 上自行 propagate |
| 本地↔云端互查 | 额外带普通 metadata 键 `request_trace_id`（本地 `X-Request-Id`），云端 trace id 由 SDK 生成、二者无天然映射，靠它跳转 |
| trace 名 | 对话链路 `agent.chat`；带外记忆保存 `memory.out_of_band_save`（与对话成本口径分开呈现） |
| 脱敏 | 走 SDK 原生 `mask=`：按键名（复用 `shared.logging.is_sensitive_field`，口径不分叉）+ 正则兜底；掩码函数自身失败则返回整车掩码占位 |
| 关闭语义 | flush（带超时）+ shutdown，任何失败只记日志；`aclose()` 幂等 |
