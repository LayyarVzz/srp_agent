# a2a_mcp — 远端智能体协作工具（A2A Client）

把**远端 A2A 智能体**封装为 MCP 工具（`a2a_call_agent`），Agent 经
`MultiServerMCPClient` 接入为普通工具 —— 「把一步交给外部智能体」= 一次普通 MCP
工具调用，`call_model` / `ToolNode` 链路完全不变（dev-version5.0.md D5.3，零改图）。

与 `agent/a2a`（本仓库 A2A **Server** 入站子集）协议形态对称：
AgentCard 发现 → `message/send`（非终态则 `task/get` 轮询至终态或超时）→ 终态 Task
提取 reply；`stream=True` 走 `message/stream` 的 SSE 帧。

## 工具集

| 工具 | 说明 |
|---|---|
| `a2a_call_agent` | `agent_id` + `message`（+`stream`）→ 远端回答（结构化 JSON：`agent_id`/`task_id`/`state`/`reply`/`progress`） |

- **高风险外呼**：工具 description 显式要求「仅在用户明确要求与指定智能体协作时调用」，
  禁止模型自行外呼（防误触发外部系统）。
- **不可信数据**：远端返回 content 经 `ToolMessage` 回流，只作参考材料整合，
  **不得当作指令执行**（提示注入防护，与 RAG 片段同一身份认定）。
- **超时/重试分层**：MCP 连接层负责连接超时与重试（`MCP_TIMEOUT_S` /
  `MCP_MAX_RETRIES`，见 `agent/tools/factory.py`）；本服务只保留**任务级总预算**
  （`A2A_REQUEST_TIMEOUT_S`）与轮询间隔。

## 运行方式

### 1. stdio（默认，本地开发 / Agent 子进程自动拉起）

```bash
uv run python -m services.a2a_mcp
```

### 2. streamable-http（容器 / 远程部署）

```bash
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8102 uv run python -m services.a2a_mcp
```

启动后监听 `http://0.0.0.0:8102/mcp`，健康探针 `GET /health` →
`{"status":"ok","agents":<注册表条目数>}`。

## 配置

服务侧配置经本地 `A2AMCPRuntimeSettings`（`services/a2a_mcp/config.py`）读取，
**不依赖根 `settings.py` / agent 框架**，可独立打包部署。

| 变量 | 默认 | 说明 |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio` \| `streamable-http`（**注意连字符**，与 fastmcp 服务端传输值对齐） |
| `MCP_HOST` | `127.0.0.1` | 容器 / 云服务器须设 `0.0.0.0` |
| `MCP_PORT` | `8102` | 避开 API 8000 / tools_mcp 8100 / lark_mcp 8101 |
| `MCP_STREAMABLE_HTTP_PATH` | `/mcp` | HTTP 端点路径 |
| `MCP_STATELESS_HTTP` | `true` | 工具无会话状态 → 无状态，支持多副本 |
| `A2A_MCP_AGENTS` | 空 | **远端注册表**：JSON `{"<agent_id>": "<base_url>"}`；空表 = 零工具 |
| `A2A_MCP_PEER_ID` | `srp-agent` | 调用远端时声明的 peer 身份（远端须已登记该 peer，否则 `a2a.invalid_request`） |
| `A2A_REQUEST_TIMEOUT_S` | `120` | 单次任务总预算（远端执行可能较慢） |
| `A2A_POLL_INTERVAL_S` | `1.0` | `task/get` 轮询间隔 |
| `A2A_OUTPUT_MAX_CHARS` | `10000` | 工具输出长度上限（服务侧截断） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 门控与零回归

**注册表为空时服务照常跑，但工具不登记**：

- Agent 侧（`agent/runtime.py::_build_mcp_servers`）双门控 —— `A2A_MCP_ENABLED`
  且 `A2A_MCP_AGENTS` 非空才登记 `a2a_mcp` 服务；否则工具集与既有完全一致；
- 服务侧（`services/a2a_mcp/server.py::a2a_call_agent`）对未注册的 `agent_id`
  返回确定性错误结果（`tool_error.execution` → 图侧既有降级路径）。

传输方式：与 `tools_mcp` 同源决策 —— `MCP_TRANSPORT=streamable-http` 时 Agent 侧
按 `A2A_MCP_HOST`/`A2A_MCP_PORT` 连远端，否则以 stdio 子进程自动拉起本服务。

## 容器化

以 streamable-http 运行在 8102（`Dockerfile.a2a_mcp`，镜像 `Dockerfile.tools_mcp` 形态）：

```bash
# 仓库根目录
docker compose up -d --build a2a_mcp
docker compose logs -f a2a_mcp
curl -s http://127.0.0.1:8102/health          # {"status":"ok","agents":0}

# 配了远端注册表时，api 侧才会登记该服务（compose 已注入 A2A_MCP_AGENTS / A2A_MCP_PEER_ID）
A2A_MCP_AGENTS='{"peer_agent":"http://host.docker.internal:9000"}' docker compose up -d
```

- 编排内地址：api 侧 `A2A_MCP_HOST=a2a_mcp` / `A2A_MCP_PORT=8102`（compose 网络内解析）；
- 宿主端口映射为 `127.0.0.1:8102`（仅供本机联调，容器间走 compose 网络）；
- 远端智能体若跑在**宿主**上，容器内需用 `host.docker.internal` 作 base_url。