# srp-agent

## 环境配置

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

安装依赖：

```
uv add xxx
```

安同步依赖：

```bash
uv sync
```

同步/更新依赖到锁文件：

```bash
uv lock --upgrade
```

## Git 使用

```bash
# 克隆仓库
git clone xxx

# 添加所有更改
git add .

# 提交
git commit -m "<提交信息>"

# 推送到远程
git push origin fearture
```

建议使用可视化的git提交，比如vscode或PyCharm等软件自带的git提交。

分支合并不需要使用rebase，使用commit就可以。

## Agent 交互接口

当前 FastAPI 入口提供以下接口：

```text
POST   /api/v1/sessions            # 创建会话
GET    /api/v1/sessions            # 会话列表
DELETE /api/v1/sessions/{id}       # 删除会话
POST   /api/v1/interactions/text   # 文字输入
POST   /api/v1/interactions/voice  # 语音输入
GET    /api/v1/logs/recent         # 最近交互事件（按用户过滤；?limit=1..200）
GET    /api/v1/logs/trace/{id}     # 按 trace_id 取一条链路（响应头 X-Request-Id）
GET    /healthz                    # 健康检查
```

会话和交互接口需要带请求头 `X-User-Id`。MVP 阶段只是用于区分用户，不做真实登录。

## 日志与可观测性

每个请求都会分配 `trace_id` 并回写响应头 `X-Request-Id`，该 id 贯穿 api → Agent → MCP 子进程，
可用它把一次交互在各进程日志里串起来：

```bash
docker compose logs -f api | grep trace=req_xxxx      # 容器聚合
uv run pytest tests/test_log_events.py -q             # 事件链路与脱敏口径
```

- **形态**：`LOG_FORMAT=text`（默认，人眼可读）/ `json`（一行一条 JSONL，供采集器）；
  4 个进程共用同一开关，避免混合形态。
- **结构化事件**：`request.received / intent.classified / tool.called / answer.generated /
  memory.saved / request.finished` 同时落库到 `interaction_events` 表，支持跨进程、跨重启回溯。
- **脱敏**：日志与事件只记消息体的 `len=` 与 `sha256=` 摘要，从不记原文；token/密钥/邮箱/手机号
  在写出前统一替换为 `***`。

语音输入目前接入讯飞语音听写 IAT，需要在 `.env` 中配置：

```env
XF_IAT_APP_ID=
XF_IAT_API_KEY=
XF_IAT_API_SECRET=
```

音频文件使用 16kHz、16bit、单声道 PCM。

可用脚本单独测试语音识别：

```bash
python scripts/test_xfyun_iat.py path/to/audio.pcm
```
