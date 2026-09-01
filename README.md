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
POST   /api/v1/sessions           # 创建会话
GET    /api/v1/sessions           # 会话列表
DELETE /api/v1/sessions/{id}      # 删除会话
POST   /api/v1/interactions/text  # 文字输入
POST   /api/v1/interactions/voice # 语音输入
GET    /api/v1/logs/recent        # 最近交互日志
GET    /healthz                   # 健康检查
```

会话和交互接口需要带请求头 `X-User-Id`。MVP 阶段只是用于区分用户，不做真实登录。

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
