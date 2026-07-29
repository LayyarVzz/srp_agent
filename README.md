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
