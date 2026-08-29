# tools_mcp 独立运行镜像
# 仅用于 compose 中以 streamable-http 方式跑工具服务；Agent 本体仍在本机 uv 环境运行。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# 安装 uv（官方 uv 镜像提供现成二进制）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先复制依赖清单并同步依赖（利用 Docker 层缓存：源码改动不触发重新解析依赖）。
# --no-install-project：只装依赖、不装本项目包（运行靠源码目录，见 CMD）。
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 只复制 tools_mcp 及其包结构（services/__init__.py 提供 services 命名空间包）
COPY services/__init__.py ./services/__init__.py
COPY services/tools_mcp ./services/tools_mcp

# streamable-http 默认端口；host/port/path 由 compose 的 MCP_* 环境变量覆盖
EXPOSE 8100

CMD [".venv/bin/python", "-m", "services.tools_mcp"]
