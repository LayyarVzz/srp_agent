"""P5-3/P5-4 验收脚本：A2A 双实例互联 demo（dev-version5.0.md V5-M3 / V5-M4）。

阶段一（T5 Server，外部客户端视角）：进程内 uvicorn 拉起 B 实例（127.0.0.1:8002，
peer=agent-a 已登记）→ 发现 AgentCard → message/send 同步收答 → task/get 复查 →
message/stream SSE 逐帧（受理帧宣告 task_id，过程帧进度，末帧终态 Task）。

阶段二（T6 Client，A 实例视角）：装配 AgentRuntime（a2a_mcp 已按注册表登记），
用户一句「请调用智能体 b…」驱动 A 经 `a2a_call_agent` 一次普通 MCP 工具调用
把子任务交给 B，并把 B 的回答整合进 A 的回答（tool_trace 可见调用轨迹）。

配置注入（进程 env，须在 get_settings() 单例缓存前生效）：
- `MCP_TRANSPORT=stdio`：demo 本地自包含，强制三个 MCP 服务（rag/tools_mcp/a2a_mcp）
  均以 stdio 子进程自拉起——否则跟随 .env 的 streamable-http 会去连不存在的远端容器
  （a2a_mcp 默认 8102 无人监听），MCP 连接失败拖垮整个工具集；
- B 侧 `A2A_PEER_MAP={"agent-a": ""}`（user_id 空 → 匿名命名空间 a2a:agent-a）；
- A 侧 `A2A_MCP_AGENTS={"b": "http://127.0.0.1:8002"}`（注册表非空才登记 a2a_mcp）。

用法（在仓库根目录执行）：
    uv run python -m scripts.demo_a2a                     # 未配置 LLM_API_KEY 时跳过
    LLM_API_KEY=sk-xxx uv run python -m scripts.demo_a2a  # DeepSeek
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
import uvicorn

import agent.core.config  # noqa: F401  # 先于 agent.runtime 初始化 core 链（避免 response→tools→core 冷入口循环导入；demo_chat 同款惯例）
from agent.runtime import AgentRuntime
from agent.share.eventloop import ensure_selector_event_loop
from app.main import create_app
from settings import configure_logging, get_settings

# Windows：psycopg 异步需 SelectorEventLoop，须在任何事件循环创建之前设置。
ensure_selector_event_loop()

logger = logging.getLogger(__name__)

# B 实例（被调用方远端智能体）地址；A 实例只建 runtime 不起 HTTP。
B_HOST, B_PORT = "127.0.0.1", 8002
B_BASE_URL = f"http://{B_HOST}:{B_PORT}"
PEER_HEADERS = {"X-A2a-Peer-Id": "agent-a"}

# JSON-RPC 单端点与 peer 身份头（与 agent/a2a 协议子集一致的演示客户端）。
B_A2A_URL = f"{B_BASE_URL}/a2a"


def _setup_demo_env() -> None:
    """注入演示配置（进程 env；setdefault 尊重外部已设值）。

    WHY 强制 MCP_TRANSPORT=stdio：demo 必须自包含（本地无 a2a_mcp 容器），
    若跟随 .env 的 streamable-http，a2a_mcp 会按 HTTP 连 8102 无人监听 →
    MCP 连接失败 → 工具集整体降级为空（build_tools_from_mcp 的 TaskGroup 语义）。
    """
    os.environ.setdefault("MCP_TRANSPORT", "stdio")
    os.environ.setdefault("A2A_PEER_MAP", json.dumps({"agent-a": ""}))
    os.environ.setdefault("A2A_MCP_AGENTS", json.dumps({"b": B_BASE_URL}))
    # A 调用远端时声明的 peer 身份（与 B 侧 A2A_PEER_MAP 登记项对应）。
    os.environ.setdefault("A2A_MCP_PEER_ID", "agent-a")


def _rpc(method: str, params: dict[str, object], request_id: str) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def _message_params(text: str) -> dict[str, object]:
    return {"message": {"role": "user", "parts": [{"kind": "text", "text": text}]}}


async def _start_b() -> tuple[uvicorn.Server, asyncio.Task[None]]:
    """进程内拉起 B 实例（lifespan 装配完整 AgentRuntime，与生产同构）。"""
    config = uvicorn.Config(
        create_app(), host=B_HOST, port=B_PORT, log_level="warning", loop="asyncio"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # 嵌入式运行：不接管进程信号
    serve_task = asyncio.create_task(server.serve())
    waited = 0.0
    while not server.started:
        await asyncio.sleep(0.1)
        waited += 0.1
        if waited > 60:  # 防御：装配失败（如 MCP 连接异常）不应悬挂 demo
            raise RuntimeError("B 实例 60s 未完成启动")
    logger.info("B 实例已就绪：%s（peer=agent-a 已登记）", B_BASE_URL)
    return server, serve_task


async def _stage1_server_demo() -> None:
    """阶段一：外部测试客户端发现并调用 B（Server 侧全部对外面）。"""
    async with httpx.AsyncClient(timeout=180) as client:
        card = (await client.get(f"{B_BASE_URL}/.well-known/agent.json")).json()
        logger.info(
            "[阶段一] 发现 AgentCard：name=%s url=%s streaming=%s",
            card["name"],
            card["url"],
            card["capabilities"]["streaming"],
        )

        # message/send：同步执行到终态
        resp = (
            await client.post(
                B_A2A_URL,
                headers=PEER_HEADERS,
                json=_rpc("message/send", _message_params("用一句话介绍你自己能做什么"), "d1"),
            )
        ).json()
        task = resp["result"]
        task_id = task["id"]
        logger.info("[阶段一] message/send → task=%s state=%s", task_id, task["state"])
        logger.info("[阶段一] B 的回答：%s", task["message"]["parts"][0]["text"])

        # task/get：凭 task_id 复查终态（含 citations 附注）
        got = (
            await client.post(
                B_A2A_URL, headers=PEER_HEADERS, json=_rpc("task/get", {"id": task_id}, "d2")
            )
        ).json()
        logger.info(
            "[阶段一] task/get → state=%s finished_reason=%s",
            got["result"]["state"],
            got["result"]["finished_reason"],
        )

        # message/stream：SSE 帧序 = 受理（宣告 task_id）→ 过程进度 → 终态 Task
        task_id_stream: str | None = None
        async with client.stream(
            "POST",
            B_A2A_URL,
            headers=PEER_HEADERS,
            json=_rpc("message/stream", _message_params("再补充一个你支持的使用场景"), "d3"),
        ) as resp:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                frame = json.loads(line[len("data: ") :])
                result = frame.get("result") or {}
                if result.get("kind") == "status-update":
                    status = result.get("status") or {}
                    if status.get("message"):
                        logger.info("[阶段一] 进度帧：%s", status["message"])
                    task_id_stream = result.get("taskId") or task_id_stream
                elif result.get("state"):
                    task_id_stream = result.get("id") or task_id_stream
                    logger.info("[阶段一] 流式末帧：state=%s", result["state"])
        logger.info(
            "[阶段一] 流式任务 %s 已登记（可 task/get 轮询 / task/cancel 取消）", task_id_stream
        )


async def _stage2_client_demo() -> None:
    """阶段二：A 经 a2a_call_agent 一次 MCP 工具调用驱动 B 并整合回答。"""
    runtime = await AgentRuntime.create()
    try:
        session = await runtime.sessions.create(user_id="demo-user")
        text = "请调用智能体 b，让它用一句话介绍它的能力，然后把它的回答告诉我"
        logger.info("[阶段二] 用户 → A（session=%s）：%s", session.session_id, text)
        resp = await runtime.chat(user_id="demo-user", session_id=session.session_id, text=text)
        for record in resp.tool_trace:
            logger.info("[阶段二] A 调用工具：%s（%s）", record.tool_name, record.status)
        logger.info("[阶段二] A 的整合回答：%s", resp.reply)
    finally:
        await runtime.aclose()


async def main() -> None:
    _setup_demo_env()
    get_settings.cache_clear()  # app.main 导入期可能已缓存：让 env 注入生效
    settings = get_settings()
    configure_logging(settings)
    if not settings.llm_api_key.get_secret_value():
        logger.warning("未配置 LLM_API_KEY，跳过 A2A 双实例 demo（可先运行 pytest 验证离线路径）")
        return

    server, serve_task = await _start_b()
    try:
        await _stage1_server_demo()
        await _stage2_client_demo()
        logger.info("A2A 双实例 demo 完成（发现 → 调用 → 流式 → 整合 全链路通过）")
    finally:
        server.should_exit = True  # 触发 uvicorn 优雅关闭（lifespan aclose runtime）
        await serve_task


if __name__ == "__main__":
    asyncio.run(main())
