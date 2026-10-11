"""P0 验收脚本：调用一次结构化输出（真实 LLM）。

用法（在仓库根目录执行）：
    uv run python -m scripts.demo_structured                 # 未配置 LLM_API_KEY 时优雅跳过
    LLM_API_KEY=sk-xxx uv run python -m scripts.demo_structured                # DeepSeek
    LLM_PROVIDER=qwen LLM_API_KEY=sk-xxx uv run python -m scripts.demo_structured  # Qwen
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from agent.core.config import LLMConfig
from agent.llm import LLMService
from settings import configure_logging, get_settings

logger = logging.getLogger(__name__)


class DemoIntent(BaseModel):
    """演示用结构化输出 schema。"""

    intent: str
    confidence: float
    reason: str


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    api_key = settings.llm_api_key.get_secret_value()
    if not api_key:
        logger.warning("未配置 LLM_API_KEY，跳过真实调用（可先运行 pytest 验证离线路径）")
        return
    config = LLMConfig.from_runtime(
        provider=settings.llm_provider,
        api_key=api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    service = LLMService(config=config)
    result = await service.ainvoke_structured(DemoIntent, "请用一句话说明现在几点了。")
    logger.info("结构化输出: %s", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
