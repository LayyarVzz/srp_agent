"""Agent 框架内部核心行为配置。

本模块只承载「框架行为」——状态图/编排、LLM 调用参数、工具注册策略、记忆行为，
  全部为纯 Pydantic 代码默认值，**不读环境变量、不持有密钥**；

新增框架行为项 → 修改本模块；新增环境项 → 修改根 `settings.py`。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr


class LLMProvider(StrEnum):
    """LLM 提供方（provider 无关：任何 OpenAI 兼容端点均可用）。"""

    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True)
class _LLMProviderPreset:
    """某提供方的默认接入预设（base_url 与默认模型）。"""

    base_url: str
    default_model: str


# provider 预设表：新增 provider = 枚举加值 + 此处加一行（零改下游）。
# 注意：Qwen 兼容模式的 base_url 必须带 `/compatible-mode/v1` 路径，否则 404。
_LLM_PROVIDER_PRESETS: dict[LLMProvider, _LLMProviderPreset] = {
    LLMProvider.DEEPSEEK: _LLMProviderPreset(
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-v4-flash",
    ),
    LLMProvider.QWEN: _LLMProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
    ),
    # 开放端点没有预设，必须由运行时显式提供 base_url / model（见 effective_*）。
    LLMProvider.OPENAI_COMPATIBLE: _LLMProviderPreset(base_url="", default_model=""),
}

# DeepSeek V4 系列（v4-flash / v4-pro）模型名标识：
# V4 默认开启思考模式，而思考模式拒绝显式 tool_choice（HTTP 400 "Thinking mode does
# not support this tool_choice"），与 function_calling 结构化输出冲突。
_DEEPSEEK_V4_MODEL_HINT = "v4"


class LLMBehaviorConfig(BaseModel):
    """LLM 调用行为默认值（框架侧，非环境配置）。"""

    # 结构化输出更低温更稳；普通对话可由调用方按需覆盖。
    temperature: float = 0.0
    max_tokens: int | None = None
    request_timeout: float = 60.0
    max_retries: int = Field(default=2, ge=0)
    # langchain-openai >= 0.3 默认 method 是 json_schema，DeepSeek 会拒绝
    # function_calling 是 DeepSeek / Qwen / 任意兼容端点最广的公共能力。
    structured_method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling"
    # DeepSeek V4 默认思考模式且拒绝显式 tool_choice；结构化输出需关闭思考模式。
    disable_thinking_for_structured: bool = True


class LLMConfig(LLMBehaviorConfig):
    """合并后的完整 LLM 配置 = 框架行为默认 + 装配层注入的运行选择。

    密钥与端点选择由入口层（app / demo）从根 `settings.py` 读出后，经 `from_runtime`
    注入；本模型自身不读取任何环境变量。
    """

    provider: LLMProvider = LLMProvider.DEEPSEEK
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str | None = None  # 覆盖预设
    model: str | None = None  # 覆盖预设

    @classmethod
    def from_runtime(
        cls,
        *,
        provider: LLMProvider,
        api_key: SecretStr | str,
        base_url: str | None = None,
        model: str | None = None,
        behavior: LLMBehaviorConfig | None = None,
    ) -> LLMConfig:
        """由运行选择 + 可选行为覆盖合并出完整配置（装配层调用）。"""
        behavior_fields = behavior.model_dump() if behavior else {}
        key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        return cls(
            provider=provider,
            api_key=key,
            base_url=base_url,
            model=model,
            **behavior_fields,
        )

    @property
    def effective_base_url(self) -> str:
        """生效的接入地址：显式覆盖优先，否则回退预设。

        惰性校验：`OPENAI_COMPATIBLE` 未提供 base_url 时仅在使用处抛错，
        避免「未实际使用 LLM 的启动」被配置错误打断。
        """
        if self.base_url:
            return self.base_url
        preset = _LLM_PROVIDER_PRESETS[self.provider]
        if not preset.base_url:
            raise ValueError("provider=openai_compatible 时必须提供 base_url")
        return preset.base_url

    @property
    def effective_model(self) -> str:
        """生效的模型名：显式覆盖优先，否则回退预设。"""
        if self.model:
            return self.model
        preset = _LLM_PROVIDER_PRESETS[self.provider]
        if not preset.default_model:
            raise ValueError("provider=openai_compatible 时必须提供 model")
        return preset.default_model

    @property
    def structured_extra_body(self) -> dict[str, object] | None:
        """结构化输出请求需附加的 provider 特定 body 参数。

        WHY 仅 DeepSeek V4：V4 默认思考模式拒绝显式 tool_choice，须附带
        `thinking={"type": "disabled"}` 才能用 function_calling；非 V4 端点（Qwen /
        兼容端点 / 旧版 deepseek-chat）不认识该参数，一律不附带。
        """
        if (
            self.provider == LLMProvider.DEEPSEEK
            and self.disable_thinking_for_structured
            and _DEEPSEEK_V4_MODEL_HINT in self.effective_model
        ):
            return {"thinking": {"type": "disabled"}}
        return None


class AgentGraphConfig(BaseModel):
    """状态图 / 编排护栏行为。"""

    max_tool_iterations: int = Field(default=3, ge=1)  # 工具迭代硬上限（MAX_TOOL_ITERATIONS）
    max_input_chars: int = Field(default=8000, ge=1)  # 输入长度上限
    trim_token_budget: int = Field(default=8000, ge=1)  # 短期上下文 token 预算
    trim_keep_recent_rounds: int = Field(default=10, ge=0)  # 裁剪保留最近 N 轮


class ToolRegistryConfig(BaseModel):
    """工具注册与调用护栏行为（MCP 服务地址属运行环境，见根 settings.py）。"""

    mcp_timeout_s: float = 10.0  # MCP 调用超时（默认 10s）
    mcp_max_retries: int = Field(default=2, ge=0)
    mcp_max_content_chars: int = Field(default=10_000, ge=1)  # 远端内容长度上限
    allowed_local_tools: tuple[str, ...] = ()  # 本地工具白名单


class MemoryBehaviorConfig(BaseModel):
    """记忆行为（长期记忆抽取 / 召回参数）。"""

    top_k: int = Field(default=5, ge=1)
    backend: Literal["in_memory", "mcp"] = "in_memory"  # 开发默认内存实现，生产切 MCP
    max_recall_chars: int = Field(default=8000, ge=1)  # 单次召回内容长度上限


class AgentFrameworkConfig(BaseModel):
    """Agent 框架行为配置聚合（纯代码默认值，供装配层读取）。"""

    graph: AgentGraphConfig = Field(default_factory=AgentGraphConfig)
    llm_behavior: LLMBehaviorConfig = Field(default_factory=LLMBehaviorConfig)
    tools: ToolRegistryConfig = Field(default_factory=ToolRegistryConfig)
    memory: MemoryBehaviorConfig = Field(default_factory=MemoryBehaviorConfig)

    @classmethod
    def get_default(cls) -> AgentFrameworkConfig:
        """返回框架行为默认配置。"""
        return cls()

