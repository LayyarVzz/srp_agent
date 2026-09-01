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

from agent.share.models import MemoryRecallConfig
from shared.embeddings import EmbeddingConfig


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
    disable_thinking: bool = True


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
        这里先只处理deepseek
        """
        if (
            self.provider == LLMProvider.DEEPSEEK
            and self.disable_thinking
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
    # 短期上下文字符预算（无 tokenizer，用字符近似 token）：窗口内消息按旧→新累计长度，
    # 超预算继续裁旧轮；是摘要预算之上的一次性上下文预算。
    max_context_chars: int = Field(default=24000, ge=1)


class PlanConfig(BaseModel):
    """多步任务编排行为（Plan-and-Solve）。

    `max_plan_steps` 是主收敛约束（步骤数上限，规划期校验超限 → 回退 ReAct）；
    `max_tool_calls_per_plan` 是 plan 模式下的工具迭代总预算（默认 = 6×2），
    `tool_iterations` 跨步累计充当总预算；ReAct 模式的全局预算
    （`AgentGraphConfig.max_tool_iterations`）不受影响（零回归）。
    """

    enabled: bool = True  # 关闭时 PLAN 意图回退 ReAct（call_model）
    max_plan_steps: int = Field(default=6, ge=1, le=10)  # 步骤数上限（主收敛约束）
    # plan 模式工具迭代上限（默认 = max_plan_steps×2）；tool_iterations 跨步累计充当总预算。
    max_tool_calls_per_plan: int = Field(default=12, ge=1)


class ClarifyConfig(BaseModel):
    """澄清式追问行为。

    触发源：
    ① 意图置信度低：`route_intent` 前检查 `intent_meta.confidence < min_confidence`
       （严格小于；规则兜底 CHAT=0.5 恰不触发）；
    ② 工具参数缺失：`route_after_tool` 检查 `tool_error.missing_argument`
       （plan 模式除外 → 走重规划）。
    `max_asks_per_turn` 是防澄清循环的硬上限（`clarify_asked` 状态位，每轮重置）。
    """

    enabled: bool = True  # 关闭时两个触发源均不反问（零回归）
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # 意图置信度阈值（严格小于触发）
    max_asks_per_turn: int = Field(default=1, ge=1)  # 每轮追问上限（防循环）


class MCPToolsConfig(BaseModel):
    """MCP 工具接入护栏行为（MCP 服务地址属运行环境，见根 settings.py）。

    超时/重试在 MCP 客户端连接配置（langchain-mcp-adapters）生效，不由图内节点承担；
    `mcp_max_content_chars` 供响应适配层截断 ToolMessage 内容（长度上限护栏）。
    所有工具一律经 MCP 客户端接入，无进程内本地工具直连。
    """

    mcp_timeout_s: float = 10.0  # MCP 客户端连接/调用超时（默认 10s）
    mcp_max_retries: int = Field(default=2, ge=0)  # MCP 调用重试上限
    mcp_max_content_chars: int = Field(default=10_000, ge=1)  # 工具返回内容长度上限


class DedupConfig(BaseModel):
    """带外保存去重（content-hash 精确层 + 语义近似层，D4）。

    阈值默认取保守值：长记忆条目是个人事实
    误并（把两条近似但不同的偏好合并）比漏并（重复存一条）代价更高；
    `semantic_threshold` 只在 embeddings 可用时生效，未配 embedding 自动降级为
    仅 content-hash 精确去重。
    """

    enabled: bool = True
    semantic_threshold: float = Field(default=0.92, ge=0.0, le=1.0)


class SummarizeConfig(BaseModel):
    """滚动摘要行为（短期上下文管理）。

    `enabled` 关闭时 `summarize_history` 恒 no-op（零回归）；`max_summary_chars` 是
    摘要字符预算：提示词要求压缩到预算内，超限以硬截断兜底（滚动重写失败保底）。
    """

    enabled: bool = True
    max_summary_chars: int = Field(default=1500, ge=1)


class KeyFactsConfig(BaseModel):
    """会话关键信息行为（短期上下文管理）。

    `max_items` 是写入 state 的关键信息条数上限（超过按序截断，遗忘策略②的确定性兜底）。
    """

    enabled: bool = True
    max_items: int = Field(default=8, ge=1)


class MemoryBehaviorConfig(BaseModel):
    """记忆行为（长期记忆抽取 / 召回 / 保存去重参数）。"""

    top_k: int = Field(default=5, ge=1)
    # 长期记忆统一走 langgraph Store，dev=InMemoryStore / prod=PostgresStore。
    store_type: Literal["in_memory", "postgres"] = "in_memory"
    max_recall_chars: int = Field(default=8000, ge=1)  # 单次召回内容长度上限
    preload_profile: bool = True  # load_context 预加载 preference 记忆
    # 语义召回开关/模型对齐。
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # 语义召回混合重排参数（模型本体在 agent/share/models.py，core 与 memory 共用，防循环导入）。
    recall: MemoryRecallConfig = Field(default_factory=MemoryRecallConfig)
    # 带外保存去重（保存端 L1 content-hash + L2 语义）。
    dedup: DedupConfig = Field(default_factory=DedupConfig)

    # 短期上下文管理：滚动摘要 + 会话关键信息。
    summarize: SummarizeConfig = Field(default_factory=SummarizeConfig)
    keyfacts: KeyFactsConfig = Field(default_factory=KeyFactsConfig)


class SessionBehaviorConfig(BaseModel):
    """会话元数据行为（消息本体由 checkpointer 承载，本配置只管元数据层）。

    TTL 不做运行时命名空间约定：过期语义由 `sessions` 表（SQLAlchemy）的
    `expires_at` 承载，读路径过滤过期行（见 agent/session/repository.py）。
    """

    # TTL：折算为 expires_at 落库；None 表示永不过期。SQLite/Postgres 两方言均生效。
    ttl_minutes: int | None = Field(default=None, ge=1)


class AgentFrameworkConfig(BaseModel):
    """Agent 框架行为配置聚合（纯代码默认值，供装配层读取）。"""

    graph: AgentGraphConfig = Field(default_factory=AgentGraphConfig)
    plan: PlanConfig = Field(default_factory=PlanConfig)  # 多步任务编排
    clarify: ClarifyConfig = Field(default_factory=ClarifyConfig)  # 澄清式追问
    llm_behavior: LLMBehaviorConfig = Field(default_factory=LLMBehaviorConfig)
    tools: MCPToolsConfig = Field(default_factory=MCPToolsConfig)
    memory: MemoryBehaviorConfig = Field(default_factory=MemoryBehaviorConfig)
    session: SessionBehaviorConfig = Field(default_factory=SessionBehaviorConfig)

    @classmethod
    def get_default(cls) -> AgentFrameworkConfig:
        """返回框架行为默认配置。"""
        return cls()
