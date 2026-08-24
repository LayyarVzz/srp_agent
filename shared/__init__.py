"""根级共享基础设施（agent 与 services 共用；仅放 ≥2 消费方的单元）。

与 `agent/share/`（agent 内部跨子模块共享）区分：本包是 agent 与 services 共同的中性层。
"""

from shared.embeddings import EmbeddingConfig, EmbeddingsFactory

__all__ = ["EmbeddingConfig", "EmbeddingsFactory"]
