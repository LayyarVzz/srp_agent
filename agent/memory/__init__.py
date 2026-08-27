"""记忆：长期记忆存储/召回，统一经 MemoryStore（langgraph Store 适配）访问。

P4-2 起额外导出：结构化抽取（MemoryExtractor / MemoryExtraction*）与
带外非阻塞保存（submit_memory_save / save_conversation_memory / wait_pending_saves）。
"""

from agent.memory.adapter import (
    KIND_EPISODE,
    KIND_FACT,
    KIND_OTHER,
    KIND_PREFERENCE,
    KNOWN_KINDS,
    LONG_TERM_NAMESPACE,
    MemoryStore,
    store_has_embeddings,
)
from agent.memory.extractor import MemoryExtractor
from agent.memory.factory import MemoryBackends, build_memory_backends
from agent.memory.judge import MemoryRelationJudge
from agent.memory.models import (
    MERGEABLE_RELATIONS,
    RELATION_EXACT,
    RELATION_NOT_DUPLICATE,
    RELATION_OVERLAP,
    CandidateVerdict,
    MemoryExtraction,
    MemoryExtractionResult,
    MemoryItem,
    MemoryRecallResult,
    MergeDecisionResult,
    SaveOutcome,
    normalize_content_hash,
)
from agent.memory.persist import (
    PROVENANCE_CONVERSATION,
    save_conversation_memory,
    submit_memory_save,
    wait_pending_saves,
)
from agent.share.models import MemoryRecallConfig

__all__ = [
    "KIND_EPISODE",
    "KIND_FACT",
    "KIND_OTHER",
    "KIND_PREFERENCE",
    "KNOWN_KINDS",
    "LONG_TERM_NAMESPACE",
    "MERGEABLE_RELATIONS",
    "PROVENANCE_CONVERSATION",
    "RELATION_EXACT",
    "RELATION_NOT_DUPLICATE",
    "RELATION_OVERLAP",
    "CandidateVerdict",
    "MemoryBackends",
    "MemoryExtraction",
    "MemoryExtractionResult",
    "MemoryExtractor",
    "MemoryItem",
    "MemoryRecallConfig",
    "MemoryRecallResult",
    "MemoryRelationJudge",
    "MemoryStore",
    "MergeDecisionResult",
    "SaveOutcome",
    "build_memory_backends",
    "normalize_content_hash",
    "save_conversation_memory",
    "store_has_embeddings",
    "submit_memory_save",
    "wait_pending_saves",
]
