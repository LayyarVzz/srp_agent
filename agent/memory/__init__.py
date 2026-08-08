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
)
from agent.memory.extractor import MemoryExtractor
from agent.memory.factory import build_store
from agent.memory.models import (
    MemoryExtraction,
    MemoryExtractionResult,
    MemoryItem,
    MemoryRecallResult,
)
from agent.memory.persist import (
    PROVENANCE_CONVERSATION,
    save_conversation_memory,
    submit_memory_save,
    wait_pending_saves,
)

__all__ = [
    "KIND_EPISODE",
    "KIND_FACT",
    "KIND_OTHER",
    "KIND_PREFERENCE",
    "KNOWN_KINDS",
    "LONG_TERM_NAMESPACE",
    "PROVENANCE_CONVERSATION",
    "MemoryExtraction",
    "MemoryExtractionResult",
    "MemoryExtractor",
    "MemoryItem",
    "MemoryRecallResult",
    "MemoryStore",
    "build_store",
    "save_conversation_memory",
    "submit_memory_save",
    "wait_pending_saves",
]
