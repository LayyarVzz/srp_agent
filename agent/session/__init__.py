"""会话与作用域管理。

`SessionManager` 在 langgraph BaseStore 上登记会话元数据（session_id↔user_id 映射 +
created_at + meta），消息本体由 checkpointer 按 `thread_id == session_id` 承载
（见 agent/core/graph.py 的 load_context）。
"""

from agent.session.manager import SESSIONS_NAMESPACE, SessionManager
from agent.session.models import SessionContext

__all__ = [
    "SESSIONS_NAMESPACE",
    "SessionContext",
    "SessionManager",
]
