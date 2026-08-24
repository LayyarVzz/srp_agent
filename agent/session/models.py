"""会话元数据模型。

会话**消息本体**由 LangGraph checkpointer 按 `thread_id == session_id` 承载；
本模块只承载「会话身份与作用域」元数据（归属映射 + 创建时间 + 扩展位），不重复存储消息。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SessionContext(BaseModel):
    """一条会话元数据记录（session_id↔user_id 映射 + 创建时间 + 扩展位）。

    存储值 = `model_dump(mode="json")`（datetime → ISO 串，InMemoryStore / PostgresStore
    均可 JSON 序列化）；读回 `SessionContext.model_validate(item.value)`。
    """

    session_id: str
    user_id: str
    created_at: datetime
    meta: dict[str, object] = Field(default_factory=dict)  # 扩展位（语言 / 设备等）
