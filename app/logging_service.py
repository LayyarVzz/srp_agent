from __future__ import annotations

import json
import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "agent_interaction.log"
MAX_RECENT_LOGS = 200

_recent_logs: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_LOGS)


def setup_interaction_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("srp_agent.interaction")
    if any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        return
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(file_handler)


def record_event(
    event: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    payload: dict[str, Any] | None = None,
    level: str = "INFO",
) -> dict[str, Any]:
    item = {
        "id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "event": event,
        "session_id": session_id,
        "user_id": user_id,
        "payload": payload or {},
    }
    _recent_logs.appendleft(item)

    logger = logging.getLogger("srp_agent.interaction")
    message = json.dumps(item, ensure_ascii=False)
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)
    return item


def recent_logs(limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(limit, MAX_RECENT_LOGS))
    return list(_recent_logs)[:limit]
