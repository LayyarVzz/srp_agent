# 根 main.py —— 仅做 re-export（真正的 app 在 app/main.py），保持 `uvicorn main:app` 兼容。
from app.main import app  # noqa: F401
