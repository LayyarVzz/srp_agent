from fastapi import FastAPI

from app.errors import InteractionError, interaction_error_handler
from app.logging_service import setup_interaction_logging
from app.routes import router as interaction_router

setup_interaction_logging()

app = FastAPI()
app.add_exception_handler(InteractionError, interaction_error_handler)
app.include_router(interaction_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Hello from srp-agent!",
        "modules": ["text interaction", "voice interaction", "recent logs"],
    }


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "app": "srp-agent",
        "env": "dev",
        "llm_configured": True,
    }
