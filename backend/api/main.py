"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import core, executions, memory
from backend.bootstrap import bootstrap
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.telemetry import memory_exporter
from backend.services.dispatcher import ExecutionDispatcher

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    platform = await bootstrap()
    app.state.platform = platform
    app.state.dispatcher = ExecutionDispatcher(platform.engine, platform.platform, platform.sessionmaker)
    from backend.services.scheduler import Scheduler

    scheduler = Scheduler(app.state.dispatcher, platform.sessionmaker)
    if get_settings().run_worker_in_api:
        scheduler.start()
    log.info("API ready")
    try:
        yield
    finally:
        await scheduler.stop()
        await platform.engine.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AI E2E Testing Platform", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(core.router)
    app.include_router(executions.router)
    app.include_router(memory.router)

    @app.get("/api/v1/health")
    async def health() -> dict[str, object]:
        platform = getattr(app.state, "platform", None)
        return {"status": "ok" if platform else "starting",
                "checkpointer": platform.engine.handle.kind if platform else None,
                "llm_provider": settings.llm_provider,
                "active_executions": len([r for r in (platform.engine.runs.values() if platform else [])
                                          if not r.done.is_set()])}

    @app.get("/api/v1/diagnostics/traces")
    async def traces(limit: int = 200) -> list[dict[str, object]]:
        return memory_exporter.spans[-limit:]

    return app


app = create_app()
