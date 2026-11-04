"""Wires process-level services (DB, secrets, graph services, engine) for API, worker and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.agents.governance.secret_manager import SecretManager, set_secret_manager
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.core.telemetry import configure_tracing
from backend.graph.services import GraphServices, set_graph_services
from backend.models.database import get_sessionmaker, init_db
from backend.services.execution_engine import ExecutionEngine
from backend.services.platform_service import PlatformService
from backend.services.secret_store import DatabaseSecretBackend


@dataclass
class Platform:
    engine: ExecutionEngine
    platform: PlatformService
    secrets: SecretManager
    sessionmaker: Any


async def bootstrap(*, checkpointer: str = "postgres", json_logs: bool | None = None) -> Platform:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json if json_logs is None else json_logs)
    configure_tracing(settings.service_name)
    await init_db()
    sessionmaker = get_sessionmaker()
    secrets = SecretManager(DatabaseSecretBackend(sessionmaker))
    set_secret_manager(secrets)
    set_graph_services(GraphServices(secrets=secrets, sessionmaker=sessionmaker))
    platform = PlatformService(sessionmaker, secrets)
    engine = ExecutionEngine(checkpointer=checkpointer, platform=platform)
    await engine.start()
    return Platform(engine=engine, platform=platform, secrets=secrets, sessionmaker=sessionmaker)
