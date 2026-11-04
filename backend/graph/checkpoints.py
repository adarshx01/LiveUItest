"""Durable LangGraph checkpointing in PostgreSQL (spec section 6.3).

Production uses ``AsyncPostgresSaver``; the in-memory saver exists only for the
offline developer CLI and is refused in production.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger

log = get_logger(__name__)


class CheckpointerHandle:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.saver: Any = None
        self.kind = "none"

    async def open_postgres(self) -> Any:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        settings = get_settings()
        pool = AsyncConnectionPool(conninfo=settings.checkpoint_database_url, max_size=10, open=False,
                                   kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        await pool.open()
        self._stack.push_async_callback(pool.close)
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()
        self.saver, self.kind = saver, "postgres"
        return saver

    def open_memory(self) -> Any:
        if get_settings().is_production:
            raise ConfigurationError("The in-memory checkpointer is not allowed in production")
        from langgraph.checkpoint.memory import InMemorySaver

        log.warning("using in-memory LangGraph checkpointer (development only; executions are not durable)")
        self.saver, self.kind = InMemorySaver(), "memory"
        return self.saver

    async def close(self) -> None:
        await self._stack.aclose()
