"""Durable LangGraph checkpointing (spec section 6.3).

``auto`` follows the database: a SQLite file next to ``aitest.db`` locally (durable, no server),
``AsyncPostgresSaver`` in production. The in-memory saver exists only for throwaway CLI runs and is
refused in production.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
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
        pool = AsyncConnectionPool(conninfo=settings.resolved_checkpoint_url(), max_size=10, open=False,
                                   kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        await pool.open()
        self._stack.push_async_callback(pool.close)
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()
        self.saver, self.kind = saver, "postgres"
        return saver

    async def open_sqlite(self) -> Any:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        url = get_settings().resolved_checkpoint_url()
        path = Path(url.split("///", 1)[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(path), timeout=30)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=30000")
        self._stack.push_async_callback(conn.close)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        self.saver, self.kind = saver, "sqlite"
        return saver

    async def open(self, kind: str = "auto") -> Any:
        """``auto`` -> SQLite or PostgreSQL depending on the checkpoint URL; ``memory`` for throwaway runs."""

        if kind == "memory":
            return self.open_memory()
        if kind == "postgres" or (kind == "auto" and get_settings().resolved_checkpoint_url().startswith("postgres")):
            return await self.open_postgres()
        return await self.open_sqlite()

    def open_memory(self) -> Any:
        if get_settings().is_production:
            raise ConfigurationError("The in-memory checkpointer is not allowed in production")
        from langgraph.checkpoint.memory import InMemorySaver

        log.warning("using in-memory LangGraph checkpointer (development only; executions are not durable)")
        self.saver, self.kind = InMemorySaver(), "memory"
        return self.saver

    async def close(self) -> None:
        await self._stack.aclose()
