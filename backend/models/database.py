"""Async database engine, sessions and schema bootstrap.

Local-first by default: a single SQLite file (``data/aitest.db``) needs no server, Docker or admin
rights. PostgreSQL (with pgvector) is the production option and is selected purely by URL. The schema
is created and upgraded automatically on startup: missing tables are created and missing columns are
added, so pulling a newer version never needs a manual migration step.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger
from backend.models.entities import Base

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _sqlite_path(url: str) -> Path | None:
    if not url.startswith("sqlite"):
        return None
    raw = url.split("///", 1)[1] if "///" in url else ""
    return None if raw in ("", ":memory:") else Path(raw)


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        url = get_settings().database_url
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            path = _sqlite_path(url)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
            # one writer at a time; wait instead of failing while another task commits
            kwargs["connect_args"] = {"timeout": 30}
        else:
            kwargs.update(pool_size=10, max_overflow=10)
        _engine = create_async_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(_engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:  # pragma: no cover - driver hook
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")  # readers never block the writer
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


def dialect_name() -> str:
    return get_engine().dialect.name


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


def _add_missing_columns(sync_conn: Any) -> list[str]:
    """Lightweight forward migration: add columns that the models have and the database lacks."""

    inspector = inspect(sync_conn)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            col_type = column.type.compile(dialect=sync_conn.dialect)
            sync_conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'))
            added.append(f"{table.name}.{column.name}")
    return added


async def init_db(*, drop: bool = False) -> None:
    engine = get_engine()
    postgres = engine.dialect.name == "postgresql"
    try:
        await _init_schema(engine, postgres, drop)
    except (OSError, SQLAlchemyError) as exc:
        url = engine.url
        where = f"{url.host}:{url.port or 5432}/{url.database}" if postgres else str(url.database)
        hint = ("start the PostgreSQL server (with the pgvector extension), or remove AITEST_DATABASE_URL from .env "
                "to use the local SQLite database" if postgres else "check that the data folder is writable and "
                                                                  "not opened exclusively by another program")
        raise ConfigurationError(f"Cannot open the database at {where}: {str(exc).splitlines()[0][:200]}",
                                 details={"fix": hint}) from exc
    log.info("database schema ready", dialect=engine.dialect.name)


async def _init_schema(engine: AsyncEngine, postgres: bool, drop: bool) -> None:
    async with engine.begin() as conn:
        if postgres:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        if drop:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        added = await conn.run_sync(_add_missing_columns)
        if added:
            log.info("database schema upgraded", added_columns=added)
        if postgres:
            for table in ("locator_memory", "page_memory"):
                await conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS ix_{table}_embedding ON {table} "
                    "USING hnsw (embedding vector_cosine_ops)"))


async def clear_tables(session: AsyncSession, tables: list[str] | tuple[str, ...]) -> None:
    """Empty tables portably (TRUNCATE on PostgreSQL, DELETE on SQLite)."""

    if session.bind.dialect.name == "postgresql":
        await session.execute(text(f"TRUNCATE {', '.join(tables)}"))
    else:
        for table in tables:
            await session.execute(text(f'DELETE FROM "{table}"'))


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine, _sessionmaker = None, None
