"""Standalone execution worker: consumes queued executions from Redis Streams.

Run several workers for horizontal scale; each execution gets its own isolated
browser context and the engine bounds per-worker concurrency.
"""

from __future__ import annotations

import asyncio
import os
import socket

from backend.bootstrap import bootstrap
from backend.services.dispatcher import ExecutionDispatcher, worker_loop


async def main() -> None:
    platform = await bootstrap()
    dispatcher = ExecutionDispatcher(platform.engine, platform.platform, platform.sessionmaker)
    worker_id = os.environ.get("AITEST_WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}"
    try:
        await worker_loop(dispatcher, worker_id)
    finally:
        await platform.engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
