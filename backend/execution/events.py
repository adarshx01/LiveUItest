"""Execution event bus (spec section 21.1).

Events are published to in-process subscribers (WebSocket handlers in the same
process), to Redis pub/sub (WebSockets served by other API replicas) and
persisted to ``agent_events`` for replay. Payloads are redacted.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any

from backend.agents.governance.redaction import redact_value
from backend.core.config import get_settings
from backend.core.logging import get_logger

log = get_logger(__name__)

EVENT_TYPES = {
    "EXECUTION_CREATED", "PLAN_COMPILED", "PLAN_INVALID", "BROWSER_INITIALIZED", "PAGE_NAVIGATED", "PAGE_OBSERVED",
    "ELEMENT_DISCOVERED", "TARGET_GROUNDED", "ACTION_PLANNED", "ACTION_VALIDATED", "ACTION_BLOCKED", "ACTION_STARTED",
    "ACTION_COMPLETED", "ACTION_VERIFIED", "ACTION_FAILED", "MEMORY_RETRIEVED", "MEMORY_USED", "MEMORY_UPDATED",
    "RECOVERY_STARTED", "RECOVERY_COMPLETED", "RECOVERY_FAILED", "HUMAN_APPROVAL_REQUIRED", "EXECUTION_PAUSED",
    "EXECUTION_RESUMED", "ASSERTION_PASSED", "ASSERTION_FAILED", "STEP_STARTED", "STEP_COMPLETED",
    "SCREENSHOT_CAPTURED", "SECURITY_WARNING", "EXECUTION_COMPLETED", "EXECUTION_CANCELLED", "REPORT_READY",
    "RUNNER_SELECTED",
}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._seq: dict[str, int] = defaultdict(int)
        self._redis: Any = None
        self._redis_checked = False
        self.persist_hook: Any = None  # async callable(event dict)

    async def _get_redis(self) -> Any:
        if self._redis_checked:
            return self._redis
        self._redis_checked = True
        settings = get_settings()
        if not settings.use_redis:
            return None
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(settings.redis_url, decode_responses=True)
            await asyncio.wait_for(client.ping(), timeout=1.5)
            self._redis = client
        except Exception as exc:  # pragma: no cover - environment dependent
            log.warning("redis unavailable; events stay in-process", error=str(exc)[:120])
            self._redis = None
        return self._redis

    async def publish(self, execution_id: str, event: str, message: str = "", *, step_id: str | None = None,
                      **payload: Any) -> dict[str, Any]:
        self._seq[execution_id] += 1
        record = {"event": event, "execution_id": execution_id, "seq": self._seq[execution_id], "ts": time.time(),
                  "step_id": step_id, "message": str(redact_value(message))[:1000],
                  "payload": redact_value(payload)}
        for queue in list(self._subscribers.get(execution_id, ())) + list(self._subscribers.get("*", ())):
            if queue.qsize() < 1000:
                queue.put_nowait(record)
        redis = await self._get_redis()
        if redis is not None:
            try:
                await redis.publish(f"aitest:events:{execution_id}", json.dumps(record, default=str))
            except Exception:  # pragma: no cover
                pass
        if self.persist_hook is not None:
            try:
                await self.persist_hook(record)
            except Exception as exc:  # pragma: no cover - persistence must never break execution
                log.warning("event persistence failed", error=str(exc)[:200])
        return record

    def subscribe(self, execution_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers[execution_id].add(queue)
        return queue

    def unsubscribe(self, execution_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers[execution_id].discard(queue)


event_bus = EventBus()
