"""Execution dispatch: run locally or enqueue for worker processes (Redis Streams).

Approvals created when an execution pauses are stored in ``approval_events`` so
reviewers can act on them from the API/UI; resuming routes back to the worker
that owns the live browser session.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any

from sqlalchemy import select

from backend.core.config import get_settings
from backend.core.exceptions import NotFound
from backend.core.ids import new_id
from backend.core.logging import get_logger
from backend.models.entities import Approval, AuditLog, ExecutionRun
from backend.services.execution_engine import ExecutionEngine, RunHandle, initial_state
from backend.services.platform_service import PlatformService

log = get_logger(__name__)

STREAM = "aitest:executions"
GROUP = "aitest-workers"


class ExecutionDispatcher:
    def __init__(self, engine: ExecutionEngine, platform: PlatformService, sessionmaker: Any) -> None:
        self.engine = engine
        self.platform = platform
        self.sessionmaker = sessionmaker
        self.engine_on_pause_installed = False
        self._watchers: set[asyncio.Task[Any]] = set()

    async def start_execution(self, *, org_id: str, test_case_id: str, version: int | None, variables: dict[str, str],
                              options: dict[str, Any], allowed_domains: list[str], environment_id: str | None,
                              actor: str, retry_of: str | None = None, suite_run_id: str | None = None) -> ExecutionRun:
        tc, tcv = await self.platform.get_test_case(org_id, test_case_id, version)
        run = await self.platform.create_run(org_id=org_id, tc=tc, version=tcv.version, options=options,
                                             environment_id=environment_id, retry_of=retry_of,
                                             suite_run_id=suite_run_id, created_by=actor)
        state = initial_state(execution_id=run.id, organization_id=org_id, project_id=tc.project_id,
                              application_id=tc.application_id, test_case_id=tc.id, test_case_version=tcv.version,
                              spec=tcv.spec, environment_id=environment_id,
                              raw_input={"name": tc.name, "variables": variables, "options": options,
                                         "allowed_domains": allowed_domains})
        await self.audit(org_id, actor, "execution.start", "execution", run.id, {"test_case_id": tc.id})
        settings = get_settings()
        if settings.run_worker_in_api:
            handle = self.engine.submit(state)
            self._watch(handle, org_id)
        else:
            await self._enqueue(state)
        return run

    def _watch(self, handle: RunHandle, org_id: str) -> None:
        async def watcher() -> None:
            while True:
                done = asyncio.ensure_future(handle.done.wait())
                paused = asyncio.ensure_future(handle.paused.wait())
                await asyncio.wait([done, paused], return_when=asyncio.FIRST_COMPLETED)
                done.cancel()
                paused.cancel()
                if handle.done.is_set():
                    return
                if handle.paused.is_set() and handle.interrupt is not None:
                    await self._create_approval(org_id, handle)
                    handle.paused.clear()
                    while handle.interrupt is not None and not handle.done.is_set():
                        await asyncio.sleep(0.5)

        task = asyncio.create_task(watcher())
        self._watchers.add(task)
        task.add_done_callback(self._watchers.discard)

    async def _create_approval(self, org_id: str, handle: RunHandle) -> None:
        request = handle.interrupt or {}
        async with self.sessionmaker() as s:
            existing = (await s.execute(select(Approval).where(Approval.execution_id == handle.execution_id,
                                                               Approval.status == "pending"))).scalars().first()
            if existing is not None:
                return
            s.add(Approval(id=new_id("apr"), organization_id=org_id, execution_id=handle.execution_id,
                           step_id=request.get("step_id"), kind=request.get("kind", "approval"), status="pending",
                           request=json.loads(json.dumps(request, default=str))))
            await s.commit()

    async def decide(self, org_id: str, approval_id: str, *, approved: bool, actor: str, note: str | None) -> Approval:
        async with self.sessionmaker() as s:
            approval = await s.get(Approval, approval_id)
            if approval is None or approval.organization_id != org_id:
                raise NotFound("Approval not found")
            if approval.status != "pending":
                return approval
            approval.status = "approved" if approved else "rejected"
            approval.decided_by = actor
            approval.decision_note = note
            approval.decided_at = dt.datetime.now(dt.UTC)
            await s.commit()
        await self.audit(org_id, actor, f"approval.{approval.status}", "approval", approval_id,
                         {"execution_id": approval.execution_id, "note": note})
        await self.engine.resume(approval.execution_id, {"approved": approved, "by": actor, "note": note})
        return approval

    async def audit(self, org_id: str, actor: str, action: str, resource_type: str, resource_id: str,
                    details: dict[str, Any]) -> None:
        try:
            async with self.sessionmaker() as s:
                s.add(AuditLog(organization_id=org_id, actor=actor, action=action, resource_type=resource_type,
                               resource_id=resource_id, details=details))
                await s.commit()
        except Exception as exc:  # pragma: no cover
            log.warning("audit write failed", error=str(exc)[:200])

    # ------------------------------------------------------------------ worker queue
    async def _enqueue(self, state: dict[str, Any]) -> None:
        import redis.asyncio as aioredis

        client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
        await client.xadd(STREAM, {"state": json.dumps(state, default=str)})
        await client.aclose()


async def worker_loop(dispatcher: ExecutionDispatcher, worker_id: str) -> None:  # pragma: no cover - process entry
    """Consume execution requests from Redis Streams (one browser context per execution)."""

    import redis.asyncio as aioredis

    client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        await client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception:
        pass
    while True:
        entries = await client.xreadgroup(GROUP, worker_id, {STREAM: ">"}, count=1, block=5000)
        for _stream, messages in entries or []:
            for message_id, fields in messages:
                state = json.loads(fields["state"])
                handle = dispatcher.engine.submit(state)
                dispatcher._watch(handle, state["organization_id"])
                await client.xack(STREAM, GROUP, message_id)
