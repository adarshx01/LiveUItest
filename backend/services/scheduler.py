"""Execution schedules: periodically start test cases or suites (spec section 20)."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

from sqlalchemy import select

from backend.core.logging import get_logger
from backend.models.entities import Schedule, TestSuite

log = get_logger(__name__)


class Scheduler:
    def __init__(self, dispatcher: Any, sessionmaker: Any, interval_s: float = 30.0) -> None:
        self.dispatcher = dispatcher
        self.sessionmaker = sessionmaker
        self.interval_s = interval_s
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as exc:  # pragma: no cover - keep scheduling
                log.warning("scheduler tick failed", error=str(exc)[:200])
            await asyncio.sleep(self.interval_s)

    async def tick(self, now: dt.datetime | None = None) -> list[str]:
        now = now or dt.datetime.now(dt.UTC)
        started: list[str] = []
        async with self.sessionmaker() as s:
            schedules = (await s.execute(select(Schedule).where(Schedule.enabled.is_(True)))).scalars().all()
        for schedule in schedules:
            due = schedule.last_run_at is None or now - schedule.last_run_at >= dt.timedelta(
                minutes=schedule.interval_minutes)
            if not due:
                continue
            test_case_ids = [schedule.target_id]
            if schedule.target_type == "suite":
                async with self.sessionmaker() as s:
                    suite = await s.get(TestSuite, schedule.target_id)
                    test_case_ids = list(suite.test_case_ids) if suite else []
            for tc_id in test_case_ids:
                run = await self.dispatcher.start_execution(
                    org_id=schedule.organization_id, test_case_id=tc_id, version=None,
                    variables=schedule.options.get("variables", {}), options=schedule.options.get("options", {}),
                    allowed_domains=[], environment_id=None, actor=f"schedule:{schedule.id}")
                started.append(run.id)
            async with self.sessionmaker() as s:
                row = await s.get(Schedule, schedule.id)
                row.last_run_at = now
                await s.commit()
        return started
