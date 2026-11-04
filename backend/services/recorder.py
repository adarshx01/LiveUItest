"""Persists execution evidence: steps, MCP audit log, events, traces, artifacts, recoveries.

Persistence is best-effort and never breaks a running execution. When no
database is configured (offline CLI), ``NullRecorder`` is used.
"""

from __future__ import annotations

import time
from typing import Any

from backend.agents.governance.redaction import redact_text, redact_value
from backend.core.logging import get_logger
from backend.dsl.results import RecoveryAttempt, StepResult
from backend.models.entities import (
    AgentTrace,
    Artifact,
    AssertionResultRecord,
    BrowserSessionEvent,
    BrowserSessionRecord,
    ExecutionEvent,
    ExecutionStep,
    RecoveryAttemptRecord,
)
from backend.services.artifact_service import ArtifactRef

log = get_logger(__name__)


class NullRecorder:
    async def audit_tool_call(self, record: dict[str, Any]) -> None: ...

    async def browser_session(self, session: Any) -> None: ...

    async def artifact(self, ref: ArtifactRef, step_id: str | None) -> None: ...

    async def step_result(self, result: StepResult) -> None: ...

    async def recovery(self, step_id: str, attempt: RecoveryAttempt) -> None: ...

    async def event(self, record: dict[str, Any]) -> None: ...

    async def trace(self, **kwargs: Any) -> None: ...


class DatabaseRecorder(NullRecorder):
    def __init__(self, sessionmaker: Any, execution_id: str) -> None:
        self.sessionmaker = sessionmaker
        self.execution_id = execution_id

    async def _add(self, *rows: Any) -> None:
        try:
            async with self.sessionmaker() as session:
                session.add_all(rows)
                await session.commit()
        except Exception as exc:  # pragma: no cover - persistence is best effort
            log.warning("recorder write failed", error=str(exc)[:200])

    async def audit_tool_call(self, record: dict[str, Any]) -> None:
        await self._add(BrowserSessionEvent(execution_id=self.execution_id, ts=record["ts"], tool=record["tool"],
                                            ok=record["ok"], latency_ms=record["latency_ms"], args=record["args"],
                                            error=record.get("error"), failure_category=record.get("failure_category"),
                                            page_url=record.get("page_url")))

    async def browser_session(self, session: Any) -> None:
        await self._add(BrowserSessionRecord(id=session.session_id, execution_id=self.execution_id,
                                             server_version=session.connection.server_version,
                                             options={"viewport": session.options.viewport,
                                                      "device": session.options.device,
                                                      "locale": session.options.locale}))

    async def artifact(self, ref: ArtifactRef, step_id: str | None) -> None:
        await self._add(Artifact(execution_id=self.execution_id, step_id=step_id, kind=ref.kind, name=ref.name,
                                 uri=ref.uri, content_type=ref.content_type, size=ref.size))

    async def step_result(self, result: StepResult) -> None:
        from sqlalchemy import delete

        try:
            async with self.sessionmaker() as session:
                await session.execute(delete(ExecutionStep).where(ExecutionStep.execution_id == self.execution_id,
                                                                  ExecutionStep.step_id == result.step_id))
                session.add(ExecutionStep(execution_id=self.execution_id, step_id=result.step_id, index=result.index,
                                          original_text=result.original_text, action=result.action,
                                          status=result.status.value, message=redact_text(result.message)[:4000],
                                          attempts=result.attempts, recovered=result.recovered,
                                          memory_used=result.memory_used, failure_category=result.failure_category,
                                          duration_ms=result.duration_ms,
                                          result=redact_value(result.model_dump(mode="json"))))
                for a in result.assertions:
                    session.add(AssertionResultRecord(
                        execution_id=self.execution_id, step_id=result.step_id, assertion_id=a.assertion_id,
                        type=a.type, status=a.status, expected=redact_value({"value": a.expected}),
                        actual=redact_value({"value": a.actual}), message=redact_text(a.message),
                        evidence=redact_value([e.model_dump() for e in a.evidence])))
                await session.commit()
        except Exception as exc:  # pragma: no cover
            log.warning("step persistence failed", error=str(exc)[:200])

    async def recovery(self, step_id: str, attempt: RecoveryAttempt) -> None:
        await self._add(RecoveryAttemptRecord(execution_id=self.execution_id, step_id=step_id,
                                              attempt=attempt.attempt, failure_category=attempt.failure_category,
                                              strategy=attempt.strategy, outcome=attempt.outcome,
                                              rationale=attempt.rationale, details=attempt.details))

    async def event(self, record: dict[str, Any]) -> None:
        if record.get("execution_id") != self.execution_id:
            return
        await self._add(ExecutionEvent(execution_id=self.execution_id, seq=record["seq"], ts=record["ts"],
                                       event=record["event"], step_id=record.get("step_id"),
                                       message=record.get("message", ""), payload=record.get("payload") or {}))

    async def trace(self, *, agent: str, node: str | None, step_id: str | None, status: str, duration_ms: float,
                    confidence: float | None = None, summary: str = "", details: dict[str, Any] | None = None) -> None:
        await self._add(AgentTrace(execution_id=self.execution_id, ts=time.time(), agent=agent, node=node,
                                   step_id=step_id, status=status, duration_ms=duration_ms, confidence=confidence,
                                   summary=summary[:2000], details=details or {}))
