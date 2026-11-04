"""Execution engine: owns the compiled graph + durable checkpointer and runs executions.

Executions run as background tasks bounded by a concurrency limit (parallelism
across isolated browser contexts, never across steps of one page). Interrupts
(approvals, human verification) park the execution with its live browser; the
API resumes it with ``Command(resume=...)``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from backend.agents.governance.redaction import redact_text
from backend.core.config import get_settings
from backend.core.exceptions import NotFound
from backend.core.logging import bind_log_context, get_logger, reset_log_context
from backend.execution.events import event_bus
from backend.execution.runtime import runtime_registry
from backend.graph.checkpoints import CheckpointerHandle
from backend.graph.graph_builder import compile_graph
from backend.graph.services import get_graph_services

log = get_logger(__name__)


@dataclass
class RunHandle:
    execution_id: str
    task: asyncio.Task[Any] | None = None
    status: str = "QUEUED"
    interrupt: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    paused: asyncio.Event = field(default_factory=asyncio.Event)


class ExecutionEngine:
    def __init__(self, *, checkpointer: str = "postgres", platform: Any = None) -> None:
        self.checkpointer_kind = checkpointer
        self.handle = CheckpointerHandle()
        self.graph: Any = None
        self.platform = platform
        self.runs: dict[str, RunHandle] = {}
        self._semaphore = asyncio.Semaphore(get_settings().max_concurrent_executions)

    async def start(self) -> None:
        saver = await self.handle.open_postgres() if self.checkpointer_kind == "postgres" else self.handle.open_memory()
        self.graph = compile_graph(saver)

    async def stop(self) -> None:
        for runtime in runtime_registry.all():
            await runtime.close()
        await self.handle.close()

    def _config(self, execution_id: str) -> dict[str, Any]:
        steps = get_settings().budget.max_steps
        return {"configurable": {"thread_id": execution_id}, "recursion_limit": 40 * steps + 200}

    def submit(self, state: dict[str, Any]) -> RunHandle:
        execution_id = state["execution_id"]
        handle = RunHandle(execution_id)
        self.runs[execution_id] = handle
        handle.task = asyncio.create_task(self._run(handle, state), name=f"exec-{execution_id}")
        return handle

    async def run_to_completion(self, state: dict[str, Any]) -> RunHandle:
        handle = self.submit(state)
        await asyncio.wait([asyncio.ensure_future(handle.done.wait()), asyncio.ensure_future(handle.paused.wait())],
                           return_when=asyncio.FIRST_COMPLETED)
        return handle

    async def _run(self, handle: RunHandle, payload: Any, *, resume: bool = False) -> None:
        execution_id = handle.execution_id
        token = bind_log_context(execution_id=execution_id)
        handle.paused.clear()
        try:
            async with self._semaphore:
                handle.status = "RUNNING"
                await self._update_run(execution_id, status="RUNNING", started_at=_now() if not resume else None)
                await self.graph.ainvoke(payload, self._config(execution_id))
                snapshot = await self.graph.aget_state(self._config(execution_id))
                interrupts = [i for task in snapshot.tasks for i in (task.interrupts or [])]
                if interrupts:
                    value = interrupts[0].value
                    handle.interrupt = value if isinstance(value, dict) else {"value": value}
                    handle.status = "AWAITING_HUMAN" if handle.interrupt.get("kind") == "captcha" else "AWAITING_APPROVAL"
                    await self._update_run(execution_id, status=handle.status)
                    await event_bus.publish(execution_id, "EXECUTION_PAUSED", handle.status, **handle.interrupt)
                    handle.paused.set()
                    return
                values = snapshot.values
                handle.final = {"final_result": values.get("final_result"), "report": values.get("report"),
                                "step_results": values.get("step_results"), "plan_issues": values.get("plan_issues"),
                                "execution_status": values.get("execution_status")}
                handle.status = values.get("execution_status") or "COMPLETED"
                await self._finish_run(execution_id, values)
        except asyncio.CancelledError:
            handle.status = "CANCELLED"
            await self._update_run(execution_id, status="CANCELLED", finished_at=_now())
            runtime = runtime_registry.get(execution_id)
            if runtime is not None:
                await runtime.close()
            raise
        except Exception as exc:  # pragma: no cover - surfaced in run status
            log.exception("execution crashed")
            handle.status = "ERROR"
            await self._update_run(execution_id, status="ERROR", reason=redact_text(str(exc))[:500], finished_at=_now())
            runtime = runtime_registry.get(execution_id)
            if runtime is not None:
                await runtime.close()
        finally:
            reset_log_context(token)
            if handle.status not in ("AWAITING_APPROVAL", "AWAITING_HUMAN"):
                runtime_registry.remove(execution_id)
                handle.done.set()

    async def _finish_run(self, execution_id: str, values: dict[str, Any]) -> None:
        final = values.get("final_result") or {}
        report = values.get("report") or {}
        runtime = runtime_registry.get(execution_id)
        metrics: dict[str, Any] = {}
        if runtime is not None:
            from backend.agents.reporting.report_generator import ReportGenerator

            steps = list(runtime.results.values())
            metrics = ReportGenerator._metrics(steps, runtime).model_dump()
        flaky = await self._is_flaky(values, final.get("outcome"))
        await self._update_run(execution_id, status=values.get("execution_status") or "COMPLETED",
                               outcome=final.get("outcome"), reason=redact_text(final.get("reason", ""))[:2000],
                               flaky=flaky,
                               runner=values.get("runner"), metrics=metrics, report_uri=report.get("html_uri"),
                               report_json_uri=report.get("json_uri"), video_uri=report.get("video_uri"),
                               finished_at=_now())

    async def _is_flaky(self, values: dict[str, Any], outcome: str | None) -> bool:
        """Same test version, same environment, diverging pass/fail outcomes in recent runs => flaky."""

        if self.platform is None or outcome not in ("PASSED", "FAILED"):
            return False
        try:
            runs = await self.platform.list_runs(values["organization_id"], test_case_id=values["test_case_id"],
                                                 limit=10)
        except Exception:  # pragma: no cover
            return False
        recent = {r.outcome for r in runs if r.test_case_version == values.get("test_case_version")
                  and r.outcome in ("PASSED", "FAILED") and r.id != values.get("execution_id")}
        return bool(recent) and (recent | {outcome}) == {"PASSED", "FAILED"}

    async def _update_run(self, execution_id: str, **fields: Any) -> None:
        if self.platform is None:
            return
        fields = {k: v for k, v in fields.items() if v is not None}
        try:
            await self.platform.update_run(execution_id, **fields)
        except Exception as exc:  # pragma: no cover
            log.warning("run status update failed", error=str(exc)[:200])

    async def resume(self, execution_id: str, value: dict[str, Any]) -> RunHandle:
        handle = self.runs.get(execution_id)
        if handle is None or handle.status not in ("AWAITING_APPROVAL", "AWAITING_HUMAN"):
            raise NotFound("No paused execution with this id in this worker")
        if runtime_registry.get(execution_id) is None:
            raise NotFound("The paused browser session is no longer available; re-run the test")
        handle.interrupt = None
        await event_bus.publish(execution_id, "EXECUTION_RESUMED", "resumed by a human")
        handle.task = asyncio.create_task(self._run(handle, Command(resume=value), resume=True))
        return handle

    async def cancel(self, execution_id: str) -> bool:
        runtime = runtime_registry.get(execution_id)
        if runtime is not None:
            runtime.cancelled = True
            runtime.budget.cancelled = True
        handle = self.runs.get(execution_id)
        if handle is not None and handle.status in ("AWAITING_APPROVAL", "AWAITING_HUMAN"):
            await self.resume(execution_id, {"approved": False, "abort": True})
        await event_bus.publish(execution_id, "EXECUTION_CANCELLED", "cancellation requested")
        return runtime is not None or handle is not None

    async def state(self, execution_id: str) -> dict[str, Any]:
        snapshot = await self.graph.aget_state(self._config(execution_id))
        return dict(snapshot.values or {})


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def initial_state(*, execution_id: str, organization_id: str, project_id: str, application_id: str,
                  test_case_id: str, test_case_version: int, spec: dict[str, Any] | None,
                  raw_input: dict[str, Any], environment_id: str | None = None) -> dict[str, Any]:
    from backend.agents.governance.secret_manager import seal_variables

    plain, sealed = seal_variables(raw_input.get("variables"))
    raw_input = {**raw_input, "variables": plain}
    if sealed is not None:  # merge with a bundle the caller already sealed (retry / re-run of a run)
        previous = raw_input.get("sealed_variables")
        if previous:
            from backend.agents.governance.secret_manager import unseal_variables

            _, sealed = seal_variables({**unseal_variables(previous), **unseal_variables(sealed)})
        raw_input["sealed_variables"] = sealed
    return {
        "run_id": execution_id, "execution_id": execution_id, "organization_id": organization_id,
        "project_id": project_id, "application_id": application_id, "environment_id": environment_id,
        "test_case_id": test_case_id, "test_case_version": test_case_version,
        "test_spec": spec or {}, "raw_input": raw_input, "queue": [], "step_results": {},
        "execution_history": [], "approval_decisions": {}, "recovery_attempts": 0, "recovery_total": 0,
        "execution_status": "QUEUED", "phase": "start",
    }


_ = get_graph_services
