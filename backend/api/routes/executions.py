"""Executions, steps, artifacts, reports, events (WebSocket), approvals, live browser."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from sqlalchemy import select

from backend.api.deps import PlatformDep, Principal, PrincipalDep, principal
from backend.api.schemas import ApprovalDecision, ExecutionCreate, HumanBrowserAction, ResumeRequest, SuiteRun
from backend.bootstrap import Platform
from backend.core.exceptions import NotFound
from backend.core.security import verify_api_key
from backend.execution.events import event_bus
from backend.execution.runtime import runtime_registry
from backend.models.entities import AgentTrace, Approval, Artifact, BrowserSessionEvent, ExecutionEvent, ExecutionStep
from backend.services.artifact_service import get_artifact_service

router = APIRouter(prefix="/api/v1")


def _dispatcher(request: Request) -> Any:
    return request.app.state.dispatcher


def _run(run: Any) -> dict[str, Any]:
    return {"id": run.id, "test_case_id": run.test_case_id, "test_case_version": run.test_case_version,
            "application_id": run.application_id, "status": run.status, "outcome": run.outcome, "reason": run.reason,
            "runner": run.runner, "metrics": run.metrics, "flaky": run.flaky, "retry_of": run.retry_of,
            "created_at": run.created_at, "started_at": run.started_at, "finished_at": run.finished_at,
            "report": f"/api/v1/executions/{run.id}/report" if run.report_uri else None,
            "video": run.video_uri}


@router.post("/executions", status_code=201)
async def create_execution(body: ExecutionCreate, request: Request, who: Principal = PrincipalDep) -> Any:
    dispatcher = _dispatcher(request)
    datasets = body.datasets or [body.variables]
    runs = []
    try:
        for dataset in datasets:  # data-driven runner: one isolated execution per data set
            runs.append(await dispatcher.start_execution(
                org_id=who.organization_id, test_case_id=body.test_case_id, version=body.version,
                variables={**body.variables, **dataset}, options=body.options, allowed_domains=body.allowed_domains,
                environment_id=body.environment_id, actor=who.actor))
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc
    return _run(runs[0]) if len(runs) == 1 else {"executions": [_run(r) for r in runs]}


@router.post("/test-suites/{suite_id}/run", status_code=201)
async def run_suite(suite_id: str, body: SuiteRun, request: Request, who: Principal = PrincipalDep,
                    pf: Platform = PlatformDep) -> dict[str, Any]:
    from backend.core.ids import new_id

    suite = await pf.platform.get_suite(who.organization_id, suite_id)
    suite_run_id = new_id("suite_run")
    runs = [await _dispatcher(request).start_execution(
        org_id=who.organization_id, test_case_id=tc_id, version=None, variables=body.variables, options=body.options,
        allowed_domains=[], environment_id=None, actor=who.actor, suite_run_id=suite_run_id)
        for tc_id in suite.test_case_ids]  # isolated contexts, bounded by the engine's concurrency limit
    return {"suite_run_id": suite_run_id, "executions": [_run(r) for r in runs]}


@router.get("/executions")
async def list_executions(test_case_id: str | None = None, limit: int = 50, who: Principal = PrincipalDep,
                          pf: Platform = PlatformDep) -> list[dict[str, Any]]:
    return [_run(r) for r in await pf.platform.list_runs(who.organization_id, test_case_id=test_case_id, limit=limit)]


async def _get_run(pf: Platform, who: Principal, execution_id: str) -> Any:
    try:
        return await pf.platform.get_run(who.organization_id, execution_id)
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc


@router.get("/executions/{execution_id}")
async def get_execution(execution_id: str, request: Request, who: Principal = PrincipalDep,
                        pf: Platform = PlatformDep) -> dict[str, Any]:
    run = await _get_run(pf, who, execution_id)
    out = _run(run)
    handle = request.app.state.platform.engine.runs.get(execution_id)
    if handle is not None and handle.interrupt:
        out["pending"] = handle.interrupt
    async with pf.sessionmaker() as s:
        approvals = (await s.execute(select(Approval).where(Approval.execution_id == execution_id))).scalars().all()
    out["approvals"] = [{"id": a.id, "kind": a.kind, "status": a.status, "step_id": a.step_id, "request": a.request}
                        for a in approvals]
    return out


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str, request: Request, who: Principal = PrincipalDep,
                           pf: Platform = PlatformDep) -> dict[str, Any]:
    await _get_run(pf, who, execution_id)
    ok = await pf.engine.cancel(execution_id)
    await _dispatcher(request).audit(who.organization_id, who.actor, "execution.cancel", "execution", execution_id, {})
    return {"cancel_requested": ok}


@router.post("/executions/{execution_id}/resume")
async def resume_execution(execution_id: str, body: ResumeRequest, request: Request, who: Principal = PrincipalDep,
                           pf: Platform = PlatformDep) -> dict[str, Any]:
    await _get_run(pf, who, execution_id)
    async with pf.sessionmaker() as s:
        pending = (await s.execute(select(Approval).where(Approval.execution_id == execution_id,
                                                          Approval.status == "pending"))).scalars().first()
    try:
        if pending is not None:
            await _dispatcher(request).decide(who.organization_id, pending.id, approved=body.approved, actor=who.actor,
                                              note=body.note)
        else:
            await pf.engine.resume(execution_id, {"approved": body.approved, "by": who.actor})
    except NotFound as exc:
        raise HTTPException(409, exc.message) from exc
    return {"resumed": True}


@router.post("/executions/{execution_id}/retry", status_code=201)
async def retry_execution(execution_id: str, request: Request, who: Principal = PrincipalDep,
                          pf: Platform = PlatformDep) -> dict[str, Any]:
    run = await _get_run(pf, who, execution_id)
    new = await _dispatcher(request).start_execution(
        org_id=who.organization_id, test_case_id=run.test_case_id, version=run.test_case_version, variables={},
        options=run.options, allowed_domains=[], environment_id=run.environment_id, actor=who.actor, retry_of=run.id)
    return _run(new)


@router.get("/executions/{execution_id}/steps")
async def execution_steps(execution_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> list[dict]:
    await _get_run(pf, who, execution_id)
    runtime = runtime_registry.get(execution_id)
    if runtime is not None:  # live view
        return [r.model_dump(mode="json") for r in sorted(runtime.results.values(), key=lambda r: r.index)]
    async with pf.sessionmaker() as s:
        rows = (await s.execute(select(ExecutionStep).where(ExecutionStep.execution_id == execution_id)
                                .order_by(ExecutionStep.index))).scalars().all()
    return [r.result for r in rows]


@router.get("/executions/{execution_id}/artifacts")
async def execution_artifacts(execution_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> list:
    await _get_run(pf, who, execution_id)
    async with pf.sessionmaker() as s:
        rows = (await s.execute(select(Artifact).where(Artifact.execution_id == execution_id)
                                .order_by(Artifact.id))).scalars().all()
    return [{"kind": a.kind, "name": a.name, "step_id": a.step_id, "uri": a.uri, "size": a.size,
             "url": f"/api/v1/artifacts/{execution_id}/{a.uri.split('/')[-2]}/{a.name}"} for a in rows]


@router.get("/executions/{execution_id}/report")
async def execution_report(execution_id: str, format: str = "html", who: Principal = PrincipalDep,
                           pf: Platform = PlatformDep) -> Response:
    run = await _get_run(pf, who, execution_id)
    uri = run.report_json_uri if format == "json" else run.report_uri
    if not uri:
        raise HTTPException(404, "report not available yet")
    data = await get_artifact_service().get_bytes(uri)
    return Response(data, media_type="application/json" if format == "json" else "text/html")


@router.get("/artifacts/{execution_id}/{kind}/{name}")
async def get_artifact(execution_id: str, kind: str, name: str, who: Principal = PrincipalDep,
                       pf: Platform = PlatformDep) -> Response:
    await _get_run(pf, who, execution_id)
    service = get_artifact_service()
    uri = f"artifact://{execution_id}/{kind}/{name}"
    path = service.local_path(uri)
    if path is not None:
        if not path.exists():
            raise HTTPException(404, "artifact not found")
        return FileResponse(path)
    return Response(await service.get_bytes(uri))


@router.get("/executions/{execution_id}/agent-trace")
async def agent_trace(execution_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict[str, Any]:
    await _get_run(pf, who, execution_id)
    async with pf.sessionmaker() as s:
        traces = (await s.execute(select(AgentTrace).where(AgentTrace.execution_id == execution_id)
                                  .order_by(AgentTrace.id))).scalars().all()
        tools = (await s.execute(select(BrowserSessionEvent).where(BrowserSessionEvent.execution_id == execution_id)
                                 .order_by(BrowserSessionEvent.id))).scalars().all()
    return {"nodes": [{"ts": t.ts, "agent": t.agent, "node": t.node, "step_id": t.step_id, "status": t.status,
                       "duration_ms": t.duration_ms, "summary": t.summary} for t in traces],
            "tool_calls": [{"ts": e.ts, "tool": e.tool, "ok": e.ok, "latency_ms": e.latency_ms, "args": e.args,
                            "error": e.error, "failure_category": e.failure_category} for e in tools]}


@router.get("/executions/{execution_id}/events")
async def execution_events(execution_id: str, after: int = 0, who: Principal = PrincipalDep,
                           pf: Platform = PlatformDep) -> list[dict[str, Any]]:
    await _get_run(pf, who, execution_id)
    async with pf.sessionmaker() as s:
        rows = (await s.execute(select(ExecutionEvent).where(ExecutionEvent.execution_id == execution_id,
                                                             ExecutionEvent.seq > after)
                                .order_by(ExecutionEvent.seq))).scalars().all()
    return [{"event": r.event, "seq": r.seq, "ts": r.ts, "step_id": r.step_id, "message": r.message,
             "payload": r.payload} for r in rows]


@router.websocket("/executions/{execution_id}/stream")
async def execution_stream(websocket: WebSocket, execution_id: str) -> None:
    key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
    if not verify_api_key(key):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    queue = event_bus.subscribe(execution_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
                await websocket.send_json(event)
            except TimeoutError:
                await websocket.send_json({"event": "PING"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        event_bus.unsubscribe(execution_id, queue)


# --------------------------------------------------------------------------- approvals
@router.get("/approvals")
async def list_approvals(status: str = "pending", who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> list:
    async with pf.sessionmaker() as s:
        rows = (await s.execute(select(Approval).where(Approval.organization_id == who.organization_id,
                                                       Approval.status == status).order_by(Approval.created_at))
                ).scalars().all()
    return [{"id": a.id, "execution_id": a.execution_id, "kind": a.kind, "status": a.status, "step_id": a.step_id,
             "request": a.request, "created_at": a.created_at} for a in rows]


@router.post("/approvals/{approval_id}/approve")
async def approve(approval_id: str, body: ApprovalDecision, request: Request, who: Principal = PrincipalDep) -> dict:
    try:
        approval = await _dispatcher(request).decide(who.organization_id, approval_id, approved=True, actor=who.actor,
                                                     note=body.note)
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc
    return {"id": approval.id, "status": approval.status}


@router.post("/approvals/{approval_id}/reject")
async def reject(approval_id: str, body: ApprovalDecision, request: Request, who: Principal = PrincipalDep) -> dict:
    try:
        approval = await _dispatcher(request).decide(who.organization_id, approval_id, approved=False, actor=who.actor,
                                                     note=body.note)
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc
    return {"id": approval.id, "status": approval.status}


# --------------------------------------------------------------------------- live browser (human handoff)
@router.get("/executions/{execution_id}/live")
async def live_screenshot(execution_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    await _get_run(pf, who, execution_id)
    runtime = runtime_registry.get(execution_id)
    if runtime is None or runtime.session is None:
        raise HTTPException(404, "no live browser for this execution")
    shot = await runtime.session.adapter.screenshot()
    if not shot.ok or not shot.images:
        raise HTTPException(409, "screenshot unavailable (a native dialog may be open)")
    return {"url": runtime.observation.url if runtime.observation else None,
            "image": "data:image/png;base64," + base64.b64encode(shot.images[0].data).decode("ascii")}


@router.post("/executions/{execution_id}/browser/human-action")
async def human_action(execution_id: str, body: HumanBrowserAction, request: Request, who: Principal = PrincipalDep,
                       pf: Platform = PlatformDep) -> dict:
    """An authorized human operates the paused live browser (e.g. to complete a CAPTCHA).

    Only allowed while the execution is paused for human verification; every action is audited.
    """

    await _get_run(pf, who, execution_id)
    handle = pf.engine.runs.get(execution_id)
    runtime = runtime_registry.get(execution_id)
    if runtime is None or runtime.session is None or handle is None or handle.status != "AWAITING_HUMAN":
        raise HTTPException(409, "human browser actions are only allowed while the execution awaits a human")
    adapter = runtime.session.adapter
    if body.type == "click" and body.x is not None and body.y is not None:
        response = await adapter.mouse_click(body.x, body.y)
    elif body.type == "type" and body.text is not None:
        response = await adapter.press_key(body.text) if len(body.text) == 1 else None
        if response is None:
            for ch in body.text:
                response = await adapter.press_key(ch)
    elif body.type == "press" and body.key:
        response = await adapter.press_key(body.key)
    else:
        raise HTTPException(422, "unsupported action")
    runtime.dirty = True
    await _dispatcher(request).audit(who.organization_id, who.actor, "browser.human_action", "execution",
                                     execution_id, {"type": body.type, "x": body.x, "y": body.y})
    await event_bus.publish(execution_id, "ACTION_COMPLETED", f"human {body.type}", human=True)
    return {"ok": bool(response and response.ok)}


_ = principal


@router.get("/executions/{execution_id}/observation")
async def current_observation(execution_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    """Browser inspection view: page state, forms, dialogs and detected elements with bounding boxes."""

    await _get_run(pf, who, execution_id)
    runtime = runtime_registry.get(execution_id)
    if runtime is None or runtime.observation is None:
        raise HTTPException(404, "no live observation for this execution")
    obs = runtime.observation
    return {"summary": obs.summary(), "page_state": obs.page_state.model_dump(),
            "fingerprint": obs.fingerprint.model_dump() if obs.fingerprint else None,
            "forms": [f.model_dump() for f in obs.forms], "dialogs": [d.model_dump() for d in obs.dialogs],
            "tabs": [t.model_dump() for t in obs.tabs], "headings": obs.headings[:20],
            "elements": [{"element_id": e.element_id, "role": e.role, "name": e.name, "label": e.label,
                          "purpose": e.semantic_purpose, "visible": e.visible, "enabled": e.enabled,
                          "checked": e.checked, "bbox": e.bbox.model_dump() if e.bbox else None,
                          "form": e.context.form_name, "dialog": e.context.dialog_name, "source": e.source,
                          "test_id": e.test_id} for e in obs.elements if e.role not in ("generic", "paragraph")
                         or e.clickable_hint],
            "snapshot": obs.snapshot_text[:20000]}
