"""Projects, applications, test cases, suites, secrets."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.agents.governance.secret_manager import SecretScope
from backend.api.deps import PlatformDep, Principal, PrincipalDep
from backend.api.schemas import ApplicationCreate, ProjectCreate, SecretCreate, SuiteCreate, TestCaseCreate, TestCasePatch
from backend.bootstrap import Platform
from backend.core.exceptions import NotFound, ParseError
from backend.dsl.models import TestSpec

router = APIRouter(prefix="/api/v1")


def _project(p: Any) -> dict[str, Any]:
    return {"id": p.id, "name": p.name, "description": p.description, "created_at": p.created_at}


@router.post("/projects", status_code=201)
async def create_project(body: ProjectCreate, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    return _project(await pf.platform.create_project(who.organization_id, body.name, body.description))


@router.get("/projects")
async def list_projects(who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> list[dict]:
    return [_project(p) for p in await pf.platform.list_projects(who.organization_id)]


@router.post("/applications", status_code=201)
async def create_application(body: ApplicationCreate, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    app = await pf.platform.ensure_application(who.organization_id, body.project_id, body.base_url, body.name)
    return {"id": app.id, "name": app.name, "base_url": app.base_url, "host": app.host}


@router.get("/applications/{application_id}")
async def get_application(application_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    try:
        app = await pf.platform.get_application(who.organization_id, application_id)
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc
    envs = await pf.platform.list_environments(who.organization_id, application_id)
    return {"id": app.id, "name": app.name, "base_url": app.base_url, "host": app.host,
            "known_routes": app.known_routes,
            "environments": [{"id": e.id, "name": e.name, "base_url": e.base_url, "type": e.environment_type,
                              "allowed_domains": e.allowed_domains, "policy": e.policy} for e in envs]}


def _test_case(tc: Any, tcv: Any | None = None) -> dict[str, Any]:
    out = {"id": tc.id, "name": tc.name, "project_id": tc.project_id, "application_id": tc.application_id,
           "tags": tc.tags, "current_version": tc.current_version, "updated_at": tc.updated_at}
    if tcv is not None:
        spec = TestSpec.model_validate(tcv.spec)
        out.update({"version": tcv.version, "source_text": tcv.source_text, "source_format": tcv.source_format,
                    "spec": tcv.spec, "issues": tcv.issues,
                    "steps": [{"id": s.id, "index": s.index, "text": s.original_text, "action": s.action.value,
                               "summary": s.summary(), "confidence": s.confidence} for s in spec.flat_steps()]})
    return out


@router.post("/test-cases", status_code=201)
async def create_test_case(body: TestCaseCreate, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    try:
        tc, tcv = await pf.platform.save_test_case(who.organization_id, body.project_id, content=body.content,
                                                   fmt=body.format, base_url=body.base_url, name=body.name,
                                                   tags=body.tags, environment_id=body.environment_id)
    except (ParseError, NotFound) as exc:
        raise HTTPException(422, exc.message) from exc
    return _test_case(tc, tcv)


@router.get("/test-cases")
async def list_test_cases(project_id: str | None = None, who: Principal = PrincipalDep,
                          pf: Platform = PlatformDep) -> list[dict]:
    return [_test_case(tc) for tc in await pf.platform.list_test_cases(who.organization_id, project_id)]


@router.get("/test-cases/{test_case_id}")
async def get_test_case(test_case_id: str, version: int | None = None, who: Principal = PrincipalDep,
                        pf: Platform = PlatformDep) -> dict:
    try:
        tc, tcv = await pf.platform.get_test_case(who.organization_id, test_case_id, version)
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc
    return _test_case(tc, tcv)


@router.patch("/test-cases/{test_case_id}")
async def patch_test_case(test_case_id: str, body: TestCasePatch, who: Principal = PrincipalDep,
                          pf: Platform = PlatformDep) -> dict:
    try:
        tc, tcv = await pf.platform.get_test_case(who.organization_id, test_case_id)
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc
    if body.content is not None:
        tc, tcv = await pf.platform.save_test_case(who.organization_id, tc.project_id, content=body.content,
                                                   fmt=body.format, base_url=body.base_url, name=body.name or tc.name,
                                                   test_case_id=tc.id)
    if body.tags is not None or body.archived is not None:
        from backend.models.entities import TestCase

        async with pf.sessionmaker() as s:
            row = await s.get(TestCase, tc.id)
            if body.tags is not None:
                row.tags = body.tags
            if body.archived is not None:
                row.archived = body.archived
            await s.commit()
    return _test_case(*(await pf.platform.get_test_case(who.organization_id, test_case_id)))


@router.post("/test-suites", status_code=201)
async def create_suite(body: SuiteCreate, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    suite = await pf.platform.create_suite(who.organization_id, body.project_id, body.name, body.test_case_ids)
    return {"id": suite.id, "name": suite.name, "test_case_ids": suite.test_case_ids}


@router.get("/test-suites/{suite_id}")
async def get_suite(suite_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    try:
        suite = await pf.platform.get_suite(who.organization_id, suite_id)
    except NotFound as exc:
        raise HTTPException(404, exc.message) from exc
    return {"id": suite.id, "name": suite.name, "test_case_ids": suite.test_case_ids}


@router.post("/secrets", status_code=201)
async def create_secret(body: SecretCreate, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    await pf.secrets.store(SecretScope(who.organization_id, body.project_id, body.environment_id), body.name, body.value)
    return {"name": body.name, "stored": True}  # values are never returned


@router.get("/secrets")
async def list_secrets(project_id: str | None = None, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    names = await pf.secrets.available_names(SecretScope(who.organization_id, project_id))
    return {"names": sorted(names)}


@router.post("/schedules", status_code=201)
async def create_schedule(body: dict[str, Any], who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    """Run a test case or suite every ``interval_minutes`` (body: target_type, target_id, interval_minutes)."""

    from backend.core.ids import new_id
    from backend.models.entities import Schedule

    if body.get("target_type") not in ("test_case", "suite") or int(body.get("interval_minutes", 0)) < 5:
        raise HTTPException(422, "target_type must be test_case|suite and interval_minutes >= 5")
    async with pf.sessionmaker() as s:
        schedule = Schedule(id=new_id("sched"), organization_id=who.organization_id,
                            project_id=body.get("project_id", ""), target_type=body["target_type"],
                            target_id=body["target_id"], interval_minutes=int(body["interval_minutes"]),
                            options={"variables": body.get("variables", {}), "options": body.get("options", {})})
        s.add(schedule)
        await s.commit()
    return {"id": schedule.id, "interval_minutes": schedule.interval_minutes}
