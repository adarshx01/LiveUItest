"""Application services: projects, applications, environments, versioned test cases, runs.

All queries are scoped by ``organization_id`` (tenant isolation at the
application layer; the API derives it from the authenticated principal).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.agents.governance.redaction import MASK, register_sensitive
from backend.agents.governance.secret_manager import SecretManager, SecretScope
from backend.agents.planner.requirement_analyzer import RequirementAnalyzer
from backend.agents.planner.test_compiler import TestCompiler
from backend.agents.planner.test_ingestion import ingest
from backend.core.exceptions import NotFound
from backend.core.ids import new_id, stable_hash
from backend.dsl.models import TestSpec
from backend.models.entities import (
    Application,
    ApplicationEnvironment,
    ExecutionRun,
    Organization,
    Project,
    TestCase,
    TestCaseVersion,
    TestDataSet,
    TestStep,
    TestSuite,
    utcnow,
)


class PlatformService:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], secrets: SecretManager | None = None) -> None:
        self.sessionmaker = sessionmaker
        self.secrets = secrets

    # ------------------------------------------------------------------ tenancy
    async def ensure_organization(self, org_id: str, name: str | None = None) -> Organization:
        async with self.sessionmaker() as s:
            org = await s.get(Organization, org_id)
            if org is None:
                org = Organization(id=org_id, name=name or org_id)
                s.add(org)
                await s.commit()
            return org

    async def create_project(self, org_id: str, name: str, description: str = "") -> Project:
        await self.ensure_organization(org_id)
        async with self.sessionmaker() as s:
            project = Project(id=new_id("prj"), organization_id=org_id, name=name, description=description)
            s.add(project)
            await s.commit()
            return project

    async def ensure_project(self, org_id: str, name: str) -> Project:
        await self.ensure_organization(org_id)
        async with self.sessionmaker() as s:
            row = (await s.execute(select(Project).where(Project.organization_id == org_id, Project.name == name))
                   ).scalars().first()
            if row is not None:
                return row
        return await self.create_project(org_id, name)

    async def list_projects(self, org_id: str) -> list[Project]:
        async with self.sessionmaker() as s:
            return list((await s.execute(select(Project).where(Project.organization_id == org_id)
                                         .order_by(Project.created_at))).scalars())

    async def ensure_application(self, org_id: str, project_id: str, base_url: str, name: str | None = None
                                 ) -> Application:
        host = (urlparse(base_url).netloc or base_url).lower()
        app_id = "app_" + stable_hash(org_id, project_id, host, length=14)
        async with self.sessionmaker() as s:
            app = await s.get(Application, app_id)
            if app is None:
                app = Application(id=app_id, organization_id=org_id, project_id=project_id, name=name or host,
                                  base_url=base_url, host=host)
                s.add(app)
                s.add(ApplicationEnvironment(id="env_" + stable_hash(app_id, "default", length=12),
                                             organization_id=org_id, application_id=app_id, name="default",
                                             base_url=base_url, environment_type="test"))
                await s.commit()
            return app

    async def get_application(self, org_id: str, application_id: str) -> Application:
        async with self.sessionmaker() as s:
            app = await s.get(Application, application_id)
            if app is None or app.organization_id != org_id:
                raise NotFound("Application not found")
            return app

    async def list_environments(self, org_id: str, application_id: str) -> list[ApplicationEnvironment]:
        async with self.sessionmaker() as s:
            return list((await s.execute(select(ApplicationEnvironment).where(
                ApplicationEnvironment.organization_id == org_id,
                ApplicationEnvironment.application_id == application_id))).scalars())

    # ------------------------------------------------------------------ test cases
    async def save_test_case(self, org_id: str, project_id: str, *, content: str | dict[str, Any],
                             fmt: str | None = None, base_url: str | None = None, name: str | None = None,
                             variables: dict[str, str] | None = None, tags: list[str] | None = None,
                             test_case_id: str | None = None, environment_id: str | None = None,
                             analyzer: RequirementAnalyzer | None = None) -> tuple[TestCase, TestCaseVersion]:
        """Create or version a test case: ingest -> analyze -> compile -> vault sensitive literals."""

        cases = ingest(content, fmt=fmt, base_url=base_url, name=name, variables=variables)
        raw = cases[0]
        analyzer = analyzer or RequirementAnalyzer()
        secret_names = await self.secrets.available_names(SecretScope(org_id, project_id, environment_id)) \
            if self.secrets else set()
        spec = TestCompiler().compile(await analyzer.analyze(raw, secret_names=secret_names))
        if not spec.base_url:
            raise NotFound("The test case needs an application URL")
        app = await self.ensure_application(org_id, project_id, spec.base_url)
        test_case_id = test_case_id or "tc_" + stable_hash(org_id, app.id, re.sub(r"\s+", " ", spec.test_name.lower()),
                                                           length=14)
        spec, masked_source = await self._vault_sensitive_literals(org_id, project_id, test_case_id, spec,
                                                                   raw.source_text or (content if isinstance(content, str)
                                                                                       else json.dumps(content)))
        spec_json = spec.model_dump(mode="json")
        spec_hash = stable_hash(json.dumps(spec_json, sort_keys=True), length=32)
        async with self.sessionmaker() as s:
            tc = await s.get(TestCase, test_case_id)
            if tc is None:
                tc = TestCase(id=test_case_id, organization_id=org_id, project_id=project_id, application_id=app.id,
                              name=spec.test_name, tags=tags or spec.tags, current_version=0)
                s.add(tc)
                await s.flush()
            latest = (await s.execute(select(TestCaseVersion).where(TestCaseVersion.test_case_id == tc.id)
                                      .order_by(desc(TestCaseVersion.version)).limit(1))).scalar_one_or_none()
            if latest is not None and latest.spec_hash == spec_hash:
                await s.commit()
                return tc, latest
            version = TestCaseVersion(id=new_id("tcv"), test_case_id=tc.id, version=(latest.version + 1) if latest else 1,
                                      source_text=masked_source, source_format=raw.source_format, spec=spec_json,
                                      spec_hash=spec_hash,
                                      issues=[a.model_dump() for a in spec.ambiguities])
            s.add(version)
            await s.flush()
            for step in spec.flat_steps():
                s.add(TestStep(test_case_version_id=version.id, step_id=step.id, index=step.index,
                               action=step.action.value, original_text=step.original_text,
                               spec=step.model_dump(mode="json", exclude={"then_steps", "else_steps"})))
            tc.current_version = version.version
            tc.name = spec.test_name
            tc.updated_at = utcnow()
            if variables:
                s.add(TestDataSet(id=new_id("ds"), organization_id=org_id, test_case_id=tc.id, name="default",
                                  variables=variables))
            await s.commit()
            return tc, version

    async def _vault_sensitive_literals(self, org_id: str, project_id: str, test_case_id: str, spec: TestSpec,
                                        source: str) -> tuple[TestSpec, str]:
        """Literal passwords in a test case are moved into the encrypted secret store."""

        if self.secrets is None:
            return spec, source
        masked = source
        vaulted: list[str] = []
        for step in spec.flat_steps():
            value = step.value
            if value is not None and value.sensitive and value.source == "literal" and value.value:
                name = f"{test_case_id}_{step.id}"
                await self.secrets.store(SecretScope(org_id, project_id), name, value.value)
                register_sensitive(value.value)
                vaulted.append(value.value)
                masked = masked.replace(value.value, MASK)
                value.source, value.reference, value.value = "secret", name, None
        if vaulted:
            # the instruction text of every step ("Enter S3cret!pass as Password") is stored, reported and
            # checkpointed: it keeps its wording with the secret masked, the value lives only in the vault
            spec = TestSpec.model_validate(_mask_strings(spec.model_dump(mode="json"), vaulted))
        if spec.secret_refs is not None:
            spec.secret_refs = sorted({s.value.reference for s in spec.flat_steps()
                                       if s.value and s.value.source == "secret" and s.value.reference})
        return spec, masked

    async def get_test_case(self, org_id: str, test_case_id: str, version: int | None = None
                            ) -> tuple[TestCase, TestCaseVersion]:
        async with self.sessionmaker() as s:
            tc = await s.get(TestCase, test_case_id)
            if tc is None or tc.organization_id != org_id:
                raise NotFound("Test case not found")
            query = select(TestCaseVersion).where(TestCaseVersion.test_case_id == test_case_id)
            query = query.where(TestCaseVersion.version == (version or tc.current_version))
            tcv = (await s.execute(query)).scalar_one_or_none()
            if tcv is None:
                raise NotFound("Test case version not found")
            return tc, tcv

    async def list_test_cases(self, org_id: str, project_id: str | None = None) -> list[TestCase]:
        async with self.sessionmaker() as s:
            query = select(TestCase).where(TestCase.organization_id == org_id, TestCase.archived.is_(False))
            if project_id:
                query = query.where(TestCase.project_id == project_id)
            return list((await s.execute(query.order_by(desc(TestCase.updated_at)))).scalars())

    async def create_suite(self, org_id: str, project_id: str, name: str, test_case_ids: list[str]) -> TestSuite:
        async with self.sessionmaker() as s:
            suite = TestSuite(id=new_id("suite"), organization_id=org_id, project_id=project_id, name=name,
                              test_case_ids=test_case_ids)
            s.add(suite)
            await s.commit()
            return suite

    async def get_suite(self, org_id: str, suite_id: str) -> TestSuite:
        async with self.sessionmaker() as s:
            suite = await s.get(TestSuite, suite_id)
            if suite is None or suite.organization_id != org_id:
                raise NotFound("Suite not found")
            return suite

    # ------------------------------------------------------------------ runs
    async def create_run(self, *, org_id: str, tc: TestCase, version: int, options: dict[str, Any],
                         environment_id: str | None = None, retry_of: str | None = None,
                         suite_run_id: str | None = None, created_by: str | None = None) -> ExecutionRun:
        async with self.sessionmaker() as s:
            run = ExecutionRun(id=new_id("exec"), organization_id=org_id, project_id=tc.project_id,
                               application_id=tc.application_id, environment_id=environment_id, test_case_id=tc.id,
                               test_case_version=version, status="QUEUED", options=options, retry_of=retry_of,
                               suite_run_id=suite_run_id, created_by=created_by)
            s.add(run)
            await s.commit()
            return run

    async def update_run(self, execution_id: str, **fields: Any) -> None:
        async with self.sessionmaker() as s:
            run = await s.get(ExecutionRun, execution_id)
            if run is None:
                return
            for key, value in fields.items():
                setattr(run, key, value)
            await s.commit()

    async def get_run(self, org_id: str, execution_id: str) -> ExecutionRun:
        async with self.sessionmaker() as s:
            run = await s.get(ExecutionRun, execution_id)
            if run is None or run.organization_id != org_id:
                raise NotFound("Execution not found")
            return run

    async def list_runs(self, org_id: str, *, test_case_id: str | None = None, limit: int = 50) -> list[ExecutionRun]:
        async with self.sessionmaker() as s:
            query = select(ExecutionRun).where(ExecutionRun.organization_id == org_id)
            if test_case_id:
                query = query.where(ExecutionRun.test_case_id == test_case_id)
            return list((await s.execute(query.order_by(desc(ExecutionRun.created_at)).limit(limit))).scalars())


def now_utc() -> dt.datetime:
    return utcnow()


def _mask_strings(data: Any, secrets: list[str]) -> Any:
    if isinstance(data, str):
        for secret in sorted(secrets, key=len, reverse=True):
            data = data.replace(secret, MASK)
        return data
    if isinstance(data, list):
        return [_mask_strings(item, secrets) for item in data]
    if isinstance(data, dict):
        return {key: _mask_strings(item, secrets) for key, item in data.items()}
    return data
