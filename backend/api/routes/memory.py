"""Memory inspection/invalidation and agent catalogue."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import desc, select

from backend.api.deps import PlatformDep, Principal, PrincipalDep
from backend.bootstrap import Platform
from backend.harness.agent_registry import AGENTS
from backend.memory.memory_manager import MemoryManager, MemoryScope
from backend.models.entities import FailureMemory, LocatorMemory, MemoryVersion, PageMemory, TestExecutionMemory
from backend.runners.runner_registry import RUNNERS

router = APIRouter(prefix="/api/v1")


def _locator(row: LocatorMemory) -> dict[str, Any]:
    return {"memory_id": row.id, "route_pattern": row.route_pattern, "action_signature": row.action_signature,
            "intent": row.intent_text, "strategy": row.target_strategy, "successes": row.successful_execution_count,
            "failures": row.failed_execution_count, "validated_runs": row.validated_run_count,
            "confidence": row.confidence, "status": row.status, "version": row.memory_version,
            "supersedes": row.supersedes, "test_case_id": row.test_case_id, "last_verified_at": row.last_verified_at}


@router.get("/memory/applications/{application_id}")
async def application_memory(application_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    async with pf.sessionmaker() as s:
        locators = (await s.execute(select(LocatorMemory).where(
            LocatorMemory.organization_id == who.organization_id, LocatorMemory.application_id == application_id)
            .order_by(LocatorMemory.route_pattern, LocatorMemory.action_signature, desc(LocatorMemory.memory_version))
        )).scalars().all()
        pages = (await s.execute(select(PageMemory).where(PageMemory.organization_id == who.organization_id,
                                                          PageMemory.application_id == application_id))).scalars().all()
        failures = (await s.execute(select(FailureMemory).where(
            FailureMemory.organization_id == who.organization_id, FailureMemory.application_id == application_id)
            .order_by(desc(FailureMemory.created_at)).limit(50))).scalars().all()
    return {"locators": [_locator(r) for r in locators],
            "pages": [{"route": p.route_pattern, "kind": p.page_kind, "structure_hash": p.structure_hash,
                       "seen": p.seen_count, "last_verified_at": p.last_verified_at} for p in pages],
            "failures": [{"category": f.failure_category, "action_signature": f.action_signature,
                          "route": f.route_pattern, "root_cause": f.root_cause,
                          "successful_alternative": f.successful_alternative, "execution_id": f.execution_id,
                          "created_at": f.created_at} for f in failures]}


@router.get("/memory/test-cases/{test_case_id}")
async def test_case_memory(test_case_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> dict:
    async with pf.sessionmaker() as s:
        traj = (await s.execute(select(TestExecutionMemory).where(
            TestExecutionMemory.organization_id == who.organization_id,
            TestExecutionMemory.test_case_id == test_case_id))).scalars().all()
        locators = (await s.execute(select(LocatorMemory).where(
            LocatorMemory.organization_id == who.organization_id, LocatorMemory.test_case_id == test_case_id)
        )).scalars().all()
    return {"trajectories": [{"version": t.test_case_version, "success_count": t.success_count,
                              "failure_count": t.failure_count, "confidence": t.confidence,
                              "trajectory": t.trajectory} for t in traj],
            "locators": [_locator(r) for r in locators]}


@router.get("/memory/{memory_id}/versions")
async def memory_versions(memory_id: str, who: Principal = PrincipalDep, pf: Platform = PlatformDep) -> list:
    async with pf.sessionmaker() as s:
        row = await s.get(LocatorMemory, memory_id)
        if row is None or row.organization_id != who.organization_id:
            raise HTTPException(404, "memory not found")
        versions = (await s.execute(select(MemoryVersion).where(MemoryVersion.memory_id == memory_id)
                                    .order_by(MemoryVersion.id))).scalars().all()
    return [{"version": v.version, "reason": v.change_reason, "snapshot": v.snapshot, "execution_id": v.execution_id,
             "created_at": v.created_at} for v in versions]


@router.post("/memory/{memory_id}/invalidate")
async def invalidate_memory(memory_id: str, reason: str = "invalidated by user", who: Principal = PrincipalDep,
                            pf: Platform = PlatformDep) -> dict:
    async with pf.sessionmaker() as s:
        row = await s.get(LocatorMemory, memory_id)
        if row is None or row.organization_id != who.organization_id:
            raise HTTPException(404, "memory not found")
        scope = MemoryScope(row.organization_id, row.project_id, row.application_id, row.environment_id)
    ok = await MemoryManager(pf.sessionmaker, scope).invalidate(memory_id, reason)
    return {"memory_id": memory_id, "invalidated": ok}


@router.get("/agents")
async def list_agents(who: Principal = PrincipalDep) -> dict:
    return {"agents": [a.to_dict() for a in AGENTS], "runners": RUNNERS}
