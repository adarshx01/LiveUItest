"""Agentic memory engine (spec section 13).

Memory is *validated execution knowledge*, never unquestionable truth:

* only semantic strategies are stored (role/name/label/test-id/context), never
  snapshot refs or coordinates, and never sensitive values;
* retrieval is exact first (application + route + action signature, preferring
  the same test case), then page-level, then pgvector semantic similarity;
* every retrieved strategy must be re-grounded on the live page by the grounder
  before it can influence an action;
* confidence follows successes/failures/recency; a strategy that stops working is
  versioned (superseded) rather than deleted, and the replacement found by
  healing becomes the new version.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.agents.grounding.locator_strategy import LocatorStrategy
from backend.agents.planner.lexicon import normalize_text
from backend.core.ids import new_id, stable_hash
from backend.core.logging import get_logger
from backend.dsl.models import Step
from backend.dsl.observation import Observation
from backend.memory.embeddings import embed
from backend.models.entities import (
    ApplicationMemory,
    FailureMemory,
    InteractionPattern,
    LocatorMemory,
    MemoryVersion,
    PageFingerprintRecord,
    PageMemory,
    TestExecutionMemory,
    utcnow,
)

log = get_logger(__name__)


@dataclass
class MemoryScope:
    organization_id: str
    project_id: str
    application_id: str
    environment_id: str | None = None


@dataclass
class MemoryCandidate:
    memory_id: str
    strategy: LocatorStrategy
    confidence: float
    match_level: str  # exact_test_case | exact_route | page | semantic
    successes: int
    failures: int
    version: int
    similarity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "match_level": self.match_level, "confidence": round(self.confidence, 3),
                "successes": self.successes, "failures": self.failures, "version": self.version,
                "strategy": self.strategy.describe(), "similarity": round(self.similarity, 3)}


@dataclass
class LearningRecord:
    step: Step
    route_pattern: str
    page_fingerprint: str
    strategy: LocatorStrategy
    healed: bool = False
    previous_memory_id: str | None = None
    used_memory_id: str | None = None
    failed_strategies: list[dict[str, Any]] = field(default_factory=list)
    failure_category: str | None = None
    recovery_attempts: int = 0


def action_signature(step: Step) -> str:
    target = step.target
    if target is None:
        return step.action.value
    key = target.purpose or normalize_text(target.exact_name or target.intent)[:60]
    name = normalize_text(target.exact_name or "")[:40]
    option = "option" if target.option_text else ""
    return ":".join(filter(None, [step.action.value, key, name, option]))


def step_text(step: Step) -> str:
    target = step.target
    bits = [step.action.value]
    if target is not None:
        bits += [target.exact_name or "", target.intent or "", target.purpose or "", target.element_kind or ""]
    return " ".join(b for b in bits if b)


def confidence_from(successes: int, failures: int, last_verified: dt.datetime | None) -> float:
    base = (successes + 1) / (successes + failures + 2)
    if last_verified is not None:
        age_days = (utcnow() - last_verified).total_seconds() / 86400
        base *= max(0.7, 1 - age_days / 365)  # gentle decay for stale knowledge
    return round(min(0.99, base), 4)


class MemoryManager:
    name = "memory_agent"

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], scope: MemoryScope) -> None:
        self.sessionmaker = sessionmaker
        self.scope = scope

    def _scope_filter(self, model: Any) -> Any:
        return and_(model.organization_id == self.scope.organization_id,
                    model.application_id == self.scope.application_id)

    # ------------------------------------------------------------------ retrieval
    async def application_memory(self) -> dict[str, Any]:
        async with self.sessionmaker() as session:
            rows = (await session.execute(select(ApplicationMemory).where(
                self._scope_filter(ApplicationMemory)))).scalars().all()
            pages = (await session.execute(select(PageMemory.route_pattern, PageMemory.page_kind, PageMemory.seen_count)
                                           .where(self._scope_filter(PageMemory)))).all()
            locators = (await session.execute(select(LocatorMemory.id).where(
                self._scope_filter(LocatorMemory), LocatorMemory.status == "active"))).all()
        return {"facts": {r.key: r.value for r in rows},
                "known_pages": [{"route": p[0], "kind": p[1], "seen": p[2]} for p in pages],
                "active_locator_memories": len(locators)}

    async def retrieve_for_step(self, step: Step, observation: Observation | None, test_case_id: str | None,
                                *, limit: int = 4) -> list[MemoryCandidate]:
        if step.target is None:
            return []
        signature = action_signature(step)
        route = observation.fingerprint.route_pattern if observation and observation.fingerprint else None
        candidates: dict[str, MemoryCandidate] = {}
        async with self.sessionmaker() as session:
            base = select(LocatorMemory).where(self._scope_filter(LocatorMemory), LocatorMemory.status == "active",
                                               LocatorMemory.action_signature == signature)
            exact_rows = (await session.execute(base.order_by(LocatorMemory.confidence.desc()).limit(20))).scalars().all()
            for row in exact_rows:
                if route and row.route_pattern != route:
                    level = "signature_other_route"
                elif test_case_id and row.test_case_id == test_case_id:
                    level = "exact_test_case"
                else:
                    level = "exact_route"
                if level == "signature_other_route":
                    continue
                candidates[row.id] = self._candidate(row, level)
            if not candidates:
                vector = embed(step_text(step), purpose=step.target.purpose)
                distance = LocatorMemory.embedding.cosine_distance(vector)
                query = (select(LocatorMemory, distance.label("d"))
                         .where(self._scope_filter(LocatorMemory), LocatorMemory.status == "active",
                                LocatorMemory.action_signature.like(f"{step.action.value}:%"))
                         .order_by(distance).limit(5))
                if route:
                    query = query.where(LocatorMemory.route_pattern == route)
                for row, d in (await session.execute(query)).all():
                    sim = 1 - float(d)
                    if sim >= 0.78:
                        cand = self._candidate(row, "semantic")
                        cand.similarity = sim
                        cand.confidence *= sim
                        candidates[row.id] = cand
        ranked = sorted(candidates.values(), key=lambda c: ({"exact_test_case": 0, "exact_route": 1, "semantic": 2}
                                                            .get(c.match_level, 3), -c.confidence))
        return ranked[:limit]

    @staticmethod
    def _candidate(row: LocatorMemory, level: str) -> MemoryCandidate:
        return MemoryCandidate(memory_id=row.id, strategy=LocatorStrategy.model_validate(row.target_strategy),
                               confidence=row.confidence, match_level=level,
                               successes=row.successful_execution_count, failures=row.failed_execution_count,
                               version=row.memory_version)

    async def trajectory(self, test_case_id: str, version: int) -> dict[str, Any] | None:
        async with self.sessionmaker() as session:
            row = (await session.execute(select(TestExecutionMemory).where(
                TestExecutionMemory.organization_id == self.scope.organization_id,
                TestExecutionMemory.test_case_id == test_case_id,
                TestExecutionMemory.test_case_version == version))).scalar_one_or_none()
            if row is None:
                return None
            return {"trajectory": row.trajectory, "success_count": row.success_count,
                    "failure_count": row.failure_count, "confidence": row.confidence}

    # ------------------------------------------------------------------ learning
    async def record_page(self, observation: Observation) -> None:
        fp = observation.fingerprint
        if fp is None:
            return
        async with self.sessionmaker() as session:
            existing = (await session.execute(select(PageMemory).where(
                self._scope_filter(PageMemory), PageMemory.route_pattern == fp.route_pattern,
                PageMemory.structure_hash == fp.structure_hash))).scalar_one_or_none()
            if existing is not None:
                existing.seen_count += 1
                existing.last_verified_at = utcnow()
            else:
                session.add(PageMemory(**self._scope_kwargs(), route_pattern=fp.route_pattern,
                                       structure_hash=fp.structure_hash, page_kind=observation.page_state.page_kind,
                                       title_pattern=fp.title_pattern, signature=fp.signature,
                                       forms={k: v for k, v in fp.form_hashes.items()},
                                       embedding=embed(" ".join(fp.signature[:40]) or fp.route_pattern)))
            fprow = (await session.execute(select(PageFingerprintRecord).where(
                PageFingerprintRecord.application_id == self.scope.application_id,
                PageFingerprintRecord.structure_hash == fp.structure_hash))).scalar_one_or_none()
            if fprow is None:
                session.add(PageFingerprintRecord(**self._scope_kwargs(), route_pattern=fp.route_pattern,
                                                  structure_hash=fp.structure_hash, title_pattern=fp.title_pattern,
                                                  signature=fp.signature))
            else:
                fprow.seen_count += 1
                fprow.last_seen = utcnow()
            await session.commit()

    async def commit_learning(self, records: list[LearningRecord], *, execution_id: str, test_case_id: str,
                              test_passed: bool) -> dict[str, int]:
        """Persist verified step strategies; healed replacements supersede old versions."""

        stats = {"created": 0, "reinforced": 0, "superseded": 0, "failures_recorded": 0}
        async with self.sessionmaker() as session:
            for rec in records:
                signature = action_signature(rec.step)
                strategy_json = rec.strategy.model_dump(exclude_none=True)
                identity = rec.strategy.identity_key()
                rows = (await session.execute(select(LocatorMemory).where(
                    self._scope_filter(LocatorMemory), LocatorMemory.status == "active",
                    LocatorMemory.route_pattern == rec.route_pattern,
                    LocatorMemory.action_signature == signature))).scalars().all()
                same = next((r for r in rows if LocatorStrategy.model_validate(r.target_strategy).identity_key()
                             == identity), None)
                if same is not None:
                    same.successful_execution_count += 1
                    same.validated_run_count += 1 if test_passed else 0
                    same.last_verified_at = utcnow()
                    same.last_execution_id = execution_id
                    same.page_fingerprint = rec.page_fingerprint
                    same.target_strategy = strategy_json  # refresh context details
                    same.confidence = confidence_from(same.successful_execution_count, same.failed_execution_count,
                                                      same.last_verified_at)
                    if test_case_id and not same.test_case_id:
                        same.test_case_id = test_case_id
                    stats["reinforced"] += 1
                    continue
                version = 1
                supersedes = None
                for old in rows:
                    # a different strategy for the same intent on the same page: version it
                    old.status = "superseded"
                    version = max(version, old.memory_version + 1)
                    supersedes = old.id
                    session.add(MemoryVersion(memory_id=old.id, memory_type="locator", version=old.memory_version,
                                              snapshot={"strategy": old.target_strategy,
                                                        "successes": old.successful_execution_count,
                                                        "failures": old.failed_execution_count},
                                              change_reason="superseded by healed strategy" if rec.healed else
                                              "superseded by newly verified strategy", execution_id=execution_id))
                    stats["superseded"] += 1
                memory_id = new_id("mem")
                session.add(LocatorMemory(
                    id=memory_id, **self._scope_kwargs(), test_case_id=test_case_id, route_pattern=rec.route_pattern,
                    page_fingerprint=rec.page_fingerprint, action_signature=signature,
                    intent_text=step_text(rec.step)[:500], target_strategy=strategy_json,
                    successful_execution_count=1, validated_run_count=1 if test_passed else 0,
                    confidence=confidence_from(1, 0, utcnow()), memory_version=version, supersedes=supersedes,
                    embedding=embed(step_text(rec.step), purpose=rec.step.target.purpose if rec.step.target else None),
                    last_execution_id=execution_id))
                session.add(MemoryVersion(memory_id=memory_id, memory_type="locator", version=version,
                                          snapshot={"strategy": strategy_json},
                                          change_reason="healed replacement" if rec.healed else "first verified use",
                                          execution_id=execution_id))
                stats["created"] += 1
                if rec.healed:
                    session.add(FailureMemory(
                        **self._scope_kwargs(), execution_id=execution_id, route_pattern=rec.route_pattern,
                        action_signature=signature, failure_category=rec.failure_category or "UNKNOWN_FAILURE",
                        failure_signature=stable_hash(signature, rec.failure_category),
                        context={"step": rec.step.original_text}, failed_strategies=rec.failed_strategies,
                        successful_alternative=strategy_json,
                        root_cause="locator changed; re-grounded semantically", recovery_attempts=rec.recovery_attempts,
                        evidence=[{"previous_memory_id": rec.previous_memory_id}]))
                    stats["failures_recorded"] += 1
            await session.commit()
        return stats

    async def record_strategy_failure(self, memory_id: str, *, execution_id: str, reason: str) -> None:
        async with self.sessionmaker() as session:
            row = await session.get(LocatorMemory, memory_id)
            if row is None or row.organization_id != self.scope.organization_id:
                return
            row.failed_execution_count += 1
            row.confidence = confidence_from(row.successful_execution_count, row.failed_execution_count,
                                             row.last_verified_at)
            session.add(MemoryVersion(memory_id=memory_id, memory_type="locator", version=row.memory_version,
                                      snapshot={"failed": True, "reason": reason[:300]},
                                      change_reason="strategy failed to re-ground", execution_id=execution_id))
            await session.commit()

    async def record_failure(self, *, execution_id: str, step: Step, route_pattern: str, category: str,
                             root_cause: str, evidence: list[str], failed_strategies: list[dict[str, Any]],
                             attempts: int) -> None:
        async with self.sessionmaker() as session:
            session.add(FailureMemory(**self._scope_kwargs(), execution_id=execution_id, route_pattern=route_pattern,
                                      action_signature=action_signature(step), failure_category=category,
                                      failure_signature=stable_hash(action_signature(step), category),
                                      context={"step": step.original_text}, failed_strategies=failed_strategies,
                                      root_cause=root_cause, recovery_attempts=attempts, evidence=evidence[:10]))
            await session.commit()

    async def record_pattern(self, pattern_type: str, signature: str, details: dict[str, Any], success: bool) -> None:
        async with self.sessionmaker() as session:
            row = (await session.execute(select(InteractionPattern).where(
                InteractionPattern.organization_id == self.scope.organization_id,
                InteractionPattern.application_id == self.scope.application_id,
                InteractionPattern.pattern_type == pattern_type,
                InteractionPattern.signature == signature))).scalar_one_or_none()
            if row is None:
                row = InteractionPattern(organization_id=self.scope.organization_id,
                                         application_id=self.scope.application_id, pattern_type=pattern_type,
                                         signature=signature, details=details)
                session.add(row)
            row.success_count = (row.success_count or 0) + (1 if success else 0)
            row.failure_count = (row.failure_count or 0) + (0 if success else 1)
            row.confidence = confidence_from(row.success_count, row.failure_count, None)
            row.details = {**(row.details or {}), **details}
            await session.commit()

    async def record_trajectory(self, *, test_case_id: str, version: int, trajectory: list[dict[str, Any]],
                                success: bool, execution_id: str) -> None:
        async with self.sessionmaker() as session:
            row = (await session.execute(select(TestExecutionMemory).where(
                TestExecutionMemory.organization_id == self.scope.organization_id,
                TestExecutionMemory.test_case_id == test_case_id,
                TestExecutionMemory.test_case_version == version))).scalar_one_or_none()
            if row is None:
                row = TestExecutionMemory(**self._scope_kwargs(), test_case_id=test_case_id, test_case_version=version)
                session.add(row)
            if success:
                row.trajectory = trajectory
                row.success_count = (row.success_count or 0) + 1
                row.last_verified_at = utcnow()
            else:
                row.failure_count = (row.failure_count or 0) + 1
            row.confidence = confidence_from(row.success_count or 0, row.failure_count or 0, None)
            row.last_execution_id = execution_id
            await session.commit()

    async def set_fact(self, key: str, value: dict[str, Any], confidence: float = 0.8) -> None:
        async with self.sessionmaker() as session:
            row = (await session.execute(select(ApplicationMemory).where(
                self._scope_filter(ApplicationMemory), ApplicationMemory.key == key))).scalar_one_or_none()
            if row is None:
                session.add(ApplicationMemory(**self._scope_kwargs(), key=key, value=value, confidence=confidence))
            else:
                row.value = {**(row.value or {}), **value}
                row.confidence = max(row.confidence, confidence)
            await session.commit()

    async def invalidate(self, memory_id: str, reason: str) -> bool:
        async with self.sessionmaker() as session:
            result = await session.execute(update(LocatorMemory).where(
                LocatorMemory.id == memory_id, LocatorMemory.organization_id == self.scope.organization_id)
                .values(status="invalidated", confidence=0.0))
            session.add(MemoryVersion(memory_id=memory_id, memory_type="locator", version=0,
                                      snapshot={}, change_reason=f"invalidated: {reason}"))
            await session.commit()
            return bool(result.rowcount)

    def _scope_kwargs(self) -> dict[str, Any]:
        return {"organization_id": self.scope.organization_id, "project_id": self.scope.project_id,
                "application_id": self.scope.application_id, "environment_id": self.scope.environment_id}
