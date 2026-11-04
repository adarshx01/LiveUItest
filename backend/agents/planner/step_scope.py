"""Partial re-runs (spec section 20, "re-run individual steps").

A user can re-run selected steps of a test version. Steps rarely stand alone:
"Click Sign In" needs the page opened and the fields filled first. By default
the earlier steps run as *prerequisites* to rebuild that state, and the run
stops after the last selected step. In isolated mode only the selected steps
run, after the entry navigation, so they start on the application.

Selection works on top-level steps. Selecting a step inside an IF/ELSE block
selects the whole block, because a branch cannot run without its condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.dsl.models import ActionType, TestSpec


@dataclass(frozen=True)
class StepScope:
    step_ids: tuple[str, ...]
    with_prerequisites: bool = True
    rerun_of: str | None = None

    @classmethod
    def from_options(cls, options: dict[str, Any] | None) -> StepScope | None:
        raw = (options or {}).get("scope") or {}
        ids = tuple(str(i) for i in raw.get("step_ids") or [] if i)
        if not ids:
            return None
        return cls(ids, bool(raw.get("with_prerequisites", True)), raw.get("rerun_of"))

    def to_options(self) -> dict[str, Any]:
        return {"step_ids": list(self.step_ids), "with_prerequisites": self.with_prerequisites,
                "rerun_of": self.rerun_of}


@dataclass
class ScopedPlan:
    queue: list[str]  # top-level step ids to execute, in order
    selected: list[str]  # top-level ids the user asked for
    prerequisites: list[str] = field(default_factory=list)  # earlier steps run to rebuild page state
    not_run: list[str] = field(default_factory=list)  # top-level steps outside the scope
    unknown: list[str] = field(default_factory=list)  # requested ids absent from this test version
    has_assertions: bool = False  # does the scoped run verify anything?

    def to_state(self, scope: StepScope) -> dict[str, Any]:
        return {**scope.to_options(), "queue": self.queue, "selected": self.selected,
                "prerequisites": self.prerequisites, "not_run": self.not_run, "unknown": self.unknown,
                "has_assertions": self.has_assertions}

    def describe(self) -> str:
        parts = [f"selected {', '.join(self.selected)}"]
        if self.prerequisites:
            parts.append(f"{len(self.prerequisites)} prerequisite step(s) re-run to rebuild page state")
        if self.not_run:
            parts.append(f"{len(self.not_run)} later/unselected step(s) not run")
        return "; ".join(parts)


def plan_scope(spec: TestSpec, scope: StepScope) -> ScopedPlan:
    top = spec.steps
    order = [s.id for s in top]
    owner = {flat.id: step.id for step in top for flat in step.iter_flat()}
    unknown = [i for i in scope.step_ids if i not in owner]
    wanted = {owner[i] for i in scope.step_ids if i in owner}
    if not wanted:
        return ScopedPlan(queue=[], selected=[], not_run=order, unknown=unknown)
    last = max(order.index(w) for w in wanted)
    queue: list[str] = []
    prerequisites: list[str] = []
    for i, step in enumerate(top[: last + 1]):
        if step.id in wanted:
            queue.append(step.id)
        elif scope.with_prerequisites or (i == 0 and step.action == ActionType.NAVIGATE):
            queue.append(step.id)
            prerequisites.append(step.id)
    by_id = {s.id: s for s in top}
    has_assertions = any(flat.action == ActionType.ASSERT for sid in queue for flat in by_id[sid].iter_flat())
    return ScopedPlan(queue=queue, selected=[s for s in order if s in wanted], prerequisites=prerequisites,
                      not_run=[s for s in order if s not in queue], unknown=unknown, has_assertions=has_assertions)
