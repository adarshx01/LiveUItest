"""Runner selection (spec section 14).

Runners are execution *strategies* applied by the same step pipeline: they set
how grounding is attempted first (validated memory, semantic DOM, visual) and
how forms are handled. Selection is deterministic and explainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.dsl.models import ActionType, TestSpec

RUNNERS = {
    "memory_replay": "re-grounds previously validated strategies before reasoning",
    "semantic": "DOM/accessibility-first grounding for standard accessible controls",
    "visual": "screenshot-based grounding for spatial interfaces (canvas, maps, charts)",
    "form": "form-centric execution with semantic field mapping",
    "hybrid": "memory + semantic + visual combined",
    "exploratory": "discovery of unknown page structures under bounded permissions",
    "data_driven": "one test executed for each data set (one execution per data set)",
    "parallel": "independent tests in isolated browser contexts",
    "debug": "extra diagnostics: screenshots and observations for every phase",
}


@dataclass
class RunnerChoice:
    name: str
    reason: str


def select_runner(spec: TestSpec, application_memory: dict[str, Any], *, forced: str | None = None) -> RunnerChoice:
    if forced:
        return RunnerChoice(forced, "requested explicitly")
    steps = spec.flat_steps()
    trajectory = (application_memory or {}).get("test_trajectory") or {}
    visual_steps = [s for s in steps if s.target and (s.target.visual_hint or "canvas" in s.target.preferred_roles)]
    form_steps = [s for s in steps if s.action in (ActionType.FILL, ActionType.SELECT, ActionType.CHECK,
                                                   ActionType.FILL_FORM, ActionType.UPLOAD)]
    if trajectory and trajectory.get("success_count", 0) >= 1 and trajectory.get("confidence", 0) >= 0.6:
        if visual_steps:
            return RunnerChoice("hybrid", "validated memory exists and some steps are spatial")
        return RunnerChoice("memory_replay", f"validated trajectory ({trajectory.get('success_count')} successful "
                                             "runs); strategies will be re-grounded before use")
    if visual_steps and len(visual_steps) >= len(steps) / 3:
        return RunnerChoice("visual", f"{len(visual_steps)} steps target spatial surfaces")
    if visual_steps:
        return RunnerChoice("hybrid", "mix of accessible controls and spatial targets")
    if len(form_steps) >= 5 or any(s.action == ActionType.FILL_FORM for s in steps):
        return RunnerChoice("form", f"{len(form_steps)} form interactions")
    if (application_memory or {}).get("known_pages", 0) == 0:
        return RunnerChoice("semantic", "no prior knowledge of this application; DOM-first discovery")
    return RunnerChoice("semantic", "standard accessible controls")
