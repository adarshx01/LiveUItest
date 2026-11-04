"""Execution comparison (spec section 20).

Lines two executions up step by step and states what changed between them:
outcome, step status (regressions and fixes), the element each step acted on,
how it was found (semantic, memory, visual, recovered), duration and assertion
results. Typical uses: a run that passed yesterday and fails today, a replay
against its first run, or a run before and after a UI change.

Steps are matched by id when both runs use the same test version. Otherwise
they are matched by normalized step text, then by position.
"""

from __future__ import annotations

import re
from typing import Any

SLOWER_RATIO = 1.5
SLOWER_MIN_MS = 750.0
_PASSING = {"PASSED"}
_FAILING = {"FAILED", "BLOCKED"}
_METRICS = ("total_steps", "passed_steps", "failed_steps", "blocked_steps", "recovered_steps", "retries",
            "recoveries", "memory_hits", "memory_lookups", "tool_calls", "llm_calls", "llm_cost_usd", "duration_ms")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s{}]", " ", (text or "").lower())).strip()


def _target(step: dict[str, Any]) -> dict[str, Any] | None:
    """Element the step finally acted on (last successful action, else the last attempt)."""

    actions = step.get("actions") or []
    chosen = next((a for a in reversed(actions) if a.get("ok")), actions[-1] if actions else None)
    if not chosen:
        return None
    grounding = chosen.get("grounding") or {}
    strategy = grounding.get("locator_strategy") or {}
    return {"display": chosen.get("target_display") or grounding.get("display") or None,
            "method": grounding.get("method"), "confidence": grounding.get("confidence"),
            "role": strategy.get("role") or grounding.get("role"),
            "name": strategy.get("name") or grounding.get("name"),
            "test_id": strategy.get("test_id"), "attempts": len(actions)}


def _step_view(step: dict[str, Any] | None) -> dict[str, Any] | None:
    if step is None:
        return None
    assertions = step.get("assertions") or []
    return {"step_id": step.get("step_id"), "index": step.get("index"), "status": step.get("status"),
            "message": (step.get("message") or "")[:300], "duration_ms": step.get("duration_ms") or 0.0,
            "attempts": step.get("attempts") or 0, "recovered": bool(step.get("recovered")),
            "memory_used": bool(step.get("memory_used")), "failure_category": step.get("failure_category"),
            "target": _target(step),
            "assertions": [{"type": a.get("type"), "status": a.get("status"), "expected": a.get("expected"),
                            "actual": a.get("actual")} for a in assertions]}


def align_steps(base: list[dict[str, Any]], other: list[dict[str, Any]], *,
                same_version: bool) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    base = sorted(base, key=lambda s: s.get("index") or 0)
    other = sorted(other, key=lambda s: s.get("index") or 0)
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    used: set[int] = set()

    def take(predicate: Any) -> dict[str, Any] | None:
        for i, candidate in enumerate(other):
            if i not in used and predicate(candidate):
                used.add(i)
                return candidate
        return None

    for step in base:
        match = None
        if same_version:
            match = take(lambda c, s=step: c.get("step_id") == s.get("step_id"))
        if match is None:
            match = take(lambda c, s=step: _norm(c.get("original_text", "")) == _norm(s.get("original_text", "")))
        if match is None and not same_version:
            match = take(lambda c, s=step: c.get("index") == s.get("index") and c.get("action") == s.get("action"))
        pairs.append((step, match))
    pairs.extend((None, candidate) for i, candidate in enumerate(other) if i not in used)
    return pairs


def _changes(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[str]:
    if a is None:
        return ["only in the compared execution"]
    if b is None:
        return ["not run in the compared execution"]
    out: list[str] = []
    sa, sb = a["status"], b["status"]
    if sa != sb:
        if sa in _PASSING and sb in _FAILING:
            out.append(f"regression: {sa} -> {sb}")
        elif sa in _FAILING and sb in _PASSING:
            out.append(f"fixed: {sa} -> {sb}")
        else:
            out.append(f"status: {sa} -> {sb}")
    ta, tb = a.get("target") or {}, b.get("target") or {}
    if ta.get("display") and tb.get("display") and ta["display"] != tb["display"]:
        out.append(f"target changed: {ta['display']} -> {tb['display']}")
    if ta.get("method") != tb.get("method") and ta.get("method") and tb.get("method"):
        out.append(f"grounding: {ta['method']} -> {tb['method']}")
    if a["recovered"] != b["recovered"]:
        out.append("needed recovery" if b["recovered"] else "no longer needs recovery")
    if a["memory_used"] != b["memory_used"]:
        out.append("now uses validated memory" if b["memory_used"] else "no longer uses memory")
    if (a.get("failure_category") or None) != (b.get("failure_category") or None) and b.get("failure_category"):
        out.append(f"failure: {b['failure_category']}")
    da, db = float(a["duration_ms"] or 0), float(b["duration_ms"] or 0)
    if da and db > da * SLOWER_RATIO and db - da >= SLOWER_MIN_MS:
        out.append(f"slower: {da / 1000:.1f}s -> {db / 1000:.1f}s")
    for x, y in zip(a["assertions"], b["assertions"], strict=False):
        if x.get("status") != y.get("status"):
            out.append(f"assertion {y.get('type')}: {x.get('status')} -> {y.get('status')} "
                       f"(actual {str(y.get('actual'))[:80]!r})")
    return out


def compare_executions(base_run: dict[str, Any], base_steps: list[dict[str, Any]], other_run: dict[str, Any],
                       other_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """``*_run``: run summaries (id, outcome, reason, test_case_id, test_case_version, metrics, created_at);
    ``*_steps``: persisted (redacted) ``StepResult`` dumps."""

    same_test = base_run.get("test_case_id") == other_run.get("test_case_id")
    same_version = same_test and base_run.get("test_case_version") == other_run.get("test_case_version")
    rows = []
    for a_raw, b_raw in align_steps(base_steps, other_steps, same_version=same_version):
        a, b = _step_view(a_raw), _step_view(b_raw)
        ref = b_raw or a_raw or {}
        rows.append({"index": ref.get("index"), "text": ref.get("original_text", ""), "action": ref.get("action"),
                     "base": a, "other": b, "changes": _changes(a, b)})
    metrics_a, metrics_b = base_run.get("metrics") or {}, other_run.get("metrics") or {}
    deltas = {k: {"base": metrics_a.get(k), "other": metrics_b.get(k)} for k in _METRICS
              if metrics_a.get(k) != metrics_b.get(k)}
    regressions = [r for r in rows if any(c.startswith("regression") for c in r["changes"])]
    fixes = [r for r in rows if any(c.startswith("fixed") for c in r["changes"])]
    target_changes = [r for r in rows if any(c.startswith("target changed") for c in r["changes"])]
    summary = []
    if base_run.get("outcome") != other_run.get("outcome"):
        summary.append(f"outcome {base_run.get('outcome')} -> {other_run.get('outcome')}")
    if regressions:
        summary.append(f"{len(regressions)} step(s) regressed (first: step {regressions[0]['index']})")
    if fixes:
        summary.append(f"{len(fixes)} step(s) fixed")
    if target_changes:
        summary.append(f"{len(target_changes)} step(s) acted on a different element")
    if not same_test:
        summary.append("different test cases: steps matched by text and position")
    elif not same_version:
        summary.append(f"different test versions (v{base_run.get('test_case_version')} vs "
                       f"v{other_run.get('test_case_version')}): steps matched by text and position")
    return {"base": {k: base_run.get(k) for k in ("id", "outcome", "reason", "test_case_id", "test_case_version",
                                                  "created_at", "runner")},
            "other": {k: other_run.get(k) for k in ("id", "outcome", "reason", "test_case_id", "test_case_version",
                                                    "created_at", "runner")},
            "same_test_case": same_test, "same_version": same_version,
            "summary": summary or ["no differences in outcome, step status or targets"],
            "metric_deltas": deltas, "steps": rows,
            "counts": {"regressions": len(regressions), "fixes": len(fixes), "target_changes": len(target_changes),
                       "changed_steps": sum(1 for r in rows if r["changes"])}}
