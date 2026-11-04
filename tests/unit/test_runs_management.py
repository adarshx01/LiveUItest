"""Test management (spec section 20): partial re-runs, execution comparison, retry data sets."""

from __future__ import annotations

from backend.agents.governance import redaction
from backend.agents.governance.secret_manager import seal_variables, unseal_variables
from backend.agents.planner.requirement_analyzer import RequirementAnalyzer
from backend.agents.planner.step_scope import StepScope, plan_scope
from backend.agents.planner.test_compiler import TestCompiler
from backend.agents.planner.test_ingestion import ingest
from backend.services.comparison import align_steps, compare_executions
from backend.services.execution_engine import initial_state

LOGIN = """URL: http://127.0.0.1:8800
1. Click on the Login button
2. Enter username as {{username}}
3. Enter password as {{test_password}}
4. Click on the Sign In button
5. Verify that the user is redirected to the dashboard
6. IF the variable plan equals pro:
   Click Upgrade
   ELSE:
   Click Stay on free
7. Verify that "Welcome" is displayed
"""


async def _spec():
    (case,) = ingest(LOGIN)
    return TestCompiler().compile(await RequirementAnalyzer().analyze(case))


async def test_scope_runs_prerequisites_and_stops_after_selection():
    spec = await _spec()
    top = [s.id for s in spec.steps]
    sign_in = spec.steps[4].id  # implicit navigation is step 1
    plan = plan_scope(spec, StepScope((sign_in,)))
    assert plan.queue == top[:5] and plan.selected == [sign_in]
    assert plan.prerequisites == top[:4] and plan.not_run == top[5:]
    assert not plan.has_assertions  # the selection verifies nothing on its own


async def test_isolated_scope_keeps_only_entry_navigation():
    spec = await _spec()
    verify = spec.steps[5].id
    plan = plan_scope(spec, StepScope((verify,), with_prerequisites=False))
    assert plan.queue == [spec.steps[0].id, verify] and plan.has_assertions


async def test_selecting_a_branch_child_selects_the_whole_conditional():
    spec = await _spec()
    conditional = next(s for s in spec.steps if s.then_steps)
    child = conditional.then_steps[0].id
    plan = plan_scope(spec, StepScope((child, "step_999")))
    assert plan.selected == [conditional.id] and plan.unknown == ["step_999"]


async def test_unknown_steps_select_nothing():
    spec = await _spec()
    plan = plan_scope(spec, StepScope(("step_404",)))
    assert plan.queue == [] and plan.selected == [] and plan.unknown == ["step_404"]


def test_scope_round_trips_through_run_options():
    scope = StepScope(("step_003",), with_prerequisites=False, rerun_of="exec_1")
    assert StepScope.from_options({"scope": scope.to_options()}) == scope
    assert StepScope.from_options({}) is None and StepScope.from_options({"scope": {"step_ids": []}}) is None


def test_credential_variables_are_sealed_out_of_graph_state():
    saved = set(redaction._sensitive_values)
    try:
        state = initial_state(execution_id="e", organization_id="o", project_id="p", application_id="a",
                              test_case_id="t", test_case_version=1, spec={},
                              raw_input={"variables": {"username": "bob", "test_password": "bobpass", "pin": "1234"}})
        raw = state["raw_input"]
        assert raw["variables"] == {"username": "bob"}
        assert "bobpass" not in repr(state) and "1234" not in repr(state)
        assert raw["sealed_variables"]["names"] == ["pin", "test_password"]
        assert unseal_variables(raw["sealed_variables"]) == {"test_password": "bobpass", "pin": "1234"}
        # a retry carries the sealed bundle over and merges new values into it
        plain, sealed = seal_variables({"username": "alice"})
        assert sealed is None and plain == {"username": "alice"}
        again = initial_state(execution_id="e2", organization_id="o", project_id="p", application_id="a",
                              test_case_id="t", test_case_version=1, spec={},
                              raw_input={"variables": {"otp": "999"}, "sealed_variables": raw["sealed_variables"]})
        assert unseal_variables(again["raw_input"]["sealed_variables"]) == {"test_password": "bobpass", "pin": "1234",
                                                                            "otp": "999"}
    finally:
        redaction._sensitive_values.clear()
        redaction._sensitive_values.update(saved)


def _step(sid, index, text, status, *, target="button 'Sign In'", method="semantic", recovered=False,
          memory=False, duration=900.0, assertion=None):
    actions = [{"ok": status != "FAILED", "target_display": target,
                "grounding": {"display": target, "method": method, "confidence": 0.97,
                              "locator_strategy": {"role": "button", "name": target.split("'")[1]}}}]
    return {"step_id": sid, "index": index, "original_text": text, "action": "click", "status": status,
            "message": "", "duration_ms": duration, "attempts": 1, "recovered": recovered, "memory_used": memory,
            "failure_category": "ASSERTION_FAILURE" if status == "FAILED" else None, "actions": actions,
            "assertions": [assertion] if assertion else []}


def test_comparison_reports_regression_target_change_and_slowdown():
    base_steps = [_step("step_001", 1, "Open the app", "PASSED"),
                  _step("step_002", 2, "Click on the Sign In button", "PASSED"),
                  _step("step_003", 3, "Verify Welcome is displayed", "PASSED",
                        assertion={"type": "TEXT_VISIBLE", "status": "PASSED", "expected": "Welcome",
                                   "actual": "Welcome"})]
    other_steps = [_step("step_001", 1, "Open the app", "PASSED", memory=True),
                   _step("step_002", 2, "Click on the Sign In button", "PASSED", target="button 'Log in'",
                         recovered=True, duration=4000.0),
                   _step("step_003", 3, "Verify Welcome is displayed", "FAILED",
                         assertion={"type": "TEXT_VISIBLE", "status": "FAILED", "expected": "Welcome",
                                    "actual": "Invalid credentials"})]
    run = {"test_case_id": "tc", "test_case_version": 2}
    out = compare_executions({**run, "id": "a", "outcome": "PASSED", "metrics": {"recoveries": 0}}, base_steps,
                             {**run, "id": "b", "outcome": "FAILED", "metrics": {"recoveries": 1}}, other_steps)
    assert out["same_version"] and out["counts"] == {"regressions": 1, "fixes": 0, "target_changes": 1,
                                                     "changed_steps": 3}
    by_index = {r["index"]: r["changes"] for r in out["steps"]}
    assert by_index[1] == ["now uses validated memory"]
    assert "target changed: button 'Sign In' -> button 'Log in'" in by_index[2]
    assert "needed recovery" in by_index[2] and any(c.startswith("slower") for c in by_index[2])
    assert by_index[3][0] == "regression: PASSED -> FAILED"
    assert any("Invalid credentials" in c for c in by_index[3])
    assert out["metric_deltas"] == {"recoveries": {"base": 0, "other": 1}}
    assert out["summary"][0] == "outcome PASSED -> FAILED"


def test_alignment_across_versions_uses_text_then_position():
    base = [_step("step_001", 1, "Open the app", "PASSED"), _step("step_002", 2, "Click Save", "PASSED")]
    other = [_step("step_001", 1, "Open the app", "PASSED"), _step("step_002", 2, "Click Cancel", "PASSED"),
             _step("step_003", 3, "Click Save", "PASSED")]
    pairs = align_steps(base, other, same_version=False)
    assert [(a and a["original_text"], b and b["original_text"]) for a, b in pairs] == [
        ("Open the app", "Open the app"), ("Click Save", "Click Save"), (None, "Click Cancel")]
