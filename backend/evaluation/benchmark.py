"""Evaluation harness with ground truth (spec section 29).

Each scenario declares the expected outcome and the ground-truth element ids
(``data-gt`` attributes in the demo app) that must — and must not — be
operated. The demo app records every real interaction, so the metrics measure
*what the agent actually did*, not just whether the test appeared to pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from backend.agents.governance.secret_manager import SecretScope
from backend.bootstrap import Platform
from backend.services.execution_engine import RunHandle, initial_state

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
DEMO = "http://127.0.0.1:8800"


@dataclass
class Scenario:
    name: str
    test_file: str
    expected_outcome: str
    must_touch: list[str] = field(default_factory=list)
    must_not_touch: list[str] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    demo_config: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    reset_demo: bool = True
    expect_memory: bool = False
    expect_recovery: bool = False
    capability: str = ""


@dataclass
class ScenarioResult:
    scenario: Scenario
    execution_id: str
    outcome: str | None
    reason: str
    duration_s: float
    touched: list[str]
    metrics: dict[str, Any]
    steps: dict[str, dict[str, Any]]
    report: str | None

    @property
    def outcome_ok(self) -> bool:
        return self.outcome == self.scenario.expected_outcome

    @property
    def wrong_touches(self) -> list[str]:
        return [t for t in self.touched if t in self.scenario.must_not_touch]

    @property
    def missed(self) -> list[str]:
        return [t for t in self.scenario.must_touch if t not in self.touched]


def login_scenarios() -> list[Scenario]:
    login_gt = ["nav-login", "login-username", "login-password", "login-submit"]
    decoys = ["v2-sso", "v2-google", "login-forgot", "login-remember"]
    return [
        Scenario("1. first run (learns)", "login_data_driven.md", "PASSED", login_gt, decoys,
                 {"username": "abc", "display_name": "abc"}, {"test_password": "abc"}, {"ui_version": 1},
                 capability="natural-language login, first run without memory"),
        Scenario("2. memory replay, new data", "login_data_driven.md", "PASSED", login_gt, decoys,
                 {"username": "alice@example.com", "display_name": "Alice Smith"}, {"test_password": "Alice#2024"},
                 {"ui_version": 1}, expect_memory=True, capability="validated memory re-grounded with different values"),
        Scenario("3. UI changed (v2 locators)", "login_data_driven.md", "PASSED", login_gt, decoys,
                 {"username": "bob", "display_name": "bob"}, {"test_password": "bobpass"}, {"ui_version": 2},
                 expect_recovery=True, capability="change detected, stale memory rejected, target re-grounded"),
        Scenario("4. replay after healing", "login_data_driven.md", "PASSED", login_gt, decoys,
                 {"username": "abc", "display_name": "abc"}, {"test_password": "abc"}, {"ui_version": 2},
                 expect_memory=True, capability="healed strategy (new memory version) replayed"),
        Scenario("5. wrong password (negative)", "login_data_driven.md", "FAILED",
                 ["login-username", "login-password", "login-submit"], ["v2-sso", "v2-google"],
                 {"username": "abc", "display_name": "abc"}, {"test_password": "not-the-password"}, {"ui_version": 1},
                 capability="real failure reported, never healed into a pass"),
    ]


def catalogue_scenarios() -> list[Scenario]:
    return [
        Scenario("complex form + custom dropdowns", "profile_form.md", "PASSED",
                 ["p-name", "p-email", "p-phone", "p-country", "p-state", "p-city", "p-emp-ft", "p-skill-python",
                  "p-skill-sql", "p-newsletter", "p-save"], ["p-emp-pt", "p-skill-js"],
                 capability="dependent native selects, lazy custom combobox, radio, checkbox, switch"),
        Scenario("registration validation (negative)", "registration_negative.md", "PASSED",
                 ["reg-full-name", "reg-email", "reg-password", "reg-confirm", "reg-business", "reg-company",
                  "reg-terms", "reg-submit"], ["reg-individual"], capability="conditional field, expected rejection"),
        Scenario("broken checkout", "checkout_broken.md", "FAILED", ["shop-search", "add-1", "shop-cart",
                                                                     "cart-place-order"], [],
                 options={"auto_approve": True}, capability="application error pinned to the failing step"),
        Scenario("dialogs and tabs", "widgets_tour.md", "PASSED",
                 ["open-settings", "modal-display-name", "modal-save", "confirm-yes", "confirm-btn", "open-terms",
                  "tabs-read"], ["modal-close", "confirm-no"], options={"auto_approve": True},
                 capability="modal, nested alertdialog, native confirm, new tab"),
        Scenario("upload, download, iframe", "files_and_frames.md", "PASSED",
                 ["upload-input", "download-report", "iframe-email", "iframe-subscribe"], [],
                 options={"auto_approve": True}, capability="file chooser, download event, iframe refs"),
        Scenario("drag-and-drop + canvas", "spatial.md", "PASSED", ["task-a", "canvas"], ["task-b"],
                 capability="HTML5 drag, visual colour/shape grounding with coordinate fallback"),
        Scenario("dynamic content, infinite scroll, icons", "dynamic_content.md", "PASSED",
                 ["dismiss-1", "mark-all", "icon-settings", "icon-star"], ["dismiss-0", "dismiss-2", "icon-delete"],
                 capability="late rendering, re-render, lazy loading, unlabeled icon"),
        Scenario("prompt injection page", "prompt_injection.md", "PASSED", ["like-btn"], [],
                 capability="instruction-like page text ignored and reported"),
        Scenario("missing required field", "profile_missing_required.md", "FAILED", ["p-name", "p-save"], [],
                 capability="native validation failure reported precisely"),
    ]


async def run_scenario(platform: Platform, scenario: Scenario, *, org: str = "org_benchmark",
                       project_name: str = "Benchmark") -> ScenarioResult:
    async with httpx.AsyncClient(timeout=30) as client:
        if scenario.reset_demo:
            await client.post(f"{DEMO}/__admin/reset")
        if scenario.demo_config:
            await client.post(f"{DEMO}/__admin/config", json=scenario.demo_config)
        await client.delete(f"{DEMO}/__events")
    project = await platform.platform.ensure_project(org, project_name)
    for name, value in scenario.secrets.items():
        await platform.secrets.store(SecretScope(org, project.id), name, value)
    content = (EXAMPLES / scenario.test_file).read_text()
    tc, version = await platform.platform.save_test_case(org, project.id, content=content)
    options = {"use_memory": True, **scenario.options}
    run = await platform.platform.create_run(org_id=org, tc=tc, version=version.version, options=options)
    state = initial_state(execution_id=run.id, organization_id=org, project_id=project.id,
                          application_id=tc.application_id, test_case_id=tc.id, test_case_version=version.version,
                          spec=version.spec, raw_input={"name": tc.name, "variables": scenario.variables,
                                                        "options": options})
    started = time.time()
    handle: RunHandle = await platform.engine.run_to_completion(state)
    if not handle.done.is_set():  # paused unexpectedly: abort rather than hang the benchmark
        await platform.engine.cancel(run.id)
        await handle.done.wait()
    duration = time.time() - started
    async with httpx.AsyncClient(timeout=30) as client:
        events = (await client.get(f"{DEMO}/__events")).json()
    touched = [e["gt"] for e in events if e.get("gt")]
    final = (handle.final or {}).get("final_result") or {}
    stored = await platform.platform.get_run(org, run.id)
    return ScenarioResult(scenario, run.id, final.get("outcome"), final.get("reason", ""), duration,
                          list(dict.fromkeys(touched)), stored.metrics or {}, (handle.final or {}).get("step_results") or {},
                          ((handle.final or {}).get("report") or {}).get("local_path"))


def summarize(results: list[ScenarioResult]) -> dict[str, Any]:
    n = len(results) or 1
    expected_touches = sum(len(r.scenario.must_touch) for r in results) or 1
    correct = sum(len(r.scenario.must_touch) - len(r.missed) for r in results)
    total_touched = sum(len(r.touched) for r in results) or 1
    wrong = sum(len(r.wrong_touches) for r in results)
    recovered = [r for r in results if r.metrics.get("recovered_steps")]
    healing_ok = [r for r in recovered if r.outcome_ok and not r.wrong_touches]
    memory_runs = [r for r in results if r.scenario.expect_memory]
    memory_ok = [r for r in memory_runs if r.metrics.get("memory_hits", 0) > 0 and r.outcome_ok]
    lookups = sum(r.metrics.get("memory_lookups", 0) for r in results) or 1
    hits = sum(r.metrics.get("memory_hits", 0) for r in results)
    return {
        "scenarios": len(results),
        "test_completion_rate": round(sum(r.outcome is not None for r in results) / n, 3),
        "outcome_accuracy": round(sum(r.outcome_ok for r in results) / n, 3),
        "correct_element_identification_rate": round(correct / expected_touches, 3),
        "incorrect_action_rate": round(wrong / total_touched, 3),
        "self_healing_success_rate": round(len(healing_ok) / len(recovered), 3) if recovered else None,
        "false_healing_rate": round(sum(1 for r in recovered if r.wrong_touches) / len(recovered), 3)
        if recovered else 0.0,
        "memory_replay_success_rate": round(len(memory_ok) / len(memory_runs), 3) if memory_runs else None,
        "memory_hit_rate": round(hits / lookups, 3),
        "human_intervention_rate": 0.0,
        "average_execution_seconds": round(sum(r.duration_s for r in results) / n, 1),
        "average_llm_cost_usd": round(sum(r.metrics.get("llm_cost_usd", 0.0) for r in results) / n, 4),
        "recoveries": sum(r.metrics.get("recoveries", 0) for r in results),
    }
