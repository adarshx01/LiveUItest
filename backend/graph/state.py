"""Strongly typed, compact LangGraph state (spec section 6.2).

Large artifacts (DOM snapshots, screenshots, inventories) are *references*; the
full objects live in the runtime and in object storage.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TestExecutionState(TypedDict, total=False):
    run_id: str
    organization_id: str
    project_id: str
    application_id: str
    environment_id: str | None
    test_case_id: str
    test_case_version: int
    execution_id: str

    raw_input: dict[str, Any]
    test_spec: dict[str, Any]
    compiled_steps: list[dict[str, Any]]
    queue: list[str]
    current_step_id: str | None
    current_step_index: int
    plan_issues: list[dict[str, Any]]

    browser_session_id: str | None
    active_tab_id: str | None
    current_url: str
    current_page_title: str
    page_fingerprint: dict[str, Any]

    dom_snapshot: str | None  # artifact reference
    accessibility_tree: dict[str, Any]  # summary only
    element_inventory: list[dict[str, Any]]  # compact top candidates only
    observation_id: str | None
    screenshot_reference: str | None
    visual_context: dict[str, Any]

    application_memory: dict[str, Any]
    retrieved_memory: list[dict[str, Any]]
    selected_memory: dict[str, Any] | None
    runner: str

    grounded_target: dict[str, Any] | None
    planned_action: dict[str, Any] | None
    executed_action: dict[str, Any] | None
    phase: str

    assertions: list[dict[str, Any]]
    verification_result: dict[str, Any] | None
    failure_context: dict[str, Any] | None
    recovery_plan: dict[str, Any] | None
    recovery_attempts: int
    recovery_total: int

    step_results: dict[str, dict[str, Any]]
    execution_history: list[dict[str, Any]]
    pending_approval: dict[str, Any] | None
    approval_decisions: dict[str, Any]

    execution_status: str
    blocked_reason: str | None
    final_result: dict[str, Any] | None
    report: dict[str, Any] | None
    execution_budget: dict[str, Any]
    learning_stats: dict[str, Any]
