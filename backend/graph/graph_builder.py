"""Builds the stateful LangGraph execution graph (spec section 6.1)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.graph import nodes as n
from backend.graph.state import TestExecutionState


def _after_validate_plan(state: TestExecutionState) -> str:
    return "evaluate_test_result" if state.get("phase") == "plan_blocked" else "check_permissions"


def _after_permissions(state: TestExecutionState) -> str:
    return "evaluate_test_result" if state.get("phase") == "plan_blocked" else "initialize_browser_session"


def _abort_or(next_node: str) -> Any:
    def route(state: TestExecutionState) -> str:
        if state.get("phase") == "abort":
            return "final_assertions"
        if state.get("phase") == "error" and next_node in ("observe_initial_page", "retrieve_application_memory"):
            return "final_assertions"
        return next_node

    return route


def _after_browser(state: TestExecutionState) -> str:
    if state.get("phase") in ("abort", "error"):
        return "evaluate_test_result"
    return "observe_initial_page"


def _after_prepare(state: TestExecutionState) -> str:
    if state.get("phase") == "abort":
        return "final_assertions"
    return "final_assertions" if state.get("phase") == "steps_done" else "retrieve_step_memory"


def _after_observe(state: TestExecutionState) -> str:
    phase = state.get("phase")
    if phase == "abort":
        return "final_assertions"
    if phase == "captcha":
        return "await_human"
    if phase == "error":
        return "classify_failure"
    return "resolve_target_element"


def _after_ground(state: TestExecutionState) -> str:
    phase = state.get("phase")
    if phase == "abort":
        return "final_assertions"
    if phase in ("ground_failed", "error"):
        return "classify_failure"
    return "validate_action"


def _after_validate(state: TestExecutionState) -> str:
    phase = state.get("phase")
    if phase == "validated":
        return "execute_action"
    if phase == "approval_required":
        return "human_approval"
    if phase == "abort":
        return "final_assertions"
    return "classify_failure"


def _after_approval(state: TestExecutionState) -> str:
    return "validate_action" if state.get("phase") == "approved" else "fail_step"


def _after_execute(state: TestExecutionState) -> str:
    if state.get("phase") == "abort":
        return "final_assertions"
    return "wait_for_state_change"


def _after_verify(state: TestExecutionState) -> str:
    phase = state.get("phase")
    if phase == "verified":
        return "learn_pattern"
    if phase == "abort":
        return "final_assertions"
    return "classify_failure"


def _after_classify(state: TestExecutionState) -> str:
    ctx = state.get("failure_context") or {}
    if state.get("phase") == "abort":
        return "final_assertions"
    if ctx.get("category") == "CAPTCHA_DETECTED":
        return "await_human"
    if ctx.get("healable"):
        return "recovery_planner"
    return "fail_step"


def _after_recovery_plan(state: TestExecutionState) -> str:
    return "recovery_policy_check" if state.get("phase") == "recovery_planned" else "fail_step"


def _after_recovery_policy(state: TestExecutionState) -> str:
    return "recovery_executor" if state.get("phase") == "recovery_allowed" else "fail_step"


def _after_recovery_exec(state: TestExecutionState) -> str:
    phase = state.get("phase")
    if phase == "recovered":
        return "observe_current_page"
    if phase == "recovered_done":
        return "verify_action_result"
    if phase == "abort":
        return "final_assertions"
    return "recovery_planner"


def _after_human(state: TestExecutionState) -> str:
    return "observe_current_page" if state.get("phase") == "human_resolved" else "fail_step"


def _after_next(state: TestExecutionState) -> str:
    return "final_assertions" if state.get("phase") == "abort" else "prepare_next_action"


def build_graph() -> StateGraph:
    g = StateGraph(TestExecutionState)
    for name in ("ingest_test_case", "normalize_test_case", "analyze_requirements", "compile_test_plan",
                 "validate_test_plan", "check_permissions", "initialize_browser_session", "observe_initial_page",
                 "retrieve_application_memory", "select_execution_strategy", "prepare_next_action",
                 "retrieve_step_memory", "observe_current_page", "resolve_target_element", "ground_target",
                 "validate_action", "human_approval", "execute_action", "wait_for_state_change",
                 "verify_action_result", "learn_pattern", "classify_failure", "recovery_planner",
                 "recovery_policy_check", "recovery_executor", "await_human", "fail_step", "next_test_action",
                 "final_assertions", "evaluate_test_result", "persist_test_memory", "generate_report"):
        fn = getattr(n, name if name != "classify_failure" else "classify_failure_node")
        g.add_node(name, fn)

    g.add_edge(START, "ingest_test_case")
    g.add_edge("ingest_test_case", "normalize_test_case")
    g.add_edge("normalize_test_case", "analyze_requirements")
    g.add_edge("analyze_requirements", "compile_test_plan")
    g.add_edge("compile_test_plan", "validate_test_plan")
    g.add_conditional_edges("validate_test_plan", _after_validate_plan)
    g.add_conditional_edges("check_permissions", _after_permissions)
    g.add_conditional_edges("initialize_browser_session", _after_browser)
    g.add_conditional_edges("observe_initial_page", _abort_or("retrieve_application_memory"))
    g.add_conditional_edges("retrieve_application_memory", _abort_or("select_execution_strategy"))
    g.add_conditional_edges("select_execution_strategy", _abort_or("prepare_next_action"))
    g.add_conditional_edges("prepare_next_action", _after_prepare)
    g.add_conditional_edges("retrieve_step_memory", _abort_or("observe_current_page"))
    g.add_conditional_edges("observe_current_page", _after_observe)
    g.add_conditional_edges("resolve_target_element", _abort_or("ground_target"))
    g.add_conditional_edges("ground_target", _after_ground)
    g.add_conditional_edges("validate_action", _after_validate)
    g.add_conditional_edges("human_approval", _after_approval)
    g.add_conditional_edges("execute_action", _after_execute)
    g.add_conditional_edges("wait_for_state_change", _abort_or("verify_action_result"))
    g.add_conditional_edges("verify_action_result", _after_verify)
    g.add_edge("learn_pattern", "next_test_action")
    g.add_conditional_edges("next_test_action", _after_next)
    g.add_conditional_edges("classify_failure", _after_classify)
    g.add_conditional_edges("recovery_planner", _after_recovery_plan)
    g.add_conditional_edges("recovery_policy_check", _after_recovery_policy)
    g.add_conditional_edges("recovery_executor", _after_recovery_exec)
    g.add_conditional_edges("await_human", _after_human)
    g.add_edge("fail_step", "final_assertions")
    g.add_edge("final_assertions", "evaluate_test_result")
    g.add_edge("evaluate_test_result", "persist_test_memory")
    g.add_edge("persist_test_memory", "generate_report")
    g.add_edge("generate_report", END)
    return g


def compile_graph(checkpointer: Any) -> Any:
    return build_graph().compile(checkpointer=checkpointer)
