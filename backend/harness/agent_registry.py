"""Agent harness: registry, protocol and permissions (spec sections 5 / 26).

The catalogue lists the logical reasoning agents and the deterministic
specialist services they are composed of. Each entry declares its kind
(deterministic / hybrid / llm), the permissions it needs, the model tier it may
use and its budget, so the harness can decide *whether* a model call is needed
at all and enforce what each agent may touch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    run_id: str
    test_case_id: str
    execution_id: str
    page_context_id: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    available_permissions: list[str] = Field(default_factory=list)
    budget_remaining: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    agent_name: str
    status: Literal["ok", "error", "skipped", "needs_human"]
    output: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: str | None = None
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requires_human_approval: bool = False


@dataclass(frozen=True)
class AgentSpec:
    name: str
    kind: Literal["deterministic", "hybrid", "llm"]
    responsibility: str
    permissions: tuple[str, ...] = ()
    model_tier: str | None = None
    graph_nodes: tuple[str, ...] = ()
    services: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AGENTS: list[AgentSpec] = [
    AgentSpec("supervisor_agent", "deterministic", "orchestrates the LangGraph lifecycle, budgets and routing",
              ("execution.control",), None, ("select_execution_strategy", "prepare_next_action", "next_test_action")),
    AgentSpec("requirement_analyzer_agent", "hybrid", "understands natural-language test instructions",
              ("spec.read",), "classification", ("normalize_test_case", "analyze_requirements"),
              ("test_ingestion", "nl_parser", "variable_resolver")),
    AgentSpec("scenario_planner_agent", "deterministic", "compiles the executable plan and form groups",
              ("spec.read",), None, ("compile_test_plan",), ("test_compiler",)),
    AgentSpec("plan_validator", "deterministic", "validates plans: targets, values, secrets, URLs, assertions",
              ("spec.read", "secrets.list"), None, ("validate_test_plan",)),
    AgentSpec("policy_guard_agent", "deterministic", "action regulator: domains, risk, approvals, captcha, secrets",
              ("policy.evaluate",), None, ("check_permissions", "validate_action", "recovery_policy_check"),
              ("domain_allowlist", "browser_action_policy", "sensitive_data_guard", "prompt_injection_guard")),
    AgentSpec("page_state_analyzer_agent", "deterministic", "loading, errors, challenges, overlays, page kind",
              ("browser.observe",), None, ("observe_current_page",), ("page_observer", "page_fingerprint")),
    AgentSpec("dom_intelligence_agent", "hybrid", "element inventory, classification and contextual grounding",
              ("browser.observe",), "reasoning", ("resolve_target_element", "ground_target"),
              ("snapshot_parser", "element_inventory_builder", "element_classifier", "element_grounder")),
    AgentSpec("visual_grounding_agent", "hybrid", "screenshot-based grounding for spatial / unlabeled targets",
              ("browser.observe", "browser.screenshot"), "vision", ("ground_target",),
              ("screenshot_analyzer", "coordinate_mapper", "visual_dom_correlator")),
    AgentSpec("logical_action_agent", "deterministic", "maps the logical step onto the concrete interaction",
              ("browser.act",), None, ("execute_action",), ("interaction_executor", "dropdown_resolver")),
    AgentSpec("locator_strategy_agent", "deterministic", "generates and validates replayable locator strategies",
              ("browser.observe",), None, ("resolve_target_element", "learn_pattern"), ("locator_strategy",)),
    AgentSpec("form_analyzer_agent", "deterministic", "form structure, semantic field mapping, generated data",
              ("browser.act",), None, ("execute_action",), ("form_filler",)),
    AgentSpec("assertion_agent", "hybrid", "deterministic assertions with evidence; LLM may only cite page text",
              ("browser.observe",), "classification", ("execute_action", "verify_action_result"),
              ("assertion_engine",)),
    AgentSpec("verification_agent", "deterministic", "post-action postcondition checks",
              ("browser.observe",), None, ("verify_action_result",), ("action_verifier",)),
    AgentSpec("failure_classifier_agent", "deterministic", "classifies failures and separates product from "
              "automation failures", (), None, ("classify_failure",), ("failure_classifier",)),
    AgentSpec("root_cause_analyzer_agent", "deterministic", "root cause and evidence for failures", (), None,
              ("classify_failure",)),
    AgentSpec("healing_agent", "hybrid", "locator healing via re-grounding without stale memory",
              ("browser.observe",), "reasoning", ("recovery_planner", "recovery_executor")),
    AgentSpec("recovery_agent", "deterministic", "bounded recovery strategies with idempotency protection",
              ("browser.act",), None, ("recovery_planner", "recovery_executor")),
    AgentSpec("memory_agent", "deterministic", "retrieves validated memory and re-grounds it",
              ("memory.read",), None, ("retrieve_application_memory", "retrieve_step_memory"),
              ("memory_retriever", "pgvector")),
    AgentSpec("learning_agent", "deterministic", "learns only from verified outcomes; versions strategies",
              ("memory.write",), None, ("learn_pattern", "persist_test_memory"), ("memory_writer",)),
    AgentSpec("test_result_evaluator_agent", "deterministic", "independent oracle: PASSED/FAILED/BLOCKED/INCONCLUSIVE",
              (), None, ("evaluate_test_result",), ("test_oracle",)),
    AgentSpec("report_generator_agent", "deterministic", "evidence-backed HTML/JSON reports with screenshots",
              ("artifacts.write",), None, ("generate_report",), ("screenshot_annotator",)),
]

REGISTRY: dict[str, AgentSpec] = {a.name: a for a in AGENTS}
NODE_TO_AGENT: dict[str, str] = {}
for _agent in AGENTS:
    for _node in _agent.graph_nodes:
        NODE_TO_AGENT.setdefault(_node, _agent.name)


def agent_for_node(node: str) -> str:
    return NODE_TO_AGENT.get(node, "supervisor_agent")
