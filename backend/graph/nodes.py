"""LangGraph node implementations (spec section 6.1).

Nodes are thin: they fetch the live runtime for the execution, run one phase,
and return a compact state update. Each node is traced (agent + duration) and
checkpointed by LangGraph, so an interrupted execution knows exactly which
phase completed — in particular whether an action was already executed.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from backend.agents.planner.lexicon import normalize_text
from backend.agents.planner.plan_validator import blocking
from backend.agents.planner.test_ingestion import RawLine, RawTestCase, ingest
from backend.agents.governance.secret_manager import SecretScope
from backend.agents.verification.test_oracle import evaluate_outcome
from backend.core.exceptions import BudgetExceeded, ExecutionCancelled, PlatformError
from backend.core.logging import get_logger
from backend.dsl.models import ActionType, Step, TestSpec
from backend.dsl.results import FailureCategory, StepResult, StepStatus, TestOutcome
from backend.execution.events import event_bus
from backend.execution.runtime import ExecutionOptions, ExecutionRequest, ExecutionRuntime, runtime_registry
from backend.graph.services import get_graph_services
from backend.graph.state import TestExecutionState
from backend.harness.agent_registry import agent_for_node
from backend.memory.memory_manager import MemoryManager, MemoryScope
from backend.services.recorder import DatabaseRecorder, NullRecorder

log = get_logger(__name__)

NodeFn = Callable[[TestExecutionState, RunnableConfig | None], Awaitable[dict[str, Any]]]


def node(name: str) -> Callable[[NodeFn], NodeFn]:
    """Trace, time and guard a node; convert step-phase crashes into step failures."""

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: TestExecutionState, config: RunnableConfig | None = None) -> dict[str, Any]:
            started = time.perf_counter()
            execution_id = state.get("execution_id", "")
            runtime = runtime_registry.get(execution_id)
            if runtime is not None:
                runtime.bind()
            status = "ok"
            try:
                update = await fn(state, config)
            except (ExecutionCancelled, BudgetExceeded) as exc:
                status = "stopped"
                update = {"execution_status": "CANCELLED" if isinstance(exc, ExecutionCancelled) else "BLOCKED",
                          "blocked_reason": exc.message, "phase": "abort"}
            except PlatformError as exc:
                status = "error"
                log.warning("node failed", node=name, error=exc.message)
                update = {"phase": "error", "failure_context": {"category": FailureCategory.UNKNOWN_FAILURE.value,
                                                                "root_cause": exc.message, "node": name}}
            except Exception as exc:  # noqa: BLE001 - never crash the graph on a single node
                if exc.__class__.__name__ in ("GraphInterrupt", "Interrupt", "ParentCommand"):
                    raise
                status = "error"
                log.exception("node crashed", node=name)
                update = {"phase": "error", "failure_context": {"category": FailureCategory.UNKNOWN_FAILURE.value,
                                                                "root_cause": f"{type(exc).__name__}: {exc}"[:300],
                                                                "node": name}}
            duration = (time.perf_counter() - started) * 1000
            runtime = runtime_registry.get(execution_id)
            if runtime is not None and runtime.recorder is not None:
                await runtime.recorder.trace(agent=agent_for_node(name), node=name,
                                             step_id=state.get("current_step_id"), status=status,
                                             duration_ms=round(duration, 1),
                                             summary=str(update.get("phase", ""))[:200])
            return update

        wrapper.__name__ = name
        return wrapper

    return decorator


def _runtime(state: TestExecutionState) -> ExecutionRuntime:
    runtime = runtime_registry.get(state["execution_id"])
    if runtime is None:
        raise PlatformError("No live runtime for this execution (process restarted?)",
                            details={"execution_id": state["execution_id"]})
    return runtime


def _step(state: TestExecutionState) -> Step:
    return _runtime(state).step(state["current_step_id"])  # type: ignore[index]


def _results_dict(runtime: ExecutionRuntime) -> dict[str, dict[str, Any]]:
    return {sid: {"status": r.status.value, "message": r.message[:300], "index": r.index,
                  "recovered": r.recovered, "memory_used": r.memory_used}
            for sid, r in runtime.results.items()}


# =========================================================================== planning
@node("ingest_test_case")
async def ingest_test_case(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    raw = state.get("raw_input") or {}
    await event_bus.publish(state["execution_id"], "EXECUTION_CREATED", raw.get("name") or "test execution",
                            test_case_id=state.get("test_case_id"))
    if state.get("test_spec"):
        return {"phase": "ingested", "execution_status": "RUNNING"}
    cases = ingest(raw.get("content", ""), fmt=raw.get("format"), base_url=raw.get("base_url"),
                   name=raw.get("name"), variables=raw.get("variables"))
    case = cases[0]
    return {"phase": "ingested", "execution_status": "RUNNING",
            "raw_input": {**raw, "_case": _case_to_dict(case)}}


def _case_to_dict(case: RawTestCase) -> dict[str, Any]:
    def line(l: RawLine) -> dict[str, Any]:
        return {"text": l.text, "line_no": l.line_no, "condition": l.condition,
                "then": [line(x) for x in l.then_lines], "else": [line(x) for x in l.else_lines]}

    return {"name": case.name, "base_url": case.base_url, "description": case.description,
            "lines": [line(l) for l in case.lines], "expected_results": case.expected_results,
            "preconditions": case.preconditions, "variables": case.variables, "tags": case.tags,
            "source_format": case.source_format, "structured_steps": case.structured_steps}


def _case_from_dict(data: dict[str, Any]) -> RawTestCase:
    def line(d: dict[str, Any]) -> RawLine:
        return RawLine(text=d["text"], line_no=d.get("line_no", 0), condition=d.get("condition"),
                       then_lines=[line(x) for x in d.get("then", [])], else_lines=[line(x) for x in d.get("else", [])])

    return RawTestCase(name=data["name"], base_url=data.get("base_url"), description=data.get("description", ""),
                       lines=[line(x) for x in data.get("lines", [])], expected_results=data.get("expected_results", []),
                       preconditions=data.get("preconditions", []), variables=data.get("variables", {}),
                       tags=data.get("tags", []), source_format=data.get("source_format", "text"),
                       structured_steps=data.get("structured_steps"))


@node("normalize_test_case")
async def normalize_test_case(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    if state.get("test_spec"):
        return {"phase": "normalized"}
    services = get_graph_services()
    case = _case_from_dict(state["raw_input"]["_case"])
    secret_names = await services.secrets.available_names(
        SecretScope(state["organization_id"], state["project_id"], state.get("environment_id")))
    spec = await services.analyzer().analyze(case, secret_names=secret_names)
    return {"phase": "normalized", "test_spec": spec.model_dump(mode="json")}


@node("analyze_requirements")
async def analyze_requirements(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    spec = TestSpec.model_validate(state["test_spec"])
    flat = spec.flat_steps()
    summary = {
        "steps": len(flat),
        "actions": sorted({s.action.value for s in flat}),
        "assertions": sum(1 for s in flat if s.action == ActionType.ASSERT),
        "sensitive_steps": [s.id for s in flat if s.sensitive],
        "high_risk_steps": [s.id for s in flat if s.risk_hint in ("destructive", "financial", "messaging")],
        "conditional_steps": [s.id for s in flat if s.action == ActionType.CONDITIONAL],
        "ambiguities": [a.model_dump() for a in spec.ambiguities],
        "missing_information": spec.missing_information,
    }
    await event_bus.publish(state["execution_id"], "PLAN_COMPILED", f"{len(flat)} steps understood", **summary)
    return {"phase": "analyzed", "accessibility_tree": {"requirements": summary}}


@node("compile_test_plan")
async def compile_test_plan(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    services = get_graph_services()
    spec = TestSpec.model_validate(state["test_spec"])
    if not spec.analyzer.get("compiled"):
        spec = services.compiler.compile(spec)
    compiled = [{"id": s.id, "index": s.index, "action": s.action.value, "text": s.original_text[:200],
                 "summary": s.summary()[:200], "confidence": s.confidence} for s in spec.flat_steps()]
    return {"phase": "compiled", "test_spec": spec.model_dump(mode="json"), "compiled_steps": compiled,
            "queue": [s.id for s in spec.steps], "current_step_index": 0}


@node("validate_test_plan")
async def validate_test_plan(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    services = get_graph_services()
    spec = TestSpec.model_validate(state["test_spec"])
    variables = set((state.get("raw_input") or {}).get("variables") or {}) | set(spec.variables)
    secrets = await services.secrets.available_names(
        SecretScope(state["organization_id"], state["project_id"], state.get("environment_id")))
    issues = services.validator.validate(spec, available_variables=variables, available_secrets=secrets)
    issue_dicts = [i.to_dict() for i in issues]
    blockers = blocking(issues)
    if blockers:
        reason = "; ".join(i.message for i in blockers)[:600]
        await event_bus.publish(state["execution_id"], "PLAN_INVALID", reason, issues=issue_dicts)
        return {"phase": "plan_blocked", "plan_issues": issue_dicts, "execution_status": "BLOCKED",
                "blocked_reason": f"test plan cannot be executed: {reason}"}
    return {"phase": "plan_valid", "plan_issues": issue_dicts}


@node("check_permissions")
async def check_permissions(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    from backend.agents.governance.policy_engine import EnvironmentPolicy, host_allowed
    from backend.core.config import get_settings

    spec = TestSpec.model_validate(state["test_spec"])
    options = ExecutionOptions.model_validate((state.get("raw_input") or {}).get("options") or {})
    policy = EnvironmentPolicy.from_settings(get_settings(), spec.base_url, environment_type=options.environment_type)
    extra = (state.get("raw_input") or {}).get("allowed_domains") or []
    policy.allowed_domains = sorted(set(policy.allowed_domains + list(extra)))
    for step in spec.flat_steps():
        if step.action == ActionType.NAVIGATE and step.target and step.target.url and "{{" not in step.target.url \
                and not step.target.url.startswith("/"):
            if not host_allowed(step.target.url, policy.allowed_domains):
                reason = f"step {step.index} navigates to {step.target.url}, outside the allowed domains"
                await event_bus.publish(state["execution_id"], "ACTION_BLOCKED", reason, step_id=step.id)
                return {"phase": "plan_blocked", "execution_status": "BLOCKED", "blocked_reason": reason}
    high = [s.id for s in spec.flat_steps() if s.risk_hint in ("destructive", "financial", "messaging")]
    return {"phase": "permitted", "pending_approval": None,
            "execution_budget": {"high_risk_steps": high, "allowed_domains": policy.allowed_domains}}


# =========================================================================== browser setup
@node("initialize_browser_session")
async def initialize_browser_session(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    services = get_graph_services()
    runtime = runtime_registry.get(state["execution_id"])
    if runtime is None:
        raw = state.get("raw_input") or {}
        spec = TestSpec.model_validate(state["test_spec"])
        options = ExecutionOptions.model_validate(raw.get("options") or {})
        request = ExecutionRequest(
            execution_id=state["execution_id"], organization_id=state["organization_id"],
            project_id=state["project_id"], application_id=state["application_id"],
            environment_id=state.get("environment_id"), test_case_id=state["test_case_id"],
            test_case_version=state.get("test_case_version", 1), spec=spec,
            variables={**spec.variables, **(raw.get("variables") or {})}, options=options,
            allowed_domains=list(raw.get("allowed_domains") or []))
        memory = None
        recorder: Any = NullRecorder()
        if services.sessionmaker is not None:
            memory = MemoryManager(services.sessionmaker, MemoryScope(
                state["organization_id"], state["project_id"], state["application_id"], state.get("environment_id")))
            recorder = DatabaseRecorder(services.sessionmaker, state["execution_id"])
            event_bus.persist_hook = _persist_event_router
        runtime = ExecutionRuntime(request, memory=memory, gateway=services.gateway, secrets=services.secrets,
                                   recorder=recorder)
        runtime_registry.register(runtime)
    session = await runtime.start_browser()
    return {"phase": "browser_ready", "browser_session_id": session.session_id, "execution_status": "RUNNING"}


async def _persist_event_router(record: dict[str, Any]) -> None:
    runtime = runtime_registry.get(record.get("execution_id", ""))
    if runtime is not None and runtime.recorder is not None:
        await runtime.recorder.event(record)


@node("observe_initial_page")
async def observe_initial_page(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    spec = runtime.request.spec
    first = spec.flat_steps()[0] if spec.flat_steps() else None
    if first is not None and first.action == ActionType.NAVIGATE:
        return {"phase": "initial_observed", "current_url": "about:blank"}
    obs = await runtime.observe(force=True)
    return {"phase": "initial_observed", "current_url": obs.url, "current_page_title": obs.title,
            "observation_id": obs.observation_id}


@node("retrieve_application_memory")
async def retrieve_application_memory(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    if runtime.memory is None:
        return {"phase": "memory_loaded", "application_memory": {"enabled": False}}
    memory = await runtime.memory.application_memory()
    trajectory = await runtime.memory.trajectory(state["test_case_id"], state.get("test_case_version", 1))
    summary = {"enabled": True, "known_pages": len(memory["known_pages"]),
               "active_locator_memories": memory["active_locator_memories"],
               "test_trajectory": {"success_count": trajectory["success_count"], "confidence": trajectory["confidence"]}
               if trajectory else None}
    await event_bus.publish(state["execution_id"], "MEMORY_RETRIEVED", "application memory loaded", **summary)
    return {"phase": "memory_loaded", "application_memory": summary}


@node("select_execution_strategy")
async def select_execution_strategy(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    from backend.runners.runner_registry import select_runner

    runtime = _runtime(state)
    choice = select_runner(runtime.request.spec, state.get("application_memory") or {},
                           forced=runtime.request.options.runner)
    runtime.runner = choice.name
    await event_bus.publish(state["execution_id"], "RUNNER_SELECTED", f"{choice.name}: {choice.reason}",
                            runner=choice.name, reason=choice.reason)
    return {"phase": "strategy_selected", "runner": choice.name}


# =========================================================================== step loop
@node("prepare_next_action")
async def prepare_next_action(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    queue = list(state.get("queue") or [])
    while queue:
        step_id = queue.pop(0)
        step = runtime.step(step_id)
        if step.action != ActionType.CONDITIONAL:
            await runtime.begin_step(step)
            return {"phase": "step_ready", "queue": queue, "current_step_id": step_id,
                    "current_step_index": step.index, "recovery_attempts": 0, "failure_context": None,
                    "verification_result": None, "grounded_target": None, "executed_action": None,
                    "step_results": _results_dict(runtime)}
        await runtime.begin_step(step)
        branch, detail = await _evaluate_condition(runtime, step)
        chosen = step.then_steps if branch else step.else_steps
        for child in reversed(chosen):
            queue.insert(0, child.id)
        for child in (step.else_steps if branch else step.then_steps):
            runtime.skip_remaining([c.id for c in child.iter_flat()])
            runtime.results[child.id].message = "branch not taken"
        result = runtime.results[step.id]
        result.branch = "then" if branch else "else"
        await runtime.finish_step(step, StepStatus.PASSED, f"condition {'true' if branch else 'false'}: {detail}")
    return {"phase": "steps_done", "queue": [], "current_step_id": None, "step_results": _results_dict(runtime)}


async def _evaluate_condition(runtime: ExecutionRuntime, step: Step) -> tuple[bool, str]:
    cond = step.condition
    if cond is None or cond.kind == "always":
        return True, "no condition"
    if cond.kind == "variable_equals":
        actual = runtime.request.variables.get(cond.variable or "", runtime.captured.get(cond.variable or ""))
        result = normalize_text(actual) == normalize_text(cond.expected)
        return result != cond.negate, f"{cond.variable}={actual!r}"
    obs = await runtime.observe(force=True)
    if cond.kind == "url_contains":
        result = (cond.expected or "").lower() in obs.url.lower()
    elif cond.kind == "text_visible":
        from backend.browser.js_probes import VISIBLE_TEXT
        from backend.agents.grounding.text_match import contains_text

        text = (await runtime.session.adapter.probe(VISIBLE_TEXT) or {}).get("text", "")  # type: ignore[union-attr]
        result = contains_text(text, cond.expected or "")
    else:
        probe = Step(id=f"{step.id}_cond", index=step.index, original_text=cond.description, action=ActionType.ASSERT,
                     target=cond.target)
        decision = runtime.grounder.ground(probe, obs, action_override="hover")
        identity = decision.candidates[0].features.get("identity", 0) if decision.candidates else 0
        result = decision.element_id is not None and identity >= 0.6
    return result != cond.negate, cond.description


@node("retrieve_step_memory")
async def retrieve_step_memory(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    candidates = await runtime.retrieve_memory(_step(state))
    return {"phase": "step_memory", "retrieved_memory": [c.to_dict() for c in candidates]}


@node("observe_current_page")
async def observe_current_page(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    if step.action == ActionType.NAVIGATE and runtime.observation is None:
        return {"phase": "observed"}
    obs = await runtime.observe()
    update: dict[str, Any] = {"phase": "observed", "current_url": obs.url, "current_page_title": obs.title,
                              "observation_id": obs.observation_id,
                              "page_fingerprint": obs.fingerprint.model_dump(include={"route_pattern", "structure_hash"})
                              if obs.fingerprint else {}}
    if obs.page_state.captcha_detected and step.action not in (ActionType.ASSERT, ActionType.WAIT):
        update["failure_context"] = {"category": FailureCategory.CAPTCHA_DETECTED.value,
                                     "root_cause": "human verification required",
                                     "evidence": obs.page_state.captcha_evidence, "blocker": True}
        update["phase"] = "captcha"
    return update


@node("resolve_target_element")
async def resolve_target_element(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    await runtime.ground(_step(state))
    decision = runtime.compact_decision()
    return {"phase": "resolved", "grounded_target": decision,
            "element_inventory": (decision or {}).get("candidates", [])}


@node("ground_target")
async def ground_target(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    work = runtime.ensure_work(_step(state))
    decision = work.decision
    if decision is None:
        return {"phase": "grounded"}
    if decision.decision in ("not_found",):
        return {"phase": "ground_failed"}
    return {"phase": "grounded", "selected_memory": {"memory_id": decision.memory_id} if decision.memory_id else None}


@node("validate_action")
async def validate_action(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    decisions = state.get("approval_decisions") or {}
    work = runtime.ensure_work(step)
    key = runtime.policy.approval_key(step, work.element)
    if decisions.get(key) == "approved":
        runtime.policy.approved.add(key)
    elif decisions.get(key) == "rejected":
        runtime.policy.rejected.add(key)
    result = await runtime.validate(step)
    planned = {"action": step.action.value, "target": work.decision.display if work.decision else None,
               "policy": result.to_dict()}
    if result.allowed:
        return {"phase": "validated", "planned_action": planned}
    if result.requires_approval:
        return {"phase": "approval_required", "planned_action": planned,
                "pending_approval": {"key": key, "step_id": step.id, "action": step.summary(),
                                     "target": planned["target"], "reasons": result.reasons,
                                     "risk": result.risk}}
    return {"phase": "denied", "planned_action": planned}


@node("human_approval")
async def human_approval(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    pending = state.get("pending_approval") or {}
    await event_bus.publish(state["execution_id"], "HUMAN_APPROVAL_REQUIRED", pending.get("action", ""),
                            step_id=pending.get("step_id"),
                            **{k: v for k, v in pending.items() if k not in ("action", "step_id")})
    runtime.budget.pause()
    decision = interrupt({"kind": "approval", **pending})
    runtime.budget.resume()
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    decisions = dict(state.get("approval_decisions") or {})
    decisions[pending.get("key", "")] = "approved" if approved else "rejected"
    await event_bus.publish(state["execution_id"], "EXECUTION_RESUMED",
                            f"approval {'granted' if approved else 'rejected'}", step_id=pending.get("step_id"))
    return {"phase": "approved" if approved else "rejected", "approval_decisions": decisions,
            "pending_approval": None}


@node("execute_action")
async def execute_action(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    outcome = await runtime.execute(_step(state))
    return {"phase": "executed", "executed_action": {"ok": outcome.ok, "tool": outcome.tool, "error": outcome.error,
                                                     "notes": outcome.notes[:4], "no_op": outcome.no_op}}


@node("wait_for_state_change")
async def wait_for_state_change(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    await runtime.settle(_step(state))
    return {"phase": "settled"}


@node("verify_action_result")
async def verify_action_result(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    verification = await runtime.verify(step)
    work = runtime.ensure_work(step)
    if verification.status != "failed" and work.healed and work.decision is not None and step.target is not None:
        identity = work.decision.candidates[0].features.get("identity", 0) if work.decision.candidates else 0
        if work.decision.method not in ("color_segmentation", "vision_model") and identity < 0.55:
            verification.status = "failed"
            verification.message = (f"healing rejected: substitute {work.decision.display} lacks evidence of being "
                                     f"the intended element (identity {identity:.2f})")
            verification.failure_category = FailureCategory.ELEMENT_NOT_FOUND
    obs = runtime.observation
    return {"phase": "verified" if verification.status != "failed" else "verify_failed",
            "verification_result": verification.to_dict(),
            "current_url": obs.url if obs else state.get("current_url", ""),
            "current_page_title": obs.title if obs else ""}


@node("learn_pattern")
async def learn_pattern(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    runtime.learn(step)
    verification = state.get("verification_result") or {}
    work = runtime.ensure_work(step)
    status = StepStatus.PASSED
    message = verification.get("message") or "ok"
    if verification.get("status") == "unverified":
        if step.action == ActionType.ASSERT:
            status = StepStatus.INCONCLUSIVE
            message = work.assertion_results[-1].message if work.assertion_results else message
        else:
            message = f"completed; {message}"
    if work.healed or (runtime.results[step.id].recovery and status == StepStatus.PASSED):
        message = f"recovered after {len(runtime.results[step.id].recovery)} recovery step(s); {message}"
    await runtime.finish_step(step, status, message)
    history = list(state.get("execution_history") or [])
    history.append({"step_id": step.id, "status": status.value, "url": runtime.observation.url
                    if runtime.observation else None})
    return {"phase": "learned", "execution_history": history[-200:], "step_results": _results_dict(runtime)}


@node("classify_failure")
async def classify_failure_node(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    analysis = runtime.classify(step)
    existing = state.get("failure_context") or {}
    if existing.get("category") == FailureCategory.CAPTCHA_DETECTED.value:
        analysis.category = FailureCategory.CAPTCHA_DETECTED
        analysis.blocker, analysis.healable = True, False
    if existing.get("node") and existing.get("category") == FailureCategory.UNKNOWN_FAILURE.value:
        analysis.evidence.append(existing.get("root_cause", ""))
    return {"phase": "classified", "failure_context": analysis.to_dict()}


@node("recovery_planner")
async def recovery_planner(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    analysis = runtime.ensure_work(step).analysis
    plan = runtime.plan_recovery(step, analysis) if analysis is not None else None
    return {"phase": "recovery_planned" if plan else "recovery_exhausted", "recovery_plan": plan}


@node("recovery_policy_check")
async def recovery_policy_check(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    plan = state.get("recovery_plan") or {}
    # Recovery may never repeat a non-idempotent action blindly; the idempotency check runs first.
    work = runtime.ensure_work(step)
    if plan.get("strategy") in ("reground_without_memory", "reground_excluding_previous", "reobserve_and_reground") \
            and step.risk_hint in ("destructive", "financial", "messaging") and work.outcome is not None \
            and work.outcome.ok:
        return {"phase": "recovery_denied", "failure_context": {**(state.get("failure_context") or {}),
                                                                "root_cause": "refusing to re-target a high-risk "
                                                                              "action that already executed"}}
    return {"phase": "recovery_allowed"}


@node("recovery_executor")
async def recovery_executor(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    ok, detail = await runtime.execute_recovery(step, state.get("recovery_plan") or {})
    work = runtime.ensure_work(step)
    already_done = ok and work.outcome is not None and work.outcome.ok and "already" in detail
    return {"phase": "recovered_done" if already_done else ("recovered" if ok else "recovery_failed"),
            "recovery_attempts": (state.get("recovery_attempts") or 0) + 1,
            "recovery_total": (state.get("recovery_total") or 0) + 1}


@node("await_human")
async def await_human(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    context = state.get("failure_context") or {}
    await event_bus.publish(state["execution_id"], "HUMAN_APPROVAL_REQUIRED",
                            "human verification (CAPTCHA) detected: a person must complete it in the live browser",
                            step_id=step.id, kind="captcha", evidence=context.get("evidence"))
    await event_bus.publish(state["execution_id"], "EXECUTION_PAUSED", "awaiting human verification",
                            step_id=step.id)
    runtime.budget.pause()
    shot = await runtime.screenshot(step.id, "challenge")
    decision = interrupt({"kind": "captcha", "step_id": step.id, "url": runtime.observation.url
                          if runtime.observation else None, "screenshot": shot})
    runtime.budget.resume()
    obs = await runtime.observe(force=True)
    resolved = not obs.page_state.captcha_detected and not (isinstance(decision, dict) and decision.get("abort"))
    await event_bus.publish(state["execution_id"], "EXECUTION_RESUMED",
                            "challenge cleared" if resolved else "challenge still present", step_id=step.id)
    if resolved:
        runtime.dirty = True
        runtime.results[step.id].message = "paused for human verification, resumed after it was completed"
        return {"phase": "human_resolved", "failure_context": None}
    return {"phase": "human_unresolved", "blocked_reason": "human verification was not completed"}


@node("fail_step")
async def fail_step(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    step = _step(state)
    context = state.get("failure_context") or {}
    work = runtime.ensure_work(step)
    category = context.get("category", FailureCategory.UNKNOWN_FAILURE.value)
    blocker = context.get("blocker") or category in (FailureCategory.CAPTCHA_DETECTED.value,
                                                     FailureCategory.POLICY_BLOCKED.value,
                                                     FailureCategory.LOW_CONFIDENCE.value,
                                                     FailureCategory.AMBIGUOUS_TARGET.value,
                                                     FailureCategory.UNEXPECTED_NAVIGATION.value,
                                                     FailureCategory.BROWSER_DISCONNECTED.value)
    if state.get("phase") == "rejected":
        blocker = True
        context = {**context, "root_cause": "a reviewer rejected the action"}
    if category == FailureCategory.ELEMENT_NOT_FOUND.value and work.decision is not None and \
            work.decision.decision == "ask_human":
        blocker = True
    message = context.get("root_cause", "step failed")
    evidence = [e for e in (context.get("evidence") or []) if e]
    if evidence:
        message += f" — {evidence[0]}"[:400]
    if work.policy is not None and not work.policy.allowed:
        message = "; ".join(work.policy.reasons)
    status = StepStatus.BLOCKED if blocker else StepStatus.FAILED
    if work.screenshot_after is None and runtime.session is not None:
        work.screenshot_after = await runtime.screenshot(step.id, "failure")
        if runtime.results[step.id].actions:
            runtime.results[step.id].actions[-1].screenshot_after = work.screenshot_after
    await runtime.finish_step(step, status, message)
    if runtime.memory is not None and work.before is not None and work.before.fingerprint is not None and \
            not context.get("product_failure"):
        await runtime.memory.record_failure(execution_id=runtime.execution_id, step=step,
                                            route_pattern=work.before.fingerprint.route_pattern, category=category,
                                            root_cause=context.get("root_cause", ""), evidence=evidence,
                                            failed_strategies=work.failed_strategies, attempts=work.recovery_cycles)
    remaining = [sid for sid in (state.get("queue") or [])]
    flat_remaining: list[str] = []
    for sid in remaining:
        flat_remaining.extend(s.id for s in runtime.step(sid).iter_flat())
    runtime.skip_remaining(flat_remaining)
    return {"phase": "step_failed", "queue": [], "step_results": _results_dict(runtime),
            "blocked_reason": message if status == StepStatus.BLOCKED else state.get("blocked_reason")}


@node("next_test_action")
async def next_test_action(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = _runtime(state)
    runtime.budget.check_alive()
    return {"phase": "next", "execution_budget": {**(state.get("execution_budget") or {}),
                                                  **runtime.budget.snapshot()}}


# =========================================================================== finalization
@node("final_assertions")
async def final_assertions(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = runtime_registry.get(state["execution_id"])
    if runtime is None or runtime.session is None:
        return {"phase": "final_checked"}
    diagnostics: dict[str, Any] = {}
    try:
        console = await runtime.session.adapter.console_errors()
        diagnostics["console_errors"] = (console.result_text or "")[:2000]
    except Exception:  # pragma: no cover
        pass
    return {"phase": "final_checked", "visual_context": {"diagnostics": diagnostics,
                                                         "security_warnings": runtime.security_warnings}}


@node("evaluate_test_result")
async def evaluate_test_result(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = runtime_registry.get(state["execution_id"])
    spec = TestSpec.model_validate(state["test_spec"]) if state.get("test_spec") else None
    if runtime is not None:
        for result in runtime.results.values():
            if result.status in (StepStatus.RUNNING, StepStatus.PENDING):
                result.status = StepStatus.BLOCKED
                result.message = state.get("blocked_reason") or "execution stopped before this step completed"
    steps: list[StepResult] = []
    if runtime is not None:
        for s in (spec.flat_steps() if spec else []):
            if s.id in runtime.results:
                steps.append(runtime.results[s.id])
    plan_blocked = state.get("blocked_reason") if state.get("phase") in ("plan_blocked",) or (
        state.get("execution_status") == "BLOCKED" and not steps) else None
    cancelled = state.get("execution_status") == "CANCELLED"
    outcome, reason = evaluate_outcome(steps, has_assertions=bool(spec and spec.has_verifiable_outcome()),
                                       plan_blocked_reason=plan_blocked, cancelled=cancelled)
    if state.get("phase") == "abort" and outcome == TestOutcome.PASSED:
        outcome, reason = TestOutcome.BLOCKED, state.get("blocked_reason") or "execution stopped"
    original_ok = all(not r.recovery for r in steps)
    final = {"outcome": outcome.value, "reason": reason, "original_strategy_succeeded": original_ok,
             "recovered_steps": sum(1 for r in steps if r.recovered)}
    update: dict[str, Any] = {"phase": "evaluated", "final_result": final,
                              "execution_status": "COMPLETED" if state.get("execution_status") not in ("CANCELLED",)
                              else "CANCELLED"}
    if runtime is not None:
        update["step_results"] = _results_dict(runtime)
    return update


@node("persist_test_memory")
async def persist_test_memory(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    runtime = runtime_registry.get(state["execution_id"])
    if runtime is None or runtime.memory is None:
        return {"phase": "memory_persisted", "learning_stats": {"enabled": False}}
    outcome = (state.get("final_result") or {}).get("outcome")
    passed = outcome == TestOutcome.PASSED.value
    stats = await runtime.memory.commit_learning(runtime.learning, execution_id=runtime.execution_id,
                                                 test_case_id=state["test_case_id"], test_passed=passed)
    trajectory = [{"step_id": r.step_id, "action": r.action, "status": r.status.value, "url": r.page_url}
                  for r in sorted(runtime.results.values(), key=lambda r: r.index)]
    await runtime.memory.record_trajectory(test_case_id=state["test_case_id"], version=state.get("test_case_version", 1),
                                           trajectory=trajectory, success=passed, execution_id=runtime.execution_id)
    if passed and runtime.request.spec.base_url:
        login_steps = [r for r in runtime.results.values() if r.action == "fill" and "password" in r.summary.lower()]
        if login_steps:
            await runtime.memory.set_fact("authentication", {"login_route": login_steps[0].page_url,
                                                             "verified_by": runtime.execution_id})
    await event_bus.publish(state["execution_id"], "MEMORY_UPDATED", f"learning committed: {stats}", **stats)
    return {"phase": "memory_persisted", "learning_stats": stats}


@node("generate_report")
async def generate_report(state: TestExecutionState, config: Any = None) -> dict[str, Any]:
    from backend.agents.reporting.report_generator import ReportGenerator

    runtime = runtime_registry.get(state["execution_id"])
    if runtime is not None:
        await runtime.close()
    generator = ReportGenerator()
    report = await generator.generate(state, runtime)
    await event_bus.publish(state["execution_id"], "REPORT_READY", report.get("html_uri", ""), **report)
    await event_bus.publish(state["execution_id"], "EXECUTION_COMPLETED",
                            (state.get("final_result") or {}).get("outcome", "UNKNOWN"),
                            outcome=(state.get("final_result") or {}).get("outcome"))
    return {"phase": "reported", "report": report}
