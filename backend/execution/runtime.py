"""Execution runtime: the live, per-execution engine driven by the LangGraph nodes.

The graph state (checkpointed in Postgres) stays small and serializable; this
runtime owns the non-serializable parts — the isolated browser session, cached
observations, resolved secrets and buffered learning — and implements each
pipeline phase: observe -> retrieve memory -> ground -> validate (policy) ->
execute -> settle -> verify -> learn | classify -> recover.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from backend.agents.governance.policy_engine import (
    ActionRequest,
    EnvironmentPolicy,
    PolicyDecision,
    PolicyEngine,
)
from backend.agents.governance.redaction import redact_text
from backend.agents.governance.secret_manager import (
    SecretManager,
    SecretScope,
    get_secret_manager,
)
from backend.agents.grounding.element_grounder import (
    ElementGrounder,
    GroundingContext,
    element_key,
)
from backend.agents.grounding.locator_strategy import (
    LocatorStrategy,
    validated_selectors,
)
from backend.agents.observation.page_observer import PageObserver
from backend.agents.planner.lexicon import normalize_text
from backend.agents.planner.variable_resolver import ResolvedValue, VariableResolver
from backend.agents.recovery.failure_classifier import FailureAnalysis, classify_failure
from backend.agents.reporting.screenshot_annotator import annotate_target
from backend.agents.verification.action_verifier import (
    ActionVerifier,
    VerificationResult,
    page_changed,
)
from backend.agents.verification.assertion_engine import (
    AssertionContext,
    AssertionEngine,
)
from backend.agents.visual.visual_grounding import VisualGroundingAgent
from backend.browser.js_probes import scroll_into_view_probe
from backend.browser.mcp_client import BrowserLaunchOptions
from backend.browser.session_manager import (
    BrowserSession,
    SessionManager,
    get_session_manager,
)
from backend.core.config import Settings, get_settings
from backend.core.exceptions import (
    BrowserDisconnected,
    ModelError,
    ModelUnavailable,
    PlatformError,
)
from backend.core.ids import new_id
from backend.core.logging import get_logger
from backend.dsl.models import ActionType, Step, TestSpec
from backend.dsl.observation import InventoryElement, Observation
from backend.dsl.results import (
    ActionRecord,
    FailureCategory,
    GroundingCandidate,
    GroundingDecision,
    RecoveryAttempt,
    StepResult,
    StepStatus,
)
from backend.execution.action_executor import (
    ActionOutcome,
    InteractionExecutor,
    target_of,
)
from backend.execution.events import EventBus, event_bus
from backend.harness.budget_manager import BudgetTracker, bind_tracker, new_tracker
from backend.harness.model_gateway import ModelGateway, ModelTier, get_model_gateway
from backend.memory.memory_manager import LearningRecord, MemoryCandidate, MemoryManager
from backend.services.artifact_service import ArtifactService, get_artifact_service

log = get_logger(__name__)

NON_IDEMPOTENT_RISKS = {"submit", "destructive", "financial", "messaging"}


class ExecutionOptions(BaseModel):
    headless: bool | None = None
    viewport: tuple[int, int] | None = None
    browser: str | None = None
    device: str | None = None
    locale: str | None = None
    timezone: str | None = None
    record_video: bool | None = None
    screenshots: str | None = None  # every_action | on_failure | off
    runner: str | None = None  # force a runner
    debug: bool = False
    use_memory: bool = True
    storage_state_path: str | None = None
    environment_type: str = "test"
    allow_destructive: bool | None = None
    allow_financial: bool | None = None
    allow_messaging: bool | None = None
    auto_approve: bool | None = None
    approval_timeout_s: float | None = None


@dataclass
class ExecutionRequest:
    execution_id: str
    organization_id: str
    project_id: str
    application_id: str
    test_case_id: str
    test_case_version: int
    spec: TestSpec
    variables: dict[str, str] = field(default_factory=dict)
    environment_id: str | None = None
    options: ExecutionOptions = field(default_factory=ExecutionOptions)
    allowed_domains: list[str] = field(default_factory=list)


class LLMChoice(BaseModel):
    element_number: int | None = Field(description="Number of the candidate that matches, or null if none does")
    confidence: float = Field(ge=0, le=1)
    reason: str


@dataclass
class StepWork:
    """Per-step working memory (runtime-only)."""

    step: Step
    started: float = field(default_factory=time.time)
    memory: list[MemoryCandidate] = field(default_factory=list)
    decision: GroundingDecision | None = None
    element: InventoryElement | None = None
    destination: InventoryElement | None = None
    value: ResolvedValue | None = None
    outcome: ActionOutcome | None = None
    before: Observation | None = None
    after: Observation | None = None
    verification: VerificationResult | None = None
    policy: PolicyDecision | None = None
    excluded: set[str] = field(default_factory=set)
    drop_memory: bool = False
    force_visual: bool = False
    force_llm: bool = False
    attempts: int = 0
    recovery_cycles: int = 0
    healing_attempts: int = 0
    tried_strategies: list[str] = field(default_factory=list)
    failed_strategies: list[dict[str, Any]] = field(default_factory=list)
    healed: bool = False
    used_memory_id: str | None = None
    analysis: FailureAnalysis | None = None
    screenshot_before: str | None = None
    screenshot_after: str | None = None
    url_before: str | None = None
    assertion_results: list[Any] = field(default_factory=list)
    llm_calls: int = 0


class ExecutionRuntime:
    def __init__(self, request: ExecutionRequest, *, settings: Settings | None = None,
                 session_manager: SessionManager | None = None, artifacts: ArtifactService | None = None,
                 memory: MemoryManager | None = None, gateway: ModelGateway | None = None,
                 secrets: SecretManager | None = None, events: EventBus | None = None,
                 recorder: Any = None) -> None:
        self.request = request
        self.settings = settings or get_settings()
        self.sessions = session_manager or get_session_manager()
        self.artifacts = artifacts or get_artifact_service()
        self.memory = memory if request.options.use_memory else None
        self.gateway = gateway or get_model_gateway()
        self.secrets = secrets or get_secret_manager()
        self.events = events or event_bus
        self.recorder = recorder
        self.execution_id = request.execution_id
        self.budget: BudgetTracker = new_tracker(self.execution_id, self.settings.budget)
        self.session: BrowserSession | None = None
        self.observer: PageObserver | None = None
        self.executor: InteractionExecutor | None = None
        self.verifier: ActionVerifier | None = None
        self.assertions: AssertionEngine | None = None
        self.grounder = ElementGrounder(self.settings.thresholds)
        self.visual = VisualGroundingAgent(self.gateway if self.gateway.available else None)
        opts = request.options
        self.env_policy = EnvironmentPolicy.from_settings(
            self.settings, request.spec.base_url, environment_type=opts.environment_type,
            allow_destructive=opts.allow_destructive, allow_financial=opts.allow_financial,
            allow_messaging=opts.allow_messaging, auto_approve_high_risk=opts.auto_approve)
        if request.allowed_domains:
            self.env_policy.allowed_domains = sorted(set(self.env_policy.allowed_domains + request.allowed_domains))
        self.policy = PolicyEngine(self.env_policy, self.settings)
        self.captured: dict[str, str] = {}
        self.resolver = VariableResolver(variables=request.variables, secret_manager=self.secrets,
                                         scope=SecretScope(request.organization_id, request.project_id,
                                                           request.environment_id),
                                         run_token=self.execution_id, captured=self.captured)
        self.observation: Observation | None = None
        self.dirty = True
        self.work: StepWork | None = None
        self.previous_element: InventoryElement | None = None
        self.previous_form_ref: str | None = None
        self.learning: list[LearningRecord] = []
        self.downloads: list[dict[str, str]] = []
        self.results: dict[str, StepResult] = {}
        self.steps_by_id: dict[str, Step] = {s.id: s for s in request.spec.flat_steps()}
        self.memory_lookups = 0
        self.memory_hits = 0
        self.memory_fastpath = 0  # steps replayed straight from validated memory (no scoring, no model)
        self.recoveries = 0
        self.retries = 0
        self.security_warnings: list[str] = []
        self.started = time.time()
        self.cancelled = False
        self.runner = "semantic"
        self.screenshot_policy = opts.screenshots or self.settings.screenshot_policy
        self.video_artifact: str | None = None
        self.closed = False
        self.announced: set[str] = set()

    # ------------------------------------------------------------------ lifecycle
    def bind(self) -> None:
        bind_tracker(self.budget)

    async def emit(self, event: str, message: str = "", *, step_id: str | None = None, **payload: Any) -> None:
        await self.events.publish(self.execution_id, event, message, step_id=step_id, **payload)

    async def start_browser(self) -> BrowserSession:
        self.bind()
        if self.session is not None and self.session.alive:
            return self.session
        opts = self.request.options
        launch = BrowserLaunchOptions(viewport=opts.viewport, locale=opts.locale, timezone=opts.timezone,
                                      device=opts.device, storage_state_path=opts.storage_state_path,
                                      headless=opts.headless, browser_name=opts.browser, record_video=opts.record_video)
        audit = self.recorder.audit_tool_call if self.recorder is not None else None
        self.session = await self.sessions.create(self.execution_id, options=launch, audit_hook=audit)
        viewport = opts.viewport or self.settings.viewport_size()
        self.observer = PageObserver(self.session.adapter, viewport)
        self.executor = InteractionExecutor(self.session, self.observer, self.grounder)
        self.verifier = ActionVerifier(self.session.adapter)
        self.assertions = AssertionEngine(self.session.adapter, self.observer, self.grounder, self.resolver,
                                          self.gateway if self.gateway.available else None)
        if self.recorder is not None:
            await self.recorder.browser_session(self.session)
        await self.emit("BROWSER_INITIALIZED", f"isolated browser context {self.session.session_id}",
                        session_id=self.session.session_id, server_version=self.session.connection.server_version,
                        tools=len(self.session.connection.tools), video=self.session.recording)
        return self.session

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.session is not None:
            session = self.session
            await session.stop_recording()
            await self.sessions.close(session.session_id)
            video = await session.finalized_video()
            if video is not None and video.exists():
                ref = await self.artifacts.put_file(self.execution_id, "video", video, "execution-video.webm")
                self.video_artifact = ref.uri
                if self.recorder is not None:
                    await self.recorder.artifact(ref, None)

    # ------------------------------------------------------------------ observation
    async def observe(self, *, force: bool = False) -> Observation:
        assert self.observer is not None
        if self.observation is not None and not self.dirty and not force:
            return self.observation
        self.budget.check_alive()
        obs = await self.observer.observe()
        self.observation = obs
        self.dirty = False
        if obs.page_state.injection_findings:
            for finding in obs.page_state.injection_findings:
                if finding not in self.security_warnings:
                    self.security_warnings.append(finding)
                    await self.emit("SECURITY_WARNING", "page contains instruction-like text; treated as untrusted "
                                    "content and ignored", finding=finding[:200], url=obs.url)
        if self.memory is not None and obs.fingerprint is not None:
            try:
                await self.memory.record_page(obs)
            except Exception as exc:  # pragma: no cover
                log.warning("page memory write failed", error=str(exc)[:200])
        if (self.runner == "debug" or self.request.options.debug) and obs.snapshot_text:
            ref = await self.artifacts.put_bytes(self.execution_id, "snapshots", f"{obs.observation_id}.yml",
                                                 redact_text(obs.snapshot_text).encode("utf-8"), "text/yaml")
            obs.snapshot_artifact = ref.uri
            if self.recorder is not None:
                await self.recorder.artifact(ref, self.work.step.id if self.work else None)
        await self.emit("PAGE_OBSERVED", f"{obs.title} ({len(obs.elements)} elements)", **obs.summary())
        return obs

    async def screenshot(self, step_id: str | None, label: str, *, decision: GroundingDecision | None = None,
                         full_page: bool = False) -> str | None:
        if self.session is None or self.screenshot_policy == "off":
            return None
        response = await self.session.adapter.screenshot(full_page=full_page)
        if not response.ok or not response.images:
            return None
        png = response.images[0].data
        if decision is not None and (decision.bbox or decision.coordinates):
            try:
                png = annotate_target(png, box=decision.bbox, point=decision.coordinates,
                                      label=decision.display or None)
            except Exception:  # pragma: no cover - annotation is best effort
                pass
        name = f"{step_id or 'page'}-{label}-{int(time.time() * 1000)}.png"
        ref = await self.artifacts.put_bytes(self.execution_id, "screenshots", name, png, "image/png")
        if self.recorder is not None:
            await self.recorder.artifact(ref, step_id)
        await self.emit("SCREENSHOT_CAPTURED", label, step_id=step_id, uri=ref.uri, api_path=ref.api_path)
        return ref.uri

    # ------------------------------------------------------------------ steps
    def step(self, step_id: str) -> Step:
        return self.steps_by_id[step_id]

    async def begin_step(self, step: Step) -> StepResult:
        self.work = StepWork(step=step)
        result = self.results.get(step.id) or StepResult(step_id=step.id, index=step.index,
                                                          original_text=step.original_text, action=step.action.value,
                                                          summary=step.summary())
        result.status = StepStatus.RUNNING
        result.started_at = result.started_at or time.time()
        result.runner = self.runner
        self.results[step.id] = result
        await self.emit("STEP_STARTED", step.original_text, step_id=step.id, action=step.action.value,
                        index=step.index)
        return result

    def ensure_work(self, step: Step) -> StepWork:
        if self.work is None or self.work.step.id != step.id:
            self.work = StepWork(step=step)
            if step.id not in self.results:
                self.results[step.id] = StepResult(step_id=step.id, index=step.index, original_text=step.original_text,
                                                   action=step.action.value, summary=step.summary(),
                                                   status=StepStatus.RUNNING, started_at=time.time())
        return self.work

    async def retrieve_memory(self, step: Step) -> list[MemoryCandidate]:
        work = self.ensure_work(step)
        if self.memory is None or step.target is None or step.action in (ActionType.NAVIGATE, ActionType.ASSERT,
                                                                         ActionType.WAIT):
            work.memory = []
            return []
        obs = await self.observe()
        self.memory_lookups += 1
        try:
            candidates = await self.memory.retrieve_for_step(step, obs, self.request.test_case_id)
        except Exception as exc:  # pragma: no cover
            log.warning("memory retrieval failed", error=str(exc)[:200])
            candidates = []
        min_conf = self.settings.memory_min_confidence_for_replay
        # validated knowledge only: a strategy needs a fully passed run or repeated verified successes
        work.memory = [c for c in candidates if c.confidence >= min_conf * (0.8 if c.match_level == "semantic" else 0.6)
                       and c.successes >= self.settings.memory_min_successes_for_replay
                       and (c.validated_runs >= 1 or c.successes >= 2)]
        if work.memory:
            await self.emit("MEMORY_RETRIEVED", f"{len(work.memory)} remembered strategies", step_id=step.id,
                            memories=[c.to_dict() for c in work.memory])
        return work.memory

    # ------------------------------------------------------------------ grounding
    async def ground(self, step: Step) -> GroundingDecision | None:
        work = self.ensure_work(step)
        if step.action not in GROUNDED_ACTIONS or step.target is None:
            work.decision = None
            return None
        obs = await self.observe()
        work.before = obs
        if obs.modal_state is not None and step.action not in (ActionType.ACCEPT_DIALOG, ActionType.DISMISS_DIALOG):
            work.decision = GroundingDecision(decision="not_found", reason=f"native {obs.modal_state.kind} dialog open: "
                                                                          f"{obs.modal_state.message[:80]}")
            return work.decision
        ctx = GroundingContext(previous_element=self.previous_element, previous_form_ref=self.previous_form_ref,
                               excluded_keys=set(work.excluded), frame_hint=step.target.frame_hint)
        if not work.drop_memory:
            ctx.memory_strategies = [(c.strategy, c.confidence, c.memory_id) for c in work.memory]
        fast = self._memory_fastpath(step, obs, work, ctx)
        if fast is not None:
            decision = fast
            self.memory_fastpath += 1
            await self.emit("MEMORY_FASTPATH", f"replayed validated strategy for {fast.display} directly",
                            step_id=step.id, memory_id=fast.memory_id, confidence=fast.confidence)
        else:
            decision = self.grounder.ground(step, obs, ctx)
            if decision.decision in ("refine", "ask_human", "not_found") and not work.force_visual:
                decision = await self._refine(step, obs, ctx, decision)
            if self._needs_visual(step, decision) or work.force_visual:
                decision = await self._visual_ground(step, obs, decision)
        work.decision = decision
        work.element = obs.element(decision.element_id) if decision.element_id else None
        if step.action == ActionType.DRAG and step.destination is not None:
            dst = self.grounder.ground(step, obs, GroundingContext(), target=step.destination, action_override="drop")
            work.destination = obs.element(dst.element_id) if dst.element_id else None
            decision.evidence.append(f"destination: {dst.display} ({dst.confidence:.2f})")
            decision.confidence = round(min(decision.confidence, max(dst.confidence, 0.66)), 4)
        if decision.memory_id:
            work.used_memory_id = decision.memory_id
            self.memory_hits += 1
            await self.emit("MEMORY_USED", f"re-grounded remembered strategy for {decision.display}",
                            step_id=step.id, memory_id=decision.memory_id)
        elif work.memory and not work.drop_memory and decision.element_id is not None:
            await self._memory_miss(step, work, decision)
        if work.element is not None:
            strategy = LocatorStrategy.from_element(work.element, obs)
            decision.locator_strategy = {**strategy.model_dump(exclude_none=True),
                                         "selectors": strategy.selectors()[:6],
                                         "validation": validated_selectors(strategy, obs)}
        await self.emit("TARGET_GROUNDED", f"{decision.display or 'no target'} "
                        f"(confidence {decision.confidence:.2f}, {decision.decision})", step_id=step.id,
                        decision=decision.decision, confidence=decision.confidence, method=decision.method,
                        target=decision.display, evidence=decision.evidence[:5],
                        candidates=[{"display": c.display, "score": c.score} for c in decision.candidates[:4]])
        return decision

    def _memory_fastpath(self, step: Step, obs: Observation, work: StepWork,
                         ctx: GroundingContext) -> GroundingDecision | None:
        """Replay a validated strategy for the same application page and step without re-deciding.

        The strategy must re-ground on the *live* page uniquely and almost perfectly, the element must be
        usable for this action and must still mean what the step asks (name/purpose/test id). Anything less
        returns None and the full grounding - scoring, refinement and, when configured, the model - takes
        over; a remembered element that then fails verification is penalised and dropped by recovery.
        """

        from backend.agents.grounding.element_grounder import (
            ACTION_FOR_GROUNDING,
            _hard_excluded,
            name_score,
        )
        from backend.agents.grounding.locator_strategy import resolve_strategy
        from backend.agents.planner.lexicon import purpose_compatible
        from backend.dsl.results import GroundingCandidate

        if work.drop_memory or not work.memory or work.excluded or work.force_visual or step.target is None:
            return None
        if step.target.ordinal is not None or step.target.position_hint or step.action == ActionType.DRAG:
            return None  # positional instructions are decided on the live layout every time
        top = work.memory[0]
        if top.match_level not in ("exact_test_case", "exact_route") or top.confidence < 0.6:
            return None
        resolution = resolve_strategy(top.strategy, obs)
        el = resolution.element
        if el is None or resolution.ambiguous or resolution.quality < 0.85:
            return None
        action = ACTION_FOR_GROUNDING.get(step.action, "click")
        if _hard_excluded(step, step.target, el, action, ctx) or not el.enabled or el.obstructed:
            return None
        meaning, _ = name_score(step.target, el)
        same_purpose = purpose_compatible(step.target.purpose, el.semantic_purpose) > 0
        if meaning < 0.5 and not same_purpose and "test_id" not in resolution.matched_on:
            return None  # the remembered element no longer means what this step asks
        confidence = round(min(0.99, 0.9 + 0.09 * resolution.quality * min(1.0, top.confidence / 0.9)), 4)
        return GroundingDecision(
            element_id=el.element_id, ref=el.ref, display=el.display(), role=el.role, name=el.name or el.label or "",
            confidence=confidence, decision=self.grounder.decide(confidence), method="memory_fastpath",
            candidates=[GroundingCandidate(element_id=el.element_id, ref=el.ref, display=el.display(),
                                           score=confidence, features={"memory": resolution.quality},
                                           evidence=[f"matched on {', '.join(resolution.matched_on[:5])}"],
                                           bbox=el.bbox.model_dump() if el.bbox else None)],
            evidence=[f"validated memory replayed directly (re-grounded {resolution.quality:.2f} on "
                      f"{', '.join(resolution.matched_on[:5])}; {top.successes} verified uses)",
                      "no candidate scoring or model call needed"],
            bbox=el.bbox.model_dump() if el.bbox else None, memory_id=top.memory_id,
            reason="memory fast path")

    async def _memory_miss(self, step: Step, work: StepWork, decision: GroundingDecision) -> None:
        """Validated memory exists but no longer matches the live page: the UI changed.

        The step is re-grounded semantically (already done by the grounder); this records
        the detection as a recovery, penalizes the stale strategy and marks the new
        strategy to supersede it once verified.
        """

        stale = work.memory[0]
        if stale.match_level == "semantic":
            # memory of a *similar* step that does not fit this one: not applicable, not a UI change
            work.drop_memory = True
            decision.evidence.append(f"memory of a similar step ({stale.strategy.describe()}) does not apply here")
            return
        work.healed = True
        work.drop_memory = True
        work.failed_strategies.append(stale.strategy.model_dump(exclude_none=True))
        attempt = RecoveryAttempt(
            attempt=len(self.results[step.id].recovery) + 1,
            failure_category=FailureCategory.PAGE_STRUCTURE_CHANGED.value, strategy="reground_without_memory",
            rationale=f"remembered strategy ({stale.strategy.describe()}) no longer matches the live page; "
                      f"re-grounded semantically", outcome="recovered" if decision.decision != "ask_human" else "failed",
            details={"detail": f"new target: {decision.display} (confidence {decision.confidence:.2f})",
                     "stale_memory_id": stale.memory_id})
        self.results[step.id].recovery.append(attempt)
        self.recoveries += 1
        if self.recorder is not None:
            await self.recorder.recovery(step.id, attempt)
        if self.memory is not None:
            await self.memory.record_strategy_failure(stale.memory_id, execution_id=self.execution_id,
                                                      reason="did not re-ground on the current page")
        decision.evidence.append(f"UI change detected: remembered {stale.strategy.describe()} not found")
        await self.emit("RECOVERY_COMPLETED", f"UI change detected; re-grounded to {decision.display}",
                        step_id=step.id, strategy="reground_without_memory", stale_memory=stale.memory_id)

    async def _refine(self, step: Step, obs: Observation, ctx: GroundingContext,
                      decision: GroundingDecision) -> GroundingDecision:
        """Re-observe (dynamic content), then ask the LLM to disambiguate among real candidates."""

        if obs.page_state.loading or decision.decision == "not_found":
            await asyncio.sleep(0.8)
            obs = await self.observe(force=True)
            self.work.before = obs  # type: ignore[union-attr]
            decision = self.grounder.ground(step, obs, ctx)
        if decision.decision in ("refine", "ask_human") and decision.candidates and self.gateway.available:
            decision = await self._llm_disambiguate(step, obs, decision)
        return decision

    async def _llm_disambiguate(self, step: Step, obs: Observation, decision: GroundingDecision) -> GroundingDecision:
        candidates = decision.candidates[:8]
        lines = []
        for i, cand in enumerate(candidates, start=1):
            el = obs.element(cand.element_id)
            if el is None:
                continue
            ctx = el.context
            lines.append(f"{i}. {el.role} name={el.name!r} label={el.label!r} placeholder={el.placeholder!r} "
                         f"form={ctx.form_name!r} dialog={ctx.dialog_name!r} section={ctx.section_heading!r} "
                         f"landmark={ctx.landmark} enabled={el.enabled}")
        try:
            result = await self.gateway.structured(
                tier=ModelTier.REASONING, agent="dom_intelligence_agent", output_model=LLMChoice, max_tokens=2000,
                system="You select which page element a test instruction refers to. Answer with a candidate number "
                       "only if the evidence clearly supports it; otherwise answer null.",
                prompt=f"Instruction: {step.original_text}\nAction: {step.action.value}\nCandidates:\n" +
                       "\n".join(lines), untrusted=f"Page title: {obs.title}\nHeadings: {obs.headings[:6]}")
        except (ModelError, ModelUnavailable) as exc:
            decision.evidence.append(f"LLM disambiguation unavailable: {str(exc)[:80]}")
            return decision
        if self.work is not None:
            self.work.llm_calls += 1
        choice = result.output
        if choice.element_number is None or not (1 <= choice.element_number <= len(candidates)):
            decision.evidence.append(f"LLM found no matching candidate: {choice.reason[:120]}")
            return decision
        chosen = candidates[choice.element_number - 1]
        el = obs.element(chosen.element_id)
        if el is None:
            return decision
        combined = round(min(0.97, 0.5 * chosen.score + 0.5 * choice.confidence), 4)
        if chosen.element_id == decision.element_id:
            combined = max(combined, decision.confidence)
        return decision.model_copy(update={
            "element_id": el.element_id, "ref": el.ref, "display": el.display(), "role": el.role, "name": el.name,
            "confidence": combined, "decision": self.grounder.decide(combined), "method": "llm",
            "bbox": el.bbox.model_dump() if el.bbox else None,
            "evidence": decision.evidence + [f"LLM selected candidate {choice.element_number}: {choice.reason[:160]}"]})

    @staticmethod
    def _needs_visual(step: Step, decision: GroundingDecision) -> bool:
        target = step.target
        if target is None:
            return False
        if decision.role == "canvas":
            return True
        return bool(target.visual_hint and "canvas" in target.preferred_roles and decision.decision != "execute")

    async def _visual_ground(self, step: Step, obs: Observation, decision: GroundingDecision) -> GroundingDecision:
        target = step.target
        assert target is not None and self.session is not None
        surface = None
        if decision.role == "canvas" and decision.element_id:
            surface = obs.element(decision.element_id)
        if surface is None:
            surface = next((e for e in obs.elements if e.role == "canvas" and e.visible), None)
        shot = await self.session.adapter.screenshot()
        if not shot.ok or not shot.images:
            return decision
        png = shot.images[0].data
        description = target.visual_hint or target.intent
        if surface is not None and surface.bbox is not None:
            match = self.visual.locate_in_surface(png, surface.bbox, description)
            if match is None and self.gateway.supports_vision():
                match = await self.visual.locate_with_model(png, surface.bbox, description)
            if match is not None:
                return decision.model_copy(update={
                    "element_id": surface.element_id, "ref": surface.ref, "display": f"{description} on canvas",
                    "role": "canvas", "name": description, "confidence": match.confidence,
                    "decision": self.grounder.decide(match.confidence), "method": match.method,
                    "coordinates": (round(match.x, 1), round(match.y, 1)),
                    "bbox": match.box.model_dump() if match.box else surface.bbox.model_dump(),
                    "evidence": decision.evidence + match.evidence})
            return decision.model_copy(update={"decision": "not_found", "confidence": 0.0,
                                               "reason": f"no '{description}' found visually in the canvas"})
        if decision.candidates and self.gateway.supports_vision():
            elements = [obs.element(c.element_id) for c in decision.candidates[:6]]
            chosen, conf, reason = await self.visual.choose_mark(png, [e for e in elements if e], description)
            if chosen is not None:
                return decision.model_copy(update={"element_id": chosen.element_id, "ref": chosen.ref,
                                                   "display": chosen.display(), "confidence": round(conf * 0.95, 4),
                                                   "decision": self.grounder.decide(conf * 0.95), "method": "visual",
                                                   "evidence": decision.evidence + [f"vision: {reason[:120]}"]})
        return decision

    # ------------------------------------------------------------------ policy
    async def validate(self, step: Step) -> PolicyDecision:
        work = self.ensure_work(step)
        obs = self.observation
        url = None
        if step.action == ActionType.NAVIGATE and step.target and step.target.url:
            url = await self._resolve_url(step.target.url)
        value_sensitive = bool(step.value and step.value.sensitive) or step.sensitive
        decision = work.decision
        confidence = decision.confidence if decision is not None else 1.0
        request = ActionRequest(step=step, url=url, element=work.element, observation=obs, confidence=confidence,
                                value_sensitive=value_sensitive, is_recovery=work.recovery_cycles > 0)
        result = self.policy.evaluate(request)
        work.policy = result
        await self.emit("ACTION_VALIDATED" if result.allowed else "ACTION_BLOCKED",
                        "; ".join(result.reasons)[:300], step_id=step.id, **result.to_dict())
        return result

    async def _resolve_url(self, raw: str) -> str:
        url = await self.resolver.interpolate(raw)
        if url.startswith("/"):
            base = (self.request.spec.base_url or "").rstrip("/")
            if not base and self.observation is not None:
                from urllib.parse import urlparse

                parsed = urlparse(self.observation.url)
                base = f"{parsed.scheme}://{parsed.netloc}"
            url = base + url
        return url

    # ------------------------------------------------------------------ execution
    async def execute(self, step: Step) -> ActionOutcome:
        assert self.executor is not None and self.session is not None
        work = self.ensure_work(step)
        work.attempts += 1
        if work.attempts > 1:
            self.retries += 1
        self.budget.check_alive()
        obs = self.observation
        work.url_before = obs.url if obs else None
        work.before = obs
        decision = work.decision
        if self.screenshot_policy == "every_action" and step.action not in (ActionType.WAIT,):
            work.screenshot_before = await self.screenshot(step.id, f"before-{work.attempts}", decision=decision)
        await self.emit("ACTION_STARTED", step.summary(), step_id=step.id, attempt=work.attempts,
                        target=decision.display if decision else None)
        try:
            outcome = await self._dispatch(step, work)
        except BrowserDisconnected as exc:
            outcome = ActionOutcome(False, error=str(exc), failure_category=FailureCategory.BROWSER_DISCONNECTED)
        except PlatformError as exc:
            outcome = ActionOutcome(False, error=exc.message, failure_category=FailureCategory.UNKNOWN_FAILURE)
        work.outcome = outcome
        self.dirty = True
        if outcome.downloads:
            self.downloads.extend(outcome.downloads)
            for dl in outcome.downloads:
                from pathlib import Path

                path = Path(dl.get("path", ""))
                if path.exists():
                    ref = await self.artifacts.put_file(self.execution_id, "downloads", path)
                    if self.recorder is not None:
                        await self.recorder.artifact(ref, step.id)
        record = ActionRecord(action_id=new_id("act"), step_id=step.id, attempt=work.attempts, action=step.action.value,
                              tool=outcome.tool, tool_args=outcome.args, target_display=decision.display if decision else None,
                              grounding=decision, ok=outcome.ok, error=outcome.error,
                              failure_category=outcome.failure_category.value if outcome.failure_category else None,
                              playwright_code=outcome.code, screenshot_before=work.screenshot_before,
                              url_before=work.url_before, duration_ms=round((time.time() - work.started) * 1000, 1))
        self.results[step.id].actions.append(record)
        await self.emit("ACTION_COMPLETED" if outcome.ok else "ACTION_FAILED",
                        outcome.error or "; ".join(outcome.notes) or "ok", step_id=step.id, tool=outcome.tool,
                        ok=outcome.ok, failure_category=record.failure_category, code=outcome.code)
        return outcome

    async def _dispatch(self, step: Step, work: StepWork) -> ActionOutcome:
        ex = self.executor
        assert ex is not None and self.session is not None
        el = work.element
        decision = work.decision
        action = step.action
        if action in GROUNDED_ACTIONS and step.target is not None and (decision is None or decision.element_id is None):
            return ActionOutcome(False, error=decision.reason if decision else "target not grounded",
                                 failure_category=FailureCategory.ELEMENT_NOT_FOUND)
        if decision is not None and decision.coordinates and action in (ActionType.CLICK, ActionType.DOUBLE_CLICK,
                                                                         ActionType.RIGHT_CLICK, ActionType.HOVER):
            x, y = decision.coordinates
            if action == ActionType.HOVER:
                response = await self.session.adapter.mouse_move(x, y)
            else:
                response = await self.session.adapter.mouse_click(
                    x, y, button="right" if action == ActionType.RIGHT_CLICK else "left",
                    click_count=2 if action == ActionType.DOUBLE_CLICK else 1)
            return ActionOutcome.from_response(response, args={"x": x, "y": y}, notes=["coordinate fallback "
                                                                                          "(visually grounded)"])
        if action in (ActionType.FILL, ActionType.TYPE, ActionType.SELECT, ActionType.UPLOAD, ActionType.CLEAR):
            work.value = await self.resolver.resolve(step.value) if step.value else ResolvedValue("", False, "none")
        if action == ActionType.NAVIGATE:
            url = await self._resolve_url(step.target.url if step.target and step.target.url else
                                          (self.request.spec.base_url or ""))
            return await ex.navigate(url)
        if action == ActionType.OPEN_PAGE:
            return await self._open_page(step, work)
        if action == ActionType.GO_BACK:
            return ActionOutcome.from_response(await self.session.adapter.back())
        if action == ActionType.RELOAD:
            return await ex.navigate(self.observation.url if self.observation else "about:blank")
        if action in (ActionType.CLICK, ActionType.DOWNLOAD):
            assert el is not None
            return await ex.click(el)
        if action == ActionType.DOUBLE_CLICK:
            assert el is not None
            return await ex.click(el, double=True)
        if action == ActionType.RIGHT_CLICK:
            assert el is not None
            return await ex.click(el, right=True)
        if action == ActionType.HOVER:
            assert el is not None
            return await ex.hover(el)
        if action in (ActionType.FILL, ActionType.TYPE):
            assert el is not None and work.value is not None
            return await ex.fill(el, work.value.text, sensitive=work.value.sensitive,
                                 slowly=action == ActionType.TYPE, observation=self.observation)
        if action == ActionType.CLEAR:
            assert el is not None
            return await ex.fill(el, "", sensitive=False)
        if action == ActionType.SELECT:
            assert el is not None and work.value is not None
            return await ex.select(el, work.value.text, self.observation)
        if action in (ActionType.CHECK, ActionType.UNCHECK, ActionType.TOGGLE):
            assert el is not None
            desired = True if action == ActionType.CHECK else False if action == ActionType.UNCHECK else None
            if action == ActionType.CHECK and el.role == "radio":
                desired = True
            return await ex.set_checked(el, desired)
        if action == ActionType.PRESS_KEY:
            focus = el is not None and (self.previous_element is None or
                                        element_key(self.previous_element) != element_key(el))
            return await ex.press_key(step.key or "Enter", el, focus=focus)
        if action == ActionType.DRAG:
            assert el is not None
            if work.destination is None:
                return ActionOutcome(False, error="drop destination not found",
                                     failure_category=FailureCategory.ELEMENT_NOT_FOUND)
            return await ex.drag(el, work.destination)
        if action == ActionType.UPLOAD:
            assert el is not None and work.value is not None
            return await ex.upload(el, work.value.text)
        if action == ActionType.SCROLL:
            return await ex.scroll(step)
        if action == ActionType.WAIT:
            return await ex.wait(step)
        if action == ActionType.SWITCH_TAB:
            return await ex.switch_tab(step, self.observation or await self.observe())
        if action == ActionType.SWITCH_FRAME:
            self.session.frame_hint = step.target.frame_hint if step.target else None
            return ActionOutcome(True, notes=["frame context updated (iframe elements are addressed by ref)"])
        if action in (ActionType.ACCEPT_DIALOG, ActionType.DISMISS_DIALOG):
            obs = await self.observe(force=True)
            text = await self.resolver.resolve(step.value) if step.value else None
            return await ex.handle_dialog(step, obs, action == ActionType.ACCEPT_DIALOG, text.text if text else None)
        if action == ActionType.CAPTURE_SCREENSHOT:
            uri = await self.screenshot(step.id, "capture", full_page="full_page" in step.notes)
            return ActionOutcome(uri is not None, notes=[f"screenshot {uri}"])
        if action == ActionType.CAPTURE_VALUE:
            assert el is not None and step.capture_as
            value = await ex.read_value(el)
            if value is None:
                return ActionOutcome(False, error="no value to capture", failure_category=FailureCategory.ELEMENT_NOT_FOUND)
            self.captured[step.capture_as] = value
            return ActionOutcome(True, notes=[f"captured {step.capture_as}='{value[:60]}'"])
        if action == ActionType.ASSERT:
            return await self._assert(step, work)
        if action == ActionType.FILL_FORM:
            return await self._fill_form(step, work)
        return ActionOutcome(False, error=f"unsupported action {action}", failure_category=FailureCategory.UNKNOWN_FAILURE)

    async def _open_page(self, step: Step, work: StepWork) -> ActionOutcome:
        ex = self.executor
        assert ex is not None and step.target is not None
        obs = await self.observe()
        page = normalize_text(step.target.intent.replace(" page", ""))
        hints = {normalize_text(h) for h in step.target.semantic_hints} | {page}
        texts = [normalize_text(obs.url), normalize_text(obs.title), *(normalize_text(h) for h in obs.headings[:3])]
        purpose_match = step.target.purpose == "submit_login" and obs.page_state.page_kind == "login"
        if purpose_match or any(h and any(h in t for t in texts) for h in hints):
            return ActionOutcome(True, no_op=True, notes=[f"already on the {page} page"],
                                 expected={"page": page})
        opener = next((n for n in step.notes if n.startswith("opened_by_next_step:")), None)
        if opener:
            return ActionOutcome(True, no_op=True, notes=[f"the next step ({opener.split(':')[1]}) opens the {page} "
                                                          f"page; application entry page is loaded"],
                                 expected={"page": page, "deferred": True})
        decision = self.grounder.ground(step, obs, GroundingContext(), action_override="click")
        el = obs.element(decision.element_id) if decision.element_id else None
        if el is None or decision.confidence < self.settings.thresholds.refine:
            return ActionOutcome(False, error=f"no navigation control for the {page} page",
                                 failure_category=FailureCategory.ELEMENT_NOT_FOUND)
        work.decision, work.element = decision, el
        outcome = await ex.click(el)
        outcome.notes.append(f"navigated via {el.display()}")
        outcome.expected = {"page": page}
        return outcome

    async def _assert(self, step: Step, work: StepWork) -> ActionOutcome:
        assert self.assertions is not None and step.assertion is not None
        ctx = AssertionContext(url_at_step_start=work.url_before, downloads=self.downloads)
        result, obs = await self.assertions.evaluate(step.assertion, step, ctx)
        if obs is not None:
            self.observation, self.dirty = obs, False
        work.assertion_results.append(result)
        self.results[step.id].assertions.append(result)
        await self.emit("ASSERTION_PASSED" if result.status == "PASSED" else "ASSERTION_FAILED",
                        f"{result.description}: expected {result.expected!r}, actual {str(result.actual)[:120]!r}",
                        step_id=step.id, status=result.status, type=result.type)
        ok = result.status == "PASSED"
        return ActionOutcome(ok, tool="assertion_engine", error=None if ok else result.message,
                             failure_category=None if ok else FailureCategory.ASSERTION_FAILURE,
                             expected={"assertion_status": result.status})

    async def _fill_form(self, step: Step, work: StepWork) -> ActionOutcome:
        """Semantic form fill: map test data (or generated non-sensitive data) onto fields by purpose."""

        from backend.agents.forms.form_filler import FormFiller

        assert self.executor is not None
        obs = await self.observe()
        filler = FormFiller(self.executor, self.resolver, self.grounder)
        outcome, filled = await filler.fill(obs, self.request.variables)
        self.dirty = True
        await self.emit("ACTION_PLANNED", f"form mapping: {filled}", step_id=step.id, mapping=filled)
        return outcome

    async def settle(self, step: Step) -> None:
        """Wait for the state change triggered by the action (navigation, rendering, network)."""

        work = self.ensure_work(step)
        if step.action in (ActionType.ASSERT, ActionType.WAIT, ActionType.CAPTURE_SCREENSHOT, ActionType.CAPTURE_VALUE):
            return
        assert self.session is not None
        deadline = time.monotonic() + (6.0 if step.expects_navigation else 1.5)
        before_url = work.url_before
        await asyncio.sleep(0.25)
        while time.monotonic() < deadline:
            info = await self.session.adapter.probe(__import__("backend.browser.js_probes",
                                                               fromlist=["PAGE_INFO"]).PAGE_INFO)
            if info is None:  # dialog open or navigation in progress
                await asyncio.sleep(0.3)
                if self.observation is not None:
                    break
                continue
            if info.get("readyState") == "complete" and (not step.expects_navigation or info.get("url") != before_url):
                break
            await asyncio.sleep(0.3)
        self.dirty = True

    # ------------------------------------------------------------------ verification
    async def verify(self, step: Step) -> VerificationResult:
        assert self.verifier is not None
        work = self.ensure_work(step)
        outcome = work.outcome
        if outcome is None:
            return VerificationResult("failed", message="action was not executed")
        after = await self.observe(force=True)
        work.after = after
        if after.page_state.injection_findings and step.action == ActionType.NAVIGATE:
            pass  # recorded as security warning in observe()
        post = self.policy.check_post_navigation(after.url)
        if not post.allowed:
            verification = VerificationResult("failed", message=post.reasons[0],
                                              failure_category=FailureCategory.UNEXPECTED_NAVIGATION)
        elif step.action == ActionType.ASSERT:
            status = outcome.expected.get("assertion_status")
            verification = VerificationResult(
                "verified" if outcome.ok else ("failed" if status == "FAILED" else "unverified"),
                message=outcome.error or "assertion passed",
                failure_category=None if outcome.ok else FailureCategory.ASSERTION_FAILURE)
            if status == "INCONCLUSIVE":
                verification.status = "unverified"
        elif not outcome.ok:
            verification = VerificationResult("failed", message=outcome.error or "action failed",
                                              failure_category=outcome.failure_category or FailureCategory.UNKNOWN_FAILURE)
        elif step.action == ActionType.OPEN_PAGE:
            page = outcome.expected.get("page", "")
            texts = [normalize_text(after.url), normalize_text(after.title), *(normalize_text(h) for h in after.headings[:3])]
            ok = outcome.no_op or outcome.expected.get("deferred") or any(page and page in t for t in texts) or (
                step.target is not None and step.target.purpose == "submit_login" and after.page_state.page_kind == "login")
            verification = VerificationResult("verified" if ok else "failed",
                                              message=f"{'on' if ok else 'not on'} the {page} page",
                                              failure_category=None if ok else FailureCategory.VERIFICATION_FAILED)
        else:
            verification = await self.verifier.verify(step, outcome, work.element, work.before, after)
        work.verification = verification
        if self.screenshot_policy == "every_action" or (self.screenshot_policy == "on_failure" and
                                                       verification.status == "failed"):
            work.screenshot_after = await self.screenshot(step.id, f"after-{work.attempts}")
        result = self.results[step.id]
        if result.actions:
            last = result.actions[-1]
            last.screenshot_after = work.screenshot_after
            last.url_after = after.url
            last.verification = verification.to_dict()
        await self.emit("ACTION_VERIFIED" if verification.status != "failed" else "ACTION_FAILED",
                        verification.message, step_id=step.id, status=verification.status,
                        checks=verification.checks[:6])
        return verification

    # ------------------------------------------------------------------ learning
    def learn(self, step: Step) -> None:
        work = self.ensure_work(step)
        el = work.element
        if el is not None:
            self.previous_element = el
            if el.context.form_ref:
                self.previous_form_ref = el.context.form_ref
        if el is None or work.before is None or work.before.fingerprint is None or step.target is None:
            return
        if step.action in (ActionType.ASSERT, ActionType.NAVIGATE, ActionType.OPEN_PAGE) or work.decision is None:
            return
        if work.decision.method in ("color_segmentation", "vision_model"):
            return  # coordinates are never stored; the surface is remembered via its element only
        strategy = LocatorStrategy.from_element(el, work.before)
        self.learning.append(LearningRecord(step=step, route_pattern=work.before.fingerprint.route_pattern,
                                            page_fingerprint=work.before.fingerprint.structure_hash, strategy=strategy,
                                            healed=work.healed, previous_memory_id=work.memory[0].memory_id
                                            if work.memory and work.healed else None,
                                            used_memory_id=work.used_memory_id,
                                            failed_strategies=work.failed_strategies,
                                            failure_category=work.analysis.category.value if work.analysis else None,
                                            recovery_attempts=work.recovery_cycles))

    # ------------------------------------------------------------------ failures & recovery
    def classify(self, step: Step) -> FailureAnalysis:
        work = self.ensure_work(step)
        category = None
        error = None
        if work.policy is not None and not work.policy.allowed:
            if work.policy.requires_approval:
                category = FailureCategory.POLICY_BLOCKED
            elif "confidence_gate" in work.policy.rules:
                category = FailureCategory.LOW_CONFIDENCE
            elif "captcha_handoff" in work.policy.rules:
                category = FailureCategory.CAPTCHA_DETECTED
            elif "domain_allowlist" in work.policy.rules:
                category = FailureCategory.UNEXPECTED_NAVIGATION
            else:
                category = FailureCategory.POLICY_BLOCKED
            error = "; ".join(work.policy.reasons)
        elif work.verification is not None and work.verification.status == "failed":
            category = work.verification.failure_category
            error = work.verification.message
        elif work.outcome is not None and not work.outcome.ok:
            category = work.outcome.failure_category
            error = work.outcome.error
        elif work.decision is not None and work.decision.decision in ("not_found", "ask_human"):
            error = work.decision.reason
        analysis = classify_failure(category=category, error=error, decision=work.decision,
                                    observation=work.after or self.observation, had_memory=bool(work.memory))
        if step.action == ActionType.ASSERT and analysis.category not in (FailureCategory.CAPTCHA_DETECTED,):
            analysis.healable = False
            analysis.product_failure = analysis.category == FailureCategory.ASSERTION_FAILURE
        work.analysis = analysis
        result = self.results[step.id]
        result.failure_category = analysis.category.value
        return analysis

    def plan_recovery(self, step: Step, analysis: FailureAnalysis) -> dict[str, Any] | None:
        work = self.ensure_work(step)
        budget = self.settings.recovery
        if not analysis.healable:
            return None
        if work.recovery_cycles >= budget.max_recovery_cycles_per_step + (1 if analysis.category in (
                FailureCategory.STALE_ELEMENT_REFERENCE,) else 0):
            return None
        if self.recoveries >= budget.max_test_recovery_cycles:
            return None
        if time.time() - work.started > budget.max_recovery_seconds:
            return None
        cat = analysis.category
        order = {
            FailureCategory.STALE_ELEMENT_REFERENCE: ["reobserve_and_reground"],
            FailureCategory.ELEMENT_NOT_FOUND: ["wait_for_stability", "reground_without_memory", "visual_grounding"],
            FailureCategory.PAGE_STRUCTURE_CHANGED: ["reground_without_memory", "wait_for_stability", "visual_grounding"],
            FailureCategory.AMBIGUOUS_TARGET: ["reground_with_context", "visual_grounding"],
            FailureCategory.LOW_CONFIDENCE: ["reground_without_memory", "wait_for_stability", "visual_grounding"],
            FailureCategory.ELEMENT_NOT_VISIBLE: ["scroll_into_view", "wait_for_stability"],
            FailureCategory.OVERLAY_BLOCKING_TARGET: ["dismiss_overlay", "wait_for_stability"],
            FailureCategory.ACTION_TIMEOUT: ["idempotency_check", "wait_for_stability"],
            FailureCategory.NAVIGATION_TIMEOUT: ["idempotency_check", "wait_for_stability"],
            FailureCategory.NETWORK_FAILURE: ["idempotency_check"],
            FailureCategory.DROPDOWN_NOT_OPEN: ["wait_for_stability", "reobserve_and_reground"],
            FailureCategory.OPTION_NOT_FOUND: ["wait_for_stability"],
            FailureCategory.POPUP_OR_DIALOG_ERROR: ["handle_unexpected_dialog", "reobserve_and_reground"],
            FailureCategory.VERIFICATION_FAILED: ["idempotency_check", "reground_excluding_previous"],
            FailureCategory.ELEMENT_NOT_ENABLED: ["wait_for_stability"],
            FailureCategory.IFRAME_CONTEXT_ERROR: ["reobserve_and_reground"],
            FailureCategory.BROWSER_DISCONNECTED: ["session_recovery"],
        }.get(cat, [])
        if cat in (FailureCategory.ELEMENT_NOT_FOUND, FailureCategory.PAGE_STRUCTURE_CHANGED,
                   FailureCategory.AMBIGUOUS_TARGET, FailureCategory.LOW_CONFIDENCE) and \
                work.healing_attempts >= budget.max_locator_healing_attempts:
            return None
        for strategy in order:
            if strategy not in work.tried_strategies or strategy in ("reobserve_and_reground",) and \
                    work.tried_strategies.count(strategy) < 2:
                rationale = {
                    "reobserve_and_reground": "the page re-rendered; capture a fresh snapshot and ground again",
                    "reground_without_memory": "the remembered locator no longer matches; ground semantically on the "
                                               "live page",
                    "reground_with_context": "several candidates match; use form/section context to disambiguate",
                    "reground_excluding_previous": "the previous target did not produce the intended state; exclude "
                                                   "it and re-ground",
                    "wait_for_stability": "content may still be loading; wait for the page to settle",
                    "visual_grounding": "use the screenshot to locate the target visually",
                    "scroll_into_view": "bring the target into the viewport",
                    "dismiss_overlay": "an overlay (cookie banner / modal) blocks the target",
                    "idempotency_check": "check whether the action already took effect before retrying",
                    "handle_unexpected_dialog": "an unexpected native dialog is open",
                    "session_recovery": "the browser connection was lost",
                }[strategy]
                return {"strategy": strategy, "rationale": rationale, "category": cat.value}
        return None

    async def execute_recovery(self, step: Step, plan: dict[str, Any]) -> tuple[bool, str]:
        assert self.session is not None
        work = self.ensure_work(step)
        strategy = plan["strategy"]
        work.tried_strategies.append(strategy)
        work.recovery_cycles += 1
        self.recoveries += 1
        await self.emit("RECOVERY_STARTED", f"{strategy}: {plan['rationale']}", step_id=step.id, **plan)
        if work.element is not None and strategy in ("reground_without_memory", "reground_excluding_previous",
                                                     "reground_with_context", "visual_grounding"):
            work.failed_strategies.append(LocatorStrategy.from_element(work.element, work.before)
                                          .model_dump(exclude_none=True))
        if work.used_memory_id and self.memory is not None and strategy in (
                "reground_without_memory", "reground_excluding_previous", "reground_with_context",
                "reobserve_and_reground", "visual_grounding"):
            # the remembered target did not work out: penalise it and let full grounding (and the model) decide
            await self.memory.record_strategy_failure(work.used_memory_id, execution_id=self.execution_id,
                                                      reason=work.analysis.root_cause if work.analysis else "")
            work.drop_memory = True
            work.used_memory_id = None
        ok, detail = True, ""
        if strategy in ("reobserve_and_reground", "reground_with_context"):
            self.dirty = True
            work.healing_attempts += 1
        elif strategy == "reground_without_memory":
            work.drop_memory = True
            work.healing_attempts += 1
            work.healed = True
            self.dirty = True
        elif strategy == "reground_excluding_previous":
            if work.element is not None:
                work.excluded.add(element_key(work.element))
            work.healing_attempts += 1
            work.healed = True
            self.dirty = True
        elif strategy == "wait_for_stability":
            for _ in range(10):
                obs = await self.observe(force=True)
                if not obs.page_state.loading:
                    break
                await asyncio.sleep(0.5)
            await asyncio.sleep(0.5)
            self.dirty = True
            detail = "page settled"
        elif strategy == "visual_grounding":
            work.force_visual = True
            work.healing_attempts += 1
            self.dirty = True
        elif strategy == "scroll_into_view":
            if work.element is not None and target_of(work.element):
                await self.session.adapter.probe(scroll_into_view_probe(), target_of(work.element),
                                                 work.element.display())
            self.dirty = True
        elif strategy == "dismiss_overlay":
            ok, detail = await self._dismiss_overlay(step)
        elif strategy == "handle_unexpected_dialog":
            obs = await self.observe(force=True)
            if obs.modal_state is not None:
                accept = obs.modal_state.kind == "alert"
                await self.session.adapter.handle_dialog(accept)
                detail = f"{'accepted' if accept else 'dismissed'} unexpected {obs.modal_state.kind}: " \
                         f"{obs.modal_state.message[:80]}"
            self.dirty = True
        elif strategy == "idempotency_check":
            ok, detail = await self._idempotency_check(step)
        elif strategy == "session_recovery":
            ok, detail = await self._session_recovery(step)
        attempt = RecoveryAttempt(attempt=work.recovery_cycles, failure_category=plan["category"], strategy=strategy,
                                  rationale=plan["rationale"], outcome="recovered" if ok else "failed",
                                  details={"detail": detail})
        self.results[step.id].recovery.append(attempt)
        if self.recorder is not None:
            await self.recorder.recovery(step.id, attempt)
        await self.emit("RECOVERY_COMPLETED" if ok else "RECOVERY_FAILED", detail or strategy, step_id=step.id,
                        strategy=strategy)
        return ok, detail

    async def _dismiss_overlay(self, step: Step) -> tuple[bool, str]:
        assert self.executor is not None
        obs = await self.observe(force=True)
        overlays = [d for d in obs.dialogs if d.element_ids]
        if not overlays:
            return False, "no overlay found"
        overlay = next((d for d in overlays if d.kind == "cookie_banner"), overlays[-1])
        if overlay.kind != "cookie_banner" and not overlay.modal:
            return False, f"overlay {overlay.name!r} is not a dismissible banner"
        outcome = await self.executor.handle_dialog(step, obs, accept=overlay.kind == "cookie_banner",
                                                    prompt_text=None)
        self.dirty = True
        return outcome.ok, f"dismissed overlay {overlay.name!r}: {outcome.notes or outcome.error}"

    async def _idempotency_check(self, step: Step) -> tuple[bool, str]:
        """Before repeating a possibly-completed action, look at what actually happened."""

        work = self.ensure_work(step)
        obs = await self.observe(force=True)
        changed, reasons = page_changed(work.before, obs)
        risky = step.risk_hint in NON_IDEMPOTENT_RISKS
        if changed and risky:
            work.outcome = ActionOutcome(True, notes=["action had already taken effect: " + ", ".join(reasons[:2])])
            return True, "action already took effect; not repeating a non-idempotent action"
        if changed and work.outcome is not None and work.verification is not None and self.verifier is not None:
            verification = await self.verifier.verify(step, work.outcome, work.element, work.before, obs)
            if verification.status == "verified":
                work.outcome.ok = True
                return True, "postcondition already satisfied"
        return True, "no prior effect detected; safe to retry"

    async def _session_recovery(self, step: Step) -> tuple[bool, str]:
        last_url = self.observation.url if self.observation else None
        last_fp = self.observation.fingerprint if self.observation else None
        if self.session is not None:
            await self.sessions.close(self.session.session_id)
        self.session = None
        await self.start_browser()
        if not last_url:
            return False, "no confirmed page to return to"
        await self.executor.navigate(last_url)  # type: ignore[union-attr]
        obs = await self.observe(force=True)
        same = bool(last_fp and obs.fingerprint and last_fp.similarity(obs.fingerprint) >= 0.8)
        if not same:
            return False, "browser restarted but the previous page state could not be restored safely"
        return True, "browser session restored and page state confirmed"

    # ------------------------------------------------------------------ step completion
    async def finish_step(self, step: Step, status: StepStatus, message: str = "") -> StepResult:
        work = self.ensure_work(step)
        result = self.results[step.id]
        result.status = status
        result.message = message
        result.attempts = work.attempts
        result.recovered = bool(result.recovery) and status == StepStatus.PASSED
        result.memory_used = bool(work.used_memory_id)
        result.memory_details = {"candidates": [c.to_dict() for c in work.memory][:3],
                                 "used": work.used_memory_id} if work.memory else {}
        result.page_url = self.observation.url if self.observation else None
        result.ended_at = time.time()
        result.duration_ms = round((result.ended_at - (result.started_at or result.ended_at)) * 1000, 1)
        result.llm_calls = work.llm_calls
        if self.recorder is not None:
            await self.recorder.step_result(result)
        await self.emit("STEP_COMPLETED", f"{status.value}: {message}"[:300], step_id=step.id, status=status.value,
                        recovered=result.recovered, memory_used=result.memory_used, duration_ms=result.duration_ms)
        return result

    def skip_remaining(self, step_ids: list[str]) -> None:
        for sid in step_ids:
            step = self.steps_by_id.get(sid)
            if step is None or sid in self.results and self.results[sid].status != StepStatus.PENDING:
                continue
            self.results[sid] = StepResult(step_id=sid, index=step.index, original_text=step.original_text,
                                           action=step.action.value, summary=step.summary(),
                                           status=StepStatus.SKIPPED, message="not executed (earlier step did not pass)")

    def compact_decision(self) -> dict[str, Any] | None:
        work = self.work
        if work is None or work.decision is None:
            return None
        d = work.decision
        return {"element_id": d.element_id, "display": d.display, "confidence": d.confidence, "decision": d.decision,
                "method": d.method, "memory_id": d.memory_id, "evidence": d.evidence[:4],
                "strategy": {k: v for k, v in d.locator_strategy.items() if k in ("role", "name", "label", "test_id",
                                                                                   "form_context")},
                "candidates": [c.model_dump(include={"display", "score"}) for c in d.candidates[:3]]}


GROUNDED_ACTIONS = {ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK, ActionType.HOVER,
                    ActionType.FILL, ActionType.TYPE, ActionType.CLEAR, ActionType.SELECT, ActionType.CHECK,
                    ActionType.UNCHECK, ActionType.TOGGLE, ActionType.UPLOAD, ActionType.DRAG, ActionType.DOWNLOAD,
                    ActionType.CAPTURE_VALUE, ActionType.PRESS_KEY}


class RuntimeRegistry:
    """Process-local registry of live runtimes, keyed by execution id."""

    def __init__(self) -> None:
        self._runtimes: dict[str, ExecutionRuntime] = {}

    def register(self, runtime: ExecutionRuntime) -> None:
        self._runtimes[runtime.execution_id] = runtime

    def get(self, execution_id: str) -> ExecutionRuntime | None:
        return self._runtimes.get(execution_id)

    def remove(self, execution_id: str) -> None:
        self._runtimes.pop(execution_id, None)

    def all(self) -> list[ExecutionRuntime]:
        return list(self._runtimes.values())


runtime_registry = RuntimeRegistry()

_ = GroundingCandidate  # re-exported type used by graph nodes
