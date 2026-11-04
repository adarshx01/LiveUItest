"""Execution result models: steps, actions, assertions, recoveries, reports."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StepStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"
    INCONCLUSIVE = "INCONCLUSIVE"


class TestOutcome(StrEnum):
    __test__ = False  # not a pytest test class
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ExecutionStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class FailureCategory(StrEnum):
    ELEMENT_NOT_FOUND = "ELEMENT_NOT_FOUND"
    STALE_ELEMENT_REFERENCE = "STALE_ELEMENT_REFERENCE"
    ELEMENT_NOT_VISIBLE = "ELEMENT_NOT_VISIBLE"
    ELEMENT_NOT_ENABLED = "ELEMENT_NOT_ENABLED"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
    ACTION_TIMEOUT = "ACTION_TIMEOUT"
    NAVIGATION_TIMEOUT = "NAVIGATION_TIMEOUT"
    UNEXPECTED_NAVIGATION = "UNEXPECTED_NAVIGATION"
    OVERLAY_BLOCKING_TARGET = "OVERLAY_BLOCKING_TARGET"
    DROPDOWN_NOT_OPEN = "DROPDOWN_NOT_OPEN"
    OPTION_NOT_FOUND = "OPTION_NOT_FOUND"
    FORM_VALIDATION_FAILED = "FORM_VALIDATION_FAILED"
    FORM_SUBMISSION_FAILED = "FORM_SUBMISSION_FAILED"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    CAPTCHA_DETECTED = "CAPTCHA_DETECTED"
    IFRAME_CONTEXT_ERROR = "IFRAME_CONTEXT_ERROR"
    POPUP_OR_DIALOG_ERROR = "POPUP_OR_DIALOG_ERROR"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    APPLICATION_ERROR = "APPLICATION_ERROR"
    ASSERTION_FAILURE = "ASSERTION_FAILURE"
    PAGE_STRUCTURE_CHANGED = "PAGE_STRUCTURE_CHANGED"
    BROWSER_DISCONNECTED = "BROWSER_DISCONNECTED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


def now() -> float:
    return time.time()


class Evidence(BaseModel):
    source: str  # accessibility_snapshot, dom_probe, url, screenshot, network, console, mcp_response
    observation_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class AssertionResult(BaseModel):
    assertion_id: str
    type: str
    description: str = ""
    expected: Any = None
    actual: Any = None
    status: Literal["PASSED", "FAILED", "INCONCLUSIVE", "ERROR"]
    message: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    derived_expectation: bool = False
    timestamp: float = Field(default_factory=now)


class GroundingCandidate(BaseModel):
    element_id: str
    ref: str | None = None
    display: str
    score: float
    features: dict[str, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    bbox: dict[str, float] | None = None


class GroundingDecision(BaseModel):
    element_id: str | None = None
    ref: str | None = None
    display: str = ""
    role: str | None = None
    name: str | None = None
    confidence: float = 0.0
    decision: Literal["execute", "execute_enhanced", "refine", "ask_human", "not_found"] = "not_found"
    method: str = "semantic"  # semantic, memory, visual, coordinate, llm, form_mapping
    locator_strategy: dict[str, Any] = Field(default_factory=dict)
    candidates: list[GroundingCandidate] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    bbox: dict[str, float] | None = None
    coordinates: tuple[float, float] | None = None
    memory_id: str | None = None
    reason: str = ""


class ActionRecord(BaseModel):
    action_id: str
    step_id: str
    attempt: int
    action: str
    tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)  # redacted
    target_display: str | None = None
    grounding: GroundingDecision | None = None
    ok: bool = False
    error: str | None = None
    failure_category: str | None = None
    playwright_code: str | None = None  # code Playwright reports it ran (redacted)
    screenshot_before: str | None = None
    screenshot_after: str | None = None
    url_before: str | None = None
    url_after: str | None = None
    verification: dict[str, Any] = Field(default_factory=dict)
    started_at: float = Field(default_factory=now)
    duration_ms: float = 0.0
    human: bool = False


class RecoveryAttempt(BaseModel):
    attempt: int
    failure_category: str
    strategy: str
    rationale: str
    outcome: Literal["recovered", "failed", "rejected", "skipped"] = "failed"
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=now)


class StepResult(BaseModel):
    step_id: str
    index: int
    original_text: str
    action: str
    summary: str = ""
    status: StepStatus = StepStatus.PENDING
    message: str = ""
    attempts: int = 0
    actions: list[ActionRecord] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)
    recovery: list[RecoveryAttempt] = Field(default_factory=list)
    recovered: bool = False
    memory_used: bool = False
    memory_details: dict[str, Any] = Field(default_factory=dict)
    runner: str | None = None
    failure_category: str | None = None
    page_url: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    duration_ms: float = 0.0
    branch: str | None = None  # "then"/"else" for conditional children
    llm_calls: int = 0

    @property
    def screenshots(self) -> list[str]:
        shots: list[str] = []
        for record in self.actions:
            for ref in (record.screenshot_before, record.screenshot_after):
                if ref and ref not in shots:
                    shots.append(ref)
        return shots


class ExecutionMetrics(BaseModel):
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    blocked_steps: int = 0
    skipped_steps: int = 0
    inconclusive_steps: int = 0
    recovered_steps: int = 0
    assertions_total: int = 0
    assertions_passed: int = 0
    assertions_failed: int = 0
    retries: int = 0
    recoveries: int = 0
    memory_hits: int = 0
    memory_fastpath: int = 0  # steps replayed directly from validated memory (no scoring or model call)
    memory_lookups: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_usd: float = 0.0
    tool_calls: int = 0
    duration_ms: float = 0.0
    model_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0


class TestResult(BaseModel):
    execution_id: str
    test_case_id: str
    test_name: str
    outcome: TestOutcome
    reason: str = ""
    steps: list[StepResult] = Field(default_factory=list)
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    flaky: bool = False
    original_strategy_succeeded: bool = True
