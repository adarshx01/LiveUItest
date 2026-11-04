"""Failure classification and root-cause analysis (spec section 17.1).

Separates *automation* failures (healable: stale refs, timing, overlays, changed
locators) from *product/test* failures (never healed: assertion failures,
application errors, validation errors, policy blocks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.dsl.observation import Observation
from backend.dsl.results import FailureCategory, GroundingDecision

HEALABLE = {
    FailureCategory.STALE_ELEMENT_REFERENCE, FailureCategory.ELEMENT_NOT_FOUND, FailureCategory.PAGE_STRUCTURE_CHANGED,
    FailureCategory.AMBIGUOUS_TARGET, FailureCategory.ELEMENT_NOT_VISIBLE, FailureCategory.OVERLAY_BLOCKING_TARGET,
    FailureCategory.ACTION_TIMEOUT, FailureCategory.NAVIGATION_TIMEOUT, FailureCategory.DROPDOWN_NOT_OPEN,
    FailureCategory.OPTION_NOT_FOUND, FailureCategory.POPUP_OR_DIALOG_ERROR, FailureCategory.LOW_CONFIDENCE,
    FailureCategory.VERIFICATION_FAILED, FailureCategory.NETWORK_FAILURE, FailureCategory.IFRAME_CONTEXT_ERROR,
    FailureCategory.ELEMENT_NOT_ENABLED, FailureCategory.BROWSER_DISCONNECTED,
}
PRODUCT_FAILURES = {
    FailureCategory.ASSERTION_FAILURE, FailureCategory.APPLICATION_ERROR, FailureCategory.FORM_VALIDATION_FAILED,
    FailureCategory.FORM_SUBMISSION_FAILED,
}
BLOCKERS = {FailureCategory.CAPTCHA_DETECTED, FailureCategory.POLICY_BLOCKED, FailureCategory.UNEXPECTED_NAVIGATION,
            FailureCategory.AUTHENTICATION_REQUIRED}


@dataclass
class FailureAnalysis:
    category: FailureCategory
    root_cause: str
    healable: bool
    product_failure: bool
    blocker: bool
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"category": self.category.value, "root_cause": self.root_cause, "healable": self.healable,
                "product_failure": self.product_failure, "blocker": self.blocker, "evidence": self.evidence}


def classify_failure(*, category: FailureCategory | None, error: str | None, decision: GroundingDecision | None,
                     observation: Observation | None, had_memory: bool = False) -> FailureAnalysis:
    evidence: list[str] = []
    cat = category or FailureCategory.UNKNOWN_FAILURE
    if observation is not None:
        state = observation.page_state
        if state.captcha_detected:
            cat = FailureCategory.CAPTCHA_DETECTED
            evidence.extend(state.captcha_evidence[:3])
        elif state.error_page and cat not in (FailureCategory.ASSERTION_FAILURE,):
            cat = FailureCategory.APPLICATION_ERROR
            evidence.append(f"error page: {observation.title} (HTTP {state.http_status})")
        elif state.validation_errors and cat in (FailureCategory.VERIFICATION_FAILED, FailureCategory.ELEMENT_NOT_ENABLED):
            cat = FailureCategory.FORM_VALIDATION_FAILED
            evidence.extend(state.validation_errors[:3])
        if observation.modal_state is not None and cat in (FailureCategory.UNKNOWN_FAILURE,
                                                           FailureCategory.ACTION_TIMEOUT):
            cat = FailureCategory.POPUP_OR_DIALOG_ERROR
            evidence.append(f"unexpected {observation.modal_state.kind}: {observation.modal_state.message[:80]}")
    if decision is not None and decision.decision in ("not_found",) and cat == FailureCategory.UNKNOWN_FAILURE:
        cat = FailureCategory.PAGE_STRUCTURE_CHANGED if had_memory else FailureCategory.ELEMENT_NOT_FOUND
    if decision is not None and decision.decision == "ask_human" and cat == FailureCategory.UNKNOWN_FAILURE:
        close = [c for c in decision.candidates[:3] if c.score >= decision.candidates[0].score - 0.05] \
            if decision.candidates else []
        cat = FailureCategory.AMBIGUOUS_TARGET if len(close) > 1 else FailureCategory.LOW_CONFIDENCE
    if error:
        evidence.append(error[:240])
    root = {
        FailureCategory.STALE_ELEMENT_REFERENCE: "page re-rendered between observation and action",
        FailureCategory.ELEMENT_NOT_FOUND: "no element on the page matches the instruction",
        FailureCategory.PAGE_STRUCTURE_CHANGED: "remembered locator no longer matches: the UI changed",
        FailureCategory.AMBIGUOUS_TARGET: "several elements match the instruction equally well",
        FailureCategory.OVERLAY_BLOCKING_TARGET: "an overlay covers the target",
        FailureCategory.ACTION_TIMEOUT: "element did not become actionable in time",
        FailureCategory.APPLICATION_ERROR: "the application returned an error",
        FailureCategory.FORM_VALIDATION_FAILED: "the application rejected the submitted data",
        FailureCategory.CAPTCHA_DETECTED: "human verification required",
        FailureCategory.ASSERTION_FAILURE: "expected outcome not observed",
    }.get(cat, cat.value.replace("_", " ").lower())
    return FailureAnalysis(category=cat, root_cause=root, healable=cat in HEALABLE,
                           product_failure=cat in PRODUCT_FAILURES, blocker=cat in BLOCKERS, evidence=evidence)
