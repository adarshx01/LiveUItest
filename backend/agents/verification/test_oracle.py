"""Independent test oracle (spec section 18.3): PASSED / FAILED / BLOCKED / INCONCLUSIVE."""

from __future__ import annotations

from backend.dsl.results import FailureCategory, StepResult, StepStatus, TestOutcome

BLOCKING_CATEGORIES = {FailureCategory.CAPTCHA_DETECTED.value, FailureCategory.POLICY_BLOCKED.value,
                       FailureCategory.LOW_CONFIDENCE.value, FailureCategory.BROWSER_DISCONNECTED.value,
                       FailureCategory.AUTHENTICATION_REQUIRED.value}


def evaluate_outcome(steps: list[StepResult], *, has_assertions: bool, plan_blocked_reason: str | None = None,
                     cancelled: bool = False) -> tuple[TestOutcome, str]:
    if plan_blocked_reason:
        return TestOutcome.BLOCKED, plan_blocked_reason
    if cancelled:
        return TestOutcome.BLOCKED, "execution was cancelled"
    for step in steps:
        if step.status == StepStatus.FAILED:
            return TestOutcome.FAILED, f"step {step.index} failed: {step.message or step.original_text}"
        if step.status == StepStatus.BLOCKED:
            return TestOutcome.BLOCKED, f"step {step.index} blocked: {step.message or step.original_text}"
    executed = [s for s in steps if s.status not in (StepStatus.SKIPPED, StepStatus.PENDING)]
    if not executed:
        return TestOutcome.INCONCLUSIVE, "no step was executed"
    if not has_assertions:
        return TestOutcome.INCONCLUSIVE, ("all actions completed but the test defines no verifiable expected outcome; "
                                          "add assertions to make the result conclusive")
    inconclusive = [s for s in steps if s.status == StepStatus.INCONCLUSIVE]
    if inconclusive:
        return TestOutcome.INCONCLUSIVE, f"step {inconclusive[0].index} inconclusive: {inconclusive[0].message}"
    assertions = [a for s in steps for a in s.assertions]
    if not assertions or not all(a.status == "PASSED" for a in assertions):
        return TestOutcome.INCONCLUSIVE, "assertions were not all evaluated"
    return TestOutcome.PASSED, f"all {len(assertions)} assertions passed"
