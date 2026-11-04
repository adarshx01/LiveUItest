"""Deterministic plan validation (graph node ``validate_test_plan``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from backend.dsl.models import TARGETED_ACTIONS, ActionType, Step, TestSpec


@dataclass
class PlanIssue:
    severity: Literal["blocking", "warning", "info"]
    code: str
    message: str
    step_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"severity": self.severity, "code": self.code, "message": self.message, "step_id": self.step_id}


class PlanValidator:
    name = "plan_validator"

    def validate(self, spec: TestSpec, *, available_variables: set[str], available_secrets: set[str],
                 min_step_confidence: float = 0.4) -> list[PlanIssue]:
        issues: list[PlanIssue] = []
        flat = spec.flat_steps()
        if not flat:
            issues.append(PlanIssue("blocking", "EMPTY_PLAN", "The test case has no executable steps"))
        for ambiguity in spec.ambiguities:
            issues.append(PlanIssue(ambiguity.severity if ambiguity.severity != "info" else "info",  # type: ignore[arg-type]
                                    "AMBIGUOUS_INSTRUCTION", f"{ambiguity.reason}: {ambiguity.line!r}"))
        captured = {s.capture_as for s in flat if s.capture_as}
        for step in flat:
            issues.extend(self._validate_step(step, available_variables | captured, available_secrets,
                                              min_step_confidence))
        if not spec.has_verifiable_outcome():
            issues.append(PlanIssue("warning", "NO_ASSERTIONS",
                                    "No verifiable expected outcome: the result cannot be PASSED and will be "
                                    "reported as INCONCLUSIVE unless assertions are added"))
        if not spec.base_url and not any(s.action == ActionType.NAVIGATE for s in flat):
            issues.append(PlanIssue("blocking", "MISSING_URL", "No application URL was provided"))
        return issues

    def _validate_step(self, step: Step, variables: set[str], secrets: set[str],
                       min_confidence: float) -> list[PlanIssue]:
        out: list[PlanIssue] = []
        if step.action in TARGETED_ACTIONS and step.target is None and step.action != ActionType.TYPE:
            out.append(PlanIssue("blocking", "MISSING_TARGET", "Action has no target element", step.id))
        if step.target is not None and step.action in TARGETED_ACTIONS:
            t = step.target
            if not (t.intent or t.exact_name or t.visual_hint or t.option_text or t.purpose):
                out.append(PlanIssue("blocking", "EMPTY_TARGET", "Target description is empty", step.id))
        if step.action in (ActionType.FILL, ActionType.TYPE) and (step.value is None or
                                                                   (step.value.source == "literal" and
                                                                    step.value.value is None)):
            out.append(PlanIssue("blocking", "MISSING_VALUE", "Fill step has no value", step.id))
        if step.action == ActionType.SELECT and not (step.value and (step.value.value or step.value.reference
                                                                    or step.value.values)):
            out.append(PlanIssue("blocking", "MISSING_OPTION", "Select step has no option", step.id))
        if step.action == ActionType.UPLOAD and not (step.value and (step.value.value or step.value.reference)):
            out.append(PlanIssue("blocking", "MISSING_FILE", "Upload step has no file", step.id))
        if step.action == ActionType.DRAG and step.destination is None:
            out.append(PlanIssue("blocking", "MISSING_DESTINATION", "Drag step has no destination", step.id))
        if step.action == ActionType.ASSERT and step.assertion is None:
            out.append(PlanIssue("blocking", "MISSING_ASSERTION", "Assertion step has no condition", step.id))
        if step.action == ActionType.NAVIGATE and step.target and step.target.url:
            url = step.target.url
            if "{{" not in url and not url.startswith("/"):
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    out.append(PlanIssue("blocking", "INVALID_URL", f"Invalid URL {url!r}", step.id))
        value = step.value
        if value is not None and value.source == "variable" and value.reference and value.reference not in variables:
            out.append(PlanIssue("blocking", "MISSING_VARIABLE",
                                 f"Test data does not define {{{{{value.reference}}}}}", step.id))
        if value is not None and value.source == "secret" and value.reference and value.reference not in secrets:
            out.append(PlanIssue("blocking", "MISSING_SECRET",
                                 f"Secret {{{{{value.reference}}}}} is not available in the secret store", step.id))
        if step.assertion and step.assertion.expected_ref:
            ref = step.assertion.expected_ref
            if ref.source in ("variable", "secret") and ref.reference and ref.reference not in variables | secrets:
                severity = "warning" if step.assertion.derived else "blocking"
                out.append(PlanIssue(severity, "MISSING_EXPECTATION_VALUE",
                                     f"Expected value {{{{{ref.reference}}}}} is not provided", step.id))
        if step.confidence < min_confidence:
            out.append(PlanIssue("warning", "LOW_CONFIDENCE_STEP",
                                 f"Instruction interpreted with low confidence ({step.confidence:.2f})", step.id))
        return out


def blocking(issues: list[PlanIssue]) -> list[PlanIssue]:
    return [i for i in issues if i.severity == "blocking"]
