"""Deterministic action regulator (spec section 16).

Every browser action is classified and checked here *before* execution. The
model has no input into this layer and cannot override it. Rules:

* only http(s) URLs on the allowlisted domains may be opened, and a redirect to
  any other domain stops the run;
* CAPTCHA / human-verification widgets are never operated by the agent;
* sensitive values may only be typed into fields meant for them (prevents
  credential exfiltration into search boxes, chat widgets, other domains...);
* destructive, financial and messaging actions require explicit approval unless
  the environment's policy grants them;
* low grounding confidence blocks submits and high-risk actions.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from backend.core.config import Settings, get_settings
from backend.dsl.models import ActionType, Step
from backend.dsl.observation import InventoryElement, Observation

Risk = Literal["low", "medium", "high", "restricted"]

SENSITIVE_TARGET_PURPOSES = {"password", "confirm_password", "otp", "cvv", "card_number"}
CAPTCHA_NAME_RE = re.compile(r"not a robot|i am human|i'm human|verify you are human|captcha|hcaptcha|recaptcha|"
                             r"turnstile|human verification", re.IGNORECASE)


@dataclass
class EnvironmentPolicy:
    """Permissions granted to one application environment (configured, never inferred)."""

    environment_type: Literal["test", "staging", "production"] = "test"
    allowed_domains: list[str] = field(default_factory=list)
    allow_destructive: bool = False
    allow_financial: bool = False
    allow_messaging: bool = False
    allow_form_submission: bool = True
    auto_approve_high_risk: bool = False

    @classmethod
    def from_settings(cls, settings: Settings, base_url: str | None, **overrides: object) -> EnvironmentPolicy:
        domains = list(settings.allowed_domains) + list(settings.always_allowed_domains)
        if base_url:
            host = (urlparse(base_url).hostname or "").lower()
            if host:
                domains.append(host)
        policy = cls(allowed_domains=sorted(set(domains)), allow_destructive=settings.allow_destructive_actions,
                     allow_financial=settings.allow_financial_actions,
                     allow_messaging=settings.allow_messaging_actions,
                     auto_approve_high_risk=settings.auto_approve_in_test_environments)
        for key, value in overrides.items():
            if hasattr(policy, key) and value is not None:
                setattr(policy, key, value)
        return policy


@dataclass
class ActionRequest:
    step: Step
    tool: str | None = None
    url: str | None = None
    element: InventoryElement | None = None
    observation: Observation | None = None
    confidence: float = 1.0
    value_sensitive: bool = False
    is_recovery: bool = False
    human_initiated: bool = False


@dataclass
class PolicyDecision:
    allowed: bool
    risk: Risk
    requires_approval: bool = False
    reasons: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"allowed": self.allowed, "risk": self.risk, "requires_approval": self.requires_approval,
                "reasons": self.reasons, "rules": self.rules}


def host_allowed(url: str, allowed: list[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    for pattern in allowed:
        pattern = pattern.lower().strip()
        if not pattern:
            continue
        if pattern.startswith("."):
            if host == pattern[1:] or host.endswith(pattern):
                return True
        elif "*" in pattern:
            if fnmatch.fnmatch(host, pattern):
                return True
        elif host == pattern:
            return True
    return False


class PolicyEngine:
    name = "policy_guard_agent"

    def __init__(self, env: EnvironmentPolicy, settings: Settings | None = None) -> None:
        self.env = env
        self.settings = settings or get_settings()
        self.approved: set[str] = set()  # approval keys granted by humans for this execution
        self.rejected: set[str] = set()

    @staticmethod
    def approval_key(step: Step, element: InventoryElement | None = None) -> str:
        target = f"{element.role}:{element.name}" if element else (step.target.intent if step.target else "")
        return f"{step.id}|{step.action.value}|{target}"

    def classify(self, request: ActionRequest) -> tuple[Risk, list[str]]:
        step = request.step
        reasons: list[str] = []
        if step.action in (ActionType.ASSERT, ActionType.WAIT, ActionType.CAPTURE_SCREENSHOT, ActionType.CAPTURE_VALUE,
                           ActionType.SCROLL, ActionType.HOVER, ActionType.SWITCH_FRAME):
            return "low", ["read-only action"]
        if step.action in (ActionType.NAVIGATE, ActionType.OPEN_PAGE, ActionType.GO_BACK, ActionType.RELOAD,
                           ActionType.SWITCH_TAB):
            return "low", ["navigation inside the test scope"]
        hint = step.risk_hint
        from backend.agents.planner.lexicon import risk_for_text

        element_risk = None
        if step.action in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.PRESS_KEY, ActionType.ACCEPT_DIALOG,
                           ActionType.TOGGLE):
            # only *activating* a control can delete, pay or send; typing into "Email" sends nothing
            text = " ".join(filter(None, [request.element.name if request.element else None,
                                          step.target.exact_name if step.target else None]))
            element_risk = risk_for_text(text) if text else None
        elif hint in ("destructive", "financial", "messaging"):
            hint = "input"
        if hint in ("destructive", "financial", "messaging") or element_risk in ("destructive", "financial",
                                                                                  "messaging"):
            kind = hint if hint in ("destructive", "financial", "messaging") else element_risk
            return "high", [f"{kind} action"]
        if hint == "submit" or element_risk == "submit":
            reasons.append("form submission")
            return "medium", reasons
        return "medium", ["test data entry / interaction"]

    def evaluate(self, request: ActionRequest) -> PolicyDecision:
        step = request.step
        risk, reasons = self.classify(request)
        rules: list[str] = []

        # 1. navigation scope
        url = request.url
        if url:
            if not host_allowed(url, self.env.allowed_domains):
                return PolicyDecision(False, "restricted", reasons=[f"URL {url!r} is outside the allowed domains "
                                                                    f"{self.env.allowed_domains}"],
                                      rules=["domain_allowlist"])
            rules.append("domain_allowlist")
        if request.observation is not None and request.observation.url and step.action not in (ActionType.NAVIGATE,):
            if request.observation.url.startswith("http") and not host_allowed(request.observation.url,
                                                                               self.env.allowed_domains):
                return PolicyDecision(False, "restricted",
                                      reasons=[f"current page {request.observation.url!r} is outside the allowed "
                                               "domains; refusing to act on it"], rules=["domain_allowlist"])

        # 2. human verification challenges are never operated by the agent
        el = request.element
        if el is not None and not request.human_initiated:
            label = " ".join(filter(None, [el.name, el.label, el.text, el.classes]))
            in_challenge = bool(request.observation and request.observation.page_state.captcha_detected
                                and el.role in ("checkbox", "button", "iframe"))
            if CAPTCHA_NAME_RE.search(label) or in_challenge:
                return PolicyDecision(False, "restricted", reasons=["target is a CAPTCHA / human-verification "
                                                                    "control; a person must complete it"],
                                      rules=["captcha_handoff"])

        # 3. secrets may only flow into fields designed for them
        if request.value_sensitive and el is not None and step.action in (ActionType.FILL, ActionType.TYPE):
            if el.input_type != "password" and el.semantic_purpose not in SENSITIVE_TARGET_PURPOSES:
                return PolicyDecision(False, "restricted",
                                      reasons=[f"refusing to type a sensitive value into {el.display()} "
                                               f"(purpose={el.semantic_purpose}, type={el.input_type})"],
                                      rules=["sensitive_value_placement"])
            rules.append("sensitive_value_placement")

        # 4. confidence gates
        t = self.settings.thresholds
        if el is not None or step.target is not None:
            if request.confidence < t.refine and step.action not in (ActionType.ASSERT, ActionType.WAIT):
                return PolicyDecision(False, risk, reasons=[f"grounding confidence {request.confidence:.2f} is below "
                                                            f"{t.refine:.2f}; human guidance required"],
                                      rules=["confidence_gate"])
            if risk in ("high",) and request.confidence < t.execute:
                key = self.approval_key(step, el)
                env_grant = self.env.auto_approve_high_risk and self.env.environment_type != "production" \
                    and request.confidence >= t.enhanced_verification
                if key not in self.approved and not env_grant:
                    reasons.append(f"high-risk action with confidence {request.confidence:.2f} < {t.execute:.2f}")
                    return PolicyDecision(False, risk, requires_approval=True, reasons=reasons,
                                          rules=["confidence_gate", "approval"])
            if reasons and "form submission" in reasons and request.confidence < t.enhanced_verification:
                if not self.env.allow_form_submission:
                    return PolicyDecision(False, risk, reasons=["form submission is not permitted"],
                                          rules=["submission_policy"])
            rules.append("confidence_gate")

        # 5. risk policy
        if risk == "high":
            key = self.approval_key(step, el)
            if key in self.rejected:
                return PolicyDecision(False, risk, reasons=["a reviewer rejected this action"], rules=["approval"])
            granted = (("destructive" in reasons[0] and self.env.allow_destructive)
                       or ("financial" in reasons[0] and self.env.allow_financial)
                       or ("messaging" in reasons[0] and self.env.allow_messaging))
            if granted and self.env.environment_type != "production":
                return PolicyDecision(True, risk, reasons=reasons + ["granted by environment policy"],
                                      rules=rules + ["risk_policy"])
            if key in self.approved or self.env.auto_approve_high_risk:
                return PolicyDecision(True, risk, reasons=reasons + ["approved by reviewer"],
                                      rules=rules + ["approval"])
            return PolicyDecision(False, risk, requires_approval=True, reasons=reasons + ["explicit approval required"],
                                  rules=rules + ["approval"])
        if "form submission" in reasons and not self.env.allow_form_submission:
            return PolicyDecision(False, risk, reasons=["form submission is not permitted in this environment"],
                                  rules=rules + ["submission_policy"])
        return PolicyDecision(True, risk, reasons=reasons, rules=rules + ["risk_policy"])

    def check_post_navigation(self, url: str) -> PolicyDecision:
        if url and url.startswith("http") and not host_allowed(url, self.env.allowed_domains):
            return PolicyDecision(False, "restricted", reasons=[f"navigation ended on a non-allowed domain: {url}"],
                                  rules=["domain_allowlist"])
        return PolicyDecision(True, "low", rules=["domain_allowlist"])
