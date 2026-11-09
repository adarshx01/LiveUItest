"""Grounding, locator strategies, policy, redaction, snapshot parsing and fingerprints (no browser needed)."""

from __future__ import annotations

from backend.agents.governance.policy_engine import ActionRequest, EnvironmentPolicy, PolicyEngine, host_allowed
from backend.agents.governance.redaction import MASK, redact_text, redact_value, register_sensitive
from backend.agents.grounding.element_grounder import ElementGrounder, GroundingContext
from backend.agents.grounding.locator_strategy import LocatorStrategy, resolve_strategy
from backend.agents.grounding.text_match import similarity, strip_badge
from backend.agents.observation.element_inventory import InventoryBuilder
from backend.agents.observation.page_fingerprint import fingerprint, normalize_route
from backend.agents.observation.snapshot_parser import parse_snapshot
from backend.agents.planner.nl_parser import ParseContext, StepParser
from backend.agents.recovery.failure_classifier import classify_failure
from backend.agents.verification.test_oracle import evaluate_outcome
from backend.dsl.models import ActionType
from backend.dsl.observation import BoundingBox, InventoryElement, Observation
from backend.dsl.results import AssertionResult, FailureCategory, StepResult, StepStatus, TestOutcome
from backend.memory.embeddings import cosine, embed

LOGIN_V1 = """- generic [ref=e1] [box=0,0,1280,800]:
  - banner [ref=e2] [box=0,0,1280,50]:
    - navigation "Main navigation" [ref=e3] [box=900,10,370,30]:
      - link "Shop" [ref=e4] [box=900,15,40,20]
      - link "Register" [ref=e5] [box=950,15,60,20]
  - main [ref=e6] [box=0,60,1280,700]:
    - heading "Sign in" [level=1] [ref=e7] [box=420,100,440,40]
    - form "Login form" [ref=e8] [box=420,150,440,300]:
      - text: Email address
      - textbox "Email address" [ref=e9] [box=420,180,440,36]:
        - /placeholder: you@example.com
      - text: Password
      - textbox "Password" [ref=e10] [box=420,240,440,36]
      - checkbox "Remember me" [ref=e11] [box=420,290,16,16]
      - link "Forgot password?" [ref=e12] [box=420,320,120,20]
      - button "Sign In" [ref=e13] [box=420,360,80,36]
"""

LOGIN_V2 = """- generic [ref=f3e1] [box=0,0,1280,800]:
  - banner [ref=f3e2] [box=0,0,1280,50]:
    - navigation "Main navigation" [ref=f3e3] [box=900,10,370,30]:
      - button "Sign in with SSO" [ref=f3e4] [box=1100,12,150,26]
  - main [ref=f3e6] [box=0,60,1280,700]:
    - region "Welcome back" [ref=f3e7] [box=400,90,480,420]:
      - heading "Welcome back" [level=2] [ref=f3e8] [box=420,100,440,30]
      - form "Account access" [ref=f3e9] [box=420,150,440,320]:
        - text: Username or email
        - textbox "Username or email" [ref=f3e10] [box=420,180,440,36]
        - text: Password
        - textbox "Password" [ref=f3e11] [box=420,240,440,36]
        - checkbox "Keep me signed in" [ref=f3e12] [box=420,290,16,16]
        - button "Continue with Google" [ref=f3e13] [box=420,360,200,36]
        - button "Log in" [ref=f3e14] [box=640,360,80,36]
"""

DOM_V1 = {"vw": 1280, "vh": 800, "elements": [
    {"tag": "input", "type": "text", "id": "email", "name": "identifier", "testid": "login-email",
     "label": "Email address", "box": [420, 180, 440, 36], "visible": True, "autocomplete": "username", "css": "#email"},
    {"tag": "input", "type": "password", "id": "password", "name": "password", "testid": "login-password",
     "label": "Password", "box": [420, 240, 440, 36], "visible": True, "css": "#password"},
    {"tag": "button", "type": "submit", "id": "login-btn", "testid": "login-submit", "box": [420, 360, 80, 36],
     "visible": True, "css": "[data-testid=\"login-submit\"]"},
]}
DOM_V2 = {"vw": 1280, "vh": 800, "elements": [
    {"tag": "input", "type": "text", "id": "user_identifier", "name": "identifier", "testid": "auth-identifier",
     "label": "Username or email", "box": [420, 180, 440, 36], "visible": True},
    {"tag": "input", "type": "password", "id": "user_secret", "name": "password", "testid": "auth-secret",
     "label": "Password", "box": [420, 240, 440, 36], "visible": True},
    {"tag": "button", "type": "submit", "id": "auth-continue", "testid": "auth-continue", "box": [640, 360, 80, 36],
     "visible": True},
    {"tag": "button", "type": "button", "box": [420, 360, 200, 36], "visible": True},
]}


def observe(snapshot: str, dom: dict, url: str = "http://app.test/login") -> Observation:
    nodes = parse_snapshot(snapshot)
    obs = Observation(observation_id="obs_test", url=url, title="Login")
    elements, forms, dialogs, headings, text = InventoryBuilder("obs_test", (1280, 800)).build(nodes, dom)
    obs.elements, obs.forms, obs.dialogs, obs.headings, obs.visible_text = elements, forms, dialogs, headings, text
    obs.fingerprint = fingerprint(url, "Login", elements, forms)
    return obs


def step(line: str):
    return StepParser().parse(line, ParseContext()).steps[0]


def test_snapshot_parser_structure_and_frames() -> None:
    nodes = parse_snapshot(LOGIN_V1)
    roles = [n.role for n in nodes]
    assert "form" in roles and roles.count("textbox") == 2
    textbox = next(n for n in nodes if n.role == "textbox")
    assert textbox.props["placeholder"] == "you@example.com"
    assert textbox.box is not None and textbox.box.width == 440
    # "f3e.." refs on a top-level page must not be mistaken for iframe content
    assert all(n.frame is None for n in parse_snapshot(LOGIN_V2))


def test_inventory_merges_dom_attributes_and_context() -> None:
    obs = observe(LOGIN_V1, DOM_V1)
    email = next(e for e in obs.elements if e.name == "Email address")
    assert email.test_id == "login-email" and email.context.form_name == "Login form"
    assert email.semantic_purpose == "username_or_email"
    assert next(e for e in obs.elements if e.name == "Password").semantic_purpose == "password"
    assert obs.forms[0].purpose == "login"


def test_grounding_prefers_semantics_over_dom_order() -> None:
    obs = observe(LOGIN_V1, DOM_V1)
    grounder = ElementGrounder()
    assert grounder.ground(step("Enter username as abc"), obs).name == "Email address"
    assert grounder.ground(step("Enter password as abc"), obs).name == "Password"
    decision = grounder.ground(step("Click on the Sign In button"), obs)
    assert decision.name == "Sign In" and decision.decision in ("execute", "execute_enhanced")


def test_grounding_on_changed_ui_avoids_decoys() -> None:
    obs = observe(LOGIN_V2, DOM_V2)
    grounder = ElementGrounder()
    form_ref = next(e for e in obs.elements if e.name == "Password").context.form_ref
    s = step("Click on the Sign In button")
    s.notes.append("submits_form_group:1")
    decision = grounder.ground(s, obs, GroundingContext(previous_form_ref=form_ref))
    assert decision.name == "Log in", decision.candidates


def test_memory_strategy_regrounds_only_when_it_still_matches() -> None:
    v1, v2 = observe(LOGIN_V1, DOM_V1), observe(LOGIN_V2, DOM_V2)
    submit = next(e for e in v1.elements if e.name == "Sign In")
    strategy = LocatorStrategy.from_element(submit, v1)
    assert resolve_strategy(strategy, v1).element is not None
    stale = resolve_strategy(strategy, v2)
    assert stale.element is None or stale.quality < 0.7  # must not silently point at a decoy
    assert 'internal:role=button[name="Sign In"s]' in strategy.selectors()


def test_password_is_not_confused_with_confirm_field() -> None:
    snapshot = """- form "Registration form" [ref=e1] [box=0,0,800,600]:
  - textbox "Password" [ref=e2] [box=10,10,300,30]
  - textbox "Confirm password" [ref=e3] [box=10,60,300,30]
"""
    dom = {"elements": [{"tag": "input", "type": "password", "box": [10, 10, 300, 30], "visible": True},
                        {"tag": "input", "type": "password", "box": [10, 60, 300, 30], "visible": True}]}
    obs = observe(snapshot, dom, "http://app.test/register")
    grounder = ElementGrounder()
    assert grounder.ground(step("Enter password as Secret1!"), obs).name == "Password"
    assert grounder.ground(step("Enter confirm password as Secret1!"), obs).name == "Confirm password"


def test_text_similarity_and_badges() -> None:
    assert similarity("Login", "Log in") >= 0.9
    assert similarity("cart", "Cart") == 1.0
    assert strip_badge("Cart (3)") == "Cart"
    assert similarity("Sign In", "Sign In") > similarity("Sign In", "Sign in with SSO")


def test_policy_domain_allowlist_and_redirects() -> None:
    env = EnvironmentPolicy(allowed_domains=["app.test", ".example.com"])
    assert host_allowed("https://app.test/x", env.allowed_domains)
    assert host_allowed("https://api.example.com/", env.allowed_domains)
    assert not host_allowed("https://evil.test/", env.allowed_domains)
    assert not host_allowed("javascript:alert(1)", env.allowed_domains)
    policy = PolicyEngine(env)
    decision = policy.evaluate(ActionRequest(step=step("Go to https://evil.test/steal"), url="https://evil.test/steal"))
    assert not decision.allowed and "domain_allowlist" in decision.rules
    assert not policy.check_post_navigation("https://evil.test/").allowed


def test_policy_blocks_secret_exfiltration_and_captcha() -> None:
    obs = observe(LOGIN_V1, DOM_V1)
    policy = PolicyEngine(EnvironmentPolicy(allowed_domains=["app.test"]))
    search_like = next(e for e in obs.elements if e.name == "Email address")
    s = step("Enter password as hunter2")
    blocked = policy.evaluate(ActionRequest(step=s, element=search_like, observation=obs, confidence=0.99,
                                            value_sensitive=True))
    assert not blocked.allowed and "sensitive_value_placement" in blocked.rules
    password = next(e for e in obs.elements if e.name == "Password")
    assert policy.evaluate(ActionRequest(step=s, element=password, observation=obs, confidence=0.99,
                                         value_sensitive=True)).allowed
    captcha = password.model_copy(update={"role": "checkbox", "name": "I'm not a robot"})
    decision = policy.evaluate(ActionRequest(step=step("Check the I'm not a robot checkbox"), element=captcha,
                                             observation=obs, confidence=0.99))
    assert not decision.allowed and "captcha_handoff" in decision.rules


def test_policy_requires_approval_for_high_risk() -> None:
    obs = observe(LOGIN_V1, DOM_V1)
    button = next(e for e in obs.elements if e.name == "Sign In").model_copy(update={"name": "Delete account"})
    s = step("Click Delete account")
    policy = PolicyEngine(EnvironmentPolicy(allowed_domains=["app.test"]))
    decision = policy.evaluate(ActionRequest(step=s, element=button, observation=obs, confidence=0.99))
    assert decision.requires_approval and decision.risk == "high"
    policy.approved.add(policy.approval_key(s, button))
    assert policy.evaluate(ActionRequest(step=s, element=button, observation=obs, confidence=0.99)).allowed


def test_low_confidence_is_never_executed() -> None:
    obs = observe(LOGIN_V1, DOM_V1)
    policy = PolicyEngine(EnvironmentPolicy(allowed_domains=["app.test"]))
    el = next(e for e in obs.elements if e.name == "Sign In")
    decision = policy.evaluate(ActionRequest(step=step("Click Sign In"), element=el, observation=obs, confidence=0.4))
    assert not decision.allowed and "confidence_gate" in decision.rules


def _box(x: float, y: float, w: float = 36, h: float = 18) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=w, height=h)


def _input(element_id: str, name: str, x: float, y: float, *, value: str | None = None) -> InventoryElement:
    return InventoryElement(element_id=element_id, ref=element_id, role="textbox", name=name, tag="input",
                            input_type="text", editable=True, visible=True, bbox=_box(x, y),
                            value_preview=value if value is not None else name)


def test_range_value_is_the_input_under_the_header_not_the_header() -> None:
    headers = [_input(letter, letter, 40 + i * 48, 200) for i, letter in enumerate("ABCDEFGHIJK")]
    values = [_input(f"v{letter}", "", 40 + i * 48, 228, value="100.000" if letter in "AK" else "")
              for i, letter in enumerate("ABCDEFGHIJK")]
    numbers = [_input(f"n{n}", str(n), 40 + (n - 1) * 48, 520) for n in range(1, 12)]
    under = [_input(f"u{n}", "", 40 + (n - 1) * 48, 548, value="0.000" if n == 1 else "") for n in range(1, 12)]
    horizontal = InventoryElement(element_id="hdrop", ref="hdrop", role="combobox", name="Horizontal", tag="select",
                                  visible=True, bbox=_box(40, 470, 80, 22))
    obs = Observation(observation_id="grid", elements=[*headers, *values, horizontal, *numbers, *under])
    grounder = ElementGrounder()

    under_h = step('Click on the second row input field under column "H" and enter "14.999"')
    decision = grounder.ground(under_h, obs, GroundingContext())
    assert decision.element_id == "vH"
    assert decision.confidence >= 0.9

    renamed = _input("n1", "01", 40, 520, value="01")
    obs.elements = [el for el in obs.elements if el.element_id != "n1"] + [renamed]
    below = step('Click on the input field below "01" and enter "10.000"')
    assert grounder.ground(below, obs, GroundingContext()).element_id == "u1"

    header = step('Click on the input field "1", clear the existing value, and enter "01" below "Horizontal" dropdown')
    fresh = Observation(observation_id="grid2", elements=[*headers, *values, horizontal, *numbers, *under])
    assert grounder.ground(header, fresh, GroundingContext()).element_id == "n1"
    assert under_h.action == ActionType.FILL


def test_redaction() -> None:
    register_sensitive("Sup3rSecret!")
    assert MASK in redact_text("await page.fill('Sup3rSecret!')")
    assert redact_value({"password": "x1", "nested": {"api_key": "abc"}}) == {"password": MASK, "nested": {"api_key": MASK}}
    assert MASK in redact_text("Authorization: Bearer abc.def.ghi")
    assert MASK in redact_text("card 4111 1111 1111 1111")


def test_route_normalization_and_fingerprint_stability() -> None:
    assert normalize_route("https://x.test/users/123/orders/9f8e7d6c5b4a3f2e1d0c") == "/users/:id/orders/:hash"
    assert normalize_route("https://x.test/users/456") == normalize_route("https://x.test/users/123")
    a, b = observe(LOGIN_V1, DOM_V1), observe(LOGIN_V2, DOM_V2)
    assert a.fingerprint.structure_hash != b.fingerprint.structure_hash
    assert a.fingerprint.similarity(a.fingerprint) == 1.0


def test_failure_classification_separates_product_failures() -> None:
    assertion = classify_failure(category=FailureCategory.ASSERTION_FAILURE, error="x", decision=None, observation=None)
    assert assertion.product_failure and not assertion.healable
    stale = classify_failure(category=FailureCategory.STALE_ELEMENT_REFERENCE, error="Ref e9 not found", decision=None,
                             observation=None)
    assert stale.healable and not stale.product_failure


def test_oracle_never_passes_without_assertions() -> None:
    passed = StepResult(step_id="s1", index=1, original_text="Click", action="click", status=StepStatus.PASSED)
    outcome, _ = evaluate_outcome([passed], has_assertions=False)
    assert outcome == TestOutcome.INCONCLUSIVE
    asserted = StepResult(step_id="s2", index=2, original_text="Verify", action="assert", status=StepStatus.PASSED,
                          assertions=[AssertionResult(assertion_id="a", type="TEXT_VISIBLE", status="PASSED")])
    assert evaluate_outcome([passed, asserted], has_assertions=True)[0] == TestOutcome.PASSED
    failed = asserted.model_copy(update={"status": StepStatus.FAILED})
    assert evaluate_outcome([passed, failed], has_assertions=True)[0] == TestOutcome.FAILED


def test_embeddings_capture_intent_similarity() -> None:
    a = embed("fill username field", purpose="username_or_email")
    b = embed("fill username", purpose="username_or_email")
    c = embed("click delete button", purpose="delete")
    assert cosine(a, b) > cosine(a, c)
