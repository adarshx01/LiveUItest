"""Natural-language test intelligence: ingestion, parsing, analysis, compilation, validation."""

from __future__ import annotations

import pytest

from backend.agents.planner.nl_parser import ParseContext, StepParser
from backend.agents.planner.plan_validator import PlanValidator, blocking
from backend.agents.planner.requirement_analyzer import RequirementAnalyzer
from backend.agents.planner.test_compiler import TestCompiler
from backend.agents.planner.test_ingestion import ingest
from backend.dsl.models import ActionType, AssertionType

SPEC_EXAMPLE = """URL: https://example.com

Test Case: Verify user login

1. Open the login page.
2. Click on the Login button.
3. Enter username as abc.
4. Enter password as abc.
5. Click on the Sign In button.
6. Verify that the user is redirected to the dashboard.
7. Verify that the dashboard displays the logged-in user's name.
"""


def parse(line: str, **ctx_kwargs):
    ctx = ParseContext(**ctx_kwargs)
    return StepParser().parse(line, ctx).steps


@pytest.mark.parametrize(
    "line,action",
    [
        ("Click on the Login button.", ActionType.CLICK),
        ("Enter username as abc.", ActionType.FILL),
        ("Type 'hello' into the search box", ActionType.FILL),
        ("Select India from the country dropdown.", ActionType.SELECT),
        ("Check the 'Remember me' checkbox", ActionType.CHECK),
        ("Uncheck the newsletter checkbox", ActionType.UNCHECK),
        ("Accept the terms and conditions", ActionType.CHECK),
        ("Press Enter", ActionType.PRESS_KEY),
        ("Hover over the Products menu", ActionType.HOVER),
        ("Drag 'Task A' to the Done column", ActionType.DRAG),
        ("Upload resume.pdf to the resume field", ActionType.UPLOAD),
        ("Switch to the new tab", ActionType.SWITCH_TAB),
        ("Accept the alert", ActionType.ACCEPT_DIALOG),
        ("Wait for 2 seconds", ActionType.WAIT),
        ("Scroll to the bottom of the page", ActionType.SCROLL),
        ("Take a screenshot", ActionType.CAPTURE_SCREENSHOT),
        ("Go to https://example.com/login", ActionType.NAVIGATE),
        ("Verify that Dashboard is visible", ActionType.ASSERT),
        ("Double-click the file name", ActionType.DOUBLE_CLICK),
        ("Clear the email field", ActionType.CLEAR),
    ],
)
def test_action_classification(line: str, action: ActionType) -> None:
    steps = parse(line)
    assert steps, line
    assert steps[0].action == action


def test_fill_extracts_semantic_target_and_value() -> None:
    (step,) = parse("Enter username as abc.")
    assert step.target.purpose == "username_or_email"
    assert step.value.value == "abc"
    assert not step.value.sensitive


def test_password_is_sensitive_and_secret_reference_detected() -> None:
    (step,) = parse("Enter password as {{test_password}}.")
    assert step.value.source == "secret"
    assert step.value.reference == "test_password"
    assert step.value.sensitive and step.sensitive
    (literal,) = parse("Enter password as hunter2")
    assert literal.value.sensitive


def test_value_first_phrasing() -> None:
    (step,) = parse("Enter abc as the username")
    assert step.target.purpose == "username_or_email" and step.value.value == "abc"
    (step,) = parse("Type 'hello world' into the search box")
    assert step.target.purpose == "search" and step.value.value == "hello world"


def test_select_option_and_target() -> None:
    (step,) = parse("Select India from the country dropdown.")
    assert step.target.purpose == "country"
    assert step.target.option_text == "India"
    assert "combobox" in step.target.preferred_roles


def test_select_list_maps_options_onto_dropdowns() -> None:
    steps = parse("Select India, Kerala, and Kochi")
    assert [s.target.purpose for s in steps] == ["country", "state", "city"]
    assert [s.value.value for s in steps] == ["India", "Kerala", "Kochi"]


def test_compound_sentences_are_split() -> None:
    steps = parse("Enter username as abc and password as xyz")
    assert [s.target.purpose for s in steps] == ["username_or_email", "password"]
    assert steps[1].value.sensitive
    steps = parse("Click Login and then verify Dashboard is visible")
    assert [s.action for s in steps] == [ActionType.CLICK, ActionType.ASSERT]


def test_click_purpose_is_action_not_field() -> None:
    (step,) = parse("Click on Login.")
    assert step.target.purpose == "submit_login"
    (step,) = parse("Click the second Delete button")
    assert step.target.ordinal == 2 and step.risk_hint == "destructive"


def test_redirect_assertion_is_composite() -> None:
    (step,) = parse("Verify that the user is redirected to the dashboard.")
    assert step.assertion.type == AssertionType.ANY_OF
    kinds = {c.type for c in step.assertion.children}
    assert {AssertionType.URL_CONTAINS, AssertionType.ELEMENT_VISIBLE} <= kinds


def test_page_is_displayed_checks_the_page_not_the_word_the() -> None:
    (step,) = parse("Verify that the page is displayed.")
    assert step.action == ActionType.ASSERT
    assert step.assertion.type == AssertionType.PAGE_LOADED
    assert step.assertion.expected is None


def test_range_grid_value_goes_under_the_column_header() -> None:
    (step,) = parse('Click on the second row input field under column "H" and enter "14.999"')
    assert step.action == ActionType.FILL
    assert step.value.value == "14.999"
    assert step.target.exact_name == "H"
    assert step.target.ordinal == 2
    assert step.target.position_hint == "column_row"

    (below,) = parse('Click on the input field below "01" and enter "10.000"')
    assert below.action == ActionType.FILL
    assert below.target.exact_name == "01"
    assert below.target.position_hint == "below_label"
    assert below.value.value == "10.000"

    (header,) = parse('Click on the input field "1", clear the existing value, and enter "01" '
                      'below "Horizontal" dropdown')
    assert header.action == ActionType.FILL
    assert header.target.exact_name == "1"
    assert header.target.position_hint == "labeled_below"
    assert header.target.context_hint == "Horizontal"
    assert header.value.value == "01"


def test_named_control_is_selected_is_a_radio_not_the_word_selected() -> None:
    (step,) = parse('Verify that "Same Period Last Year" is selected')
    assert step.assertion.type == AssertionType.RADIO_SELECTED
    assert step.assertion.target.exact_name == "Same Period Last Year"


def test_verify_else_runs_the_recovery_only_as_the_else_branch() -> None:
    (step,) = parse('Verify that "10" is displayed in the "Horizontal" field else click the "Horizontal" '
                    'dropdown and choose the "10" option')
    assert step.action == ActionType.CONDITIONAL
    assert step.condition.expected == "10"
    assert step.then_steps == []
    assert step.else_steps
    assert step.else_steps[-1].action == ActionType.SELECT


def test_value_word_is_not_part_of_the_expected_text() -> None:
    (step,) = parse('Verify that the value "100.000" is displayed in the Horizontal Range (%) input '
                    'under column "11"')
    assert step.assertion.type == AssertionType.TEXT_VISIBLE
    assert step.assertion.expected == "100.000"


def test_value_in_a_field_inside_a_section_is_an_assertion() -> None:
    (step,) = parse('Verify that "Percentage" is displayed in the "Range Type" field in the Vertical Axis section')
    assert step.action == ActionType.ASSERT
    assert step.assertion.type == AssertionType.TEXT_VISIBLE
    assert step.assertion.expected == "Percentage"
    assert step.assertion.target.context_hint == "Vertical Axis section"
    assert "Range Type" in (step.assertion.target.exact_name or step.assertion.target.intent)


def test_derived_user_name_expectation() -> None:
    parser = StepParser()
    ctx = ParseContext()
    parser.parse("Enter username as abc.", ctx)
    (step,) = parser.parse("Verify that the dashboard displays the logged-in user's name.", ctx).steps
    assert step.assertion.derived is True
    assert step.assertion.expected_ref.value == "abc"


@pytest.mark.parametrize(
    "line,atype",
    [
        ("Verify the URL contains /dashboard", AssertionType.URL_CONTAINS),
        ("Verify the page title is 'Dashboard'", AssertionType.PAGE_TITLE_EQUALS),
        ("Ensure the Submit button is disabled", AssertionType.ELEMENT_DISABLED),
        ("Verify that 3 results are displayed", AssertionType.ELEMENT_COUNT),
        ("Verify the Remember me checkbox is checked", AssertionType.CHECKBOX_CHECKED),
        ("Verify India is selected in the country dropdown", AssertionType.DROPDOWN_SELECTED),
        ("Verify the email field contains jane@x.com", AssertionType.INPUT_VALUE_EQUALS),
        ("The error message 'Invalid credentials' should be displayed", AssertionType.TEXT_VISIBLE),
        ("Verify that 'Welcome, abc' is not displayed", AssertionType.TEXT_NOT_VISIBLE),
        ("User should see the order confirmation", AssertionType.TEXT_VISIBLE),
    ],
)
def test_assertion_types(line: str, atype: AssertionType) -> None:
    (step,) = parse(line)
    assert step.action == ActionType.ASSERT
    assert step.assertion.type == atype


def test_conditional_inline() -> None:
    (step,) = parse("If a cookie banner is visible, click 'Accept all'")
    assert step.action == ActionType.CONDITIONAL
    assert step.condition.kind == "element_visible"
    assert step.then_steps[0].action == ActionType.CLICK


def test_ingest_plain_text_example() -> None:
    (case,) = ingest(SPEC_EXAMPLE)
    assert case.base_url == "https://example.com"
    assert case.name == "Verify user login"
    assert len(case.lines) == 7


def test_ingest_markdown_with_data_and_if_blocks() -> None:
    content = """# Test: Account type

## Test Data
- account_type: Business
- email: jane@example.com

## Steps
1. Go to https://app.example.com/signup
2. Enter email as {{email}}
3. IF the account type is Individual:
    Select Individual
   ELSE:
    Select Business
4. Click Continue

## Expected
- The company name field is visible
"""
    (case,) = ingest(content)
    assert case.variables == {"account_type": "Business", "email": "jane@example.com"}
    assert case.lines[2].condition == "the account type is Individual"
    assert [l.text for l in case.lines[2].then_lines] == ["Select Individual"]
    assert [l.text for l in case.lines[2].else_lines] == ["Select Business"]
    assert case.expected_results == ["The company name field is visible"]


def test_ingest_json_suite_and_csv() -> None:
    suite = {"base_url": "https://x.test", "tests": [{"name": "a", "steps": ["Click Login"]},
                                                     {"name": "b", "steps": ["Click Logout"]}]}
    cases = ingest(suite)
    assert [c.name for c in cases] == ["a", "b"] and all(c.base_url == "https://x.test" for c in cases)
    csv_text = "test_name,url,step,expected\nlogin,https://x.test,Click Login,Dashboard is visible\n"
    (case,) = ingest(csv_text)
    assert case.base_url == "https://x.test"
    assert [l.text for l in case.lines] == ["Click Login", "Verify that Dashboard is visible"]


async def test_analyzer_compiler_validator_end_to_end() -> None:
    (case,) = ingest(SPEC_EXAMPLE)
    spec = await RequirementAnalyzer().analyze(case)
    assert not [a for a in spec.ambiguities if a.severity == "blocking"]
    compiled = TestCompiler().compile(spec)
    flat = compiled.flat_steps()
    assert flat[0].action == ActionType.NAVIGATE and "implicit_navigation" in flat[0].notes
    assert [s.index for s in flat] == list(range(1, len(flat) + 1))
    issues = PlanValidator().validate(compiled, available_variables=set(), available_secrets=set())
    assert not blocking(issues)


async def test_validator_blocks_missing_secret() -> None:
    (case,) = ingest("URL: https://x.test\n1. Enter password as {{admin_password}}\n2. Verify Welcome is visible")
    spec = TestCompiler().compile(await RequirementAnalyzer().analyze(case))
    issues = PlanValidator().validate(spec, available_variables=set(), available_secrets=set())
    assert any(i.code == "MISSING_SECRET" for i in blocking(issues))
    issues = PlanValidator().validate(spec, available_variables=set(), available_secrets={"admin_password"})
    assert not blocking(issues)


async def test_unparseable_instruction_is_reported_not_guessed() -> None:
    (case,) = ingest("URL: https://x.test\n1. Frobnicate the quux thoroughly\n2. Verify Done is visible")
    spec = await RequirementAnalyzer().analyze(case)
    assert any(a.severity == "blocking" for a in spec.ambiguities)


def test_gateway_status_on_a_real_screen_is_not_an_error_page() -> None:
    from backend.agents.observation.page_state_analyzer import analyze_page_state
    from backend.dsl.observation import FormInfo, InventoryElement

    def control(role: str, name: str) -> InventoryElement:
        return InventoryElement(element_id=name, role=role, name=name, visible=True)

    sign_in = analyze_page_state(
        url="https://hl-demo.demohealthlink.in/sign-in", title="healthlink", headings=["Sign in"],
        elements=[control("textbox", "Email"), control("textbox", "Password"), control("button", "Login")],
        forms=[FormInfo(form_id="login", purpose="login")], dialogs=[], signals={"readyState": "complete"},
        dom={"httpStatus": 403})
    assert sign_in.http_status == 403
    assert sign_in.error_page is False
    assert sign_in.page_kind == "login"

    denied = analyze_page_state(
        url="https://hl-demo.demohealthlink.in/sign-in", title="403 Forbidden", headings=["Access denied"],
        elements=[control("link", "Home")], forms=[], dialogs=[], signals={"readyState": "complete"},
        dom={"httpStatus": 403})
    assert denied.error_page is True
    assert denied.page_kind == "error"
