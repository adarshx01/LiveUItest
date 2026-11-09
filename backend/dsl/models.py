"""Action DSL and intermediate test specification (spec sections 7.3 / 7.4).

Natural-language test cases compile into these strictly validated models. They
are pure data: no step can carry executable code, so neither the parser nor an
LLM can smuggle scripts into an execution plan.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ActionType(StrEnum):
    NAVIGATE = "navigate"
    OPEN_PAGE = "open_page"  # ensure a semantically described page is open
    GO_BACK = "go_back"
    RELOAD = "reload"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    HOVER = "hover"
    FILL = "fill"
    TYPE = "type"
    CLEAR = "clear"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    TOGGLE = "toggle"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    DRAG = "drag"
    DROP = "drop"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    WAIT = "wait"
    SWITCH_TAB = "switch_tab"
    SWITCH_FRAME = "switch_frame"
    ACCEPT_DIALOG = "accept_dialog"
    DISMISS_DIALOG = "dismiss_dialog"
    ASSERT = "assert"
    CAPTURE_SCREENSHOT = "capture_screenshot"
    CAPTURE_VALUE = "capture_value"
    FILL_FORM = "fill_form"  # semantic multi-field form fill from test data
    CONDITIONAL = "conditional"


# Actions that only read state.
READ_ONLY_ACTIONS = {
    ActionType.ASSERT,
    ActionType.CAPTURE_SCREENSHOT,
    ActionType.CAPTURE_VALUE,
    ActionType.WAIT,
    ActionType.HOVER,
    ActionType.SCROLL,
    ActionType.SWITCH_TAB,
    ActionType.SWITCH_FRAME,
}

# Actions that need an element target resolved on the live page.
TARGETED_ACTIONS = {
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.RIGHT_CLICK,
    ActionType.HOVER,
    ActionType.FILL,
    ActionType.TYPE,
    ActionType.CLEAR,
    ActionType.SELECT,
    ActionType.CHECK,
    ActionType.UNCHECK,
    ActionType.TOGGLE,
    ActionType.DRAG,
    ActionType.DROP,
    ActionType.UPLOAD,
    ActionType.DOWNLOAD,
}


class AssertionType(StrEnum):
    URL_EQUALS = "URL_EQUALS"
    URL_CONTAINS = "URL_CONTAINS"
    PAGE_TITLE_EQUALS = "PAGE_TITLE_EQUALS"
    PAGE_TITLE_CONTAINS = "PAGE_TITLE_CONTAINS"
    TEXT_VISIBLE = "TEXT_VISIBLE"
    TEXT_NOT_VISIBLE = "TEXT_NOT_VISIBLE"
    ELEMENT_VISIBLE = "ELEMENT_VISIBLE"
    ELEMENT_NOT_VISIBLE = "ELEMENT_NOT_VISIBLE"
    ELEMENT_ENABLED = "ELEMENT_ENABLED"
    ELEMENT_DISABLED = "ELEMENT_DISABLED"
    INPUT_VALUE_EQUALS = "INPUT_VALUE_EQUALS"
    CHECKBOX_CHECKED = "CHECKBOX_CHECKED"
    CHECKBOX_UNCHECKED = "CHECKBOX_UNCHECKED"
    RADIO_SELECTED = "RADIO_SELECTED"
    DROPDOWN_SELECTED = "DROPDOWN_SELECTED"
    ELEMENT_COUNT = "ELEMENT_COUNT"
    PAGE_LOADED = "PAGE_LOADED"
    NAVIGATION_OCCURRED = "NAVIGATION_OCCURRED"
    DOWNLOAD_STARTED = "DOWNLOAD_STARTED"
    FILE_EXISTS = "FILE_EXISTS"
    DIALOG_VISIBLE = "DIALOG_VISIBLE"
    NETWORK_RESPONSE = "NETWORK_RESPONSE"
    APPLICATION_STATE = "APPLICATION_STATE"
    VISUAL_STATE = "VISUAL_STATE"
    ANY_OF = "ANY_OF"  # composite: passes when any child passes
    ALL_OF = "ALL_OF"  # composite: passes when every child passes


class TargetSpec(BaseModel):
    """Semantic description of an element. Never a stored selector or ref."""

    model_config = ConfigDict(extra="forbid")

    intent: str = ""
    exact_name: str | None = None
    preferred_roles: list[str] = Field(default_factory=list)
    semantic_hints: list[str] = Field(default_factory=list)
    element_kind: str | None = None  # button, link, field, dropdown, checkbox, ...
    input_type: str | None = None  # password, email, tel, ...
    purpose: str | None = None  # classifier vocabulary (username_or_email, password, submit_login...)
    context_hint: str | None = None  # "login form", "dialog", "header"...
    ordinal: int | None = None  # 1-based; -1 = last
    position_hint: str | None = None  # "top right", "left"...
    visual_hint: str | None = None  # "red circle", "gear icon"
    frame_hint: str | None = None
    url: str | None = None
    option_text: str | None = None  # option to locate inside a dropdown ("dropdown containing India")
    # the name is data (a patient, customer or record name): matched literally, never typo-repaired, so
    # "Abcd Xy" never passes for "Abc Xy". Labels of controls stay typo-tolerant ("Data Perid" -> "Data period").
    literal: bool = False
    # spatial relation to another element: "next to Upload Documents" -> relation="next_to", anchor="Upload Documents"
    relation: str | None = None  # right_of, left_of, below, above, next_to, near, inside
    anchor: str | None = None


class ValueSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["literal", "variable", "secret", "generated", "previous", "none"] = "literal"
    value: str | None = None
    reference: str | None = None
    sensitive: bool = False
    generator: str | None = None  # e.g. "email", "full_name", "phone" for generated data
    values: list[str] = Field(default_factory=list)  # multi-select / multi-option

    def display(self) -> str:
        if self.sensitive:
            return "••••••"
        if self.source in ("variable", "secret") and self.reference:
            return "{{" + self.reference + "}}"
        if self.values:
            return ", ".join(self.values)
        return self.value or ""


class AssertionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AssertionType
    description: str = ""
    expected: Any = None
    expected_ref: ValueSpec | None = None  # expectation sourced from a variable/previous value
    target: TargetSpec | None = None
    negate: bool = False
    match: Literal["exact", "contains", "regex", "case_insensitive", "semantic"] = "contains"
    children: list[AssertionSpec] = Field(default_factory=list)
    derived: bool = False  # expectation inferred from context, not stated explicitly
    timeout_ms: int = 5000


class ConditionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["variable_equals", "text_visible", "element_visible", "url_contains", "assertion", "always"]
    variable: str | None = None
    expected: str | None = None
    target: TargetSpec | None = None
    # kind "assertion": the condition is a full check ("the checkbox HP is checked", "the total number of
    # range columns is 11") evaluated by the assertion engine, not a text search
    assertion: AssertionSpec | None = None
    negate: bool = False
    description: str = ""


class WaitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seconds: float | None = None
    until_text: str | None = None
    until_text_gone: str | None = None
    until_target: TargetSpec | None = None
    # "wait until the results are loaded": no busy indicator, no pending navigation, DOM no longer changing
    until_stable: bool = False
    timeout_seconds: float = 30.0


class ScrollSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["down", "up", "left", "right"] = "down"
    amount_px: int | None = None
    to: Literal["bottom", "top", "element", "until_text"] | None = None
    text: str | None = None
    max_scrolls: int = 25


class TabSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    which: Literal["new", "latest", "first", "previous", "next", "index", "title", "close_current", "url"] = "latest"
    index: int | None = None
    title: str | None = None


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    index: int
    original_text: str
    action: ActionType
    target: TargetSpec | None = None
    destination: TargetSpec | None = None  # drag/drop destination
    value: ValueSpec | None = None
    key: str | None = None
    assertion: AssertionSpec | None = None
    wait: WaitSpec | None = None
    scroll: ScrollSpec | None = None
    tab: TabSpec | None = None
    condition: ConditionSpec | None = None
    then_steps: list[Step] = Field(default_factory=list)
    else_steps: list[Step] = Field(default_factory=list)
    capture_as: str | None = None
    fields: list[FormFieldSpec] = Field(default_factory=list)  # FILL_FORM
    expected_postcondition: str | None = None
    expects_navigation: bool = False
    expects_download: bool = False
    risk_hint: Literal["read", "input", "submit", "destructive", "financial", "messaging", "navigation"] | None = None
    sensitive: bool = False
    optional: bool = False
    confidence: float = 1.0
    parse_source: Literal["deterministic", "llm", "structured"] = "deterministic"
    notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, value))

    def summary(self) -> str:
        parts = [self.action.value]
        if self.target and (self.target.exact_name or self.target.intent or self.target.url):
            parts.append(repr(self.target.exact_name or self.target.intent or self.target.url))
        if self.value is not None and self.value.source != "none":
            parts.append("= " + self.value.display())
        if self.assertion is not None:
            parts.append(self.assertion.type.value)
        return " ".join(parts)

    def iter_flat(self) -> list[Step]:
        out = [self]
        for child in [*self.then_steps, *self.else_steps]:
            out.extend(child.iter_flat())
        return out


class FormFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str  # full_name, email, phone, country...
    value: ValueSpec
    target: TargetSpec | None = None


class Ambiguity(BaseModel):
    line: str
    reason: str
    step_index: int | None = None
    severity: Literal["info", "warning", "blocking"] = "warning"


class TestSpec(BaseModel):
    """Normalized, versionable intermediate representation of one test case."""

    model_config = ConfigDict(extra="forbid")

    test_name: str
    base_url: str | None = None
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    secret_refs: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    expected_results: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_format: Literal["text", "markdown", "json", "csv", "structured"] = "text"
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    analyzer: dict[str, Any] = Field(default_factory=dict)

    def flat_steps(self) -> list[Step]:
        out: list[Step] = []
        for step in self.steps:
            out.extend(step.iter_flat())
        return out

    def has_verifiable_outcome(self) -> bool:
        return any(s.action == ActionType.ASSERT for s in self.flat_steps())


Step.model_rebuild()
TestSpec.model_rebuild()
