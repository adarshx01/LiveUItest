"""Typed page-observation models (spec section 8)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def intersection(self, other: BoundingBox) -> float:
        dx = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        dy = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        return dx * dy if dx > 0 and dy > 0 else 0.0

    def iou(self, other: BoundingBox) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def offset(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(x=self.x + dx, y=self.y + dy, width=self.width, height=self.height)


class SnapshotNode(BaseModel):
    """One node of the Playwright MCP aria snapshot."""

    ref: str | None = None
    role: str
    name: str = ""
    text: str = ""  # inline text after ":" for leaf nodes
    attrs: dict[str, Any] = Field(default_factory=dict)  # checked, disabled, expanded, level...
    props: dict[str, str] = Field(default_factory=dict)  # /url, /placeholder...
    box: BoundingBox | None = None
    depth: int = 0
    frame: str | None = None  # "f1" for iframe content
    parent_index: int | None = None
    index: int = 0
    children: list[int] = Field(default_factory=list)


class ElementContext(BaseModel):
    form_ref: str | None = None
    form_name: str | None = None
    dialog_ref: str | None = None
    dialog_name: str | None = None
    landmark: str | None = None  # banner, navigation, main, contentinfo, complementary, region
    landmark_name: str | None = None
    section_heading: str | None = None
    fieldset_legend: str | None = None
    list_ref: str | None = None
    row_text: str | None = None
    frame: str | None = None
    ancestors: list[str] = Field(default_factory=list)  # "role:name" path


class InventoryElement(BaseModel):
    """An interactive or meaningful element in one observation.

    ``element_id`` and ``ref`` are only valid for the observation that produced them.
    """

    element_id: str
    ref: str | None = None
    role: str
    name: str = ""
    text: str = ""
    label: str | None = None
    placeholder: str | None = None
    input_type: str | None = None
    tag: str | None = None
    dom_id: str | None = None
    name_attr: str | None = None
    test_id: str | None = None
    autocomplete: str | None = None
    title: str | None = None
    href: str | None = None
    css_selector: str | None = None
    classes: str | None = None
    visible: bool = True
    enabled: bool = True
    editable: bool = False
    checked: bool | None = None
    selected: bool | None = None
    expanded: bool | None = None
    required: bool = False
    readonly: bool = False
    value_preview: str | None = None  # redacted for sensitive fields
    options: list[str] = Field(default_factory=list)
    selected_options: list[str] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    in_viewport: bool = True
    obstructed: bool = False
    clickable_hint: bool = False
    context: ElementContext = Field(default_factory=ElementContext)
    nearby_text: list[str] = Field(default_factory=list)
    semantic_purpose: str | None = None
    purpose_confidence: float = 0.0
    source: Literal["aria", "dom", "aria+dom"] = "aria"
    snapshot_index: int | None = None

    def display(self) -> str:
        label = self.name or self.label or self.placeholder or self.text or self.dom_id or ""
        return f'{self.role} "{label}"'.strip()


class FormInfo(BaseModel):
    form_id: str
    ref: str | None = None
    name: str | None = None
    field_ids: list[str] = Field(default_factory=list)
    submit_ids: list[str] = Field(default_factory=list)
    purpose: str | None = None  # login, registration, search, contact, checkout, profile
    fingerprint: str = ""
    validation_messages: list[str] = Field(default_factory=list)


class DialogInfo(BaseModel):
    ref: str | None = None
    name: str = ""
    modal: bool = True
    element_ids: list[str] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    kind: Literal["modal", "cookie_banner", "alert", "popover", "unknown"] = "modal"


class ModalState(BaseModel):
    """Native browser modal states reported by Playwright MCP."""

    kind: Literal["alert", "confirm", "prompt", "beforeunload", "file_chooser", "unknown"]
    message: str = ""
    raw: str = ""


class TabInfo(BaseModel):
    index: int
    title: str = ""
    url: str = ""
    current: bool = False


class PageState(BaseModel):
    loading: bool = False
    captcha_detected: bool = False
    captcha_evidence: list[str] = Field(default_factory=list)
    error_page: bool = False
    http_status: int | None = None
    blocking_overlay: bool = False
    overlay_names: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    invalid_fields: list[str] = Field(default_factory=list)  # native constraint validation state
    page_kind: str | None = None  # login, dashboard, form, list, error, challenge...
    authenticated_hint: bool | None = None
    injection_findings: list[str] = Field(default_factory=list)
    console_errors: int = 0


class PageFingerprint(BaseModel):
    route_pattern: str
    host: str
    structure_hash: str
    signature: list[str] = Field(default_factory=list)  # role:name tokens of key elements
    form_hashes: dict[str, str] = Field(default_factory=dict)
    title_pattern: str = ""

    def similarity(self, other: PageFingerprint) -> float:
        if not self.signature and not other.signature:
            return 1.0 if self.route_pattern == other.route_pattern else 0.0
        a, b = set(self.signature), set(other.signature)
        jaccard = len(a & b) / max(1, len(a | b))
        route_bonus = 0.1 if self.route_pattern == other.route_pattern else 0.0
        return min(1.0, jaccard * 0.9 + route_bonus)


class Observation(BaseModel):
    observation_id: str
    url: str = ""
    title: str = ""
    viewport: tuple[int, int] = (1280, 800)
    tabs: list[TabInfo] = Field(default_factory=list)
    active_tab: int = 0
    modal_state: ModalState | None = None
    elements: list[InventoryElement] = Field(default_factory=list)
    forms: list[FormInfo] = Field(default_factory=list)
    dialogs: list[DialogInfo] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)
    page_state: PageState = Field(default_factory=PageState)
    fingerprint: PageFingerprint | None = None
    snapshot_artifact: str | None = None
    screenshot_artifact: str | None = None
    snapshot_text: str = ""  # kept in-process only; never placed in graph state
    timestamp: float = 0.0
    dom_probe_ok: bool = False

    def element(self, element_id: str) -> InventoryElement | None:
        for el in self.elements:
            if el.element_id == element_id:
                return el
        return None

    def by_ref(self, ref: str) -> InventoryElement | None:
        for el in self.elements:
            if el.ref == ref:
                return el
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "url": self.url,
            "title": self.title,
            "elements": len(self.elements),
            "forms": [f.name or f.purpose for f in self.forms],
            "dialogs": [d.name for d in self.dialogs],
            "modal_state": self.modal_state.kind if self.modal_state else None,
            "page_kind": self.page_state.page_kind,
            "captcha": self.page_state.captcha_detected,
            "route": self.fingerprint.route_pattern if self.fingerprint else None,
        }
