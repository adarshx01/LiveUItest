"""Locator strategies (spec section 10).

A ``LocatorStrategy`` is the durable, semantic description of how an element
was found: role + accessible name, label, placeholder, test id, stable
attributes and structural context. It never contains snapshot refs or
coordinates, so it can be stored in memory and *re-grounded* against a fresh
observation before every replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from backend.agents.grounding.text_match import similarity
from backend.agents.planner.lexicon import normalize_text
from backend.dsl.observation import InventoryElement, Observation

FIELD_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}


def _q(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class LocatorStrategy(BaseModel):
    role: str
    name: str = ""
    label: str | None = None
    placeholder: str | None = None
    test_id: str | None = None
    dom_id: str | None = None
    name_attr: str | None = None
    css: str | None = None
    input_type: str | None = None
    purpose: str | None = None
    form_context: str | None = None
    dialog_context: str | None = None
    section_heading: str | None = None
    landmark: str | None = None
    frame_context: str | None = None
    row_text: str | None = None
    ordinal_in_context: int | None = None

    @classmethod
    def from_element(cls, el: InventoryElement, observation: Observation | None = None) -> LocatorStrategy:
        strategy = cls(
            role=el.role, name=el.name or "", label=el.label, placeholder=el.placeholder, test_id=el.test_id,
            dom_id=el.dom_id, name_attr=el.name_attr, css=el.css_selector, input_type=el.input_type,
            purpose=el.semantic_purpose, form_context=el.context.form_name, dialog_context=el.context.dialog_name,
            section_heading=el.context.section_heading, landmark=el.context.landmark, frame_context=el.context.frame,
            row_text=(el.context.row_text or None) if el.context.row_text and len(el.context.row_text) < 160 else None,
        )
        if observation is not None:
            same = [e for e in observation.elements if e.role == el.role and e.name == el.name and e.visible]
            if len(same) > 1:
                strategy.ordinal_in_context = next((i for i, e in enumerate(same) if e.element_id == el.element_id), None)
        return strategy

    def identity_key(self) -> str:
        return "|".join([self.role, normalize_text(self.name), normalize_text(self.label or ""), self.test_id or "",
                         self.dom_id or "", normalize_text(self.form_context or "")])

    def selectors(self) -> list[str]:
        """Playwright selectors from most semantic to most structural."""

        out: list[str] = []
        scope = f'internal:role=form[name="{_q(self.form_context)}"s] >> ' if self.form_context else ""
        if self.name:
            out.append(f'internal:role={self.role}[name="{_q(self.name)}"s]')
            if scope:
                out.append(scope + f'internal:role={self.role}[name="{_q(self.name)}"s]')
        if self.label:
            out.append(f'internal:label="{_q(self.label)}"s')
        if self.placeholder:
            out.append(f'internal:attr=[placeholder="{_q(self.placeholder)}"s]')
        if self.test_id:
            out.append(f'internal:testid=[data-testid="{_q(self.test_id)}"s]')
        if self.dom_id:
            out.append(f"#{self.dom_id}" if self.dom_id.replace("-", "").replace("_", "").isalnum()
                       else f'[id="{_q(self.dom_id)}"]')
        if self.name_attr:
            out.append(f'[name="{_q(self.name_attr)}"]')
        if self.css:
            out.append(self.css)
        return list(dict.fromkeys(out))

    def describe(self) -> str:
        bits = [f'{self.role} "{self.name}"' if self.name else self.role]
        if self.label and self.label != self.name:
            bits.append(f"label={self.label!r}")
        if self.test_id:
            bits.append(f"testid={self.test_id}")
        if self.form_context:
            bits.append(f"in form {self.form_context!r}")
        if self.dialog_context:
            bits.append(f"in dialog {self.dialog_context!r}")
        return " ".join(bits)


@dataclass
class StrategyResolution:
    element: InventoryElement | None
    quality: float
    matched_on: list[str] = field(default_factory=list)
    ambiguous: bool = False
    candidates: int = 0


WEIGHTS = {"test_id": 0.30, "role": 0.10, "name": 0.24, "label": 0.14, "placeholder": 0.07, "dom_id": 0.10,
           "name_attr": 0.07, "css": 0.04, "form_context": 0.06, "dialog_context": 0.05, "section_heading": 0.04,
           "purpose": 0.08, "input_type": 0.05}


def _score(strategy: LocatorStrategy, el: InventoryElement) -> tuple[float, float, list[str]]:
    got = 0.0
    total = 0.0
    matched: list[str] = []

    def check(key: str, present: bool, ok: float) -> None:
        nonlocal got, total
        if not present:
            return
        total += WEIGHTS[key]
        got += WEIGHTS[key] * ok
        if ok >= 0.9:
            matched.append(key)

    check("role", True, 1.0 if el.role == strategy.role else (0.6 if {el.role, strategy.role} <= FIELD_ROLES else 0.0))
    check("name", bool(strategy.name), 1.0 if el.name == strategy.name else similarity(strategy.name, el.name or ""))
    check("label", bool(strategy.label), 1.0 if el.label == strategy.label else similarity(strategy.label or "",
                                                                                          el.label or el.name or ""))
    check("placeholder", bool(strategy.placeholder), 1.0 if el.placeholder == strategy.placeholder else 0.0)
    check("test_id", bool(strategy.test_id), 1.0 if el.test_id == strategy.test_id else 0.0)
    check("dom_id", bool(strategy.dom_id), 1.0 if el.dom_id == strategy.dom_id else 0.0)
    check("name_attr", bool(strategy.name_attr), 1.0 if el.name_attr == strategy.name_attr else 0.0)
    check("css", bool(strategy.css), 1.0 if el.css_selector == strategy.css else 0.0)
    check("form_context", bool(strategy.form_context),
          1.0 if el.context.form_name == strategy.form_context else similarity(strategy.form_context or "",
                                                                                el.context.form_name or ""))
    check("dialog_context", bool(strategy.dialog_context),
          1.0 if el.context.dialog_name == strategy.dialog_context else 0.0)
    check("section_heading", bool(strategy.section_heading),
          1.0 if el.context.section_heading == strategy.section_heading else 0.0)
    check("purpose", bool(strategy.purpose), 1.0 if el.semantic_purpose == strategy.purpose else 0.0)
    check("input_type", bool(strategy.input_type), 1.0 if el.input_type == strategy.input_type else 0.0)
    return got, total, matched


def resolve_strategy(strategy: LocatorStrategy, observation: Observation) -> StrategyResolution:
    """Re-ground a remembered strategy against the live page."""

    scored = []
    for el in observation.elements:
        if not el.visible and el.role != "option":
            continue
        got, total, matched = _score(strategy, el)
        if total <= 0:
            continue
        quality = got / total
        if strategy.role and el.role != strategy.role and not ({el.role, strategy.role} <= FIELD_ROLES):
            quality *= 0.5
        scored.append((quality, el, matched))
    if not scored:
        return StrategyResolution(None, 0.0)
    scored.sort(key=lambda item: -item[0])
    best_quality, best, matched = scored[0]
    ambiguous = False
    if len(scored) > 1 and scored[1][0] >= best_quality - 0.04:
        ambiguous = True
        if strategy.ordinal_in_context is not None:
            same = [el for q, el, _ in scored if q >= best_quality - 0.04]
            if 0 <= strategy.ordinal_in_context < len(same):
                best = same[strategy.ordinal_in_context]
                ambiguous = False
    return StrategyResolution(best if best_quality >= 0.45 else None, round(best_quality, 3), matched, ambiguous,
                              candidates=sum(1 for q, _, _ in scored if q >= 0.45))


def count_selector_matches(selector_kind: str, value: str, observation: Observation, role: str | None = None) -> int:
    """Deterministic uniqueness check of a simple selector against the inventory."""

    n = 0
    for el in observation.elements:
        if not el.visible:
            continue
        if selector_kind == "role_name" and el.role == role and el.name == value:
            n += 1
        elif selector_kind == "label" and (el.label == value or (el.role in FIELD_ROLES and el.name == value)):
            n += 1
        elif selector_kind == "test_id" and el.test_id == value:
            n += 1
        elif selector_kind == "placeholder" and el.placeholder == value:
            n += 1
        elif selector_kind == "dom_id" and el.dom_id == value:
            n += 1
    return n


def validated_selectors(strategy: LocatorStrategy, observation: Observation) -> list[dict[str, object]]:
    """Evaluate each candidate selector for existence/uniqueness on the current observation."""

    checks: list[dict[str, object]] = []
    if strategy.name:
        checks.append({"kind": "getByRole", "value": f'{strategy.role} "{strategy.name}"',
                       "matches": count_selector_matches("role_name", strategy.name, observation, strategy.role)})
    if strategy.label:
        checks.append({"kind": "getByLabel", "value": strategy.label,
                       "matches": count_selector_matches("label", strategy.label, observation)})
    if strategy.placeholder:
        checks.append({"kind": "getByPlaceholder", "value": strategy.placeholder,
                       "matches": count_selector_matches("placeholder", strategy.placeholder, observation)})
    if strategy.test_id:
        checks.append({"kind": "getByTestId", "value": strategy.test_id,
                       "matches": count_selector_matches("test_id", strategy.test_id, observation)})
    if strategy.dom_id:
        checks.append({"kind": "css#id", "value": strategy.dom_id,
                       "matches": count_selector_matches("dom_id", strategy.dom_id, observation)})
    for check in checks:
        check["unique"] = check["matches"] == 1
    return checks
