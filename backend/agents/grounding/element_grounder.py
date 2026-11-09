"""Contextual element grounding (spec sections 8.4, 9.5, 10.1).

Scores every inventory element against the step's semantic target with
explainable features (name, purpose, role, context, state, type, option,
visual, memory), applies hard safety constraints, resolves ambiguity with
context and ordinals instead of DOM order, and calibrates a confidence that
drives the execute / enhanced-verify / refine / ask-human decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.agents.grounding.fuzzy import explain_match, match_score
from backend.agents.grounding.locator_strategy import (
    LocatorStrategy,
    StrategyResolution,
    resolve_strategy,
)
from backend.agents.grounding.text_match import similarity, split_identifier, strip_badge
from backend.agents.planner.lexicon import (
    ACTION_ROLE_COMPAT,
    normalize_text,
    purpose_compatible,
)
from backend.core.config import ConfidenceThresholds, get_settings
from backend.dsl.models import ActionType, Step, TargetSpec
from backend.dsl.observation import InventoryElement, Observation
from backend.dsl.results import GroundingCandidate, GroundingDecision

EDITABLE_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}
TOGGLE_ROLES = {"checkbox", "switch", "radio", "menuitemcheckbox", "menuitemradio"}
SUBMIT_PURPOSES = {"submit_login", "submit_register", "submit_generic", "checkout"}

ACTION_FOR_GROUNDING = {
    ActionType.CLICK: "click", ActionType.DOUBLE_CLICK: "click", ActionType.RIGHT_CLICK: "click",
    ActionType.HOVER: "hover", ActionType.FILL: "fill", ActionType.TYPE: "type", ActionType.CLEAR: "clear",
    ActionType.SELECT: "select", ActionType.CHECK: "check", ActionType.UNCHECK: "uncheck",
    ActionType.TOGGLE: "toggle", ActionType.UPLOAD: "upload", ActionType.DRAG: "drag", ActionType.DROP: "drop",
    ActionType.DOWNLOAD: "download", ActionType.PRESS_KEY: "type", ActionType.CAPTURE_VALUE: "hover",
    ActionType.OPEN_PAGE: "click", ActionType.ASSERT: "hover", ActionType.CAPTURE_SCREENSHOT: "hover",
    ActionType.DISMISS_DIALOG: "click", ActionType.ACCEPT_DIALOG: "click",
}


@dataclass
class GroundingContext:
    previous_element: InventoryElement | None = None
    previous_form_ref: str | None = None
    previous_form_name: str | None = None
    memory_strategies: list[tuple[LocatorStrategy, float, str | None]] = field(default_factory=list)  # (s, conf, id)
    excluded_keys: set[str] = field(default_factory=set)
    frame_hint: str | None = None
    allow_hidden: bool = False


@dataclass
class ScoredCandidate:
    element: InventoryElement
    raw: float
    features: dict[str, float]
    evidence: list[str]
    memory_quality: float = 0.0
    memory_id: str | None = None


def element_key(el: InventoryElement) -> str:
    return "|".join([el.role, normalize_text(el.name), el.dom_id or "", el.test_id or "", el.css_selector or "",
                     str(round(el.bbox.x)) if el.bbox else "", str(round(el.bbox.y)) if el.bbox else ""])


def _target_phrases(target: TargetSpec) -> list[tuple[str, float]]:
    phrases: list[tuple[str, float]] = []
    if target.exact_name:
        phrases.append((target.exact_name, 1.0))
    intent = target.intent or ""
    if target.option_text and " containing " in intent:
        intent = intent.split(" containing ")[0]
        if intent in ("choice",):
            intent = ""
    if intent and intent.lower() not in ("url", "focused element"):
        phrases.append((intent, 0.95 if not target.exact_name else 0.8))
    for hint in target.semantic_hints:
        if hint and all(normalize_text(hint) != normalize_text(p) for p, _ in phrases):
            phrases.append((hint, 0.8))
    return phrases[:16]


def _element_texts(el: InventoryElement) -> list[tuple[str, float]]:
    texts: list[tuple[str, float]] = []
    if el.name:
        texts.append((el.name, 1.0))
        core = strip_badge(el.name)
        if core != el.name:
            texts.append((core, 0.97))
    if el.label and el.label != el.name:
        texts.append((el.label, 0.93 if el.label_inferred else 0.97))
    if el.placeholder:
        texts.append((el.placeholder, 0.9))
    if el.title and el.title != el.name:
        texts.append((el.title, 0.88))
    if el.text and el.role not in EDITABLE_ROLES and el.text != el.name and len(el.text) < 120:
        texts.append((el.text, 0.85))
    for shown in (el.name, el.text):
        prompt = PROMPT_RE.match(shown or "")
        if prompt and prompt.group("field"):
            texts.append((prompt.group("field"), 0.92))  # "Select department" names the Department field
            break
    for ident, weight in ((el.test_id, 0.72), (el.dom_id, 0.66), (el.name_attr, 0.66)):
        words = split_identifier(ident)
        if words:
            texts.append((words, weight))
    for extra in el.nearby_text[:3]:
        if extra.startswith("icon:"):
            texts.append((extra[5:], 0.72))
        elif extra != "draggable":
            texts.append((extra, 0.6))
    return texts


def name_score(target: TargetSpec, el: InventoryElement) -> tuple[float, str]:
    best, why = 0.0, ""
    if target.exact_name:
        visible_name = (el.name or el.text or "").strip()
        if visible_name and normalize_text(visible_name) == normalize_text(target.exact_name):
            return 1.0, f"name equals '{target.exact_name}'"
    best_pair = ("", "")
    for phrase, pw in _target_phrases(target):
        for text, tw in _element_texts(el):
            # an exact name ("the patient named Abc Xy", a quoted label) is data: never typo-repaired, so
            # "Abcd Xy" does not pass for "Abc Xy"
            scorer = similarity if target.exact_name else match_score
            s = scorer(phrase, text) * pw * tw
            np_, nt = normalize_text(phrase), normalize_text(text)
            if len(np_.split()) >= 2 and nt.startswith(np_ + " "):
                # a card / tile whose accessible name is its title followed by a description ("Patient Support
                # Program Enrolled patients, ...") is named by its leading phrase: close to an exact match
                s = max(s, 0.9 * pw * tw)
            if s > best:
                best, why, best_pair = s, f"'{phrase}' ~ '{text[:40]}' ({s:.2f})", (phrase, text)
    if best_pair[0]:
        repaired = explain_match(*best_pair)
        if repaired:
            why += f" [{repaired}]"
    return round(best, 4), why


def _is_specific(target: TargetSpec) -> bool:
    """True when the instruction names a particular control rather than a generic purpose."""

    if target.exact_name:
        return True
    if not target.purpose:
        return True
    from backend.agents.planner.lexicon import PURPOSES

    intent_tokens = set(normalize_text(target.intent).split())
    for phrase in PURPOSES[target.purpose].phrases:
        if intent_tokens <= set(phrase.split()) | {"field", "button", "link", "input", "box"}:
            return False
    return len(intent_tokens) > 1


NON_TEXT_INPUTS = {"checkbox", "radio", "submit", "button", "image", "file", "reset", "hidden"}
PROMPT_RE = re.compile(r"^\s*(?:-+\s*)?(?:please\s+)?(?:select|choose|pick)\s+(?:an?\s+|your\s+|the\s+)?(?P<field>[\w /&-]{2,40}?)"
                       r"\s*(?:-+|\.\.\.|…)?\s*$", re.IGNORECASE)
DROPDOWN_CLASS_RE = re.compile(r"(^|[\s_-])(select|dropdown|dd|picker|combo)([\s_-]|$)", re.IGNORECASE)


BUTTON_CLASS_RE = re.compile(r"(^|[\s_-])(btn|button|cta|action|link)([\s_-]|$)", re.IGNORECASE)


def looks_like_button(el: InventoryElement) -> bool:
    """A clickable, role-less element whose short own text reads like a command."""

    if not el.clickable_hint or el.role not in ("generic", "listitem", "cell", "paragraph", "img", "strong",
                                                 "emphasis", "text"):
        return False
    words = (el.text or el.name or "").split()
    return 0 < len(words) <= 5 or bool(BUTTON_CLASS_RE.search(el.classes or ""))


def looks_like_dropdown(el: InventoryElement) -> bool:
    return bool(el.haspopup or el.controls or DROPDOWN_CLASS_RE.search(el.classes or "")
                or PROMPT_RE.match(el.text or el.name or ""))


def _grid_label(el: InventoryElement) -> set[str]:
    return {normalize_text(part) for part in (el.name, el.label, el.value_preview, el.text) if part}


def _is_range_input(el: InventoryElement) -> bool:
    """A Range (%) cell is a text box. Column headers on this form are text boxes too; dropdown options are not."""

    if not el.visible or el.bbox is None or el.role == "option" or el.tag == "option":
        return False
    if el.role in ("textbox", "spinbutton"):
        return True
    return el.tag == "input" and (el.input_type or "text").lower() in ("text", "number", "")


def _same_column(anchor: InventoryElement, el: InventoryElement) -> bool:
    if anchor.bbox is None or el.bbox is None:
        return False
    ax, _ = anchor.bbox.center
    ex, _ = el.bbox.center
    band = max(anchor.bbox.width, el.bbox.width, 16) * 0.8
    return abs(ax - ex) <= band


def _column_stack(observation: Observation, anchor: InventoryElement) -> list[InventoryElement]:
    """Header plus the inputs directly under it, top to bottom. Overlapping duplicates are one cell."""

    if anchor.bbox is None:
        return []
    cells: list[InventoryElement] = []
    for el in observation.elements:
        if not _is_range_input(el) or not _same_column(anchor, el):
            continue
        if el.bbox.y > anchor.bbox.y + anchor.bbox.height + 80:
            continue
        if el.bbox.y + el.bbox.height < anchor.bbox.y - 6:
            continue
        if any(el.bbox.iou(kept.bbox) > 0.45 for kept in cells if kept.bbox is not None):
            continue
        cells.append(el)
    cells.sort(key=lambda el: (el.bbox.y if el.bbox else 0, el.bbox.x if el.bbox else 0))
    return cells


def _anchor_for_label(observation: Observation, label: str) -> InventoryElement | None:
    want = normalize_text(label)
    if not want:
        return None
    found = [el for el in observation.elements
             if el.visible and el.bbox is not None and el.role != "option" and want in _grid_label(el)]
    if not found:
        return None
    found.sort(key=lambda el: (0 if el.role in ("textbox", "spinbutton") else 1, el.bbox.y if el.bbox else 0))
    return found[0]


def _landmark(observation: Observation, hint: str | None) -> InventoryElement | None:
    want = normalize_text(hint or "")
    if not want:
        return None
    best: InventoryElement | None = None
    best_score = 0.0
    for el in observation.elements:
        if not el.visible or el.bbox is None or el.role == "option":
            continue
        name = normalize_text(el.name or el.label or "")
        if not name or (want != name and want not in name.split() and not name.startswith(want)):
            continue
        score = 1.0 if el.role == "combobox" else 0.7
        if score > best_score:
            best, best_score = el, score
    return best


def resolve_grid_cell(observation: Observation, target: TargetSpec) -> InventoryElement | None:
    """The Range (%) value lives in the input under the column header, not in the header box itself.

    ``column_row`` picks the Nth input in that column (the header is row 1 when it is itself an input).
    ``below_label`` picks the nearest input strictly under the box named or showing the label.
    ``labeled_below`` picks the input that currently shows the label and sits under a named control
    (the header row, which the test then renames from 1 to 01).
    """

    hint = target.position_hint
    label = target.exact_name or target.intent
    if hint == "labeled_below":
        landmark = _landmark(observation, target.context_hint)
        matches = [el for el in observation.elements
                   if _is_range_input(el) and normalize_text(label) in _grid_label(el)]
        if landmark is not None and landmark.bbox is not None:
            floor = landmark.bbox.y + landmark.bbox.height * 0.45
            matches = [el for el in matches if el.bbox is not None and el.bbox.y + 4 >= floor]
        matches.sort(key=lambda el: el.bbox.y if el.bbox else 0)
        return matches[0] if matches else None

    anchor = _anchor_for_label(observation, label)
    if anchor is None:
        return None
    stack = _column_stack(observation, anchor)
    if hint == "column_row":
        row = target.ordinal or 1
        index = row - 1 if row > 0 else len(stack) + row
        if not stack or index < 0 or index >= len(stack):
            return None
        chosen = stack[index]
        if row > 1 and chosen is anchor and len(stack) == 1:
            return None
        return chosen
    if hint == "below_label":
        below = [el for el in stack if el is not anchor and el.bbox is not None and anchor.bbox is not None
                 and el.bbox.y >= anchor.bbox.y + anchor.bbox.height * 0.5]
        return below[0] if below else None
    return None


def is_fillable(el: InventoryElement) -> bool:
    """Text fields, but also browser-rendered pickers (date, time, color, range, number) and editable regions."""

    if el.role in EDITABLE_ROLES or el.editable:
        return True
    return el.tag in ("input", "textarea") and (el.input_type or "text").lower() not in NON_TEXT_INPUTS


def _role_score(target: TargetSpec, el: InventoryElement, action: str) -> float:
    if action in ("fill", "type", "clear") and el.role not in EDITABLE_ROLES and is_fillable(el):
        return 0.85  # slider / color well / date picker exposed with a non-textbox role
    if action == "select" and el.tag == "input" and is_fillable(el) and el.role not in ("combobox",):
        return 0.6  # "set <field> to <value>" parsed as a choice
    roles = target.preferred_roles or []
    if roles and el.role == roles[0]:
        return 1.0
    if action == "select" and el.clickable_hint and looks_like_dropdown(el):
        return 0.8  # an ARIA-less dropdown trigger ("div soup")
    if action in ("click", "double_click") and looks_like_button(el):
        return 0.8  # <div onclick> / <span class="btn"> acting as a button
    if el.role in roles:
        return 0.85
    if el.role in ACTION_ROLE_COMPAT.get(action, ()):
        return 0.55
    if el.clickable_hint and action in ("click", "hover", "drag", "select"):
        return 0.45
    if action == "select" and (el.haspopup or el.controls):
        return 0.6  # a trigger that declares the popup it opens
    if action in ("drop", "hover") and el.role in ("region", "list", "group", "generic", "listitem", "article"):
        return 0.5
    return 0.0


def _context_score(target: TargetSpec, el: InventoryElement, ctx: GroundingContext, action: str,
                   modal_refs: set[str], has_modal: bool) -> tuple[float, list[str]]:
    score = 0.6
    notes: list[str] = []
    c = el.context
    if target.context_hint:
        fields = [c.form_name, c.dialog_name, c.section_heading, c.landmark_name, c.landmark, c.fieldset_legend,
                  c.row_text, *(c.ancestors or [])]
        best = max((match_score(target.context_hint, f) for f in fields if f), default=0.0)
        if best >= 0.5:
            score = 0.6 + 0.4 * best
            notes.append(f"context '{target.context_hint}' matched ({best:.2f})")
        else:
            score = 0.12
            notes.append(f"context '{target.context_hint}' not matched")
    if ctx.previous_form_ref and action in ("fill", "type", "select", "check", "uncheck", "click"):
        if c.form_ref == ctx.previous_form_ref:
            score = max(score, 0.85)
            notes.append("same form as previous step")
        elif c.form_ref and action == "click" and target.purpose in SUBMIT_PURPOSES:
            score = min(score, 0.35)
            notes.append("different form than previous step")
    prev = ctx.previous_element
    if prev is not None and action in ("fill", "type", "select", "check", "uncheck", "clear") and is_fillable(prev) \
            and not target.context_hint:
        mine, theirs = group_key(el), group_key(prev)
        if theirs and mine == theirs:
            score = max(score, 0.9)
            notes.append("same group as the field of the previous step")
        elif theirs and mine != theirs:
            score = min(score, 0.4)
            notes.append("different group than the field of the previous step")
    if (target.purpose in SUBMIT_PURPOSES or target.purpose in ("username_or_email", "password", "email")) \
            and c.form_ref:
        score += 0.1
    if has_modal:
        if c.dialog_ref and c.dialog_ref in modal_refs:
            score = max(score, 0.85)
            notes.append("inside open modal dialog")
        else:
            score = min(score, 0.05)
            notes.append("outside the open modal dialog")
    if target.frame_hint or ctx.frame_hint:
        hint = target.frame_hint or ctx.frame_hint
        if c.frame:
            score = max(score, 0.7)
            notes.append(f"inside frame (hint '{hint}')")
        else:
            score = min(score, 0.3)
    return max(0.0, min(1.0, score)), notes


def group_key(el: InventoryElement) -> str | None:
    """The visual/structural group a field belongs to: form, dialog, table-built form or web component."""

    c = el.context
    if c.form_ref:
        return f"form:{c.form_ref}"
    if c.form_name:
        return f"form:{c.form_name}"
    if c.dialog_ref:
        return f"dialog:{c.dialog_ref}"
    if c.list_ref and el.role in EDITABLE_ROLES | TOGGLE_ROLES:
        return f"list:{c.list_ref}"
    if el.in_shadow_dom:
        return f"component:{c.section_heading or c.landmark_name or ''}"
    return None


def _state_score(el: InventoryElement) -> float:
    if not el.visible:
        return 0.1
    if not el.enabled:
        return 0.35
    if el.obstructed:
        return 0.5
    if not el.in_viewport:
        return 0.85
    return 1.0


def _option_score(target: TargetSpec, el: InventoryElement, step: Step) -> float | None:
    option = target.option_text
    if not option or step.action != ActionType.SELECT:
        return None
    norm_opt = normalize_text(option)
    offered = el.option_details or el.options or el.datalist
    if offered:
        from backend.agents.forms.option_matcher import (
            OptionInfo,
            match_option,
            split_choices,
        )

        if not el.enabled or all(o.is_placeholder or "loading" in o.label.lower() for o in OptionInfo.coerce(offered)):
            return None  # a dependent list still loading ("Loading...", disabled): no evidence either way

        wanted = split_choices(option, offered) if el.multiple else [option]
        found = [match_option(offered, w) for w in wanted]
        if all(m.found and m.how in ("exact", "case_insensitive", "normalized", "value") for m in found):
            return 1.0
        if all(m.found for m in found):
            return 0.88 if all(m.how != "fuzzy" for m in found) else 0.82
        return 0.05
    if el.role in ("radio", "tab", "button", "option", "menuitemradio") and normalize_text(el.name) == norm_opt:
        return 1.0
    if el.role in ("combobox", "listbox") or looks_like_dropdown(el):
        return None  # custom/lazy dropdown: options only appear after opening, so no evidence either way
    if is_fillable(el):
        return None  # "set <input> to <value>": a value, not a list choice
    return 0.1


def _visual_score(target: TargetSpec, el: InventoryElement) -> float | None:
    if not target.visual_hint:
        return None
    words = [t[5:] for t in el.nearby_text if t.startswith("icon:")] + [el.classes or "", el.title or "",
                                                                       el.name or "", el.text or ""]
    best = max((match_score(target.visual_hint, w) for w in words if w), default=0.0)
    if el.role == "canvas" and any(r == "canvas" for r in target.preferred_roles):
        best = max(best, 0.7)
    return best


def _hard_excluded(step: Step, target: TargetSpec, el: InventoryElement, action: str,
                   ctx: GroundingContext) -> str | None:
    if element_key(el) in ctx.excluded_keys:
        return "excluded by healing"
    native_option = el.role == "option" and el.context.owner_role == "combobox"
    if el.role == "option" and not native_option and not el.visible:
        return "option inside a closed list"
    if el.role == "option" and native_option and step.action != ActionType.CLICK:
        return "option of a native select (the select itself is the target)"
    if not el.visible and not ctx.allow_hidden and not native_option:
        return "not visible"
    if action in ("fill", "type", "clear") and not is_fillable(el):
        return "not editable"
    if target.input_type == "password" and action in ("fill", "type") and el.input_type not in (None, "password"):
        return "not a password field"
    if target.input_type == "password" and action in ("fill", "type") and el.input_type is None and \
            el.semantic_purpose not in ("password", "confirm_password"):
        return "not a password field"
    if action in ("check", "uncheck") and el.role not in TOGGLE_ROLES and not (el.role == "button" and
                                                                               el.checked is not None):
        return "not a toggle"
    if el.tag == "label" and action != "click":
        return "label proxy of a form control"
    if action == "upload" and not (el.input_type == "file" or (el.role == "button" and el.semantic_purpose == "upload"
                                                               and el.tag != "label")):
        return "not a file input"
    if el.role in ("heading", "paragraph", "status", "alert", "row", "cell", "listitem", "form", "iframe", "group",
                   "region", "list", "table", "dialog", "alertdialog") and action in ("click", "fill", "type",
                                                                                      "select", "check"):
        if not (action == "click" and el.clickable_hint):
            return "non-interactive container"
    return None


class ElementGrounder:
    name = "dom_intelligence_agent"

    def __init__(self, thresholds: ConfidenceThresholds | None = None) -> None:
        self.thresholds = thresholds or get_settings().thresholds

    def ground(self, step: Step, observation: Observation, ctx: GroundingContext | None = None,
               *, target: TargetSpec | None = None, action_override: str | None = None) -> GroundingDecision:
        ctx = ctx or GroundingContext()
        target = target or step.target
        if target is None:
            return GroundingDecision(decision="not_found", reason="step has no target")
        if target.position_hint in ("column_row", "below_label", "labeled_below"):
            cell = resolve_grid_cell(observation, target)
            if cell is not None:
                return self._pinned(cell, observation, 0.96, "range cell under the column header")
        action = action_override or ACTION_FOR_GROUNDING.get(step.action, "click")
        modal_refs = {d.ref for d in observation.dialogs if d.modal and d.ref}
        has_modal = bool(modal_refs)

        memory_res: list[tuple[StrategyResolution, float, str | None]] = []
        for strategy, conf, memory_id in ctx.memory_strategies:
            res = resolve_strategy(strategy, observation)
            # only a strong, unambiguous re-grounding may influence the decision
            if res.element is not None and res.quality >= 0.7 and not res.ambiguous:
                memory_res.append((res, conf, memory_id))

        scored: list[ScoredCandidate] = []
        rejected: dict[str, str] = {}
        submit_form_ref = ctx.previous_form_ref if any(n.startswith("submits_form_group") for n in step.notes) else None
        has_name_target = bool(_target_phrases(target))
        specific_name = _is_specific(target)
        for el in observation.elements:
            reason = _hard_excluded(step, target, el, action, ctx)
            if reason:
                rejected[el.element_id] = reason
                continue
            role = _role_score(target, el, action)
            if role == 0.0:
                rejected[el.element_id] = "role incompatible"
                continue
            features: dict[str, float] = {"role": role}
            evidence: list[str] = []
            identity_parts: list[float] = []
            if has_name_target:
                n_score, why = name_score(target, el)
                features["name"] = n_score
                identity_parts.append(n_score)
                if why and n_score >= 0.3:
                    evidence.append(why)
            purpose_conflict = False
            if target.purpose:
                if el.semantic_purpose:
                    compat = purpose_compatible(target.purpose, el.semantic_purpose)
                    features["purpose"] = compat * max(0.6, el.purpose_confidence)
                    identity_parts.append(features["purpose"] * 0.95)
                    if compat:
                        evidence.append(f"purpose {el.semantic_purpose} ({el.purpose_confidence:.2f})")
                    elif el.purpose_confidence >= 0.8:
                        purpose_conflict = True
                        evidence.append(f"purpose conflict: element is {el.semantic_purpose}")
                else:
                    features["purpose"] = 0.0
            opt = _option_score(target, el, step)
            if opt is not None:
                features["option"] = opt
                identity_parts.append(opt * 0.95)
                if opt >= 0.75:
                    evidence.append(f"contains option '{target.option_text}'")
            vis = _visual_score(target, el)
            if vis is not None:
                features["visual"] = vis
                identity_parts.append(vis * 0.85)
            # Identity: the strongest evidence, corroborated by the others. When the instruction names a
            # *specific* control ("Dismiss Password changed"), purpose is shared by its siblings and may
            # only corroborate; for generic references ("the username field") purpose carries identity.
            name_part = features.get("name", 0.0)
            purpose_part = features.get("purpose", 0.0) * (0.95 if not specific_name else 0.72)
            others = [p for k, p in (("option", features.get("option")), ("visual", features.get("visual")))
                      if p is not None]
            strongest = max([name_part, purpose_part, *[o * 0.95 for o in others]]) if identity_parts else 0.5
            support = sorted([name_part, purpose_part, *others], reverse=True)
            identity = min(1.0, strongest + 0.1 * (support[1] if len(support) > 1 else 0.0)) \
                if identity_parts else 0.5
            if purpose_conflict:
                identity *= 0.65
            if el.tag == "label":
                identity *= 0.8  # clicking a label is a proxy; prefer the control it labels
            if opt is not None and opt < 0.2:
                identity *= 0.5  # this dropdown does not offer the requested option
            features["identity"] = round(identity, 4)
            context_score, notes = _context_score(target, el, ctx, action, modal_refs, has_modal)
            if submit_form_ref and action == "click":
                if el.context.form_ref == submit_form_ref and (el.input_type in ("submit", "image") or
                                                               el.semantic_purpose in SUBMIT_PURPOSES):
                    context_score = max(context_score, 0.95)
                    notes.append("submit control of the form filled in previous steps")
                elif el.context.form_ref != submit_form_ref:
                    context_score = min(context_score, 0.3)
            features["context"] = context_score
            evidence.extend(notes)
            features["state"] = _state_score(el)
            if target.input_type:
                features["type"] = 1.0 if el.input_type == target.input_type else (
                    0.6 if el.input_type in (None, "text") and target.input_type == "email" else 0.2)
            candidate = ScoredCandidate(el, 0.0, features, evidence)
            for res, conf, memory_id in memory_res:
                if res.element is el and not res.ambiguous and identity >= 0.5:
                    candidate.memory_quality = res.quality * conf
                    candidate.memory_id = memory_id
                    candidate.evidence.append(f"memory strategy re-grounded ({res.quality:.2f}, on "
                                              f"{','.join(res.matched_on[:4])})")
            scored.append(candidate)

        # prefer main-content controls over header/footer look-alikes, but only when such an alternative exists
        if target.purpose in SUBMIT_PURPOSES or target.purpose in ("username_or_email", "password", "email") or \
                action in ("fill", "type"):
            main_alternative = any(c.features["identity"] >= 0.6 and c.element.context.landmark not in
                                   ("banner", "navigation", "contentinfo") for c in scored)
            if main_alternative:
                for c in scored:
                    if c.element.context.landmark in ("banner", "navigation", "contentinfo") and not target.context_hint:
                        c.features["context"] = max(0.0, c.features["context"] - 0.3)
                        c.evidence.append("header/footer control while a main-content match exists")
        best_identity = max((c.features["identity"] for c in scored), default=0.0)
        for c in scored:
            if c.memory_quality > 0 and c.features["identity"] < best_identity - 0.15:
                # memory may confirm a live match, never overrule a clearly better one
                c.memory_quality, c.memory_id = 0.0, None
                c.evidence.append("remembered target ignored: a clearly better live match exists")
        for c in scored:
            f = c.features
            weights = {"identity": 0.56, "role": 0.12, "context": 0.14, "state": 0.12}
            if "type" in f:
                weights["type"] = 0.06
            c.raw = sum(f[k] * w for k, w in weights.items()) / sum(weights.values())

        if not scored:
            return GroundingDecision(decision="not_found", reason="no compatible element on the page",
                                     evidence=[f"{k}: {v}" for k, v in list(rejected.items())[:8]])
        scored.sort(key=lambda c: (-(c.raw + 0.25 * c.memory_quality), c.element.bbox.y if c.element.bbox else 0))
        top = scored[0]
        second = scored[1] if len(scored) > 1 else None
        ordinal_applied = False
        if target.ordinal is not None or target.position_hint:
            tier = [c for c in scored if c.raw >= top.raw - 0.08]
            tier.sort(key=lambda c: ((c.element.bbox.y if c.element.bbox else 0) // 8,
                                     c.element.bbox.x if c.element.bbox else 0))
            if target.position_hint:
                tier = _apply_position(tier, target.position_hint, observation.viewport)
            if target.ordinal is not None and tier:
                index = target.ordinal - 1 if target.ordinal > 0 else len(tier) + target.ordinal
                if 0 <= index < len(tier):
                    top = tier[index]
                    ordinal_applied = True
            elif tier:
                top = tier[0]
                ordinal_applied = True
            second = next((c for c in scored if c is not top), None)

        margin = top.raw - (second.raw if second else 0.0)
        separation = min(1.0, max(0.0, margin) / 0.25)
        confidence = top.raw * (0.6 + 0.4 * separation)
        context_resolved = bool(second and not ordinal_applied
                                and abs(top.features.get("identity", 0) - second.features.get("identity", 0)) <= 0.1
                                and top.features.get("context", 0) >= 0.85 and second.features.get("context", 1) <= 0.4)
        if ordinal_applied:
            confidence = max(confidence, top.raw * 0.95)
        elif context_resolved:
            # equally good matches, one in the group the test is working in: context decides (like an ordinal)
            confidence = max(confidence, top.raw * 0.9)
            top.evidence.append("ambiguity resolved by context: same group as the previous step")
        elif margin >= 0.4 and top.features.get("identity", 0) >= 0.75:
            confidence += (1 - confidence) * 0.35  # the only plausible match on the page
        elif top.features.get("name", 0) >= 0.94 and top.features.get("identity", 0) >= 0.9 and (
                second is None or second.features.get("name", 0) <= top.features.get("name", 0) - 0.15):
            confidence += (1 - confidence) * 0.5  # the only element whose label says exactly what the test says
            top.evidence.append("unique exact label match")
        if top.memory_quality > 0:
            confidence = min(0.995, confidence + (1 - confidence) * 0.6 * top.memory_quality)
        elif memory_res and any(not r.ambiguous and r.element is not None and r.quality > 0.8 for r, _, _ in memory_res):
            confidence *= 0.85  # live page evidence disagrees with validated memory
            top.evidence.append("conflicts with remembered strategy")
        confidence = round(min(confidence, 0.995), 4)
        decision = self.decide(confidence)
        el = top.element
        return GroundingDecision(
            element_id=el.element_id,
            ref=el.ref,
            display=el.display(),
            role=el.role,
            name=el.name or el.label or "",
            confidence=confidence,
            decision=decision,
            method="memory" if top.memory_quality > 0.5 else "semantic",
            locator_strategy=LocatorStrategy.from_element(el, observation).model_dump(exclude_none=True),
            candidates=[GroundingCandidate(element_id=c.element.element_id, ref=c.element.ref,
                                           display=c.element.display(), score=round(c.raw, 4),
                                           features={k: round(v, 3) for k, v in c.features.items()},
                                           evidence=c.evidence[:6],
                                           bbox=c.element.bbox.model_dump() if c.element.bbox else None)
                        for c in scored[:6]],
            evidence=top.evidence[:8],
            bbox=el.bbox.model_dump() if el.bbox else None,
            memory_id=top.memory_id,
            reason=f"best={top.raw:.2f} margin={margin:.2f}" + (" ordinal" if ordinal_applied else ""),
        )

    def _pinned(self, el: InventoryElement, observation: Observation, confidence: float, reason: str) -> GroundingDecision:
        return GroundingDecision(
            element_id=el.element_id, ref=el.ref, display=el.display(), role=el.role,
            name=el.name or el.label or "", confidence=confidence, decision=self.decide(confidence),
            method="semantic",
            locator_strategy=LocatorStrategy.from_element(el, observation).model_dump(exclude_none=True),
            candidates=[GroundingCandidate(element_id=el.element_id, ref=el.ref, display=el.display(),
                                           score=confidence, evidence=[reason],
                                           bbox=el.bbox.model_dump() if el.bbox else None)],
            evidence=[reason], bbox=el.bbox.model_dump() if el.bbox else None, reason=reason,
        )

    def decide(self, confidence: float) -> Any:
        t = self.thresholds
        if confidence >= t.execute:
            return "execute"
        if confidence >= t.enhanced_verification:
            return "execute_enhanced"
        if confidence >= t.refine:
            return "refine"
        return "ask_human"


def _apply_position(tier: list[ScoredCandidate], hint: str, viewport: tuple[int, int]) -> list[ScoredCandidate]:
    width, height = viewport

    def key(c: ScoredCandidate) -> float:
        if not c.element.bbox:
            return 1e9
        cx, cy = c.element.bbox.center
        score = 0.0
        if "top" in hint:
            score += cy
        if "bottom" in hint:
            score += height - cy
        if "left" in hint:
            score += cx
        if "right" in hint:
            score += width - cx
        if "center" in hint or "middle" in hint:
            score += abs(cx - width / 2) + abs(cy - height / 2)
        return score

    return sorted(tier, key=key)
