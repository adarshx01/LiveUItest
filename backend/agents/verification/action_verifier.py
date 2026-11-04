"""Post-action verification (spec sections 2.3 / 18).

A successful tool response is not proof. After each meaningful action the page is
re-observed and the *intended* postcondition is checked (value entered, option
selected, toggle state, navigation, upload, download, dialog closed, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from backend.agents.grounding.locator_strategy import LocatorStrategy, resolve_strategy
from backend.agents.grounding.text_match import similarity
from backend.browser.js_probes import (
    ELEMENT_STATE,
    SELECT_STATE,
    VALUE_FINGERPRINT,
    fnv1a,
)
from backend.browser.tool_adapter import BrowserToolAdapter
from backend.dsl.models import ActionType, Step
from backend.dsl.observation import InventoryElement, Observation
from backend.dsl.results import FailureCategory
from backend.execution.action_executor import ActionOutcome


@dataclass
class VerificationResult:
    status: Literal["verified", "unverified", "failed"]
    checks: list[dict[str, Any]] = field(default_factory=list)
    failure_category: FailureCategory | None = None
    message: str = ""
    page_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": self.checks, "message": self.message,
                "failure_category": self.failure_category.value if self.failure_category else None,
                "page_changed": self.page_changed}


def page_changed(before: Observation | None, after: Observation | None) -> tuple[bool, list[str]]:
    if before is None or after is None:
        return True, ["no baseline"]
    reasons: list[str] = []
    if before.url != after.url:
        reasons.append(f"url {before.url} -> {after.url}")
    if (before.fingerprint and after.fingerprint and before.fingerprint.structure_hash !=
            after.fingerprint.structure_hash):
        reasons.append("page structure changed")
    if len(before.tabs) != len(after.tabs):
        reasons.append(f"tabs {len(before.tabs)} -> {len(after.tabs)}")
    if bool(before.modal_state) != bool(after.modal_state):
        reasons.append("native dialog state changed")
    if len(before.dialogs) != len(after.dialogs):
        reasons.append("dialogs changed")
    b_states = {(e.role, e.name, e.checked, e.expanded, e.value_preview, e.enabled) for e in before.elements}
    a_states = {(e.role, e.name, e.checked, e.expanded, e.value_preview, e.enabled) for e in after.elements}
    if b_states != a_states:
        reasons.append("element states changed")
    if set(before.visible_text) != set(after.visible_text):
        reasons.append("visible text changed")
    return bool(reasons), reasons


class ActionVerifier:
    name = "verification_agent"

    def __init__(self, adapter: BrowserToolAdapter) -> None:
        self.adapter = adapter

    def relocate(self, element: InventoryElement | None, before: Observation | None,
                 after: Observation) -> InventoryElement | None:
        if element is None:
            return None
        if element.ref and after.by_ref(element.ref) is not None:
            candidate = after.by_ref(element.ref)
            if candidate and candidate.role == element.role:
                return candidate
        resolution = resolve_strategy(LocatorStrategy.from_element(element, before), after)
        return resolution.element if resolution.quality >= 0.6 else None

    async def verify(self, step: Step, outcome: ActionOutcome, element: InventoryElement | None,
                     before: Observation | None, after: Observation) -> VerificationResult:
        changed, change_reasons = page_changed(before, after)
        checks: list[dict[str, Any]] = [{"check": "page_change", "ok": changed, "detail": change_reasons[:4]}]
        expected = outcome.expected
        action = step.action

        def result(status: str, message: str = "", category: FailureCategory | None = None) -> VerificationResult:
            return VerificationResult(status=status, checks=checks, failure_category=category,  # type: ignore[arg-type]
                                      message=message, page_changed=changed)

        if after.page_state.captcha_detected and not (before and before.page_state.captcha_detected):
            checks.append({"check": "challenge", "ok": False, "detail": after.page_state.captcha_evidence})
            return result("failed", "a human-verification challenge appeared", FailureCategory.CAPTCHA_DETECTED)
        if after.page_state.error_page and not (before and before.page_state.error_page):
            checks.append({"check": "error_page", "ok": False, "detail": after.page_state.http_status})
            return result("failed", f"application error page ({after.page_state.http_status or after.title})",
                          FailureCategory.APPLICATION_ERROR)
        if outcome.no_op:
            checks.append({"check": "already_in_state", "ok": True, "detail": outcome.notes})
            return result("verified", "control already in the requested state")

        typed_value = expected.get("value") is not None and expected.get("selected_option") is None
        if (action in (ActionType.FILL, ActionType.TYPE, ActionType.CLEAR) or typed_value) and expected.get("kind") is None:
            el = self.relocate(element, before, after)
            if el is None or not el.ref:
                checks.append({"check": "field_located", "ok": False})
                return result("unverified", "field could not be re-located after typing")
            want = "" if action == ActionType.CLEAR else str(expected.get("value", ""))
            if expected.get("sensitive"):
                probe = await self.adapter.probe(VALUE_FINGERPRINT, el.ref, el.display()) or {}
                ok = probe.get("hash") == fnv1a(want)
                checks.append({"check": "value_fingerprint", "ok": ok, "detail": {"length": probe.get("length")}})
            else:
                state = await self.adapter.probe(ELEMENT_STATE, el.ref, el.display()) or {}
                actual = state.get("value")
                ok = actual == want or (actual is not None and actual.strip() == want.strip())
                checks.append({"check": "field_value", "ok": ok, "detail": {"expected": want, "actual": actual}})
            if not ok:
                return result("failed", "field does not contain the entered value", FailureCategory.VERIFICATION_FAILED)
            return result("verified", "field contains the entered value")

        if expected.get("kind") == "menu":
            checks.append({"check": "menu_path", "ok": True, "detail": expected.get("menu_path")})
            return result("verified", f"menu path {' > '.join(expected.get('menu_path') or [])} activated")

        if action == ActionType.SELECT or expected.get("selected_option") is not None:
            option = str(expected.get("selected_option", ""))
            wanted = [str(o) for o in (expected.get("selected_list") or [option])]
            kind = expected.get("kind")
            el = self.relocate(element, before, after)
            selected: list[str] = []
            if kind == "native" and expected.get("owner_target"):
                # read the live <select>: the snapshot can lag behind a change event
                state = await self.adapter.probe(SELECT_STATE, str(expected["owner_target"]), "native select") or {}
                selected = [str(s) for s in state.get("selected") or []]
                ok = bool(state.get("isSelect")) and all(any(similarity(w, s) >= 0.95 for s in selected)
                                                         for w in wanted)
            elif kind == "custom" and expected.get("confirmed"):
                ok = True  # the engine saw every chosen option marked selected before closing the list
                selected = wanted
            elif kind in ("radio", "checkbox", "switch", "menuitemradio"):
                ok = bool(el and el.checked)
            elif kind in ("tab", "button", "option"):
                ok = changed
            elif el is not None and el.options:
                selected = list(el.selected_options)
                ok = all(any(similarity(w, s) >= 0.95 for s in selected) for w in wanted)
            else:
                ok = all(_custom_selected(w, el, after, expected) for w in wanted)
            checks.append({"check": "option_selected", "ok": ok,
                           "detail": {"expected": wanted, "selected": selected or (el.selected_options if el else None)}})
            if not ok:
                return result("failed", f"option '{option}' is not selected", FailureCategory.VERIFICATION_FAILED)
            return result("verified", f"'{option}' selected")

        if action in (ActionType.CHECK, ActionType.UNCHECK, ActionType.TOGGLE):
            el = self.relocate(element, before, after)
            desired = expected.get("checked")
            ok = el is not None and (desired is None or bool(el.checked) == bool(desired))
            checks.append({"check": "toggle_state", "ok": ok, "detail": {"desired": desired,
                                                                         "actual": el.checked if el else None}})
            if not ok:
                return result("failed", "control did not reach the requested state", FailureCategory.VERIFICATION_FAILED)
            return result("verified", f"control is {'checked' if el and el.checked else 'unchecked'}")

        if action == ActionType.UPLOAD:
            el = self.relocate(element, before, after)
            files: list[str] = []
            if el is not None and el.ref:
                state = await self.adapter.probe(ELEMENT_STATE, el.ref, el.display()) or {}
                files = state.get("files") or []
            want = expected.get("files") or []
            ok = bool(want) and all(f in files for f in want)
            checks.append({"check": "files_attached", "ok": ok, "detail": files})
            return result("verified" if ok else "failed", "file attached" if ok else "file not attached",
                          None if ok else FailureCategory.VERIFICATION_FAILED)

        if action == ActionType.DOWNLOAD or step.expects_download:
            ok = bool(outcome.downloads)
            checks.append({"check": "download", "ok": ok, "detail": outcome.downloads})
            return result("verified" if ok else "failed", "download completed" if ok else "no download started",
                          None if ok else FailureCategory.VERIFICATION_FAILED)

        if action in (ActionType.NAVIGATE, ActionType.GO_BACK, ActionType.RELOAD):
            status = after.page_state.http_status
            ok = bool(after.url) and not after.page_state.error_page and (status is None or status < 400)
            checks.append({"check": "page_loaded", "ok": ok, "detail": {"url": after.url, "status": status}})
            if not ok:
                return result("failed", f"page did not load correctly (status {status})",
                              FailureCategory.APPLICATION_ERROR if status and status >= 400 else
                              FailureCategory.NAVIGATION_TIMEOUT)
            return result("verified", f"loaded {after.url}")

        if action == ActionType.SWITCH_TAB:
            want_title = expected.get("tab_title")
            current = next((t for t in after.tabs if t.current), None)
            ok = current is not None and (want_title is None or current.title == want_title or
                                          expected.get("tab") in ("new", "closed"))
            checks.append({"check": "active_tab", "ok": ok, "detail": current.model_dump() if current else None})
            return result("verified" if ok else "failed", "", None if ok else FailureCategory.POPUP_OR_DIALOG_ERROR)

        if action in (ActionType.ACCEPT_DIALOG, ActionType.DISMISS_DIALOG):
            ok = after.modal_state is None and (expected.get("dialog_closed") is None or not any(
                d.name == expected["dialog_closed"] for d in after.dialogs))
            checks.append({"check": "dialog_closed", "ok": ok})
            return result("verified" if ok else "failed", "", None if ok else FailureCategory.POPUP_OR_DIALOG_ERROR)

        if action == ActionType.DRAG:
            moved = str(expected.get("moved") or "")
            into = str(expected.get("into") or "")
            moved_el = next((e for e in after.elements if moved and (e.name == moved or e.text == moved)), None)
            ok = moved_el is not None and into and any(into.lower() in a.lower() for a in moved_el.context.ancestors)
            ok = bool(ok) or (changed and "element states changed" in change_reasons or "visible text changed" in
                              change_reasons)
            checks.append({"check": "item_moved", "ok": bool(ok), "detail": {"item": moved, "into": into}})
            return result("verified" if ok else "failed", "", None if ok else FailureCategory.VERIFICATION_FAILED)

        if action == ActionType.SCROLL:
            ok = bool(expected.get("text_visible")) or expected.get("before") != expected.get("after") or changed
            checks.append({"check": "scrolled", "ok": ok})
            return result("verified" if ok else "unverified", "")

        if action in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK, ActionType.PRESS_KEY):
            errors = [e for e in after.page_state.validation_errors if e not in
                      (before.page_state.validation_errors if before else [])]
            if errors:
                checks.append({"check": "validation_messages", "ok": False, "detail": errors[:5]})
            if outcome.tab_switched or after.modal_state is not None:
                checks.append({"check": "new_context", "ok": True})
            if changed:
                return result("verified", "observable effect: " + ", ".join(change_reasons[:3])
                              + (f"; messages: {errors[:2]}" if errors else ""))
            submit_like = step.risk_hint == "submit" or (element is not None and element.input_type == "submit") or \
                any(n.startswith("submits_form_group") for n in step.notes)
            if submit_like and after.page_state.invalid_fields:
                checks.append({"check": "constraint_validation", "ok": False,
                               "detail": after.page_state.invalid_fields[:5]})
                return result("failed", "form was not submitted; the browser rejected invalid fields: "
                              + "; ".join(after.page_state.invalid_fields[:3]), FailureCategory.FORM_VALIDATION_FAILED)
            if step.expects_navigation:
                return result("unverified", "no observable effect after a click that was expected to navigate")
            return result("unverified", "no observable change detected (effect may be outside the page)")

        if action in (ActionType.HOVER, ActionType.WAIT, ActionType.CAPTURE_SCREENSHOT, ActionType.CAPTURE_VALUE,
                      ActionType.SWITCH_FRAME):
            return result("verified", "")
        return result("verified" if outcome.ok else "unverified", "")


def _custom_selected(option: str, trigger: InventoryElement | None, after: Observation,
                     expected: dict[str, Any]) -> bool:
    """A custom dropdown shows its selection in one of a few ways; any reliable one is enough."""

    low = option.lower()
    if trigger is not None:
        shown = " ".join(filter(None, [trigger.name, trigger.text, trigger.value_preview, trigger.label]))
        if low in shown.lower():
            return True
    name = expected.get("trigger")
    for e in after.elements:
        if e.role == "option" and e.selected and (e.name or e.text).lower() == low:
            return True  # listbox still open with aria-selected
        if e.role in ("combobox", "button", "listbox", "generic", "textbox") and \
                (not name or name in (e.name, e.label, e.text) or (e.value_preview or "").lower() == low) and \
                low in " ".join(filter(None, [e.name, e.text, e.value_preview])).lower():
            return True
    # chips / tags of multi-selects, or the control's own echo of the choice - only right next to the trigger
    anchor = trigger.bbox if trigger is not None else None
    if anchor is None:
        return False

    def near(e: InventoryElement) -> bool:
        if e.bbox is None:
            return False
        dy = max(0.0, max(anchor.y, e.bbox.y) - min(anchor.y + anchor.height, e.bbox.y + e.bbox.height))
        dx = max(0.0, max(anchor.x, e.bbox.x) - min(anchor.x + anchor.width, e.bbox.x + e.bbox.width))
        return dy <= 60 and dx <= 40

    return any(low == (e.text or e.name or "").strip().lower() for e in after.elements
               if e.visible and e.role not in ("option", "menuitem") and near(e))
