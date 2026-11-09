"""Page state analysis: loading, challenges, errors, overlays, page kind, injection."""

from __future__ import annotations

import re
from typing import Any

from backend.dsl.observation import DialogInfo, FormInfo, InventoryElement, PageState

ERROR_TITLE_RE = re.compile(r"\b(404|not found|500|internal server error|bad gateway|service unavailable|"
                            r"something went wrong|application error|access denied|403 forbidden|401 unauthorized)\b",
                            re.IGNORECASE)

# A document status of 401/403 is not, by itself, proof the screen is unusable. Gateways often
# answer with those codes while the application (a sign-in form, a SPA shell) is what is on screen.
_USABLE_ROLES = {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "switch", "tab"}


APP_ERROR_ALERT_RE = re.compile(r"something went wrong|internal server error|unexpected error|an error occurred|"
                                r"service unavailable|failed to process", re.IGNORECASE)


def analyze_page_state(*, url: str, title: str, headings: list[str], elements: list[InventoryElement],
                       forms: list[FormInfo], dialogs: list[DialogInfo], signals: dict[str, Any] | None,
                       dom: dict[str, Any] | None, console_errors: int = 0) -> PageState:
    signals = signals or {}
    state = PageState(console_errors=console_errors)
    state.loading = (signals.get("readyState") not in (None, "complete")) or bool(signals.get("busy"))
    captcha = list(signals.get("captcha") or [])
    state.captcha_detected = bool(captcha)
    state.captcha_evidence = captcha[:5]
    state.http_status = (dom or {}).get("httpStatus")
    heading_text = " ".join(headings[:3])
    alert_text = " ".join(e.text or e.name for e in elements if e.role in ("alert", "alertdialog") and e.visible)
    text_error = bool(ERROR_TITLE_RE.search(title or "") or ERROR_TITLE_RE.search(heading_text)
                      or APP_ERROR_ALERT_RE.search(alert_text))
    status_error = bool(state.http_status and state.http_status >= 400)
    interactive = sum(1 for e in elements if e.visible and e.role in _USABLE_ROLES)
    has_entry_form = any(f.purpose in ("login", "registration") for f in forms)
    usable = interactive >= 2 or has_entry_form
    state.error_page = bool(text_error or (status_error and not usable))
    state.validation_errors = [e for e in (signals.get("validationErrors") or []) if e][:8]
    state.invalid_fields = [e for e in (signals.get("invalidFields") or []) if e][:10]
    modal_dialogs = [d for d in dialogs if d.modal]
    obstructed = sum(1 for e in elements if e.obstructed and e.visible and e.in_viewport)
    state.blocking_overlay = bool(modal_dialogs) or obstructed >= 3
    state.overlay_names = [d.name for d in dialogs][:4]
    state.injection_findings = [f"{'hidden ' if f.get('hidden') else ''}text: {f.get('text', '')[:120]}"
                                for f in (signals.get("injection") or [])]
    purposes = {e.semantic_purpose for e in elements if e.visible}
    has_logout = "logout" in purposes
    has_password = "password" in purposes
    if has_logout:
        state.authenticated_hint = True
    elif has_password and any(f.purpose == "login" for f in forms):
        state.authenticated_hint = False
    if state.captcha_detected:
        state.page_kind = "challenge"
    elif state.error_page:
        state.page_kind = "error"
    elif any(f.purpose == "login" for f in forms):
        state.page_kind = "login"
    elif any(f.purpose == "registration" for f in forms):
        state.page_kind = "registration"
    elif re.search(r"dashboard|welcome|overview|home", heading_text, re.IGNORECASE) and has_logout:
        state.page_kind = "dashboard"
    elif forms:
        state.page_kind = "form"
    elif sum(1 for e in elements if e.role in ("row", "listitem", "article")) >= 5:
        state.page_kind = "list"
    else:
        state.page_kind = "content"
    return state
