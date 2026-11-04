"""Semantic element classification (DOM intelligence, deterministic).

Maps each element to the shared purpose vocabulary using accessible name,
label, placeholder, input type, autocomplete hints and DOM attributes.
"""

from __future__ import annotations

import re

from backend.agents.planner.lexicon import PURPOSES, classify_phrase, normalize_text
from backend.dsl.observation import InventoryElement

FIELD_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}
ACTION_ROLES = {"button", "link", "menuitem", "tab"}
TOGGLE_ROLES = {"checkbox", "switch", "radio", "menuitemcheckbox"}

AUTOCOMPLETE_PURPOSE = {
    "username": "username_or_email", "email": "email", "current-password": "password",
    "new-password": "password", "tel": "phone", "tel-national": "phone", "given-name": "first_name",
    "family-name": "last_name", "name": "full_name", "street-address": "address", "address-line1": "address",
    "address-level2": "city", "address-level1": "state", "country": "country", "country-name": "country",
    "postal-code": "postal_code", "organization": "company", "bday": "date", "cc-number": "card_number",
    "cc-csc": "cvv", "one-time-code": "otp",
}


def _attr_words(el: InventoryElement) -> str:
    parts = [el.dom_id, el.name_attr, el.test_id]
    return " ".join(normalize_text(p) for p in parts if p)


NON_CLASSIFIED_ROLES = {"heading", "form", "group", "region", "list", "table", "grid", "radiogroup", "tabpanel",
                        "dialog", "alertdialog", "row", "cell", "listitem", "paragraph", "status", "alert", "iframe",
                        "option", "article", "figure", "navigation", "banner", "main", "contentinfo", "note", "log"}

# Icon glyphs commonly used as the only content of a control.
GLYPHS = {"★": "star favorite", "☆": "star favorite", "♥": "like favorite heart", "❤": "like favorite heart",
          "☰": "menu hamburger", "≡": "menu hamburger", "×": "close dismiss", "✕": "close dismiss", "✖": "close dismiss",
          "⚙": "settings gear", "🔍": "search", "🗑": "delete trash", "✎": "edit", "✏": "edit", "+": "add plus",
          "＋": "add plus", "⋮": "more options menu", "⋯": "more options menu", "…": "more options menu",
          "↻": "refresh reload", "⟳": "refresh reload", "⬇": "download", "⬆": "upload", "✓": "confirm done check",
          "✔": "confirm done check", "←": "back previous", "→": "next forward", "‹": "previous", "›": "next"}


def glyph_words(text: str | None) -> str:
    if not text:
        return ""
    stripped = text.strip()
    return " ".join(words for glyph, words in GLYPHS.items() if glyph in stripped) if len(stripped) <= 3 else ""


def classify_element(el: InventoryElement) -> tuple[str | None, float]:
    role = el.role
    if role in NON_CLASSIFIED_ROLES:
        return None, 0.0
    kinds: tuple[str, ...] | None
    if role in FIELD_ROLES:
        kinds = ("field",)
    elif role in ACTION_ROLES or (role in ("generic", "img") and el.clickable_hint):
        kinds = ("action", "page")
    elif role in TOGGLE_ROLES:
        kinds = ("toggle",)
    else:
        kinds = None

    input_type = (el.input_type or "").lower()
    if role in FIELD_ROLES or el.tag in ("input", "textarea"):
        if input_type == "password":
            text = normalize_text(" ".join(filter(None, [el.name, el.label, el.placeholder, _attr_words(el)])))
            if re.search(r"\b(confirm|repeat|re ?enter|retype|verify|again)\b", text):
                return "confirm_password", 0.97
            return "password", 0.98
        if input_type == "email":
            return ("username_or_email" if re.search(r"\b(user|login)\b", normalize_text(el.name or "")) else "email",
                    0.95)
        if input_type == "tel":
            return "phone", 0.9
        if input_type == "search" or role == "searchbox":
            return "search", 0.9
        if input_type == "file":
            return "upload", 0.9
        if input_type == "date":
            return "date", 0.9
        if el.autocomplete:
            for token in el.autocomplete.split():
                if token in AUTOCOMPLETE_PURPOSE:
                    return AUTOCOMPLETE_PURPOSE[token], 0.93

    best: tuple[str | None, float] = (None, 0.0)
    icon_words = " ".join(t[5:] for t in el.nearby_text if t.startswith("icon:"))
    for source, weight in ((el.name, 1.0), (el.label, 0.98), (el.placeholder, 0.9), (el.title, 0.85),
                           (el.text if role not in FIELD_ROLES else None, 0.9), (_attr_words(el), 0.8),
                           (glyph_words(el.text or el.name), 0.85), (icon_words, 0.75)):
        if not source:
            continue
        match = classify_phrase(source, kinds=kinds)
        if match is None:
            continue
        score = match.score * weight
        if score > best[1]:
            best = (match.purpose, score)
    purpose, score = best
    if purpose and role in TOGGLE_ROLES and PURPOSES[purpose].kind not in ("toggle", "field"):
        return None, 0.0
    return purpose, round(min(score, 0.99), 3)
