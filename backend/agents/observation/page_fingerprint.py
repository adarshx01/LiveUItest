"""Route normalization and structural page fingerprints (spec section 13.2).

``/users/123`` and ``/users/456`` normalize to ``/users/:id``; the structural
signature (landmarks, headings, forms, interactive controls) identifies the page
template independently of data values so memory can be keyed on it.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from backend.agents.planner.lexicon import normalize_text
from backend.core.ids import stable_hash
from backend.dsl.observation import FormInfo, InventoryElement, PageFingerprint

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
HEX_RE = re.compile(r"^[0-9a-f]{12,}$", re.IGNORECASE)
NUM_RE = re.compile(r"^\d+$")
SLUG_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d{3,}$", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@/]+@[^@/]+$")

SIGNATURE_ROLES = {"heading", "button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "switch",
                   "tab", "form", "dialog", "navigation", "main", "banner", "spinbutton", "listbox", "slider"}


def normalize_route(url: str) -> str:
    parsed = urlparse(url)
    segments = []
    for seg in parsed.path.split("/"):
        if not seg:
            continue
        if NUM_RE.match(seg):
            segments.append(":id")
        elif UUID_RE.match(seg):
            segments.append(":uuid")
        elif HEX_RE.match(seg):
            segments.append(":hash")
        elif EMAIL_RE.match(seg):
            segments.append(":email")
        elif SLUG_ID_RE.match(seg):
            segments.append(":slug")
        else:
            segments.append(seg.lower())
    route = "/" + "/".join(segments)
    if parsed.fragment and parsed.fragment.startswith("/"):
        route += "#" + normalize_route("http://x" + parsed.fragment)  # hash-router apps
    return route


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _token(el: InventoryElement) -> str | None:
    if el.role not in SIGNATURE_ROLES:
        return None
    name = normalize_text(el.name or el.label or el.placeholder or "")
    if el.role == "link" and len(name) > 30:
        return None  # long link texts are usually content, not structure
    if re.search(r"\d{2,}", name):
        name = re.sub(r"\d+", "#", name)
    return f"{el.role}:{name[:40]}"


def fingerprint(url: str, title: str, elements: list[InventoryElement], forms: list[FormInfo]) -> PageFingerprint:
    tokens = sorted({t for t in (_token(e) for e in elements if e.visible) if t})[:80]
    route = normalize_route(url)
    title_pattern = re.sub(r"\d+", "#", normalize_text(title))
    return PageFingerprint(
        route_pattern=route,
        host=host_of(url),
        structure_hash=stable_hash(route, *tokens),
        signature=tokens,
        form_hashes={f.purpose or f.form_id: f.fingerprint for f in forms},
        title_pattern=title_pattern,
    )
