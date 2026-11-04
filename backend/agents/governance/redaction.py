"""Sensitive-data redaction (spec sections 16.4 / 28).

Applied to logs, events, agent prompts, MCP tool arguments echoed in reports,
Playwright code returned by the MCP server, and persisted traces. Two layers:

1. exact masking of every secret value resolved during the process lifetime;
2. key- and pattern-based masking for credential-shaped data.
"""

from __future__ import annotations

import re
import threading
from typing import Any

MASK = "••••••"

_SENSITIVE_KEY = re.compile(
    r"(pass(word|wd|code)?|pwd|secret|token|api[-_]?key|authorization|cookie|session[-_]?id|"
    r"credit[-_]?card|card[-_]?number|cvv|cvc|ssn|\bpin\b|otp|private[-_]?key|access[-_]?key)",
    re.IGNORECASE,
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\bbearer\s+[a-z0-9._\-~+/]+=*"), "Bearer " + MASK),
    (re.compile(r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b"), MASK),  # JWT
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), MASK),  # AWS access key id
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), MASK),  # API-key shaped
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), MASK),
]

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_lock = threading.Lock()
_sensitive_values: set[str] = set()


def register_sensitive(value: str | None) -> None:
    """Register a concrete secret value so it is masked wherever it appears."""

    if not value:
        return
    with _lock:
        _sensitive_values.add(value)


def clear_sensitive_registry() -> None:
    with _lock:
        _sensitive_values.clear()


def _luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact_text(text: str) -> str:
    if not text:
        return text
    with _lock:
        values = sorted(_sensitive_values, key=len, reverse=True)
    for value in values:
        if len(value) >= 3:
            text = text.replace(value, MASK)
        else:
            # very short secrets are only masked when quoted, to avoid destroying unrelated text
            for quote in ("'", '"', "`"):
                text = text.replace(f"{quote}{value}{quote}", f"{quote}{MASK}{quote}")
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    def _card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return MASK if 13 <= len(digits) <= 19 and _luhn_valid(digits) else match.group(0)

    return _CARD.sub(_card, text)


def redact_value(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 12:
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_KEY.search(key) and isinstance(item, (str, int, float)) and item != "":
                out[key] = MASK
            else:
                out[key] = redact_value(item, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v, _depth=_depth + 1) for v in value)
    return value


def is_sensitive_name(name: str | None) -> bool:
    return bool(name and _SENSITIVE_KEY.search(name))
