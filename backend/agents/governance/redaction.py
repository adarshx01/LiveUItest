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

# Key names are split into words ("llm_input_tokens" -> llm/input/tokens, "apiKey" -> api/key) so that
# credential-shaped names are masked while counters such as "passed_steps" or "llm_input_tokens" are not.
_SENSITIVE_WORDS = frozenset({"password", "passwd", "passcode", "passphrase", "pass", "pwd", "secret", "token",
                              "authorization", "cookie", "cvv", "cvc", "ssn", "pin", "otp", "credential",
                              "credentials", "mfa", "totp"})
_SENSITIVE_PAIRS = frozenset({("api", "key"), ("private", "key"), ("access", "key"), ("secret", "key"),
                              ("session", "id"), ("card", "number"), ("credit", "card"), ("security", "code"),
                              ("one", "time")})
# words that are unambiguous even inside an unsplittable key ("userpassword", "accesstoken")
_SENSITIVE_INFIXES = ("password", "passwd", "passcode", "passphrase", "secret", "authorization", "apikey",
                      "privatekey", "accesskey", "sessionid", "cardnumber", "creditcard")
_SENSITIVE_SUFFIXES = ("token", "cookie")

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


def mask_values(text: str, values: list[str] | tuple[str, ...] | set[str]) -> str:
    """Mask the given concrete values (longest first) wherever they appear in ``text``."""

    if not text:
        return text
    for value in sorted({v for v in values if v}, key=len, reverse=True):
        if len(value) >= 3:
            text = text.replace(value, MASK)
        else:
            # very short secrets are only masked when quoted, to avoid destroying unrelated text
            for quote in ("'", '"', "`", "&#39;", "&#34;", "&quot;"):
                text = text.replace(f"{quote}{value}{quote}", f"{quote}{MASK}{quote}")
    return text


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
        values = set(_sensitive_values)
    text = mask_values(text, values)
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
            if (isinstance(key, str) and is_sensitive_name(key) and isinstance(item, (str, int, float))
                    and not isinstance(item, bool) and item != ""):
                out[key] = MASK
            else:
                out[key] = redact_value(item, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v, _depth=_depth + 1) for v in value)
    return value


def _name_words(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).lower())


def is_sensitive_name(name: str | None) -> bool:
    """True for credential-shaped names: ``test_password``, ``apiKey``, ``X-Auth-Token``, ``otp_code``..."""

    if not name:
        return False
    words = _name_words(name)
    if any(w in _SENSITIVE_WORDS for w in words) or any(p in _SENSITIVE_PAIRS for p in zip(words, words[1:])):
        return True
    joined = "".join(words)
    return any(i in joined for i in _SENSITIVE_INFIXES) or joined.endswith(_SENSITIVE_SUFFIXES)


_DATA_URI = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")


def redact_html(document: str) -> str:
    """Final redaction pass over rendered HTML.

    Embedded ``data:`` images are left byte-for-byte intact (masking inside base64 would corrupt them), and
    secret values are matched both raw and HTML-escaped (``a&b`` renders as ``a&amp;b``).
    """

    import html as _html

    with _lock:
        values = set(_sensitive_values)
    escaped = {_html.escape(v, quote=True) for v in values} | {_html.escape(v, quote=False) for v in values}
    variants = values | escaped

    def clean(segment: str) -> str:
        return redact_text(mask_values(segment, variants))

    out: list[str] = []
    last = 0
    for match in _DATA_URI.finditer(document):
        out.append(clean(document[last:match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(clean(document[last:]))
    return "".join(out)
