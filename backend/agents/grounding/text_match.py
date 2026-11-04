"""Text similarity used by grounding, assertions and memory re-grounding."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

from backend.agents.planner.lexicon import STOPWORDS, normalize_text


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


@lru_cache(maxsize=20000)
def token_set(text: str) -> frozenset[str]:
    return frozenset(_stem(t) for t in normalize_text(text).split() if t and t not in STOPWORDS)


BADGE_RE = re.compile(r"\s*(?:\(\s*\d+\s*\)|\[\s*\d+\s*\]|·\s*\d+|\d+\s*(?:new|items?)?)\s*$", re.IGNORECASE)


def strip_badge(text: str) -> str:
    """'Cart (3)' -> 'Cart', 'Inbox [12]' -> 'Inbox' (counts are data, not identity)."""

    stripped = BADGE_RE.sub("", text or "").strip()
    return stripped or (text or "")


@lru_cache(maxsize=50000)
def similarity(a: str, b: str) -> float:
    """Symmetric-ish similarity in [0, 1] tuned for UI labels."""

    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    compact_a, compact_b = na.replace(" ", ""), nb.replace(" ", "")
    if compact_a == compact_b:  # "log in" vs "login", "sign in" vs "signin"
        return 0.95
    best = 0.0
    # a phrase embedded in a longer label is weaker evidence the more extra words the label has
    if re.search(rf"(^|\s){re.escape(na)}(\s|$)", nb):
        best = max(best, 0.66 + 0.28 * len(na) / len(nb))
    if re.search(rf"(^|\s){re.escape(nb)}(\s|$)", na):
        best = max(best, 0.6 + 0.25 * len(nb) / len(na))
    ta, tb = token_set(na), token_set(nb)
    if ta and tb:
        inter = len(ta & tb)
        if inter:
            jaccard = inter / len(ta | tb)
            containment = inter / min(len(ta), len(tb))
            best = max(best, 0.55 * jaccard + 0.3 * containment)
    ratio = SequenceMatcher(None, compact_a, compact_b).ratio()
    if ratio >= 0.8:
        best = max(best, ratio * 0.82)
    return round(min(best, 0.99), 4)


def contains_text(haystack: str, needle: str, *, mode: str = "case_insensitive") -> bool:
    if needle is None:
        return False
    if mode == "exact":
        return needle in haystack
    h = re.sub(r"\s+", " ", haystack).lower()
    n = re.sub(r"\s+", " ", str(needle)).lower().strip()
    if mode in ("case_insensitive", "contains"):
        return n in h
    if mode == "semantic":
        tokens = [t for t in token_set(n)]
        if not tokens:
            return n in h
        htokens = token_set(h)
        hits = sum(1 for t in tokens if t in htokens or any(ht.startswith(t[:5]) for ht in htokens if len(t) >= 5))
        return hits / len(tokens) >= 0.75
    if mode == "regex":
        return re.search(str(needle), haystack) is not None
    return n in h


def split_identifier(text: str | None) -> str:
    """'loginSubmitBtn' / 'login-submit' / 'login_submit' -> 'login submit btn'."""

    if not text:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return normalize_text(re.sub(r"[_\-.#\[\]=\"']+", " ", text))
