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
        return n in h or phrase_near(h, n)
    if mode == "regex":
        return re.search(str(needle), haystack) is not None
    return n in h


def split_identifier(text: str | None) -> str:
    """'loginSubmitBtn' / 'login-submit' / 'login_submit' -> 'login submit btn'."""

    if not text:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return normalize_text(re.sub(r"[_\-.#\[\]=\"']+", " ", text))


_DIGIT_RE = re.compile(r"\d")
# words a paraphrase adds or drops without changing what it says ("has been placed" / "was placed")
_FILLER = STOPWORDS | frozenset("a an the is are was were be been being has have had will shall would should can could "
                                "may might must do does did of to for in on at by with and or now just your my our their "
                                "this that these those it its here there".split())


def phrase_near(haystack: str, needle: str, *, threshold: float = 0.84) -> bool:
    """A paraphrase-tolerant match that still means "this sentence is on the page".

    The words of the expectation must appear together, in order, in one place: a run of page words about as long
    as the expectation that reads like it (typos, plural/singular, a word added or dropped). Words scattered over
    the page never add up to a match - dropdown options "2026/04 ... 2026/09" are not "Period: 2026/04 ～ 2026/09".
    Every number, date or code in the expectation must be on the page in that same run, exactly.
    """

    want = normalize_text(needle).split()
    words = normalize_text(haystack).split()
    if not want or not words:
        return False
    must = [w for w in want if _DIGIT_RE.search(w)]
    target = " ".join(want)
    n = len(want)
    lengths = (n, n + 1, n + 2)  # never shorter: the dropped word could be the one that matters (a name)
    core = [w for w in want if w not in _FILLER or _DIGIT_RE.search(w)] or want
    # only look where the page has a word that starts like one of the expected words (pages run to 20k words)
    prefixes = {w[:3] for w in core}
    starts: set[int] = set()
    for i, word in enumerate(words):
        if word[:3] in prefixes:
            starts.update(range(max(0, i - n - 1), i + 1))
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    for start in sorted(starts):
        for length in lengths:
            window = words[start:start + length]
            if len(window) < n:
                break
            if must and not all(m in window for m in must):
                continue
            matcher.set_seq1(" ".join(window))
            if matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold and \
                    matcher.ratio() >= threshold:
                return True
        # word by word, in order, close together: "order placed successfuly" ~ "order has been placed successfully"
        if _word_like(words[start], core[0]) and _aligned(words[start:start + n + 3], core):
            return True
    return False


def _word_like(page_word: str, wanted: str) -> bool:
    if page_word == wanted:
        return True
    if _DIGIT_RE.search(wanted) or _DIGIT_RE.search(page_word):
        return False  # numbers, dates and codes are exact
    if _stem(page_word) == _stem(wanted):
        return True
    return len(wanted) >= 4 and SequenceMatcher(None, page_word, wanted).ratio() >= 0.82


def _aligned(window: list[str], core: list[str]) -> bool:
    """Each wanted word in order; the page words skipped between them may only be filler ("has been")."""

    j, skipped = 0, 0
    allowed = 0 if len(core) <= 3 else 1
    for wanted in core:
        while j < len(window) and not _word_like(window[j], wanted):
            if j > 0 and window[j] not in _FILLER:
                skipped += 1
                if skipped > allowed:
                    return False
            j += 1
        if j == len(window):
            return False
        j += 1
    return True

