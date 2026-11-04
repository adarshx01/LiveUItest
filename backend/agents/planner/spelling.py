"""Spelling repair for test instructions ("Clik on the Loign button", "Selct Indai from contry").

Only the *grammar* of an instruction is corrected: the leading verb, control nouns ("buton", "dropdwon"),
assertion predicates ("dispalyed", "redirected") and field/page words used to recognise purposes
("usrname", "pasword", "dashbaord"). Data is never touched:

* quoted text, ``{{variables}}``, URLs, e-mail addresses and anything containing digits;
* the value of an entry step ("Enter username as **abc**", "Type **abc** into ...");
* the option of a choice ("Select **Indai** from ..."): options are matched against the real list at run
  time, where "Indai" -> "India" is decided with the page in view;
* expected texts of assertions ("Verify **Welcom back** is displayed"): fixing them would turn a real
  typo in the product, or in the test's expectation, into a silent pass.

A word is only replaced when it is not a known word itself, the repair is a plausible slip (the same
weighted typo distance the matcher uses), and no other vocabulary word is about as close. Candidates
are tried in priority order, so the frequent reading wins a tie ("usrname" -> username, not surname).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.agents.grounding.fuzzy import KNOWN_WORDS, typo_allowance, typo_cost

VERBS: tuple[str, ...] = (
    "click", "enter", "type", "select", "verify", "check", "choose", "fill", "open", "navigate", "press", "uncheck",
    "input", "pick", "tap", "submit", "search", "upload", "download", "hover", "scroll", "wait", "switch", "close",
    "clear", "accept", "dismiss", "drag", "drop", "ensure", "confirm", "assert", "validate", "expect", "toggle",
    "tick", "untick", "expand", "collapse", "visit", "provide", "write", "insert", "attach", "login", "logout",
    "capture", "take", "refresh", "reload", "double", "right", "should", "go", "make", "see", "set", "add", "use",
    "put", "hit", "follow", "activate", "launch", "reveal", "show", "observe", "return", "go")

STRUCTURE_WORDS: tuple[str, ...] = (
    "button", "field", "dropdown", "checkbox", "link", "textbox", "menu", "option", "icon", "page", "form",
    "dialog", "modal", "popup", "slider", "radio", "toggle", "switch", "tab", "list", "table", "section", "header",
    "footer", "displayed", "display", "displays", "visible", "shown", "shows", "appears", "appear", "present",
    "redirected", "navigated", "loaded", "enabled", "disabled", "checked", "unchecked", "selected", "contains",
    "exists", "hidden", "invisible", "disappears", "opened", "closed", "successful", "message", "error", "title",
    "heading", "dashboard", "homepage", "settings", "profile", "account", "inbox", "orders", "should")

FIELD_WORDS: tuple[str, ...] = (
    "username", "password", "email", "address", "country", "state", "city", "phone", "mobile", "number", "first",
    "last", "name", "company", "search", "quantity", "comment", "description", "postal", "code", "birth", "date",
    "login", "logout", "register", "submit", "sign", "continue", "remember", "forgot", "terms", "conditions",
    "upload", "delete", "remove", "cancel", "save", "next", "previous", "confirm", "checkout", "cart", "basket",
    "surname")

FILL_VERBS = {"enter", "type", "input", "fill", "write", "provide", "insert", "supply", "put", "key", "set", "add",
              "use", "populate", "complete"}
SELECT_VERBS = {"select", "choose", "pick"}
ASSERT_VERBS = {"verify", "ensure", "assert", "validate", "expect", "confirm", "check", "make", "should", "see",
                "observe"}

PROTECT_RE = re.compile(r'"[^"]*"|\'[^\']*\'|`[^`]*`|\{\{[^}]*\}\}|https?://\S+|www\.\S+|\S+@\S+\.\S+|\S*\d\S*')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
AMBIGUITY = 0.35


@dataclass
class SpellingResult:
    text: str
    corrections: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.corrections)

    def note(self) -> str:
        return "spelling: " + ", ".join(f"{a}->{b}" for a, b in self.corrections)


def _best(word: str, vocabulary: tuple[str, ...]) -> str | None:
    w = word.lower()
    if len(w) < 3 or w in KNOWN_WORDS or w in vocabulary:
        return None
    candidates: list[tuple[float, int, str]] = []
    for rank, cand in enumerate(vocabulary):
        if abs(len(cand) - len(w)) > 3 or cand[0] != w[0] and not (len(w) > 1 and cand[:2] == w[1::-1]):
            continue
        budget = typo_allowance(len(w))  # judged on the word as written: "ordr" is one slip from "order", not "orders"
        cost = typo_cost(w, cand, budget)
        if cand[0] != w[0]:
            cost = max(cost, 0.8)
        if cost <= budget:
            candidates.append((cost, rank, cand))
    if not candidates:
        return None
    best_cost = min(c[0] for c in candidates)
    plausible = sorted((c for c in candidates if c[0] <= best_cost + AMBIGUITY), key=lambda c: (c[1], c[0]))
    return plausible[0][2]  # equally plausible readings: the more frequent word wins (username over surname)


def _protected_spans(line: str, verb: str | None, is_assert: bool) -> tuple[list[tuple[int, int]],
                                                                         list[tuple[int, int]]]:
    """(hard, soft) spans: hard spans are data; soft spans may hold expected text or a target noun."""

    spans = [(m.start(), m.end()) for m in PROTECT_RE.finditer(line)]
    soft: list[tuple[int, int]] = []
    low = line.lower()
    if verb in FILL_VERBS:
        m = re.search(r"\s(?:as|with|to|=)\s|:\s", low)
        if m:
            spans.append((m.start(), len(line)))  # "Enter username as <value>"
        m = re.search(r"\s(?:into|in\s+to)\s", low)
        if m:
            spans.append((len(verb), m.start()))  # "Type <value> into the field"
    elif verb in SELECT_VERBS:
        m = re.search(r"\s(?:from|in|on|under|within|for)\s", low)
        if m:
            spans.append((len(verb), m.start()))  # "Select <option> from the dropdown"
        m = re.search(r"\s(?:as|to)\s|:\s", low)
        if m and not re.search(r"\s(?:from|in|on|under|within|for)\s", low[: m.start()]):
            spans.append((m.start(), len(line)))  # "Select country as <option>"
    elif verb == "search":
        m = re.search(r"\sfor\s", low)
        if m:
            spans.append((m.start(), len(line)))
    if is_assert:
        m = re.search(r"\b(?:displays|shows|contains|reads|says|equals|matches|titled|named|with\s+text)\s", low)
        if m:
            spans.append((m.end(), len(line)))  # "...displays <expected text>"
        m = re.search(r"\bthat\s+(.+?)\s+(?:is|are|was|were|should|appears?|gets?)\b", low)
        if m:
            soft.append((m.start(1), m.end(1)))  # "Verify that <expected text | target> is displayed"
        m = re.match(r"^\w+\s+(?!that\b)(.+?)\s+(?:is|are|appears?)\s+\w+", low)
        if m:
            soft.append((m.start(1), m.end(1)))
    return spans, soft


def correct_instruction(line: str) -> SpellingResult:
    """Repair misspelled grammar words of one instruction line; data stays byte-identical."""

    words = list(WORD_RE.finditer(line))
    if not words:
        return SpellingResult(line)
    corrections: list[tuple[int, int, str, str]] = []
    first = words[0]
    verb = first.group(0).lower()
    fixed_verb = _best(verb, VERBS)
    if fixed_verb is not None:
        corrections.append((first.start(), first.end(), first.group(0), fixed_verb))
        verb = fixed_verb
    if verb in ("double", "right") and len(words) > 1:
        second = _best(words[1].group(0), ("click",))
        if second is not None:
            corrections.append((words[1].start(), words[1].end(), words[1].group(0), second))
    low = line.lower()
    is_assert = verb in ASSERT_VERBS and not (verb in ("check", "confirm") and
                                              not re.match(r"^\w+\s+(?:that|if|whether)\b", low))
    hard, soft = _protected_spans(line, verb if (is_assert or verb not in ASSERT_VERBS) else None, is_assert)
    vocabulary = STRUCTURE_WORDS if is_assert else STRUCTURE_WORDS + FIELD_WORDS
    for m in words[1:]:
        if any(s <= m.start() < e for s, e in hard):
            continue
        if any(s <= m.start() < e for s, e in soft):
            # an assertion subject may be expected text: only control/predicate nouns are repaired there
            fixed = _best(m.group(0), STRUCTURE_WORDS)
            if fixed is not None:
                corrections.append((m.start(), m.end(), m.group(0), fixed))
            continue
        if any(c[0] == m.start() for c in corrections):
            continue
        fixed = _best(m.group(0), vocabulary)
        if fixed is not None:
            corrections.append((m.start(), m.end(), m.group(0), fixed))
    if not corrections:
        return SpellingResult(line)
    out = line
    for start, end, original, fixed in sorted(corrections, reverse=True):
        replacement = fixed.capitalize() if original[:1].isupper() else fixed
        out = out[:start] + replacement + out[end:]
    return SpellingResult(out, [(o, f) for _, _, o, f in sorted(corrections)])
