"""Choosing the option a test means, among the options a control really offers.

Used for native ``<select>`` (labels, values, optgroups, disabled options), ``<datalist>``
suggestions, custom listboxes and autocomplete suggestions. The order of evidence is fixed and
reported, so a report can say *why* "Indai" became "India":

exact label -> case-insensitive -> normalized -> option value / code -> alias ("USA") ->
unique word containment -> ordinal ("the second option") -> unique typo-tolerant best match.

Never a "closest guess": a typo match must be clearly better than the runner-up, a containment match
must be unique, placeholders ("-- Select --") are never chosen by fuzzy rules, and a disabled
option is reported as such instead of being silently skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.agents.grounding.fuzzy import best_match, match_score, option_aliases
from backend.agents.planner.lexicon import ORDINALS, normalize_text

PLACEHOLDER_RE = re.compile(r"^\s*(?:[-—–_*.]+\s*)?(?:please\s+)?(?:select|choose|pick|none|any|all|--|—)"
                            r"(?:\s+(?:an?|one|the|your))?(?:\s+\w+){0,3}\s*(?:[-—–_*.]+|\.\.\.|…)?\s*$",
                            re.IGNORECASE)
ORDINAL_OPTION_RE = re.compile(r"^(?:the\s+)?(?P<ord>first|second|third|fourth|fifth|last|final|1st|2nd|3rd|4th|5th|"
                               r"\d+(?:st|nd|rd|th))\s+(?:option|item|entry|choice|value)$", re.IGNORECASE)


@dataclass
class OptionInfo:
    label: str
    value: str | None = None
    disabled: bool = False
    group: str | None = None
    selected: bool = False
    index: int = 0

    @property
    def is_placeholder(self) -> bool:
        return (not self.label.strip()) or bool(PLACEHOLDER_RE.match(self.label)) and (self.value in (None, "", "0",
                                                                                                       "-1"))

    @classmethod
    def coerce(cls, options: list[Any]) -> list[OptionInfo]:
        out: list[OptionInfo] = []
        for i, o in enumerate(options):
            if isinstance(o, OptionInfo):
                o.index = i
                out.append(o)
            elif isinstance(o, dict):
                out.append(cls(label=str(o.get("label") or o.get("text") or "").strip(),
                               value=None if o.get("value") is None else str(o.get("value")),
                               disabled=bool(o.get("disabled")), group=o.get("group") or None,
                               selected=bool(o.get("selected")), index=i))
            else:
                out.append(cls(label=str(o).strip(), index=i))
        return out


@dataclass
class OptionMatch:
    option: OptionInfo | None
    how: str  # exact | case_insensitive | normalized | value | alias | contains | ordinal | fuzzy | not_found | ambiguous
    score: float = 0.0
    alternatives: list[str] = field(default_factory=list)  # "did you mean" when nothing matched

    @property
    def label(self) -> str | None:
        return self.option.label if self.option else None

    @property
    def found(self) -> bool:
        return self.option is not None

    def describe(self) -> str:
        if self.option is None:
            hint = f"; closest: {', '.join(self.alternatives[:3])}" if self.alternatives else ""
            return f"no option matched ({self.how}){hint}"
        if self.how == "exact":
            return f"option '{self.option.label}'"
        return f"option '{self.option.label}' ({self.how}, {self.score:.2f})"


def _ordinal(wanted: str, options: list[OptionInfo]) -> OptionInfo | None:
    m = ORDINAL_OPTION_RE.match(wanted.strip())
    if not m:
        return None
    raw = m.group("ord").lower()
    index = ORDINALS.get(raw)
    if index is None:
        digits = re.match(r"\d+", raw)
        index = int(digits.group(0)) if digits else None
    real = [o for o in options if not o.is_placeholder]
    if index is None or not real:
        return None
    position = index - 1 if index > 0 else len(real) + index
    return real[position] if 0 <= position < len(real) else None


def match_option(options: list[Any], wanted: str, *, allow_fuzzy: bool = True) -> OptionMatch:
    opts = OptionInfo.coerce(options)
    want = (wanted or "").strip()
    if not opts or not want:
        return OptionMatch(None, "not_found")
    for o in opts:
        if o.label == want:
            return OptionMatch(o, "exact", 1.0)
    low = want.lower()
    for o in opts:
        if o.label.lower() == low:
            return OptionMatch(o, "case_insensitive", 0.99)
    norm = normalize_text(want)
    for o in opts:
        if normalize_text(o.label) == norm:
            return OptionMatch(o, "normalized", 0.98)
    for o in opts:
        if o.value is not None and o.value.strip() and (o.value == want or o.value.lower() == low
                                                        or normalize_text(o.value) == norm):
            return OptionMatch(o, "value", 0.97)
    aliases = option_aliases(want)
    if aliases:
        for o in opts:
            label_norm = normalize_text(o.label)
            if label_norm in aliases or (o.value and normalize_text(o.value) in aliases):
                return OptionMatch(o, "alias", 0.95)
    real = [o for o in opts if not o.is_placeholder]
    contains = [o for o in real if norm and re.search(rf"(^|\s){re.escape(norm)}(\s|$)", normalize_text(o.label))]
    if len(contains) == 1:
        return OptionMatch(contains[0], "contains", 0.9)
    ordinal = _ordinal(want, opts)
    if ordinal is not None:
        return OptionMatch(ordinal, "ordinal", 0.9)
    if allow_fuzzy and real:
        index, score, _margin = best_match(want, [o.label for o in real], min_score=0.82, min_margin=0.05)
        if index is not None:
            return OptionMatch(real[index], "fuzzy", score)
    ranked = sorted(real, key=lambda o: -match_score(want, o.label))[:3]
    how = "ambiguous" if len(contains) > 1 else "not_found"
    return OptionMatch(None, how, alternatives=[o.label for o in ranked if match_score(want, o.label) >= 0.4])


def split_choices(wanted: str, options: list[Any]) -> list[str]:
    """'Python, SQL and Go' -> three choices, unless the whole text is itself one option ('Salt and pepper')."""

    whole = match_option(options, wanted, allow_fuzzy=False)
    if whole.found:
        return [wanted]
    parts = [p.strip().strip("'\"") for p in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s*;\s*|\s*\|\s*", wanted)]
    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else [wanted]
