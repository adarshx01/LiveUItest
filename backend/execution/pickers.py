"""Custom date and time pickers (the ones a page draws itself, not ``<input type=date|time>``).

Date: an editable field with a recognisable placeholder ("dd/mm/yyyy", "DD MMM YYYY") is typed in that
format first and read back. Otherwise (read-only input, button, or the typed value did not stick) the
calendar the trigger opens is driven: month/year selects when it has them, prev/next navigation until
the header shows the wanted month, then the day cell of *that* month (cells of the adjacent months and
disabled days are never clicked).

Time: an editable free-text field takes "HH:MM" directly; otherwise the hour / minute (/ AM-PM) fields
or columns the trigger reveals are set one by one and the picker is confirmed (OK / Set / Done).

Both read the field back and only report success when it shows the requested date or time; the action
verifier re-reads it independently (``value_matches``).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from typing import TYPE_CHECKING, Any

from backend.agents.forms.value_normalizer import MONTHS, parse_date, parse_time
from backend.browser.js_probes import CALENDAR_STATE, ELEMENT_STATE, TIME_FIELDS
from backend.core.logging import get_logger
from backend.dsl.observation import InventoryElement
from backend.dsl.results import FailureCategory

if TYPE_CHECKING:  # pragma: no cover
    from backend.browser.tool_adapter import BrowserToolAdapter
    from backend.execution.action_executor import ActionOutcome

log = get_logger(__name__)

MAX_MONTH_STEPS = 240  # 20 years of prev/next clicks
OPEN_WAIT_S = 2.5
WEEKDAYS = r"(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s*"
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
DATE_HINT = re.compile(r"\b(date|dob|birth|day|calendar|from|to|start|end|due|deadline|expiry|expires|valid)\b", re.I)
TIME_HINT = re.compile(r"\b(time|hour|hh|clock|slot|at)\b", re.I)


def target_of(el: InventoryElement) -> str | None:
    return el.ref or el.css_selector


def describe(el: InventoryElement) -> str:
    return " ".join(filter(None, [el.name, el.label, el.placeholder, el.text, el.name_attr, el.dom_id, el.title]))


def looks_like_date(text: str) -> dt.date | None:
    """A value that is unambiguously a calendar date (has a year, or a month name)."""

    raw = text.strip()
    if not re.search(r"\d", raw) or not (re.search(r"\b(19|20)\d{2}\b", raw) or
                                         any(re.search(rf"\b{m}", raw, re.I) for m in MONTHS)):
        return None
    return parse_date(re.sub(rf"^{WEEKDAYS}", "", raw, flags=re.I))


def looks_like_time(text: str) -> str | None:
    raw = text.strip()
    if not re.fullmatch(r"\d{1,2}\s*[:.h]\s*\d{2}(\s*[ap]\.?m\.?)?|\d{1,2}\s*[ap]\.?m\.?", raw, re.I):
        return None
    return parse_time(re.sub(r"\s*[.h]\s*(?=\d{2})", ":", re.sub(r"\s*:\s*", ":", raw)))


def date_format_of(placeholder: str | None) -> str | None:
    """'dd/mm/yyyy' -> '%d/%m/%Y'; 'DD MMM YYYY' -> '%d %b %Y'; None when the placeholder is not a format."""

    if not placeholder:
        return None
    p = placeholder.strip()
    if not re.search(r"y{2,4}", p, re.I) or not re.search(r"d{1,2}", p, re.I):
        return None
    out = re.sub(r"y{4}", "%Y", p, flags=re.I)
    out = re.sub(r"y{2}", "%y", out, flags=re.I)
    out = re.sub(r"m{4}", "%B", out, flags=re.I)
    out = re.sub(r"m{3}", "%b", out, flags=re.I)
    out = re.sub(r"m{1,2}", "%m", out, flags=re.I)
    out = re.sub(r"d{1,2}", "%d", out, flags=re.I)
    return out if "%m" in out or "%b" in out or "%B" in out else None


def _day_first(placeholder: str | None) -> bool | None:
    fmt = date_format_of(placeholder)
    if not fmt:
        return None
    return fmt.index("%d") < (fmt.find("%m") if "%m" in fmt else 99)


def value_matches(kind: str, raw: str | None, wanted: str, *, placeholder: str | None = None) -> bool:
    """Does the field's displayed value mean the wanted date ("2026-06-09") or time ("23:15")?"""

    if not raw:
        return False
    text = raw.strip()
    if kind == "time":
        m = re.search(r"(\d{1,2})\s*[:.h]\s*(\d{2})(?:\s*:\s*\d{2})?\s*([ap]\.?m\.?)?", text, re.I)
        if not m:
            return False
        got = parse_time(f"{m.group(1)}:{m.group(2)}{(' ' + m.group(3).replace('.', '')) if m.group(3) else ''}")
        return got is not None and got[:5] == wanted[:5]
    target = dt.date.fromisoformat(wanted)
    text = re.sub(rf"^{WEEKDAYS}", "", text, flags=re.I)
    candidates: set[dt.date | None] = set()
    order = _day_first(placeholder)
    for day_first in ([order] if order is not None else [True, False]):
        candidates.add(parse_date(text, day_first=day_first, today=target))
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})", text)  # "2026-06-09T00:00"
    if m:
        candidates.add(dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return target in candidates


class PickerDriver:
    def __init__(self, adapter: BrowserToolAdapter) -> None:
        self.adapter = adapter

    # ------------------------------------------------------------------ helpers
    async def _read(self, el: InventoryElement) -> str:
        for handle in dict.fromkeys(h for h in (el.ref, el.css_selector) if h):
            state = await self.adapter.probe(ELEMENT_STATE, handle, el.display()) or {}
            if state:
                return str(state.get("value") or state.get("text") or "")
        return ""

    async def _probe_until(self, probe: Any, target: str, ready: Any, timeout: float = OPEN_WAIT_S) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        state: dict[str, Any] = {}
        while True:
            state = await self.adapter.probe(probe, target, "picker") or {}
            if ready(state) or asyncio.get_running_loop().time() >= deadline:
                return state
            await asyncio.sleep(0.25)

    def _fail(self, error: str, category: FailureCategory, expected: dict[str, Any],
              notes: list[str]) -> ActionOutcome:
        from backend.execution.action_executor import ActionOutcome

        return ActionOutcome(False, error=error, failure_category=category, expected=expected, notes=notes)

    # ------------------------------------------------------------------ date
    async def set_date(self, el: InventoryElement, wanted: dt.date, original: str) -> ActionOutcome | None:
        """Returns None when no calendar opens and the field is not typeable (caller falls back)."""

        from backend.execution.action_executor import ActionOutcome

        target = target_of(el)
        iso = wanted.isoformat()
        expected: dict[str, Any] = {"value": original, "date": iso, "kind": "date", "trigger_target": target,
                                    "trigger_css": el.css_selector, "placeholder": el.placeholder}
        notes: list[str] = []
        if target is None:
            return self._fail("date field is not addressable", FailureCategory.ELEMENT_NOT_FOUND, expected, notes)
        typeable = el.tag in ("input", "textarea") and el.editable and not el.readonly
        fmt = date_format_of(el.placeholder)
        if typeable:
            # what the tester wrote first ("2025/12/25" is usually the format the application expects), then the
            # field's own placeholder format; the calendar only when typing does not stick
            attempts = [original]
            if fmt and wanted.strftime(fmt) != original:
                attempts.append(wanted.strftime(fmt))
            for text in attempts:
                response = await self.adapter.type_text(target, el.display(), text)
                await self.adapter.press_key("Escape")  # close a calendar that opened on focus
                shown = await self._read(el)
                if response.ok and not value_matches("date", shown, iso, placeholder=el.placeholder):
                    await self.adapter.press_key("Tab")  # some fields only take the value when left
                    shown = await self._read(el)
                if response.ok and value_matches("date", shown, iso, placeholder=el.placeholder):
                    notes.append(f"typed {text!r}" + (f" in the field's own format ({el.placeholder})"
                                                      if text != original else ""))
                    expected["confirmed"] = shown
                    return ActionOutcome(True, tool="browser_type", args={"target": target, "text": text},
                                         expected=expected, notes=notes)
                notes.append(f"typing {text!r} did not stick (field shows {shown!r})")
            notes.append("using the calendar")

        await self.adapter.click(target, el.display())
        cal = await self._probe_until(CALENDAR_STATE, target, lambda s: s.get("found"))
        if not cal.get("found"):
            if typeable:
                return self._fail(f"the date field does not keep {original!r} and opens no calendar",
                                  FailureCategory.FORM_VALIDATION_FAILED, expected, notes)
            return None
        notes.append(f"calendar opened (showing {cal.get('header') or 'unknown month'})")

        # month / year dropdowns inside the calendar
        if cal.get("yearSelect") and str(wanted.year) in cal.get("yearOptions", []):
            await self.adapter.select_option(cal["yearSelect"], "calendar year", [str(wanted.year)])
        if cal.get("monthSelect"):
            label = next((o for o in cal.get("monthOptions", [])
                          if MONTHS.get(o.lower()[:3]) == wanted.month), None)
            if label:
                await self.adapter.select_option(cal["monthSelect"], "calendar month", [label])
        if cal.get("yearSelect") or cal.get("monthSelect"):
            cal = await self.adapter.probe(CALENDAR_STATE, target, "calendar") or cal

        steps = 0
        while cal.get("month") and cal.get("year") and (cal["year"], cal["month"]) != (wanted.year, wanted.month):
            forward = (wanted.year, wanted.month) > (cal["year"], cal["month"])
            nav = cal.get("next") if forward else cal.get("prev")
            if not nav or steps >= MAX_MONTH_STEPS:
                return self._fail(f"calendar cannot reach {wanted:%B %Y} (shows {cal.get('header')!r}, "
                                  f"no {'next' if forward else 'previous'} control)",
                                  FailureCategory.OPTION_NOT_FOUND, expected, notes)
            before = (cal["year"], cal["month"])
            await self.adapter.click(nav, "next month" if forward else "previous month")
            steps += 1
            cal = await self._probe_until(CALENDAR_STATE, target,
                                          lambda s: s.get("found") and (s.get("year"), s.get("month")) != before,
                                          timeout=1.5)
            if not cal.get("found"):
                return self._fail("calendar closed while navigating months", FailureCategory.DROPDOWN_NOT_OPEN,
                                  expected, notes)
        if steps:
            notes.append(f"navigated {steps} month(s) to {wanted:%B %Y}")
        if not cal.get("month"):
            notes.append("calendar header not recognised; choosing the day by its label")

        days = [d for d in cal.get("days", []) if d.get("day") == wanted.day]
        exact = [d for d in days if d.get("label") and value_matches("date", d["label"], iso)]
        in_month = [d for d in days if not d.get("outside")]
        candidates = exact or (in_month if cal.get("month") else [])
        cell = next((d for d in candidates if not d.get("disabled")), None)
        if cell is None:
            disabled = any(d.get("disabled") for d in candidates)
            return self._fail(f"{wanted:%d %B %Y} is {'disabled' if disabled else 'not offered'} in the calendar",
                              FailureCategory.ELEMENT_NOT_ENABLED if disabled else FailureCategory.OPTION_NOT_FOUND,
                              expected, notes)
        response = await self.adapter.click(cell["css"], f"day {wanted.day}")
        if not response.ok:
            return ActionOutcome.from_response(response, expected=expected, notes=notes)
        await asyncio.sleep(0.2)
        shown = await self._read(el)
        if shown and value_matches("date", shown, iso, placeholder=el.placeholder):
            expected["confirmed"] = shown
        else:
            still = await self.adapter.probe(CALENDAR_STATE, target, "calendar") or {}
            if still.get("found"):  # some pickers need an explicit confirmation
                await self.adapter.press_key("Enter")
            notes.append(f"field shows {shown!r} after picking the day")
        notes.append(f"picked {wanted:%d %B %Y} in the calendar")
        return ActionOutcome(True, tool="browser_click", args={"target": cell["css"], "date": iso},
                             expected=expected, notes=notes)

    # ------------------------------------------------------------------ time
    async def set_time(self, el: InventoryElement, wanted: str, original: str) -> ActionOutcome | None:
        from backend.execution.action_executor import ActionOutcome

        target = target_of(el)
        expected: dict[str, Any] = {"value": original, "time": wanted, "kind": "time", "trigger_target": target,
                                    "trigger_css": el.css_selector}
        notes: list[str] = []
        if target is None:
            return self._fail("time field is not addressable", FailureCategory.ELEMENT_NOT_FOUND, expected, notes)
        hour, minute = (int(x) for x in wanted.split(":")[:2])
        has = lambda s: bool((s.get("hour") and s.get("minute")) or len(s.get("columns") or []) >= 2)  # noqa: E731
        fields = await self.adapter.probe(TIME_FIELDS, target, el.display()) or {}
        if not has(fields):
            if el.tag in ("input", "textarea") and el.editable and not el.readonly:
                response = await self.adapter.type_text(target, el.display(), wanted)
                await self.adapter.press_key("Escape")
                shown = await self._read(el)
                if response.ok and value_matches("time", shown, wanted):
                    expected["confirmed"] = shown
                    return ActionOutcome(True, tool="browser_type", args={"target": target, "text": wanted},
                                         expected=expected, notes=["typed the time"])
                notes.append(f"typing {wanted} did not stick (field shows {shown!r}); opening the picker")
            await self.adapter.click(target, el.display())
            fields = await self._probe_until(TIME_FIELDS, target, has)
        if not has(fields):
            return None

        twelve = bool(fields.get("ampm"))
        h_value = (hour % 12 or 12) if twelve else hour
        if fields.get("hour") and fields.get("minute"):
            for field, number in ((fields["hour"], h_value), (fields["minute"], minute)):
                if not await self._set_field(field, number):
                    return self._fail(f"could not set {number:02d} in the {field['text'] or 'time'} field",
                                      FailureCategory.FORM_VALIDATION_FAILED, expected, notes)
            if twelve:
                await self._set_field(fields["ampm"], "PM" if hour >= 12 else "AM")
            notes.append(f"set hour field to {h_value:02d} and minute field to {minute:02d}"
                         + (f" {'PM' if hour >= 12 else 'AM'}" if twelve else ""))
        else:
            for column, number in zip(fields["columns"][:2], (h_value, minute), strict=False):
                item = next((c for c in column if c["text"].isdigit() and int(c["text"]) == number), None)
                if item is None:
                    return self._fail(f"{number:02d} is not offered by the time picker",
                                      FailureCategory.OPTION_NOT_FOUND, expected, notes)
                await self.adapter.click(item["css"], f"time {number:02d}")
            notes.append(f"picked {h_value:02d} and {minute:02d} in the time columns")
        if fields.get("confirm"):
            await self.adapter.click(fields["confirm"], "confirm time")
            notes.append("confirmed the time picker")
        await asyncio.sleep(0.2)
        shown = await self._read(el)
        if value_matches("time", shown, wanted):
            expected["confirmed"] = shown
        else:
            notes.append(f"field shows {shown!r} after setting the time")
        return ActionOutcome(True, tool="browser_type", args={"target": target, "time": wanted},
                             expected=expected, notes=notes)

    async def _set_field(self, field: dict[str, Any], value: int | str) -> bool:
        css = field["css"]
        if field.get("tag") == "select":
            options = field.get("options") or []
            label = next((o for o in options if (o.strip().isdigit() and isinstance(value, int)
                                                  and int(o) == value) or
                          (isinstance(value, str) and o.strip().upper().startswith(value[0]))), None)
            if label is None:
                return False
            return (await self.adapter.select_option(css, "time field", [label])).ok
        text = f"{value:02d}" if isinstance(value, int) else value
        return (await self.adapter.type_text(css, "time field", text)).ok
