"""Turning human-written values into what browser-rendered inputs accept.

Date, time, month, week, color and range inputs open pickers drawn by the browser or the OS. They
cannot be typed into character by character; they take one machine format (``2024-05-01``,
``14:30``, ``#ff0000``) and reject anything else. Tests are written the way people talk ("05/01/1990",
"1 May 1990", "2:30 PM", "tomorrow", "red", "50%"), so the executor normalizes before filling and the
verifier then expects the normalized value.

Ambiguous dates follow the configured locale (``en-US`` reads 05/01/1990 as May 1st); a day above 12
disambiguates either way. Anything that cannot be understood is returned unchanged with a reason, so
the failure points at the value instead of at a mysterious "Malformed value".
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], start=1)}
MONTHS.update({name[:3]: i for name, i in list(MONTHS.items())})
MONTHS["sept"] = 9

COLOR_NAMES = {
    "red": "#ff0000", "green": "#008000", "blue": "#0000ff", "black": "#000000", "white": "#ffffff",
    "yellow": "#ffff00", "orange": "#ffa500", "purple": "#800080", "pink": "#ffc0cb", "gray": "#808080",
    "grey": "#808080", "brown": "#a52a2a", "cyan": "#00ffff", "magenta": "#ff00ff", "teal": "#008080",
    "navy": "#000080", "maroon": "#800000", "olive": "#808000", "lime": "#00ff00", "silver": "#c0c0c0",
    "gold": "#ffd700", "violet": "#ee82ee", "indigo": "#4b0082",
}

PICKER_TYPES = {"date", "time", "datetime-local", "month", "week", "color", "range"}


@dataclass
class Normalized:
    value: str
    changed: bool
    note: str | None = None
    ok: bool = True


def _today() -> dt.date:
    return dt.date.today()


def parse_date(text: str, *, day_first: bool = False, today: dt.date | None = None) -> dt.date | None:
    raw = text.strip().lower().rstrip(".")
    today = today or _today()
    relative = {"today": 0, "now": 0, "tomorrow": 1, "yesterday": -1}
    if raw in relative:
        return today + dt.timedelta(days=relative[raw])
    m = re.fullmatch(r"(?:in\s+)?(\d+)\s+days?(?:\s+from\s+(?:now|today))?|\+(\d+)\s*d(?:ays?)?", raw)
    if m:
        return today + dt.timedelta(days=int(m.group(1) or m.group(2)))
    m = re.fullmatch(r"(\d+)\s+days?\s+ago|-(\d+)\s*d(?:ays?)?", raw)
    if m:
        return today - dt.timedelta(days=int(m.group(1) or m.group(2)))
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", raw)  # ISO-like, year first
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})", raw)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = year + 2000 if year < 100 and year < 70 else (year + 1900 if year < 100 else year)
        if a > 12 >= b:
            day, month = a, b
        elif b > 12 >= a:
            month, day = a, b
        else:
            day, month = (a, b) if day_first else (b, a)
        return _safe_date(year, month, day)
    words = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw).replace(",", " ")
    m = re.fullmatch(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", words.strip())  # 1 May 1990
    if m and m.group(2) in MONTHS:
        return _safe_date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(r"([a-z]+)\s+(\d{1,2})\s+(\d{4})", re.sub(r"\s+", " ", words.strip()))  # May 1 1990
    if m and m.group(1) in MONTHS:
        return _safe_date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    return None


def _safe_date(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def parse_time(text: str) -> str | None:
    raw = text.strip().lower().replace(".", "")
    if raw in ("noon", "midday"):
        return "12:00"
    if raw == "midnight":
        return "00:00"
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(am|pm|a|p)?", raw)
    if not m:
        return None
    hour, minute, second = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    meridiem = m.group(4)
    if meridiem:
        if hour > 12 or hour == 0:
            return None
        hour = hour % 12 + (12 if meridiem.startswith("p") else 0)
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}" + (f":{int(second):02d}" if second else "")


def _color(text: str) -> str | None:
    raw = text.strip().lower()
    if raw in COLOR_NAMES:
        return COLOR_NAMES[raw]
    m = re.fullmatch(r"#?([0-9a-f]{6})", raw)
    if m:
        return "#" + m.group(1)
    m = re.fullmatch(r"#?([0-9a-f])([0-9a-f])([0-9a-f])", raw)
    if m:
        return "#" + "".join(c * 2 for c in m.groups())
    m = re.fullmatch(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", raw)
    if m and all(int(v) <= 255 for v in m.groups()):
        return "#" + "".join(f"{int(v):02x}" for v in m.groups())
    return None


def _number(text: str) -> float | None:
    cleaned = re.sub(r"(?<=\d)[,_ ](?=\d{3}\b)", "", text.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def _fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def normalize_for_input(value: str, input_type: str | None, *, min_value: str | None = None,
                        max_value: str | None = None, step: str | None = None, locale: str = "en-US",
                        today: dt.date | None = None) -> Normalized:
    kind = (input_type or "").lower()
    text = (value or "").strip()
    day_first = not locale.lower().startswith(("en-us", "en_us"))
    if kind == "date":
        d = parse_date(text, day_first=day_first, today=today)
        return _result(text, d.isoformat() if d else None, f"date '{text}'")
    if kind == "time":
        return _result(text, parse_time(text), f"time '{text}'")
    if kind == "datetime-local":
        parts = re.split(r"\s*(?:T|\bat\b|,)\s*|\s+(?=\d{1,2}(?::\d{2})?\s*(?:am|pm)?$)", text, maxsplit=1,
                         flags=re.IGNORECASE)
        if len(parts) == 2:
            d, t = parse_date(parts[0], day_first=day_first, today=today), parse_time(parts[1])
            return _result(text, f"{d.isoformat()}T{t}" if d and t else None, f"date-time '{text}'")
        return _result(text, None, f"date-time '{text}'")
    if kind == "month":
        m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", text)
        if m:
            return _result(text, f"{int(m.group(1)):04d}-{int(m.group(2)):02d}", "month")
        m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", text)
        if m and m.group(1).lower() in MONTHS:
            return _result(text, f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}", "month")
        d = parse_date(text, day_first=day_first, today=today)
        return _result(text, f"{d.year:04d}-{d.month:02d}" if d else None, f"month '{text}'")
    if kind == "week":
        m = re.fullmatch(r"(\d{4})-?w(\d{1,2})", text, re.IGNORECASE) or None
        if m:
            return _result(text, f"{int(m.group(1)):04d}-W{int(m.group(2)):02d}", "week")
        m = re.fullmatch(r"week\s+(\d{1,2}),?\s+(\d{4})", text, re.IGNORECASE)
        if m:
            return _result(text, f"{int(m.group(2)):04d}-W{int(m.group(1)):02d}", "week")
        d = parse_date(text, day_first=day_first, today=today)
        if d:
            year, week, _ = d.isocalendar()
            return _result(text, f"{year:04d}-W{week:02d}", "week")
        return _result(text, None, f"week '{text}'")
    if kind == "color":
        return _result(text, _color(text), f"color '{text}'")
    if kind in ("range", "number"):
        lo, hi = _number(min_value or "") if min_value else None, _number(max_value or "") if max_value else None
        if kind == "range":
            lo = 0.0 if lo is None else lo
            hi = 100.0 if hi is None else hi
        percent = text.endswith("%")
        number = _number(text[:-1] if percent else text)
        if number is None:
            return _result(text, None, f"number '{text}'")
        if percent and kind == "range" and lo is not None and hi is not None:
            number = lo + (hi - lo) * number / 100.0
        if kind == "range":
            stride = _number(step or "") if step and step != "any" else 1.0
            stride = stride or 1.0
            number = min(max(number, lo), hi)  # type: ignore[type-var]
            number = lo + round((number - lo) / stride) * stride  # type: ignore[operator]
        return _result(text, _fmt_number(number), f"number '{text}'")
    return Normalized(text, False)


def _result(original: str, normalized: str | None, what: str) -> Normalized:
    if normalized is None:
        return Normalized(original, False, f"could not interpret {what} for this input type", ok=False)
    if normalized == original:
        return Normalized(normalized, False)
    return Normalized(normalized, True, f"{what} normalized to '{normalized}'")
