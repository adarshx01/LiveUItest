"""Test-case ingestion: plain text, Markdown, JSON, CSV and test suites (spec 7.1).

Ingestion only *structures* the input (name, URL, variables, ordered lines,
IF/ELSE blocks, expected results). Understanding the instructions is the job of
the requirement analyzer.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

from backend.core.exceptions import ParseError

HEADER_RE = re.compile(
    r"^\s*(?:[#*_>\s]*)?(?P<key>url|base\s*url|application\s*url|app\s*url|test\s*case|test\s*name|test|title|name|"
    r"description|preconditions?|pre-conditions?|expected\s*results?|expected|test\s*data|data|variables|tags|"
    r"steps|environment)\s*[:=-]\s*(?P<value>.*)$",
    re.IGNORECASE,
)
SECTION_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(?P<title>.+?)\s*#*\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:\d+\s*[.)\]:]|[-*•+]|step\s*\d+\s*[:.)-])\s+(?P<text>.+)$", re.IGNORECASE)
KV_RE = re.compile(r"^\s*[-*•]?\s*(?P<k>[A-Za-z_][\w .-]*?)\s*[:=]\s*(?P<v>.+?)\s*$")
IF_RE = re.compile(r"^\s*(?:\d+\s*[.)]\s*)?if\s+(?P<cond>.+?)\s*:\s*$", re.IGNORECASE)
ELSE_RE = re.compile(r"^\s*(?:else|otherwise)\s*:?\s*$", re.IGNORECASE)
ENDIF_RE = re.compile(r"^\s*(?:end\s*if|endif|end)\s*$", re.IGNORECASE)


@dataclass
class RawLine:
    text: str
    line_no: int = 0
    condition: str | None = None  # set for IF blocks
    then_lines: list[RawLine] = field(default_factory=list)
    else_lines: list[RawLine] = field(default_factory=list)


@dataclass
class RawTestCase:
    name: str = "Untitled test"
    base_url: str | None = None
    description: str = ""
    lines: list[RawLine] = field(default_factory=list)
    expected_results: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source_format: str = "text"
    structured_steps: list[dict[str, Any]] | None = None
    source_text: str = ""


def detect_format(content: str) -> str:
    stripped = content.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass
    first_line = stripped.splitlines()[0] if stripped else ""
    if "," in first_line and re.search(r"\b(step|steps|test[_ ]?name|url)\b", first_line, re.IGNORECASE) and \
            not re.search(r"\b(click|enter|verify|open)\b", first_line, re.IGNORECASE):
        return "csv"
    if re.search(r"^\s*#{1,6}\s", content, re.MULTILINE):
        return "markdown"
    return "text"


def ingest(content: str | dict[str, Any] | list[Any], *, fmt: str | None = None,
           base_url: str | None = None, name: str | None = None,
           variables: dict[str, str] | None = None) -> list[RawTestCase]:
    """Parse any supported input into one or more raw test cases."""

    if isinstance(content, (dict, list)):
        cases = _from_json(content)
    else:
        fmt = fmt or detect_format(content)
        if fmt == "json":
            cases = _from_json(json.loads(content))
        elif fmt == "csv":
            cases = _from_csv(content)
        else:
            cases = _from_text(content, fmt)
    if not cases:
        raise ParseError("No test case could be found in the supplied content")
    for case in cases:
        if base_url and not case.base_url:
            case.base_url = base_url
        if name and (case.name in ("", "Untitled test")):
            case.name = name
        if variables:
            case.variables = {**variables, **case.variables}
    return cases


# --------------------------------------------------------------------------- text / markdown

def _split_cases(text: str) -> list[str]:
    chunks = re.split(r"^\s*(?:---+|===+|\*\*\*+)\s*$", text, flags=re.MULTILINE)
    out: list[str] = []
    for chunk in chunks:
        # multiple "Test Case:" headers in one chunk -> split further
        positions = [m.start() for m in re.finditer(r"^\s*(?:#+\s*)?test\s*case\s*[:#]", chunk, re.IGNORECASE | re.MULTILINE)]
        if len(positions) > 1:
            positions.append(len(chunk))
            for start, end in zip(positions, positions[1:]):
                out.append(chunk[start:end])
        elif chunk.strip():
            out.append(chunk)
    return out


def _from_text(text: str, fmt: str) -> list[RawTestCase]:
    cases = []
    for chunk in _split_cases(text):
        case = _parse_text_case(chunk, fmt)
        if case.lines or case.expected_results:
            cases.append(case)
    return cases


def _parse_text_case(text: str, fmt: str) -> RawTestCase:
    case = RawTestCase(source_format="markdown" if fmt == "markdown" else "text", source_text=text.strip())
    section = "steps"
    in_code = False
    block_stack: list[tuple[RawLine, str]] = []  # (if-line, branch)
    base_indent: int | None = None

    def add_line(raw: RawLine, indent: int) -> None:
        nonlocal base_indent
        while block_stack:
            parent, branch = block_stack[-1]
            parent_indent = getattr(parent, "_indent", 0)
            if indent > parent_indent:
                (parent.then_lines if branch == "then" else parent.else_lines).append(raw)
                return
            block_stack.pop()
        case.lines.append(raw)

    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        heading = SECTION_HEADING_RE.match(line)
        if heading:
            title = heading.group("title").strip().lower()
            header = HEADER_RE.match(heading.group("title"))
            if header:
                section = _apply_header(case, header.group("key"), header.group("value"), section)
            elif re.match(r"^(steps?|test steps|procedure|actions?)$", title):
                section = "steps"
            elif re.match(r"^(expected( results?)?|assertions|verification|outcome)$", title):
                section = "expected"
            elif re.match(r"^(test data|data|variables|inputs?)$", title):
                section = "data"
            elif re.match(r"^(pre-?conditions?|setup)$", title):
                section = "preconditions"
            elif case.name == "Untitled test":
                case.name = re.sub(r"^(test( case)?\s*[:#-]\s*)", "", heading.group("title"), flags=re.IGNORECASE)
            continue
        header = HEADER_RE.match(line)
        if header and not LIST_ITEM_RE.match(line):
            section = _apply_header(case, header.group("key"), header.group("value"), section)
            continue
        item = LIST_ITEM_RE.match(line)
        content = (item.group("text") if item else line).strip()
        if section == "data":
            kv = KV_RE.match(content)
            if kv:
                case.variables[_var_name(kv.group("k"))] = kv.group("v").strip().strip("\"'")
            continue
        if section == "expected":
            case.expected_results.append(content)
            continue
        if section == "preconditions":
            case.preconditions.append(content)
            continue
        if section == "description":
            case.description = (case.description + " " + content).strip()
            continue
        # steps section -------------------------------------------------------
        if ENDIF_RE.match(content):
            if block_stack:
                block_stack.pop()
            continue
        if_match = IF_RE.match(content)
        if if_match:
            raw = RawLine(text=content, line_no=line_no, condition=if_match.group("cond"))
            raw._indent = indent  # type: ignore[attr-defined]
            add_line(raw, indent)
            block_stack.append((raw, "then"))
            continue
        if ELSE_RE.match(content) and block_stack:
            parent, _ = block_stack.pop()
            block_stack.append((parent, "else"))
            continue
        if base_indent is None:
            base_indent = indent
        add_line(RawLine(text=content, line_no=line_no), indent)
    return case


def _var_name(key: str) -> str:
    return re.sub(r"\W+", "_", key.strip().lower()).strip("_")


def _apply_header(case: RawTestCase, key: str, value: str, section: str) -> str:
    key = re.sub(r"\s+", " ", key.lower())
    value = value.strip()
    if key in ("url", "base url", "application url", "app url"):
        url_match = re.search(r"(https?://\S+|www\.\S+|localhost\S*)", value)
        if url_match:
            url = url_match.group(1).rstrip(".,;")
            case.base_url = url if url.startswith("http") else f"https://{url}"
        return section
    if key in ("test case", "test name", "title", "name", "test"):
        if value:
            case.name = value
        return "steps"
    if key == "description":
        case.description = value
        return "description" if not value else section
    if key.startswith(("precondition", "pre-condition")):
        if value:
            case.preconditions.append(value)
        return "preconditions"
    if key.startswith("expected"):
        if value:
            case.expected_results.append(value)
        return "expected"
    if key in ("test data", "data", "variables"):
        if value:
            for pair in re.split(r"[;,]\s*", value):
                kv = KV_RE.match(pair)
                if kv:
                    case.variables[_var_name(kv.group("k"))] = kv.group("v").strip().strip("\"'")
        return "data"
    if key == "tags":
        case.tags.extend(t.strip() for t in re.split(r"[,;]", value) if t.strip())
        return section
    if key == "steps":
        if value:
            case.lines.append(RawLine(text=value))
        return "steps"
    return section


# --------------------------------------------------------------------------- JSON

def _from_json(data: Any) -> list[RawTestCase]:
    if isinstance(data, list):
        cases: list[RawTestCase] = []
        for item in data:
            cases.extend(_from_json(item))
        return cases
    if not isinstance(data, dict):
        raise ParseError("JSON test case must be an object or a list of objects")
    if "test_cases" in data or "tests" in data:
        items = data.get("test_cases") or data.get("tests") or []
        suite_url = data.get("base_url") or data.get("url")
        suite_vars = data.get("variables") or {}
        cases = _from_json(items)
        for case in cases:
            case.base_url = case.base_url or suite_url
            case.variables = {**suite_vars, **case.variables}
        return cases
    case = RawTestCase(
        name=str(data.get("test_name") or data.get("name") or data.get("title") or "Untitled test"),
        base_url=data.get("base_url") or data.get("url"),
        description=str(data.get("description") or ""),
        expected_results=[str(e) for e in _as_list(data.get("expected_results") or data.get("expected"))],
        preconditions=[str(p) for p in _as_list(data.get("preconditions"))],
        variables={str(k): str(v) for k, v in (data.get("variables") or data.get("test_data") or {}).items()},
        tags=[str(t) for t in _as_list(data.get("tags"))],
        source_format="json",
        source_text=json.dumps(data, indent=2),
    )
    steps = data.get("steps") or []
    if isinstance(steps, str):
        steps = [s for s in steps.splitlines() if s.strip()]
    structured: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, str):
            case.lines.append(RawLine(text=step, line_no=idx))
        elif isinstance(step, dict) and "action" in step:
            structured.append(step)
        elif isinstance(step, dict) and ("instruction" in step or "text" in step):
            case.lines.append(RawLine(text=str(step.get("instruction") or step.get("text")), line_no=idx))
            if step.get("expected"):
                case.lines.append(RawLine(text=f"Verify that {step['expected']}", line_no=idx))
        else:
            raise ParseError(f"Unsupported step format at position {idx}")
    if structured:
        if case.lines:
            raise ParseError("A test case cannot mix structured DSL steps with natural-language steps")
        case.structured_steps = structured
        case.source_format = "structured"
    return [case]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# --------------------------------------------------------------------------- CSV

def _from_csv(content: str) -> list[RawTestCase]:
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise ParseError("CSV has no header row")
    fields = {f.lower().strip().replace(" ", "_"): f for f in reader.fieldnames}
    name_col = next((fields[k] for k in ("test_name", "test_case", "name", "title") if k in fields), None)
    url_col = next((fields[k] for k in ("url", "base_url") if k in fields), None)
    step_col = next((fields[k] for k in ("step", "steps", "instruction", "action") if k in fields), None)
    expected_col = next((fields[k] for k in ("expected", "expected_result", "expected_results") if k in fields), None)
    if step_col is None:
        raise ParseError("CSV needs a 'step' (or 'steps') column")
    cases: dict[str, RawTestCase] = {}
    order: list[str] = []
    for row_no, row in enumerate(reader, start=2):
        name = (row.get(name_col) if name_col else None) or "Imported test"
        if name not in cases:
            cases[name] = RawTestCase(name=name, source_format="csv")
            order.append(name)
        case = cases[name]
        if url_col and row.get(url_col) and not case.base_url:
            case.base_url = row[url_col].strip()
        steps_text = (row.get(step_col) or "").strip()
        for part in [p for p in re.split(r"\r?\n", steps_text) if p.strip()]:
            case.lines.append(RawLine(text=part.strip(), line_no=row_no))
        if expected_col and (row.get(expected_col) or "").strip():
            case.lines.append(RawLine(text=f"Verify that {row[expected_col].strip()}", line_no=row_no))
    return [cases[n] for n in order]
