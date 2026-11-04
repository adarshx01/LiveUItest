"""Normalizes Playwright MCP tool responses into typed structures.

The server answers with Markdown sections (``### Page``, ``### Snapshot``,
``### Modal state``, ``### Open tabs``, ``### Result``, ``### Error``, ...). This
module turns them into a ``ToolResponse`` and classifies errors so the rest of
the platform never parses raw strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.dsl.observation import ModalState, TabInfo
from backend.dsl.results import FailureCategory

SECTION_RE = re.compile(r"^### (?P<title>.+?)\s*$", re.MULTILINE)
TAB_RE = re.compile(r"^- (?P<index>\d+):\s*(?P<current>\(current\)\s*)?\[(?P<title>.*?)\]\((?P<url>.*?)\)\s*$",
                    re.MULTILINE)
MODAL_RE = re.compile(r"^- \[(?P<desc>.+?)\]:\s*(?P<hint>.*)$", re.MULTILINE)


@dataclass
class ImagePart:
    data: bytes
    mime_type: str


@dataclass
class ToolResponse:
    tool: str
    ok: bool
    text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    code: str | None = None
    page_url: str | None = None
    page_title: str | None = None
    console_errors: int = 0
    snapshot_yaml: str | None = None
    snapshot_file: str | None = None
    modal_state: ModalState | None = None
    tabs: list[TabInfo] = field(default_factory=list)
    result_text: str | None = None
    error: str | None = None
    failure_category: FailureCategory | None = None
    images: list[ImagePart] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    downloads: list[dict[str, str]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    raw: Any = None


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group("title").strip()] = text[m.end():end].strip("\n")
    return sections


def _strip_code_fence(body: str) -> str:
    m = re.search(r"```(?:\w+)?\n(?P<code>.*?)```", body, re.DOTALL)
    return (m.group("code") if m else body).strip()


def parse_modal_state(body: str) -> ModalState | None:
    m = MODAL_RE.search(body)
    if not m:
        return None
    desc = m.group("desc")
    kind_m = re.match(r'"(?P<kind>alert|confirm|prompt|beforeunload)" dialog with message "(?P<msg>.*)"$', desc,
                      re.DOTALL)
    if kind_m:
        return ModalState(kind=kind_m.group("kind"), message=kind_m.group("msg"), raw=desc)  # type: ignore[arg-type]
    if "file chooser" in desc.lower():
        return ModalState(kind="file_chooser", raw=desc)
    return ModalState(kind="unknown", message=desc, raw=desc)


def parse_tabs(body: str) -> list[TabInfo]:
    return [TabInfo(index=int(m.group("index")), title=m.group("title"), url=m.group("url"),
                    current=bool(m.group("current"))) for m in TAB_RE.finditer(body)]


def classify_error(message: str) -> FailureCategory:
    low = message.lower()
    if "ref " in low and "not found" in low or "stale" in low or "try capturing new snapshot" in low:
        return FailureCategory.STALE_ELEMENT_REFERENCE
    if "strict mode violation" in low:
        return FailureCategory.AMBIGUOUS_TARGET
    if "does not handle the modal state" in low or "dialog" in low and "modal" in low:
        return FailureCategory.POPUP_OR_DIALOG_ERROR
    if "intercepts pointer events" in low or "element is outside of the viewport" in low and "intercept" in low:
        return FailureCategory.OVERLAY_BLOCKING_TARGET
    if "element is not enabled" in low or "element is disabled" in low:
        return FailureCategory.ELEMENT_NOT_ENABLED
    if "element is not visible" in low or "not visible" in low:
        return FailureCategory.ELEMENT_NOT_VISIBLE
    if "did not find some options" in low or "no option" in low or "option not found" in low:
        return FailureCategory.OPTION_NOT_FOUND
    if ("navigat" in low or "page.goto" in low) and "timeout" in low:
        return FailureCategory.NAVIGATION_TIMEOUT
    if "net::err" in low or "err_connection" in low or "err_name_not_resolved" in low:
        return FailureCategory.NETWORK_FAILURE
    if "frame" in low and ("detached" in low or "not found" in low):
        return FailureCategory.IFRAME_CONTEXT_ERROR
    if "target closed" in low or "browser has been closed" in low or "connection closed" in low \
            or "browser closed" in low or "has been disconnected" in low:
        return FailureCategory.BROWSER_DISCONNECTED
    if "waiting for locator" in low and "timeout" in low and "resolved to" not in low:
        return FailureCategory.ELEMENT_NOT_FOUND
    if "timeout" in low:
        return FailureCategory.ACTION_TIMEOUT
    if "not found" in low or "no element" in low or "resolved to 0 elements" in low:
        return FailureCategory.ELEMENT_NOT_FOUND
    return FailureCategory.UNKNOWN_FAILURE


def parse_tool_result(tool: str, result: Any) -> ToolResponse:
    texts: list[str] = []
    images: list[ImagePart] = []
    import base64

    for part in getattr(result, "content", None) or []:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            texts.append(part.text)
        elif ptype == "image":
            try:
                images.append(ImagePart(base64.b64decode(part.data), getattr(part, "mime_type", "image/png")))
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
    text = "\n".join(texts)
    is_error = bool(getattr(result, "is_error", False))
    sections = split_sections(text)
    response = ToolResponse(tool=tool, ok=not is_error, text=text, sections=sections, images=images, raw=None)
    if "Ran Playwright code" in sections:
        response.code = _strip_code_fence(sections["Ran Playwright code"])
    page = sections.get("Page", "")
    if page:
        url_m = re.search(r"^- Page URL:\s*(.+)$", page, re.MULTILINE)
        title_m = re.search(r"^- Page Title:\s*(.*)$", page, re.MULTILINE)
        console_m = re.search(r"^- Console:\s*(\d+)\s+errors?", page, re.MULTILINE)
        response.page_url = url_m.group(1).strip() if url_m else None
        response.page_title = title_m.group(1).strip() if title_m else None
        response.console_errors = int(console_m.group(1)) if console_m else 0
    snapshot = sections.get("Snapshot")
    if snapshot:
        if "```" in snapshot:
            response.snapshot_yaml = _strip_code_fence(snapshot)
        else:
            file_m = re.search(r"\]\((?P<path>[^)]+\.ya?ml)\)", snapshot)
            if file_m:
                response.snapshot_file = file_m.group("path")
    if "Modal state" in sections:
        response.modal_state = parse_modal_state(sections["Modal state"])
    if "Open tabs" in sections:
        response.tabs = parse_tabs(sections["Open tabs"])
    if "Result" in sections:
        response.result_text = sections["Result"].strip()
        if tool == "browser_tabs":
            response.tabs = parse_tabs(sections["Result"])
        response.files = re.findall(r"\]\(([^)]+\.(?:png|jpe?g|webp|webm|zip|pdf|ya?ml|log|csv))\)", sections["Result"])
    if "Events" in sections:
        response.events = [line[2:].strip() for line in sections["Events"].splitlines() if line.startswith("- ")]
        for event in response.events:
            dl = re.match(r'Downloaded file (?P<name>.+?) to "(?P<path>.+)"$', event)
            if dl:
                response.downloads.append({"name": dl.group("name"), "path": dl.group("path")})
    if "Error" in sections or is_error:
        response.ok = False
        response.error = (sections.get("Error") or text).strip()
        response.failure_category = classify_error(response.error)
    return response
