"""Typed, policy-restricted adapter over Playwright MCP tools (spec section 12).

All browser access goes through here. The adapter:
* allows only registered tools (unknown/page-registered tools are rejected);
* only lets ``browser_evaluate`` run platform probes from ``js_probes``;
* validates arguments against the server-advertised JSON schema;
* applies timeouts, budget checks and tracing to every call;
* normalizes responses (``ToolResponse``) and classifies failures;
* emits a redacted audit record for every call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import jsonschema

from backend.agents.governance.redaction import redact_text, redact_value
from backend.browser.js_probes import PROBE_NAMES, Probe
from backend.browser.mcp_client import MCPConnection
from backend.browser.response_parser import ToolResponse, parse_tool_result
from backend.core.exceptions import BrowserDisconnected, BrowserError, ToolNotAllowed
from backend.core.logging import get_logger
from backend.core.telemetry import span
from backend.dsl.results import FailureCategory
from backend.harness.budget_manager import current_tracker

log = get_logger(__name__)


@dataclass(frozen=True)
class ToolRule:
    category: str  # observe | navigate | input | dialog | tabs | vision | diagnostics | recording
    risk: str  # low | medium | high
    read_only: bool
    timeout_kind: str = "action"  # action | navigation | quick


TOOL_RULES: dict[str, ToolRule] = {
    "browser_snapshot": ToolRule("observe", "low", True, "quick"),
    "browser_find": ToolRule("observe", "low", True, "quick"),
    "browser_take_screenshot": ToolRule("observe", "low", True, "quick"),
    "browser_console_messages": ToolRule("diagnostics", "low", True, "quick"),
    "browser_network_requests": ToolRule("diagnostics", "low", True, "quick"),
    "browser_network_request": ToolRule("diagnostics", "low", True, "quick"),
    "browser_evaluate": ToolRule("observe", "low", True, "quick"),  # probes only
    "browser_wait_for": ToolRule("observe", "low", True, "navigation"),
    "browser_navigate": ToolRule("navigate", "medium", False, "navigation"),
    "browser_navigate_back": ToolRule("navigate", "medium", False, "navigation"),
    "browser_tabs": ToolRule("tabs", "low", False, "navigation"),
    "browser_resize": ToolRule("navigate", "low", False, "quick"),
    "browser_click": ToolRule("input", "medium", False),
    "browser_type": ToolRule("input", "medium", False),
    "browser_fill_form": ToolRule("input", "medium", False),
    "browser_select_option": ToolRule("input", "medium", False),
    "browser_hover": ToolRule("input", "low", False),
    "browser_drag": ToolRule("input", "medium", False),
    "browser_drop": ToolRule("input", "medium", False),
    "browser_press_key": ToolRule("input", "medium", False),
    "browser_file_upload": ToolRule("input", "medium", False),
    "browser_handle_dialog": ToolRule("dialog", "medium", False),
    "browser_mouse_move_xy": ToolRule("vision", "low", False),
    "browser_mouse_click_xy": ToolRule("vision", "medium", False),
    "browser_mouse_drag_xy": ToolRule("vision", "medium", False),
    "browser_mouse_down": ToolRule("vision", "medium", False),
    "browser_mouse_up": ToolRule("vision", "medium", False),
    "browser_mouse_wheel": ToolRule("vision", "low", False),
    "browser_start_video": ToolRule("recording", "low", True, "quick"),
    "browser_stop_video": ToolRule("recording", "low", True, "navigation"),
    "browser_start_tracing": ToolRule("recording", "low", True, "quick"),
    "browser_stop_tracing": ToolRule("recording", "low", True, "navigation"),
    "browser_close": ToolRule("navigate", "low", False, "quick"),
}

# Never callable, even if the server advertises them.
BLOCKED_TOOLS = {"browser_run_code_unsafe", "browser_install", "browser_pdf_save", "browser_resume",
                 "browser_annotate"}

AuditHook = Callable[[dict[str, Any]], Awaitable[None] | None]


class BrowserToolAdapter:
    def __init__(self, connection: MCPConnection, *, execution_id: str, action_timeout_s: float,
                 navigation_timeout_s: float, audit_hook: AuditHook | None = None) -> None:
        self.connection = connection
        self.execution_id = execution_id
        self.action_timeout_s = action_timeout_s
        self.navigation_timeout_s = navigation_timeout_s
        self.audit_hook = audit_hook
        self.call_count = 0
        self.history: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- core
    def available(self, tool: str) -> bool:
        return tool in self.connection.tools and tool in TOOL_RULES and tool not in BLOCKED_TOOLS

    def _timeout_for(self, rule: ToolRule) -> float:
        if rule.timeout_kind == "navigation":
            return self.navigation_timeout_s + 10
        if rule.timeout_kind == "quick":
            return max(15.0, self.action_timeout_s + 5)
        return self.action_timeout_s + 10

    async def call(self, tool: str, arguments: dict[str, Any] | None = None, *, probe: Probe | None = None,
                   sensitive_args: tuple[str, ...] = ()) -> ToolResponse:
        arguments = dict(arguments or {})
        if tool in BLOCKED_TOOLS:
            raise ToolNotAllowed(f"Tool {tool!r} is blocked by platform policy", details={"tool": tool})
        rule = TOOL_RULES.get(tool)
        if rule is None:
            raise ToolNotAllowed(f"Tool {tool!r} is not on the approved list", details={"tool": tool})
        if tool not in self.connection.tools:
            raise ToolNotAllowed(f"Tool {tool!r} is not provided by the connected MCP server",
                                 details={"tool": tool, "capability_missing": True})
        if tool == "browser_evaluate":
            if probe is None or probe.name not in PROBE_NAMES:
                raise ToolNotAllowed("browser_evaluate only accepts registered platform probes",
                                     details={"tool": tool})
            arguments["function"] = probe.source
        schema = self.connection.tools[tool].get("input_schema") or {}
        try:
            jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as exc:
            raise BrowserError(f"Invalid arguments for {tool}: {exc.message}", details={"tool": tool}) from exc
        tracker = current_tracker()
        if tracker:
            tracker.check_tool()
        safe_args = redact_value({k: ("••••••" if k in sensitive_args else v) for k, v in arguments.items()
                                  if k != "function"})
        if probe is not None:
            safe_args["probe"] = probe.name
        started = time.perf_counter()
        with span("mcp.tool", tool=tool, execution_id=self.execution_id, probe=probe.name if probe else None) as sp:
            try:
                raw = await self.connection.call(tool, arguments, timeout=self._timeout_for(rule))
                response = parse_tool_result(tool, raw)
            except TimeoutError:
                response = ToolResponse(tool=tool, ok=False, error=f"{tool} timed out",
                                        failure_category=FailureCategory.NAVIGATION_TIMEOUT
                                        if rule.timeout_kind == "navigation" else FailureCategory.ACTION_TIMEOUT)
            except BrowserDisconnected:
                raise
            except BrowserError as exc:
                response = ToolResponse(tool=tool, ok=False, error=str(exc),
                                        failure_category=FailureCategory.UNKNOWN_FAILURE)
            response.latency_ms = (time.perf_counter() - started) * 1000
            sp.set_attribute("ok", response.ok)
        self.call_count += 1
        if tracker:
            tracker.record_tool(response.latency_ms)
        record = {
            "execution_id": self.execution_id,
            "tool": tool,
            "args": safe_args,
            "ok": response.ok,
            "error": redact_text(response.error or "")[:500] or None,
            "failure_category": response.failure_category.value if response.failure_category else None,
            "latency_ms": round(response.latency_ms, 1),
            "code": redact_text(response.code or "")[:500] or None,
            "page_url": response.page_url,
            "ts": time.time(),
        }
        self.history.append(record)
        if len(self.history) > 2000:
            del self.history[:500]
        if self.audit_hook is not None:
            result = self.audit_hook(record)
            if result is not None:
                await result
        if not response.ok:
            log.info("tool call failed", tool=tool, category=record["failure_category"], error=record["error"])
        return response

    # ---------------------------------------------------------------- typed helpers
    async def navigate(self, url: str) -> ToolResponse:
        return await self.call("browser_navigate", {"url": url})

    async def back(self) -> ToolResponse:
        return await self.call("browser_navigate_back", {})

    async def snapshot(self, target: str | None = None) -> ToolResponse:
        args: dict[str, Any] = {"boxes": True}
        if target:
            args["target"] = target
        return await self.call("browser_snapshot", args)

    async def screenshot(self, *, full_page: bool = False, target: str | None = None) -> ToolResponse:
        args: dict[str, Any] = {"scale": "css", "type": "png"}
        if full_page:
            args["fullPage"] = True
        if target:
            args["target"] = target
        return await self.call("browser_take_screenshot", args)

    async def click(self, target: str, description: str, *, double: bool = False, button: str | None = None,
                    modifiers: list[str] | None = None) -> ToolResponse:
        args: dict[str, Any] = {"target": target, "element": description[:200]}
        if double:
            args["doubleClick"] = True
        if button:
            args["button"] = button
        if modifiers:
            args["modifiers"] = modifiers
        return await self.call("browser_click", args)

    async def type_text(self, target: str, description: str, text: str, *, submit: bool = False,
                        slowly: bool = False, sensitive: bool = False) -> ToolResponse:
        args: dict[str, Any] = {"target": target, "element": description[:200], "text": text}
        if submit:
            args["submit"] = True
        if slowly:
            args["slowly"] = True
        return await self.call("browser_type", args, sensitive_args=("text",) if sensitive else ())

    async def fill_form(self, fields: list[dict[str, str]], *, sensitive: bool = False) -> ToolResponse:
        return await self.call("browser_fill_form", {"fields": fields}, sensitive_args=("fields",) if sensitive else ())

    async def select_option(self, target: str, description: str, values: list[str]) -> ToolResponse:
        return await self.call("browser_select_option", {"target": target, "element": description[:200],
                                                         "values": values})

    async def hover(self, target: str, description: str) -> ToolResponse:
        return await self.call("browser_hover", {"target": target, "element": description[:200]})

    async def drag(self, start: str, start_desc: str, end: str, end_desc: str) -> ToolResponse:
        return await self.call("browser_drag", {"startTarget": start, "startElement": start_desc[:200],
                                                "endTarget": end, "endElement": end_desc[:200]})

    async def press_key(self, key: str) -> ToolResponse:
        return await self.call("browser_press_key", {"key": key})

    async def upload(self, paths: list[str]) -> ToolResponse:
        return await self.call("browser_file_upload", {"paths": paths})

    async def handle_dialog(self, accept: bool, prompt_text: str | None = None) -> ToolResponse:
        args: dict[str, Any] = {"accept": accept}
        if prompt_text is not None:
            args["promptText"] = prompt_text
        return await self.call("browser_handle_dialog", args)

    async def tabs(self, action: str, index: int | None = None, url: str | None = None) -> ToolResponse:
        args: dict[str, Any] = {"action": action}
        if index is not None:
            args["index"] = index
        if url:
            args["url"] = url
        return await self.call("browser_tabs", args)

    async def wait_for(self, *, seconds: float | None = None, text: str | None = None,
                       text_gone: str | None = None) -> ToolResponse:
        args: dict[str, Any] = {}
        if seconds is not None:
            args["time"] = min(30.0, max(0.1, seconds))
        if text:
            args["text"] = text
        if text_gone:
            args["textGone"] = text_gone
        return await self.call("browser_wait_for", args)

    async def mouse_click(self, x: float, y: float, *, button: str = "left", click_count: int = 1) -> ToolResponse:
        return await self.call("browser_mouse_click_xy", {"x": round(x, 1), "y": round(y, 1), "button": button,
                                                          "clickCount": click_count})

    async def mouse_move(self, x: float, y: float) -> ToolResponse:
        return await self.call("browser_mouse_move_xy", {"x": round(x, 1), "y": round(y, 1)})

    async def mouse_drag(self, sx: float, sy: float, ex: float, ey: float) -> ToolResponse:
        return await self.call("browser_mouse_drag_xy", {"startX": round(sx, 1), "startY": round(sy, 1),
                                                         "endX": round(ex, 1), "endY": round(ey, 1)})

    async def mouse_wheel(self, dx: float, dy: float) -> ToolResponse:
        return await self.call("browser_mouse_wheel", {"deltaX": dx, "deltaY": dy})

    async def console_errors(self) -> ToolResponse:
        return await self.call("browser_console_messages", {"level": "error"})

    async def network_requests(self, url_filter: str | None = None) -> ToolResponse:
        args: dict[str, Any] = {"static": False}
        if url_filter:
            args["filter"] = url_filter
        return await self.call("browser_network_requests", args)

    async def probe(self, probe: Probe, target: str | None = None, description: str = "element") -> Any:
        """Run a registered probe and return its decoded JSON result (None on failure)."""

        args: dict[str, Any] = {}
        if probe.element:
            if not target:
                raise BrowserError("Element probe requires a target", details={"probe": probe.name})
            args["target"] = target
            args["element"] = description[:200]
        response = await self.call("browser_evaluate", args, probe=probe)
        if not response.ok or response.result_text is None:
            return None
        try:
            return json.loads(response.result_text)
        except json.JSONDecodeError:
            return None

    async def start_video(self, filename: str, size: tuple[int, int]) -> ToolResponse:
        return await self.call("browser_start_video", {"filename": filename,
                                                       "size": {"width": size[0], "height": size[1]}})

    async def stop_video(self) -> ToolResponse:
        return await self.call("browser_stop_video", {})
