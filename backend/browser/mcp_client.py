"""MCP client connection to a Playwright MCP server.

One ``MCPConnection`` == one isolated browser context. The MCP transport's
async context managers must be entered and exited by the same task, so a
dedicated runner task owns them for the connection's lifetime while tool calls
are issued from the execution's task.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.core.config import Settings, get_settings
from backend.core.exceptions import BrowserDisconnected, BrowserError, ConfigurationError
from backend.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class BrowserLaunchOptions:
    """Per-session isolation settings (spec section 12.3)."""

    viewport: tuple[int, int] | None = None
    locale: str | None = None
    timezone: str | None = None
    user_agent: str | None = None
    device: str | None = None
    storage_state_path: str | None = None
    headless: bool | None = None
    browser_name: str | None = None
    extra_http_headers: dict[str, str] = field(default_factory=dict)
    record_video: bool | None = None


class MCPConnection:
    def __init__(self, *, session_id: str, workspace: Path, options: BrowserLaunchOptions | None = None,
                 settings: Settings | None = None) -> None:
        self.session_id = session_id
        self.workspace = workspace
        self.options = options or BrowserLaunchOptions()
        self.settings = settings or get_settings()
        self._session: Any = None
        self._runner: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self.server_name: str | None = None
        self.server_version: str | None = None
        self.tools: dict[str, dict[str, Any]] = {}
        self._call_lock = asyncio.Lock()

    # ------------------------------------------------------------------ config
    def _config(self) -> dict[str, Any]:
        s = self.settings
        width, height = self.options.viewport or s.viewport_size()
        launch: dict[str, Any] = {"headless": s.browser_headless if self.options.headless is None
                                  else self.options.headless}
        if s.browser_executable_path and (self.options.browser_name or s.browser_name) == "chromium":
            launch["executablePath"] = s.browser_executable_path
        if s.browser_no_sandbox:
            launch["chromiumSandbox"] = False
        context: dict[str, Any] = {"viewport": {"width": width, "height": height}, "acceptDownloads": True}
        if self.options.locale:
            context["locale"] = self.options.locale
        if self.options.timezone:
            context["timezoneId"] = self.options.timezone
        if self.options.user_agent:
            context["userAgent"] = self.options.user_agent
        if self.options.storage_state_path:
            context["storageState"] = self.options.storage_state_path
        if self.options.extra_http_headers:
            context["extraHTTPHeaders"] = self.options.extra_http_headers
        caps = ["core", "core-navigation", "core-tabs", "core-input", "network", "devtools"]
        if s.browser_vision_enabled:
            caps.append("vision")
        return {
            "browser": {"browserName": self.options.browser_name or s.browser_name, "isolated": True,
                        "launchOptions": launch, "contextOptions": context},
            "capabilities": caps,
            "outputDir": str(self.workspace / "output"),
            "testIdAttribute": s.test_id_attribute,
            "timeouts": {"action": s.budget.action_timeout_ms, "navigation": s.budget.navigation_timeout_ms,
                         "settle": 500, "idle": 0},
            "snapshot": {"mode": "none", "boxes": True},
            "imageResponses": "allow",
            "filePaths": "absolute",
            "ignoreRoots": True,
            "webmcp": False,  # pages must not be able to register tools
            "redactUrlParams": ["access_token", "id_token", "refresh_token", "code", "token", "session", "sig"],
            "codegen": "typescript",
        }

    def _stdio_params(self) -> Any:
        from mcp import StdioServerParameters

        script = Path(self.settings.mcp_server_script)
        if not script.exists():
            raise ConfigurationError("Playwright MCP server is not installed",
                                     details={"expected": str(script), "fix": "cd browser/playwright-mcp && npm ci"})
        config_path = self.workspace / "mcp-config.json"
        config_path.write_text(json.dumps(self._config(), indent=2))
        args = [str(script), "--config", str(config_path)]
        if self.options.device:
            args += ["--device", self.options.device]
        # the browser process never needs platform credentials
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AITEST_SECRET_", "ANTHROPIC_", "OPENAI_",
                                                                       "AWS_", "GH_", "GITHUB_"))}
        browsers = self.settings.playwright_browsers_path
        if browsers is None and (Path(__file__).resolve().parents[2] / "data" / "pw-browsers").exists():
            browsers = Path(__file__).resolve().parents[2] / "data" / "pw-browsers"
        if browsers is not None:
            env["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers)
        return StdioServerParameters(command=self.settings.mcp_node_command, args=args, cwd=str(self.workspace),
                                     env=env)

    # ------------------------------------------------------------------ lifecycle
    async def start(self, timeout: float = 60.0) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "output").mkdir(exist_ok=True)
        self._runner = asyncio.create_task(self._run(), name=f"mcp-{self.session_id}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except TimeoutError as exc:
            await self.close()
            raise BrowserError("Timed out starting the Playwright MCP server") from exc
        if self._error is not None:
            raise BrowserError(f"Failed to start Playwright MCP server: {self._error}") from self._error

    async def _run(self) -> None:
        from mcp import ClientSession

        stderr_path = self.workspace / "mcp-server.log"
        try:
            async with contextlib.AsyncExitStack() as stack:
                if self.settings.mcp_transport == "http":
                    if not self.settings.mcp_http_url:
                        raise ConfigurationError("AITEST_MCP_HTTP_URL is required for the http transport")
                    from mcp.client.streamable_http import streamable_http_client

                    streams = await stack.enter_async_context(streamable_http_client(self.settings.mcp_http_url))
                    read, write = streams[0], streams[1]
                else:
                    from mcp.client.stdio import stdio_client

                    errlog = stack.enter_context(open(stderr_path, "a", encoding="utf-8"))
                    read, write = await stack.enter_async_context(stdio_client(self._stdio_params(), errlog=errlog))
                session = await stack.enter_async_context(ClientSession(read, write))
                init = await session.initialize()
                info = getattr(init, "server_info", None)
                self.server_name = getattr(info, "name", None)
                self.server_version = getattr(info, "version", None)
                listed = await session.list_tools()
                self.tools = {t.name: {"input_schema": t.input_schema or {}, "description": t.description or "",
                                       "annotations": t.annotations.model_dump() if t.annotations else {}}
                              for t in listed.tools}
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - runner must never leak exceptions
            self._error = exc
            if not isinstance(exc, asyncio.CancelledError):
                log.warning("MCP connection ended with error", session_id=self.session_id, error=repr(exc)[:300])
        finally:
            self._session = None
            self._ready.set()

    @property
    def alive(self) -> bool:
        return self._session is not None and self._runner is not None and not self._runner.done()

    async def call(self, tool: str, arguments: dict[str, Any], timeout: float) -> Any:
        if not self.alive:
            raise BrowserDisconnected("Browser session is not connected", details={"session_id": self.session_id})
        async with self._call_lock:  # strictly sequential actions against one page
            try:
                return await asyncio.wait_for(self._session.call_tool(tool, arguments), timeout=timeout)
            except TimeoutError:
                raise
            except Exception as exc:
                if not self.alive:
                    raise BrowserDisconnected("Browser session disconnected during a tool call",
                                              details={"tool": tool}) from exc
                raise BrowserError(f"MCP call failed: {exc}", details={"tool": tool}) from exc

    async def close(self) -> None:
        self._stop.set()
        if self._runner is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._runner), timeout=15)
            except (TimeoutError, asyncio.CancelledError):
                self._runner.cancel()
                with contextlib.suppress(BaseException):
                    await self._runner
        self._session = None
