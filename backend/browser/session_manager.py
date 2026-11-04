"""Browser session lifecycle, pooling and isolation (spec sections 12.3 / 12.4).

Every execution owns exactly one session, i.e. one Playwright MCP connection with
its own isolated (in-memory profile) browser context and workspace directory.
Sessions are never shared across executions or tenants.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from backend.browser.mcp_client import BrowserLaunchOptions, MCPConnection
from backend.browser.tool_adapter import AuditHook, BrowserToolAdapter
from backend.core.config import Settings, get_settings
from backend.core.exceptions import BrowserError, ConfigurationError, NotFound
from backend.core.ids import new_id
from backend.core.logging import get_logger

log = get_logger(__name__)


class BrowserSession:
    def __init__(self, *, execution_id: str, workspace: Path, options: BrowserLaunchOptions, settings: Settings,
                 audit_hook: AuditHook | None) -> None:
        self.session_id = new_id("bs")
        self.execution_id = execution_id
        self.workspace = workspace
        self.options = options
        self.settings = settings
        self.connection = MCPConnection(session_id=self.session_id, workspace=workspace, options=options,
                                        settings=settings)
        self.adapter = BrowserToolAdapter(self.connection, execution_id=execution_id,
                                          action_timeout_s=settings.budget.action_timeout_ms / 1000,
                                          navigation_timeout_s=settings.budget.navigation_timeout_ms / 1000,
                                          audit_hook=audit_hook)
        self.created_at = time.time()
        self.last_used = time.time()
        self.frame_hint: str | None = None
        self.video_file: Path | None = None
        self.recording = False
        self.downloads_dir = workspace / "downloads"
        self.uploads_dir = workspace / "uploads"

    @property
    def alive(self) -> bool:
        return self.connection.alive

    async def start(self) -> None:
        await self.connection.start()
        version = self.connection.server_version or ""
        expected = self.settings.mcp_expected_version_prefix
        if expected and not version.startswith(expected):
            message = f"Playwright MCP server version {version!r} does not match pinned {expected!r}"
            if self.settings.is_production:
                await self.connection.close()
                raise ConfigurationError(message)
            log.warning(message)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        record = self.settings.browser_record_video if self.options.record_video is None else self.options.record_video
        if record and self.adapter.available("browser_start_video"):
            self.video_file = self.workspace / "execution-video.webm"
            width, height = self.options.viewport or self.settings.viewport_size()
            response = await self.adapter.start_video(str(self.video_file), (width, height))
            self.recording = response.ok
            if not response.ok:
                log.warning("video recording unavailable", error=response.error)
        log.info("browser session started", session_id=self.session_id, execution_id=self.execution_id,
                 server_version=version, tools=len(self.connection.tools))

    def touch(self) -> None:
        self.last_used = time.time()

    async def stop_recording(self) -> Path | None:
        """Stop recording; the file is finalized by the browser when the context closes."""

        if not self.recording:
            return None
        self.recording = False
        try:
            response = await self.adapter.stop_video()
        except BrowserError:
            return None
        if response.ok and response.files:
            self.video_file = Path(response.files[0])
        return self.video_file

    async def finalized_video(self, timeout: float = 10.0) -> Path | None:
        """Wait (after close) for the recorded video to be flushed to disk."""

        if self.video_file is None:
            return None
        deadline = time.time() + timeout
        last_size = -1
        empty_checks = 0
        while time.time() < deadline:
            if self.video_file.exists():
                size = self.video_file.stat().st_size
                if size > 0 and size == last_size:
                    return self.video_file
                empty_checks = empty_checks + 1 if size == 0 else 0
                if empty_checks >= 5:  # the encoder produced nothing; do not keep the run waiting
                    return None
                last_size = size
            await asyncio.sleep(0.3)
        if self.video_file.exists() and self.video_file.stat().st_size > 0:
            return self.video_file
        return None  # recording unavailable (e.g. ffmpeg revision mismatch); the report builds a replay instead

    async def close(self) -> None:
        if self.recording:
            await self.stop_recording()
        await self.connection.close()


class SessionManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._sessions: dict[str, BrowserSession] = {}
        self._by_execution: dict[str, str] = {}
        self._pool = asyncio.Semaphore(self.settings.browser_pool_size)
        self._lock = asyncio.Lock()

    async def create(self, execution_id: str, *, options: BrowserLaunchOptions | None = None,
                     audit_hook: AuditHook | None = None, acquire_timeout: float = 300.0) -> BrowserSession:
        async with self._lock:
            existing = self._by_execution.get(execution_id)
            if existing and existing in self._sessions and self._sessions[existing].alive:
                return self._sessions[existing]
        try:
            await asyncio.wait_for(self._pool.acquire(), timeout=acquire_timeout)
        except TimeoutError as exc:
            raise BrowserError("No browser worker became available in time") from exc
        workspace = self.settings.browser_output_root / execution_id / new_id("ws")
        workspace.mkdir(parents=True, exist_ok=True)
        session = BrowserSession(execution_id=execution_id, workspace=workspace, options=options or BrowserLaunchOptions(),
                                 settings=self.settings, audit_hook=audit_hook)
        try:
            await session.start()
        except BaseException:
            self._pool.release()
            raise
        async with self._lock:
            self._sessions[session.session_id] = session
            self._by_execution[execution_id] = session.session_id
        return session

    def get(self, session_id: str) -> BrowserSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise NotFound("Browser session not found", details={"session_id": session_id})
        return session

    def for_execution(self, execution_id: str) -> BrowserSession | None:
        sid = self._by_execution.get(execution_id)
        return self._sessions.get(sid) if sid else None

    async def close(self, session_id: str, *, keep_workspace: bool = True) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None and self._by_execution.get(session.execution_id) == session_id:
                self._by_execution.pop(session.execution_id, None)
        if session is None:
            return
        try:
            await session.close()
        finally:
            self._pool.release()
            if not keep_workspace:
                shutil.rmtree(session.workspace, ignore_errors=True)
        log.info("browser session closed", session_id=session_id, execution_id=session.execution_id)

    async def close_all(self) -> None:
        for session_id in list(self._sessions):
            await self.close(session_id)

    async def reap_idle(self) -> int:
        now = time.time()
        idle = [sid for sid, s in self._sessions.items()
                if now - s.last_used > self.settings.browser_session_idle_ttl_seconds or not s.alive]
        for sid in idle:
            await self.close(sid)
        return len(idle)

    def stats(self) -> dict[str, int]:
        return {"active_sessions": len(self._sessions), "pool_size": self.settings.browser_pool_size}


_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


def set_session_manager(manager: SessionManager | None) -> None:
    global _manager
    _manager = manager
