"""End-to-end tests against the demo app through real Playwright MCP and PostgreSQL.

Run with: pytest -m "browser and db" tests/integration
Requires the demo application (python -m demo.app), PostgreSQL with pgvector and the pinned
Playwright MCP install; tests are skipped when any of them is unavailable.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text

from backend.core.config import get_settings
from backend.evaluation.benchmark import (
    Scenario,
    catalogue_scenarios,
    login_scenarios,
    run_scenario,
)

pytestmark = [pytest.mark.browser, pytest.mark.db]


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session")
def demo_app():
    proc = None
    if not _port_open(8800):
        proc = subprocess.Popen([sys.executable, "-m", "demo.app"])
        for _ in range(40):
            if _port_open(8800):
                break
            time.sleep(0.25)
    if not _port_open(8800):
        pytest.skip("demo app not available")
    yield "http://127.0.0.1:8800"
    if proc is not None:
        proc.terminate()


@pytest.fixture(scope="session")
async def platform(demo_app):
    settings = get_settings()
    if settings.database_url.startswith("postgres") and not _port_open(5432):
        pytest.skip("PostgreSQL configured but not available")
    if not get_settings().mcp_server_script.exists():
        pytest.skip("Playwright MCP not installed (cd browser/playwright-mcp && npm ci)")
    from backend.bootstrap import bootstrap

    pf = await bootstrap(json_logs=False)
    from backend.models.database import clear_tables

    async with pf.sessionmaker() as session:
        await clear_tables(session, ("locator_memory", "page_memory", "failure_memory", "memory_versions",
                                     "test_execution_memory"))
        await session.commit()
    yield pf
    await pf.engine.stop()


_DATA_URI = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")

# tables holding the output of one run (scoped by execution id) and long-lived memory (scanned in full)
_RUN_TABLES = ("agent_events", "agent_traces", "approval_events", "artifacts", "assertion_results",
               "browser_session_events", "browser_sessions", "execution_metrics", "execution_steps",
               "recovery_attempts")
_MEMORY_TABLES = ("application_memory", "page_memory", "locator_memory", "failure_memory", "memory_versions",
                  "test_execution_memory", "interaction_patterns", "page_fingerprints")


async def _assert_secret_not_leaked(platform, result, secret: str) -> None:
    """Spec 16.4: no unredacted credential in reports, events, traces, step records or memory."""

    assert result.report, "a report must be generated"
    html = Path(result.report).read_text(encoding="utf-8")
    assert secret not in _DATA_URI.sub("", html), "secret value rendered in the HTML report"
    assert secret not in (Path(result.report).parent / "report.json").read_text(encoding="utf-8")
    async with platform.sessionmaker() as session:
        for table in _RUN_TABLES:
            assert not await _rows_contain(session, table, secret, "execution_id = :eid", result.execution_id), \
                f"secret value persisted: {table}"
        for table in _MEMORY_TABLES:
            assert not await _rows_contain(session, table, secret), f"secret value persisted: {table}"
        assert not await _rows_contain(session, "execution_runs", secret, "id = :eid", result.execution_id)


async def _rows_contain(session, table: str, needle: str, where: str | None = None, eid: str | None = None) -> bool:
    """Portable leak scan (SQLite and PostgreSQL): stringify every column of the matching rows."""

    query = f'SELECT * FROM "{table}"' + (f" WHERE {where}" if where else "")
    rows = (await session.execute(text(query), {"eid": eid} if eid else {})).all()
    return any(needle in str(tuple(row)) for row in rows)


def _checkpoint_contains(thread_id: str, needle: str) -> bool:
    """Search the LangGraph checkpoints of one execution for a value (both checkpointer backends)."""

    import sqlite3

    url = get_settings().resolved_checkpoint_url()
    raw = needle.encode("utf-8")
    if url.startswith("sqlite"):
        conn = sqlite3.connect(url.split("///", 1)[1])
        try:
            rows = conn.execute("SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = ?", (thread_id,)).fetchall()
            rows += conn.execute("SELECT value, value FROM writes WHERE thread_id = ?", (thread_id,)).fetchall()
        finally:
            conn.close()
        return any(raw in (v if isinstance(v, bytes) else str(v).encode()) for row in rows for v in row if v)
    import psycopg

    with psycopg.connect(url) as conn:
        hits = conn.execute(
            "SELECT count(*) FROM checkpoint_blobs WHERE thread_id = %s AND position(%s in blob) > 0 "
            "UNION ALL SELECT count(*) FROM checkpoint_writes WHERE thread_id = %s AND position(%s in blob) > 0",
            (thread_id, raw, thread_id, raw)).fetchall()
    return any(h[0] for h in hits)


async def test_login_learn_replay_heal_and_fail(platform):
    """The spec's acceptance demonstration: learn, replay with new data, heal a UI change, fail honestly."""

    first, replay, changed, healed_replay, wrong_password = login_scenarios()
    r1 = await run_scenario(platform, first)
    assert r1.outcome == "PASSED", r1.reason
    assert not r1.wrong_touches and not r1.missed

    r2 = await run_scenario(platform, replay)
    assert r2.outcome == "PASSED", r2.reason
    assert r2.metrics["memory_hits"] >= 3, r2.metrics
    assert r2.metrics["memory_fastpath"] >= 3, "validated memory must replay directly on the same page"
    await _assert_secret_not_leaked(platform, r2, replay.secrets["test_password"])

    r3 = await run_scenario(platform, changed)
    assert r3.outcome == "PASSED", r3.reason
    assert r3.metrics["recovered_steps"] >= 1, "UI change must be detected and reported as recovery"
    assert r3.metrics["memory_fastpath"] <= 1, "changed controls must not be replayed from stale memory"
    assert not r3.wrong_touches, f"decoy operated: {r3.wrong_touches}"
    await _assert_secret_not_leaked(platform, r3, changed.secrets["test_password"])

    r4 = await run_scenario(platform, healed_replay)
    assert r4.outcome == "PASSED" and r4.metrics["memory_hits"] >= 3

    r5 = await run_scenario(platform, wrong_password)
    assert r5.outcome == "FAILED", "a wrong password must never be healed into a pass"


@pytest.mark.parametrize("scenario", catalogue_scenarios(), ids=lambda s: s.name)
async def test_capability_catalogue(platform, scenario: Scenario):
    result = await run_scenario(platform, scenario)
    assert result.outcome == scenario.expected_outcome, result.reason
    assert not result.wrong_touches, f"operated the wrong element(s): {result.wrong_touches}"
    assert not result.missed, f"did not operate: {result.missed}"
    for secret in scenario.secrets.values():
        await _assert_secret_not_leaked(platform, result, secret)


async def test_captcha_requires_a_human(platform, demo_app):
    async with httpx.AsyncClient() as client:
        await client.post(f"{demo_app}/__admin/reset")
        await client.post(f"{demo_app}/__admin/config", json={"challenge_enabled": True})
    scenario = Scenario("captcha", "login.txt", "BLOCKED", reset_demo=False)
    try:
        result = await run_scenario(platform, scenario)
    finally:
        async with httpx.AsyncClient() as client:
            await client.post(f"{demo_app}/__admin/config", json={"challenge_enabled": False})
    # without a human the run is cancelled while paused; the agent never touched the challenge
    assert result.outcome == "BLOCKED"
    assert "captcha-checkbox" not in result.touched


async def test_rerun_steps_retry_and_compare(platform, demo_app):
    """Spec 20: re-run one step with its prerequisites, retry it with the same (sealed) data set, compare runs.

    Drives the API route functions directly (same code path as HTTP, without a second app/engine)."""

    import asyncio
    import json
    from types import SimpleNamespace

    from backend.agents.governance.secret_manager import SecretScope
    from backend.api.deps import Principal
    from backend.api.routes import executions as routes
    from backend.api.schemas import ExecutionCreate, StepRerun
    from backend.evaluation.benchmark import EXAMPLES
    from backend.services.dispatcher import ExecutionDispatcher

    async with httpx.AsyncClient() as client:
        await client.post(f"{demo_app}/__admin/reset")
    org = "org_benchmark"
    project = await platform.platform.ensure_project(org, "Benchmark")
    # the vault holds a wrong value: the credential supplied with this run's data set must take precedence
    await platform.secrets.store(SecretScope(org, project.id), "test_password", "vault-value-not-used")
    tc, _ = await platform.platform.save_test_case(org, project.id,
                                                   content=(EXAMPLES / "login_data_driven.md").read_text())
    dispatcher = ExecutionDispatcher(platform.engine, platform.platform, platform.sessionmaker)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(dispatcher=dispatcher, platform=platform)))
    who = Principal(organization_id=org, actor="pytest")

    async def finished(run: dict) -> tuple[object, list[dict]]:
        await asyncio.wait_for(platform.engine.runs[run["id"]].done.wait(), timeout=300)
        row = await platform.platform.get_run(org, run["id"])
        return row, await routes.execution_steps(run["id"], who, platform)

    data = {"username": "alice@example.com", "display_name": "Alice Smith", "test_password": "Alice#2024"}
    full, full_steps = await finished(await routes.create_execution(
        ExecutionCreate(test_case_id=tc.id, variables=data, options={"use_memory": True}), request, who))
    assert full.outcome == "PASSED", full.reason
    assert full.options["dataset"] == {"username": "alice@example.com", "display_name": "Alice Smith"}
    assert "Alice#2024" not in json.dumps(full.options), "credential stored in clear on the run record"
    assert not _checkpoint_contains(full.id, "Alice#2024"), "credential persisted in the LangGraph checkpoints"

    redirect = next(s for s in full_steps if "redirected to the dashboard" in s["original_text"])
    partial, partial_steps = await finished(await routes.rerun_steps(
        full.id, StepRerun(step_ids=[redirect["step_id"]]), request, who, platform))
    assert partial.outcome == "PASSED" and "partial re-run" in partial.reason, partial.reason
    assert [s["step_id"] for s in partial_steps] == [s["step_id"] for s in full_steps
                                                     if s["index"] <= redirect["index"]]
    assert partial.options["scope"]["step_ids"] == [redirect["step_id"]] and partial.retry_of == full.id

    retried, retried_steps = await finished(await routes.retry_execution(partial.id, request, who, platform))
    assert retried.outcome == "PASSED", retried.reason  # same scope, same sealed data set
    assert len(retried_steps) == len(partial_steps)

    comparison = await routes.compare_executions(full.id, partial.id, who, platform)
    assert comparison["same_version"] and comparison["counts"]["regressions"] == 0
    not_run = [r for r in comparison["steps"] if r["changes"] == ["not run in the compared execution"]]
    assert len(not_run) == len(full_steps) - len(partial_steps)


async def test_inline_password_is_vaulted_and_never_stored(platform, tmp_path):
    """A password typed straight into the test text ("Enter X as Password") must work, and must not be
    stored anywhere in clear: test definition, step records, run tables, checkpoints, report."""

    password = "CareDemo#2026"
    case = tmp_path / "inline_password.txt"
    case.write_text("http://127.0.0.1:8800/care/sign-in\n"
                    "Enter care.user@example.com as Email\n"
                    f"Enter {password} as Password\n"
                    "Click on Login button\n"
                    "Click on Patient Support Program\n"
                    "Verify that \"Abc Xy\" is displayed\n", encoding="utf-8")
    result = await run_scenario(platform, Scenario("inline password", str(case), "PASSED",
                                                   ["care-email", "care-password", "care-login", "care-psp"]))
    assert result.outcome == "PASSED", result.reason
    await _assert_secret_not_leaked(platform, result, password)
    assert not _checkpoint_contains(result.execution_id, password), "inline password reached the checkpoints"
    async with platform.sessionmaker() as session:
        for table in ("test_case_versions", "test_steps", "execution_steps"):
            assert not await _rows_contain(session, table, password), f"inline password stored in {table}"
        masked = (await session.execute(text("SELECT original_text FROM test_steps WHERE original_text LIKE "
                                             "'%as Password'"))).scalars().all()
        assert masked and all("••••••" in t for t in masked)
