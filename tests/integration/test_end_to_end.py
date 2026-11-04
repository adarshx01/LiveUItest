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
    if not _port_open(5432):
        pytest.skip("PostgreSQL not available")
    if not get_settings().mcp_server_script.exists():
        pytest.skip("Playwright MCP not installed (cd browser/playwright-mcp && npm ci)")
    from backend.bootstrap import bootstrap

    pf = await bootstrap(json_logs=False)
    async with pf.sessionmaker() as session:
        await session.execute(text("TRUNCATE locator_memory, page_memory, failure_memory, memory_versions, "
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
    needle = {"needle": f"%{secret}%", "eid": result.execution_id}
    async with platform.sessionmaker() as session:
        queries = [f"SELECT count(*) FROM {t} t WHERE t.execution_id = :eid AND t::text LIKE :needle"
                   for t in _RUN_TABLES]
        queries += [f"SELECT count(*) FROM {t} t WHERE t::text LIKE :needle" for t in _MEMORY_TABLES]
        queries.append("SELECT count(*) FROM execution_runs t WHERE t.id = :eid AND t::text LIKE :needle")
        for query in queries:
            hits = (await session.execute(text(query), needle)).scalar_one()
            assert hits == 0, f"secret value persisted: {query.split(' FROM ')[1].split()[0]}"


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
    async with platform.sessionmaker() as session:
        leaked = (await session.execute(text(
            "SELECT count(*) FROM checkpoint_blobs WHERE thread_id = :t AND position(convert_to(:s, 'UTF8') in blob) > 0"
            " UNION ALL SELECT count(*) FROM checkpoint_writes WHERE thread_id = :t "
            "AND position(convert_to(:s, 'UTF8') in blob) > 0"), {"t": full.id, "s": "Alice#2024"})).scalars().all()
    assert not any(leaked), "credential persisted in the LangGraph checkpoints"

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
