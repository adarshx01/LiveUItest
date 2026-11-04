"""End-to-end tests against the demo app through real Playwright MCP and PostgreSQL.

Run with: pytest -m "browser and db" tests/integration
Requires the demo application (python -m demo.app), PostgreSQL with pgvector and the pinned
Playwright MCP install; tests are skipped when any of them is unavailable.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx
import pytest
from sqlalchemy import text

from backend.core.config import get_settings
from backend.evaluation.benchmark import Scenario, catalogue_scenarios, login_scenarios, run_scenario

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


async def test_login_learn_replay_heal_and_fail(platform):
    """The spec's acceptance demonstration: learn, replay with new data, heal a UI change, fail honestly."""

    first, replay, changed, healed_replay, wrong_password = login_scenarios()
    r1 = await run_scenario(platform, first)
    assert r1.outcome == "PASSED", r1.reason
    assert not r1.wrong_touches and not r1.missed

    r2 = await run_scenario(platform, replay)
    assert r2.outcome == "PASSED", r2.reason
    assert r2.metrics["memory_hits"] >= 3, r2.metrics

    r3 = await run_scenario(platform, changed)
    assert r3.outcome == "PASSED", r3.reason
    assert r3.metrics["recovered_steps"] >= 1, "UI change must be detected and reported as recovery"
    assert not r3.wrong_touches, f"decoy operated: {r3.wrong_touches}"

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
