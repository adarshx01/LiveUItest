"""End-to-end demonstration + evaluation (spec sections 29 / 31 / 32).

Runs, against the controlled demo app with a clean memory:
  * the login test on its first run (learning), again with different values
    (validated memory re-grounded), after a UI change (change detected, stale
    memory rejected, target re-grounded, new memory version), and after healing;
  * a wrong-password run that must FAIL (never healed into a pass);
  * the capability catalogue (forms, dialogs, tabs, iframe, files, drag, canvas,
    dynamic content, prompt injection, broken workflow, validation failures).

Writes docs/demo-results.md with per-scenario results and the evaluation metrics, and refreshes the
sample reports in docs/sample-reports/ from this run.
Prerequisites: PostgreSQL, the demo app (python -m demo.app) and Playwright MCP.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import shutil
import sys
from pathlib import Path

from sqlalchemy import text

from backend.bootstrap import bootstrap
from backend.evaluation.benchmark import (
    catalogue_scenarios,
    login_scenarios,
    run_scenario,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
# scenario -> published sample report (docs/sample-reports/)
SAMPLE_REPORTS = {"1. first run (learns)": "login-first-run.html",
                  "3. UI changed (v2 locators)": "ui-change-healed.html",
                  "broken checkout": "broken-checkout-failed.html",
                  "one-time code from the vault": "otp-secret-masked.html",
                  "custom dropdown zoo": "custom-dropdowns.html"}
MEMORY_TABLES = ("locator_memory", "page_memory", "page_fingerprints", "failure_memory", "memory_versions",
                 "test_execution_memory", "interaction_patterns", "application_memory")


async def main(include_catalogue: bool) -> int:
    platform = await bootstrap(json_logs=False)
    try:
        async with platform.sessionmaker() as session:  # clean memory so run 1 is a genuine first run
            await session.execute(text(f"TRUNCATE {', '.join(MEMORY_TABLES)}"))
            await session.commit()
        scenarios = login_scenarios() + (catalogue_scenarios() if include_catalogue else [])
        results = []
        for scenario in scenarios:
            print(f"▶ {scenario.name} …", flush=True)
            result = await run_scenario(platform, scenario)
            flag = "✔" if result.outcome_ok and not result.wrong_touches and not result.missed else "✘"
            print(f"  {flag} outcome={result.outcome} (expected {scenario.expected_outcome}) "
                  f"memory={result.metrics.get('memory_hits', 0)}/{result.metrics.get('memory_lookups', 0)} "
                  f"recovered={result.metrics.get('recovered_steps', 0)} touched={result.touched} "
                  f"wrong={result.wrong_touches} missed={result.missed} {result.duration_s:.0f}s", flush=True)
            results.append(result)
            if scenario.name in SAMPLE_REPORTS and result.report:
                target = ROOT / "docs" / "sample-reports" / SAMPLE_REPORTS[scenario.name]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(result.report, target)
        metrics = summarize(results)
        lines = ["# End-to-end demonstration results", "",
                 f"Generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M UTC} by `scripts/demo_e2e.py` against the demo "
                 "application, with no LLM configured (deterministic agents only) and a clean memory store.", "",
                 "| Scenario | Capability | Expected | Outcome | Memory hits | Recovered steps | Wrong element "
                 "touched | Duration |", "|---|---|---|---|---|---|---|---|"]
        for r in results:
            lines.append(f"| {r.scenario.name} | {r.scenario.capability} | {r.scenario.expected_outcome} | "
                         f"**{r.outcome}** | {r.metrics.get('memory_hits', 0)}/{r.metrics.get('memory_lookups', 0)} | "
                         f"{r.metrics.get('recovered_steps', 0)} | {', '.join(r.wrong_touches) or 'none'} | "
                         f"{r.duration_s:.0f}s |")
        lines += ["", "## Evaluation metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in metrics.items()]
        lines += ["", "Ground truth comes from the demo app's interaction recorder (`data-gt` attributes), so the "
                  "element-identification metrics reflect which controls were actually operated.", "",
                  "Sample reports from this run: " + ", ".join(f"[{name}](sample-reports/{file})"
                                                               for name, file in SAMPLE_REPORTS.items()) + "."]
        out = ROOT / "docs" / "demo-results.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"\nmetrics: {metrics}\nwritten {out}")
        ok = all(r.outcome_ok and not r.wrong_touches for r in results)
        return 0 if ok else 1
    finally:
        await platform.engine.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(include_catalogue="--login-only" not in sys.argv)))
