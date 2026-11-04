# API

Base path `/api/v1`, header `X-API-Key` (WebSocket/`<img>`: `?api_key=`), optional `X-Organization-Id`.

| Method | Path | Purpose |
|---|---|---|
| POST/GET | `/projects` | create / list projects |
| POST | `/applications` · GET `/applications/{id}` | register application (base URL) / details + environments |
| POST | `/test-cases` | ingest NL/Markdown/JSON/CSV → analyzed, compiled, versioned test case (returns the structured plan) |
| GET | `/test-cases`, `/test-cases/{id}[?version=]` | list / detail with parsed steps and issues |
| PATCH | `/test-cases/{id}` | new version (content), tags, archive |
| POST | `/test-suites` · GET `/test-suites/{id}` · POST `/test-suites/{id}/run` | suites and parallel suite runs |
| POST | `/secrets` · GET `/secrets` | store encrypted secret (value never returned) / list names |
| POST | `/executions` | start (`variables`, `datasets` for data-driven runs, `options`, `allowed_domains`, optional `step_ids` + `with_prerequisites` for a partial run) |
| GET | `/executions`, `/executions/{id}` | list / status, outcome, metrics, pending approvals, `scope` of a partial run |
| POST | `/executions/{id}/cancel` · `/resume` · `/retry` | control; retry reuses the run's data set (and the scope of a partial run) |
| POST | `/executions/{id}/rerun-steps` | re-run individual steps (`step_ids`, `with_prerequisites`, default true) with the same data set |
| GET | `/executions/{id}/compare/{other_id}` | step-by-step comparison against a baseline: regressions, fixes, changed targets, recovery, memory, timing, metric deltas |
| GET | `/executions/{id}/steps` | step results (live while running) |
| GET | `/executions/{id}/artifacts` · `/artifacts/{exec}/{kind}/{name}` | evidence |
| GET | `/executions/{id}/report[?format=json]` | HTML / JSON report |
| GET | `/executions/{id}/events[?after=]` · WS `/executions/{id}/stream` | event history / live stream |
| GET | `/executions/{id}/agent-trace` | node traces + MCP tool-call audit |
| GET | `/executions/{id}/live` · `/observation` | live screenshot · browser inspection (elements, boxes, forms, page state, snapshot) |
| POST | `/executions/{id}/browser/human-action` | audited human click/type while `AWAITING_HUMAN` |
| GET | `/approvals` · POST `/approvals/{id}/approve|reject` | approval queue |
| GET | `/memory/applications/{id}` · `/memory/test-cases/{id}` · `/memory/{id}/versions` | memory inspection |
| POST | `/memory/{id}/invalidate` | invalidate a strategy |
| GET | `/agents` · `/health` · `/diagnostics/traces` | catalogue, health, recent spans |

Event types: `EXECUTION_CREATED, PLAN_COMPILED, PLAN_INVALID, BROWSER_INITIALIZED, PAGE_OBSERVED,
TARGET_GROUNDED, ACTION_VALIDATED, ACTION_BLOCKED, ACTION_STARTED, ACTION_COMPLETED, ACTION_VERIFIED,
ACTION_FAILED, MEMORY_RETRIEVED, MEMORY_USED, MEMORY_UPDATED, RECOVERY_STARTED, RECOVERY_COMPLETED,
RECOVERY_FAILED, HUMAN_APPROVAL_REQUIRED, EXECUTION_PAUSED, EXECUTION_RESUMED, ASSERTION_PASSED,
ASSERTION_FAILED, STEP_STARTED, STEP_COMPLETED, SCREENSHOT_CAPTURED, SECURITY_WARNING, RUNNER_SELECTED,
REPORT_READY, EXECUTION_COMPLETED, EXECUTION_CANCELLED`. Payloads are redacted.

## Data sets, retries and partial runs

A run records its data set so it can be retried or partially re-run without the caller sending it
again. Plain test data (`username`, `display_name`, ...) is stored as is. Values whose names look
like credentials (`password`, `test_password`, `otp_code`, `api_key`, `pin`, ...) are sealed: they
are Fernet-encrypted with the platform key, the same protection as the credential store. Only their
names stay readable. They are never in graph checkpoints, queues or run records in clear text, and
are decrypted only when the browser session starts. A credential supplied with a run overrides a
vault secret of the same name, which is what a data-driven run with one account per data set needs.

`POST /executions/{id}/rerun-steps` with `{"step_ids": ["step_006"]}` re-runs step 6 of that
execution's test version. The earlier steps run first to rebuild the page state (open the page, fill
the form), and the run stops after step 6. With `"with_prerequisites": false`, only the entry
navigation and the selected steps run. Selecting a step inside an IF/ELSE block selects the whole
block. A partial run is judged like any other: if the selected steps contain no assertion, the
outcome is INCONCLUSIVE, never PASSED. Its verified steps reinforce locator memory, but it records
no test trajectory and never counts as a validated full run.

`GET /executions/{base}/compare/{other}` matches steps by id (same test version), otherwise by step
text, then position. Example response (abridged):

```json
{"summary": ["outcome PASSED -> FAILED", "1 step(s) regressed (first: step 7)", "1 step(s) acted on a different element"],
 "counts": {"regressions": 1, "fixes": 0, "target_changes": 1, "changed_steps": 3},
 "steps": [{"index": 5, "text": "Click on the Sign In button",
            "base": {"status": "PASSED", "target": {"display": "button 'Sign In'", "method": "memory"}},
            "other": {"status": "PASSED", "recovered": true, "target": {"display": "button 'Log in'", "method": "semantic"}},
            "changes": ["target changed: button 'Sign In' -> button 'Log in'", "grounding: memory -> semantic", "needed recovery"]}]}
```
