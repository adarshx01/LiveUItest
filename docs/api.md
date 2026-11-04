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
| POST | `/executions` | start (`variables`, `datasets` for data-driven runs, `options`, `allowed_domains`) |
| GET | `/executions`, `/executions/{id}` | list / status, outcome, metrics, pending approvals |
| POST | `/executions/{id}/cancel` · `/resume` · `/retry` | control |
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
