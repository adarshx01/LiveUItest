# AI-Powered End-to-End Testing Platform

Agentic browser testing from natural-language test cases — **LangGraph** orchestration,
**Playwright MCP** browser execution, DOM-first / visual-aware grounding, a deterministic
safety regulator, validated execution memory, self-healing that never hides real bugs, and
evidence-backed reports with a screenshot of every action. Runs entirely on one machine by default
(SQLite, files on disk, no Docker); PostgreSQL + pgvector, Redis and S3 are optional integrations.

```text
URL: https://example.com
Test Case: Verify user login
1. Open the login page.
2. Click on the Login button.
3. Enter username as abc.
4. Enter password as {{test_password}}.
5. Click on the Sign In button.
6. Verify that the user is redirected to the dashboard.
7. Verify that the dashboard displays the logged-in user's name.
```

The platform parses the instructions into a typed plan, opens an isolated browser through
Playwright MCP, observes the real page (accessibility tree + DOM + bounding boxes), grounds each
instruction to the right element with an explainable confidence score, passes every action
through a policy gate, executes it, verifies the intended postcondition, recovers from automation
failures within strict budgets, learns validated locator strategies for the next run, and writes an
HTML/JSON report showing what passed or failed — with before/after screenshots where the acted-on
element is highlighted.

## What is in the box

| Area | Where | Highlights |
|---|---|---|
| NL test intelligence | `backend/agents/planner/` | spelling repair of instructions ("Clik on the Loign button") that never rewrites data, text / Markdown / JSON / CSV / suites, IF/ELSE blocks, variables & secrets, 31-action DSL with 26 assertion types, assertions as first-class steps, compound sentences, ambiguity reporting (never guessing), optional LLM escalation through a closed schema |
| Browser layer | `backend/browser/` | MCP client (stdio or HTTP), tool allowlist (`browser_run_code_unsafe` blocked, `browser_evaluate` restricted to platform probes, page-registered WebMCP tools disabled), schema validation, timeouts, budgets, redacted audit log, isolated per-execution contexts, version pinning |
| Observation | `backend/agents/observation/` | shadow-DOM-piercing DOM probe, labels inferred from surrounding text and table layouts, role-less clickable "div soup", aria snapshot parser, DOM ↔ snapshot correlation by bounding box, semantic classifier, forms/dialogs/landmarks/sections, route-normalized page fingerprints, CAPTCHA / error / overlay / injection signals |
| Grounding | `backend/agents/grounding/` | contextual scoring (identity, role, context, state, type, option, visual, memory), typo/synonym/abbreviation-tolerant matching (`fuzzy.py`, never for assertions), form-flow context, ambiguity margins, ordinals/positions, calibrated confidence → execute / enhanced verification / refine / ask human; ref-free `LocatorStrategy` with validated Playwright selectors; memory fast path |
| Visual | `backend/agents/visual/` | colour/shape segmentation of canvas surfaces with safe click points, set-of-marks prompting for vision models, annotated evidence screenshots |
| Execution | `backend/execution/` | custom date and time pickers (`pickers.py`: types in the field's own format when it is editable, otherwise drives the calendar - month/year selects or prev/next to the right month, never an adjacent-month or disabled day - and HH / MM (/ AM-PM) popover fields or columns, then reads the field back); dropdown engine (`dropdown.py`): native OS-rendered selects via live option lists (values, optgroups, disabled, async dependent lists, multi-select), datalists, date/time/color/range pickers with value normalization, custom dropdowns (portals, search-as-you-type, virtualized lists, ARIA-less div menus, multi-select chips, nested menus); toggles (idempotent), uploads, downloads, drag-and-drop, scrolling/infinite lists, tabs, iframes, native & DOM dialogs |
| Verification | `backend/agents/verification/` | per-action postconditions, assertion engine (expected vs actual + evidence), independent oracle: PASSED / FAILED / BLOCKED / INCONCLUSIVE |
| Recovery | `backend/agents/recovery/`, `runtime.py` | failure taxonomy (spec 17.1), bounded strategies that are never repeated for the same failure, a redacted failure-context artifact per failed attempt (URL, error, decision + candidates, page signals, interactive elements, accessibility excerpt, screenshot), optional model-chosen strategy from a closed list, idempotency checks before retrying non-idempotent actions, healing validation, product failures never healed; failed checks continue with independent steps, failed actions skip their dependents |
| Governance | `backend/agents/governance/` | domain allowlist + post-redirect checks, risk classes, approvals, CAPTCHA hand-off, sensitive-value placement rule, confidence gates, Fernet-encrypted secret store, redaction everywhere |
| Memory | `backend/memory/`, `backend/models/` | application / page / test trajectory / locator / failure / pattern memory, exact → page → similarity retrieval (pgvector on PostgreSQL, in-process on SQLite), re-grounding before use, versioning & supersession |
| Orchestration | `backend/graph/` | 32-node LangGraph graph (spec 6.1), durable checkpoints (`AsyncSqliteSaver` locally, `AsyncPostgresSaver` on PostgreSQL), interrupts for approvals and human verification |
| Harness | `backend/harness/` | agent registry & protocol, model gateway (Anthropic SDK, Azure OpenAI via the OpenAI SDK with the Responses API and Chat Completions fallback, OpenAI-compatible, offline), per-tier models, timeouts, retries with backoff, schema-validated outputs, budgets |
| API | `backend/api/` | REST (spec 21), WebSocket event stream, approvals, live browser hand-off, browser inspection, retries with the stored data set, partial re-runs of individual steps, execution comparison |
| UI | `frontend/` | Next.js: test editor with plan preview, live execution (stream + live browser + element overlay), intelligence view, inspection, embedded report, memory inspector, approvals, per-step re-run, run comparison |
| Reports | `backend/agents/reporting/` | self-contained HTML + JSON, annotated before/after screenshots, assertions, recovery, memory reuse, cost, timeline, execution replay / video |
| Demo app | `demo/` | the 22 controlled scenarios of spec §29, plus `/care` - a patient-services portal (programmes, near-identical patient names, tabs, an interaction sidebar with portal / ARIA-less / editable dropdowns, a calendar and an HH:MM popover) - all with a ground-truth interaction recorder |

## Quick start (local, no Docker)

Requirements: Python 3.12+, Node 20+ (22 recommended). Nothing else - no database server, no Redis,
no admin rights. **Windows:** follow [`docs/windows-local-setup.md`](docs/windows-local-setup.md)
(exact PowerShell commands, Azure OpenAI setup, troubleshooting).

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
(cd browser/playwright-mcp && npm ci && npx playwright install chromium)
cp .env.example .env                                # Windows: copy .env.example .env
python -m backend.cli doctor                        # checks node, browser, database, secret key, model
python -m backend.cli init-db                       # data/aitest.db (also automatic on first use)
python -m demo.app &                                # demo application on :8800 (separate terminal on Windows)

# run a natural-language test
python -m backend.cli run --test examples/login.txt --headed
python -m backend.cli run --test examples/login_data_driven.md \
    --var username=alice@example.com --var "display_name=Alice Smith" --secret "test_password=Alice#2024"
python -m backend.cli run --test examples/healthlink_interaction.md \
    --var care_email=care.user@example.com --secret "care_password=CareDemo#2026"

# API + UI
python -m backend.cli serve                         # http://127.0.0.1:8000 (X-API-Key: dev-api-key)
(cd frontend && npm ci && npm run build && npm start)   # http://localhost:3000
```

Runs are headless unless `--headed` is given or `AITEST_BROWSER_HEADLESS=false` is set (the per-run
choice wins). Credentials belong in the secret store (`--secret`, or the UI) and are referenced as
`{{name}}`; an inline password in a test is vaulted and masked automatically.

Docker (team / server setup): `docker compose up --build` starts PostgreSQL+pgvector, Redis, MinIO, the
API, two workers, the Playwright MCP browser worker (HTTP transport), the UI and the demo app. See
[`docs/deployment.md`](docs/deployment.md).

## Demonstrations

* `python scripts/demo_e2e.py` — first run → memory replay with new data → UI change (v2 locators,
  decoy buttons) detected and re-grounded → replay of the healed strategy → wrong password must FAIL,
  plus the capability catalogue; writes [`docs/demo-results.md`](docs/demo-results.md) with
  ground-truth metrics (correct element identification, incorrect action rate, false healing, ...).
* `python scripts/demo_captcha_api.py` — CAPTCHA hand-off through the API: the agent detects the
  challenge, refuses to operate it, pauses with its live browser, an authorized human completes it,
  and the run resumes.
* `python scripts/demo_parallel_api.py` — one test, three data sets → three concurrent executions in
  isolated browser contexts (data-driven + parallel runners).
* `examples/` — test cases for forms, dialogs, tabs, iframes, files, drag-and-drop, canvas, dynamic
  content, autocomplete, mobile layout, slow navigation, a one-time code from the secret vault
  (masked in screenshots and reports), prompt injection, a broken checkout and validation failures,
  plus:
  * `native_controls.md` — OS/browser-rendered controls: native select with optgroups, async
    dependent select, multi-select, datalist, date/time/color/range/number pickers;
  * `custom_dropdowns.md` — portal listbox, search-as-you-type, 400-row virtualized list, ARIA-less
    div dropdown, multi-select chips, hidden native select behind a custom UI, nested menu;
  * `messy_soup.md` — unlabeled table form, `div` buttons, hidden duplicate, shadow-DOM web component;
  * `typos_and_synonyms.md`, `typo_dropdowns.md` — "Clik on the Loign button", "Clcik Sing In",
    "Selct Indai from the contry dropdown";
  * `healthlink_interaction.md` — a patient-interaction flow written the way testers write it ("Click
    on Theme Box and select Initial Call", "Click on Start Date Selection and select 09 June 2026",
    "Select Start Time and set the first field as 23 HH and second field as 15 MM", "Check whether new
    Interaction for patient Abc is added on the Interaction Logs tab") against the `/care` replica;
  * `continue_after_failed_check.md` — a failed check is reported and the independent steps still run.
* [`docs/sample-reports/`](docs/sample-reports/) — generated reports, refreshed by `demo_e2e.py`: a
  first run, the UI-change run (stale memory detected, re-grounded, recovery recorded), a failed
  broken-checkout run, a one-time-code run showing secrets masked in code, text and screenshots, and
  the custom-dropdown run (portal, search, virtualized, div-soup, multi-select, menu).

## Tests

```bash
pytest tests/unit                                # fast, no browser, no network
pytest -m "browser and db" tests/integration     # real Playwright MCP + demo app, on SQLite by default
AITEST_DATABASE_URL=postgresql+asyncpg://aitest:aitest@localhost:5432/aitest \
  pytest -m "browser and db" tests/integration   # the same suite on PostgreSQL + pgvector
```

## Models

No LLM is required: every agent has a deterministic implementation and the harness only calls a
model when evidence is insufficient (low-confidence parsing, ambiguous grounding, visual surfaces,
semantic assertions, recovery after the deterministic strategies). Providers:

* `AITEST_LLM_PROVIDER=azure_openai` — official OpenAI SDK against your Azure resource
  (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`,
  `AZURE_OPENAI_API_VERSION`). Responses API with JSON-schema structured output, falling back to Chat
  Completions where the resource or API version lacks it; separate deployments per tier
  (`AZURE_OPENAI_REASONING|CLASSIFICATION|VISION_DEPLOYMENT`, or non-Claude `AITEST_LLM_*_MODEL` names);
  timeouts, retries with backoff, rate-limit / auth / deployment / content-filter errors turned into
  actionable messages, every output validated against its schema. Verify with
  `python -m backend.cli check-llm`.
* `AITEST_LLM_PROVIDER=anthropic` — official Anthropic SDK, schema-validated structured outputs,
  `claude-opus-5-5` for reasoning/vision and `claude-haiku-5-5` for classification, server-side refusal
  fallback (`ANTHROPIC_API_KEY` in the environment or `.env`).
* `AITEST_LLM_PROVIDER=openai_compatible` — any OpenAI-compatible endpoint.

Models never produce executable code, never see secrets, and can only choose among elements and
recovery strategies the platform offers.

## Known limits

* **CAPTCHA / human verification** is detected and never solved: the run pauses for a person (live
  browser in the UI, or `--headed` in the terminal), or use a test environment without it.
* **Access controls are respected.** Sites that reject automated browsers (HTTP 403 from a WAF or an
  IP allowlist) need an allowlisted network or test account; single sign-on hosts must be allowed
  explicitly (`--allow-domain`).
* Tests run exactly as written: duplicated lines run twice, and items on another page of a table need
  an explicit paging step.

See [`docs/`](docs/) for architecture, agent catalogue, memory design, security, API and deployment.
