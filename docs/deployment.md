# Deployment

**Single user / workstation:** nothing to deploy. The defaults are a SQLite database and checkpoint file
under `data/`, filesystem artifacts, in-process events and execution, and an OS-keychain encryption key.
See [windows-local-setup.md](windows-local-setup.md) for the step-by-step guide (also valid on macOS and
Linux with the usual path syntax). PostgreSQL, Redis and S3 are integrations you switch on:

| Setting | Local default | Team / server |
|---|---|---|
| `AITEST_DATABASE_URL` | `sqlite+aiosqlite:///<repo>/data/aitest.db` | `postgresql+asyncpg://...` (pgvector required) |
| `AITEST_CHECKPOINT_DATABASE_URL` | `auto` (`data/checkpoints.db`) | `auto` (same PostgreSQL) |
| `AITEST_USE_REDIS` / `AITEST_REDIS_URL` | `false` / empty | `true` / `redis://...` |
| `AITEST_RUN_WORKER_IN_API` | `true` | `false` with separate workers |
| `AITEST_SECRET_KEY` | empty (OS keychain / `data/secret.key`) | one shared key for all processes |

The schema is created and upgraded automatically on start-up (new tables and new columns; SQLite runs in
WAL mode with a busy timeout). A configured PostgreSQL that cannot be reached is an error with the URL's
host in the message; Redis that is not enabled is simply not used.

## Containers

Set `AITEST_SECRET_KEY` in `.env` first (`python -c "from cryptography.fernet import Fernet;
print(Fernet.generate_key().decode())"`). Compose refuses to start without it. The API and every
worker must share one key: secrets stored through the API, and the sealed credential part of a run's
data set (queued through Redis), are decrypted by whichever worker picks up the run.

`docker compose up --build` runs every component independently:

* **postgres** (`pgvector/pgvector:pg16`) — system of record, LangGraph checkpoints, vector memory.
* **redis** — execution queue (Redis Streams), event fan-out to API replicas.
* **minio** — S3-compatible artifact storage (`AITEST_ARTIFACT_BACKEND=s3`).
* **browser** — Playwright MCP over HTTP (`AITEST_MCP_TRANSPORT=http`); each MCP connection gets an
  isolated browser context. Scale it horizontally behind the URL.
* **api** — FastAPI; with `AITEST_RUN_WORKER_IN_API=false` it only enqueues executions.
* **worker** — consumes the queue, runs the LangGraph engine (bounded concurrency per worker).
* **frontend** — Next.js standalone server. **demo** — the controlled demo application.

## Version pinning

`browser/playwright-mcp/package.json` pins `@playwright/mcp`; the backend checks the server's reported
version against `AITEST_MCP_EXPECTED_VERSION_PREFIX` and refuses mismatches in production. Images
install the exact browser/ffmpeg revisions the pinned Playwright expects. Where downloads are not
possible, `infrastructure/scripts/link_browsers.sh` builds a compatible local `PLAYWRIGHT_BROWSERS_PATH`
from installed builds (video recording needs the matching ffmpeg; without it the report falls back to
an animated replay assembled from the per-action screenshots).

## Production checklist

* `AITEST_ENVIRONMENT=production`, a real `AITEST_SECRET_KEY`, rotated `AITEST_API_KEYS`.
* Per-environment allowed domains and explicit grants for destructive/financial/messaging actions.
* Artifact retention (`AITEST_ARTIFACT_RETENTION_DAYS`) and S3 server-side encryption (enabled by default).
* Export OpenTelemetry spans to your collector (the in-process exporter backs `/diagnostics/traces`).
* Optional LLM: `AITEST_LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`, or `azure_openai` with the
  `AZURE_OPENAI_*` settings (verify with `python -m backend.cli check-llm`).
