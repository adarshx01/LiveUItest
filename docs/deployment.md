# Deployment

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
* Optional LLM: `AITEST_LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`.
