# Security and governance

The regulator (`backend/agents/governance/policy_engine.py`) is deterministic and has no model
input. Every action is evaluated before execution:

| Rule | Behaviour |
|---|---|
| Domain allowlist | only `http(s)` URLs on the test's base host + configured domains; post-navigation check stops the run if a redirect lands elsewhere |
| CAPTCHA / human verification | detected (widgets, iframes, challenge text); the agent never operates them; the execution pauses (`AWAITING_HUMAN`) and an authorized person completes it via audited human actions |
| Sensitive value placement | a secret may only be typed into password / OTP / card fields — prevents exfiltration into search boxes, chat widgets or third-party fields |
| Risk classes | read / navigation / input (low–medium) allowed in scope; destructive, financial and messaging actions need explicit approval unless the *environment* grants them (never in production) |
| Confidence gates | below the refine threshold an action is blocked for human guidance; high-risk actions need the execute threshold or approval |
| Tool restrictions | only allowlisted MCP tools; `browser_run_code_unsafe` blocked; `browser_evaluate` only runs platform probes; WebMCP (page-registered tools) disabled |

## Prompt injection

All page content is untrusted. It never becomes instructions: navigation targets come only from the
test spec or from clicks on grounded elements (and pass the allowlist); model prompts wrap page text in
`<untrusted_page_content>` with an explicit instruction to treat it as data; instruction-like text found
on pages is reported as a security warning.

## Secrets

Fernet-encrypted (`environment_credentials`), scoped org → project → environment; CI may inject
`AITEST_SECRET_<NAME>`. Literal sensitive values in test text are moved into the store and masked in
the saved source. Values are resolved only for an approved action, registered with the redactor, never
sent to models, never persisted in memory, events, logs, screenshots captions or reports. Field
verification for passwords compares an in-page FNV-1a fingerprint instead of reading values back.
The MCP subprocess environment is stripped of platform credentials.

## Isolation and audit

One isolated (in-memory profile) browser context per execution, own workspace for uploads/downloads,
never shared across tenants. Every MCP call (redacted), node trace, approval decision, human browser
action and execution start/cancel is persisted (`browser_session_events`, `agent_traces`,
`approval_events`, `audit_logs`).
