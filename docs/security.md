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

Redaction happens in layers, so one missed path does not leak a value:

1. **At the source.** Playwright MCP echoes the code it ran (`await page.fill(..., 'value')`) and can
   quote the value in errors. The tool adapter knows which arguments of a call are secret
   (`browser_type.text`, `browser_fill_form.fields[].value`), registers them and scrubs the echoed
   code, error and result text before the response leaves the browser layer. Action records, events,
   step persistence and reports therefore never see the raw value.
2. **At every persistence or egress boundary.** Logs, events, agent traces, step results, assertion
   records, run reasons, debug snapshots and model prompts pass through the redactor. It masks every
   value registered during the process lifetime, plus credential-shaped keys (`password`, `apiKey`,
   `X-Auth-Token`, `session_id`, ...) and patterns (bearer tokens, JWTs, cloud keys, Luhn-valid card
   numbers). Keys are matched by word, so counters such as `passed_steps` or `llm_input_tokens` stay
   readable.
3. **On the rendered report.** The HTML report gets a final pass that matches raw and HTML-escaped
   values. Embedded screenshots are skipped, because masking inside base64 would corrupt them.
4. **In screenshots.** Browsers already draw password inputs as discs. When a secret is typed into
   any other field (OTP, card number), a registered presentational probe draws that field as discs
   (`-webkit-text-security`), or makes its text transparent where that property is unsupported.
   Every later screenshot then shows the field masked. The field's value and events are not changed.

Observation data held in memory (snapshots, probe results) is not scrubbed, because assertions must
see the real page. It is redacted when it crosses a boundary.

`tests/unit/test_redaction.py` covers the rules. The integration suite checks that the typed secrets
of the login and OTP scenarios appear nowhere in the HTML/JSON report, in the run's rows of every
run table, or in any memory table.

## Encryption key and credentials

The secret store and the sealed part of run data sets are encrypted with one Fernet key. Lookup order:

1. `AITEST_SECRET_KEY` (servers, docker compose: every process must share one key). An invalid value
   stops start-up with a message that shows how to generate a valid one - it is never silently replaced.
2. An existing `data/secret.key` (installations created before the key store existed keep working).
3. The OS credential store via `keyring` (Windows Credential Manager, macOS Keychain, Secret Service),
   service `ai-testing-platform`. A key is created there on first use. `AITEST_SECRET_KEY_STORE=file`
   disables this.
4. Otherwise a new `data/secret.key`, restricted to the current user (`chmod 600`, or an `icacls` ACL on
   Windows).

`python -m backend.cli doctor` reports which source is in use without printing the key.

**Credentials in test cases.** Reference them as `{{name}}` and pass them with `--secret name=value` (or
the UI's secret store). A password written inline ("Enter Pa55word as Password") is still protected:
when the test is saved, the value is moved into the secret store, registered for redaction, and replaced
by a mask in the stored source, in every step's instruction text and in the compiled plan. Only the
compiled plan reaches the execution state, so the value never enters checkpoints, reports, events or
memory. The integration suite asserts this for the replica scenario.

**Model provider keys** (`ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `AITEST_OPENAI_API_KEY`) are read
from the environment or `.env`, registered for redaction at start-up and never logged; provider errors
are rewritten into actionable messages with the key masked. A key that was ever pasted into a chat,
ticket, screenshot or commit must be regenerated at the provider (Azure portal -> resource -> *Keys and
Endpoint* -> *Regenerate*), not just removed from the file.

**Identifiers vs. secrets.** Keys such as `session_id` are credential-shaped and masked - except when the
value is one of the platform's own identifiers (`bs_<hex>`, `exec_<hex>`), which carry no authority and
are needed to correlate logs. A real session cookie or token under the same key is still masked.

## Isolation and audit

One isolated (in-memory profile) browser context per execution, own workspace for uploads/downloads,
never shared across tenants. Every MCP call (redacted), node trace, approval decision, human browser
action and execution start/cancel is persisted (`browser_session_events`, `agent_traces`,
`approval_events`, `audit_logs`).
