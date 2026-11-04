# Running locally on Windows (no Docker, no WSL, no admin rights)

The default installation is fully local: a SQLite database file, screenshots and reports on disk,
in-process events, and a Playwright-managed Chromium. PostgreSQL, Redis, Docker, Rancher Desktop and
WSL2 are **not** needed. Everything below runs as a normal user in PowerShell.

## 1. Prerequisites (once)

| Tool | Version | Install without admin rights |
|---|---|---|
| Python | 3.12 or newer | python.org installer, **"Install for me only"** (untick "for all users"), tick "Add python.exe to PATH" |
| Node.js | 20 or newer (22 LTS recommended) | download the **Windows .zip** from nodejs.org, extract to `%LOCALAPPDATA%\node`, add that folder to your *user* PATH (Settings → "Edit environment variables for your account") |
| Git | any | Git for Windows, "Only for me" |

Open a **new** PowerShell window afterwards and check:

```powershell
python --version      # 3.12+
node --version        # v20+ / v22+
```

If `python` opens the Microsoft Store, turn off *Settings → Apps → Advanced app settings → App execution
aliases → python.exe / python3.exe*, or use the launcher: `py -3.12`.

## 2. Install

```powershell
cd C:\work                                   # any folder you own
git clone https://github.com/adarshx01/LiveUItest.git
cd LiveUItest

python -m venv .venv
# If activation is blocked ("running scripts is disabled"), allow it for your user only:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"

# pinned Playwright MCP server + its matching Chromium (downloaded into your user profile)
cd browser\playwright-mcp
npm ci
npx playwright install chromium
cd ..\..
```

Behind a corporate proxy, set it first in the same window:
`$env:HTTPS_PROXY="http://proxy.company:8080"; $env:HTTP_PROXY=$env:HTTPS_PROXY`
(pip, npm and the Chromium download all honour it).

## 3. Configure

```powershell
copy .env.example .env
notepad .env
```

Nothing is required for a first run. To use Azure OpenAI, set:

```ini
AITEST_LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key from Azure portal → your resource → Keys and Endpoint>
AZURE_OPENAI_DEPLOYMENT=<your deployment name, e.g. gpt-4.1>
AZURE_OPENAI_API_VERSION=2025-03-01-preview
# optional, per tier:
AZURE_OPENAI_CLASSIFICATION_DEPLOYMENT=<a small/cheap deployment, e.g. gpt-4.1-mini>
AZURE_OPENAI_VISION_DEPLOYMENT=<a vision-capable deployment, e.g. gpt-4o>
```

> **If a key was ever pasted into a chat, ticket, screenshot or commit, rotate it now:** Azure portal →
> your OpenAI resource → *Keys and Endpoint* → **Regenerate Key 1** (or switch your apps to Key 2 first,
> then regenerate Key 1). Put only the new key in `.env`.

## 4. Check, initialise, verify the model

```powershell
python -m backend.cli doctor       # python, node, MCP server, browser, database, secret key, model, headless mode
python -m backend.cli init-db      # creates / upgrades data\aitest.db (also happens automatically on first use)
python -m backend.cli check-llm    # only when a model is configured: one tiny call per tier, with diagnosis
```

`check-llm` prints, per tier, the deployment used, which API served it (Responses or Chat Completions),
latency and tokens - or an actionable error (wrong key, unknown deployment, unsupported API version,
rate limit, content filter). The key itself is never printed.

## 5. Run a test

Start the bundled demo application in a **second** PowerShell window (activate the venv there too):

```powershell
.\.venv\Scripts\Activate.ps1
python -m demo.app                 # http://127.0.0.1:8800
```

Then, in the first window:

```powershell
python -m backend.cli run --test examples\login.txt --headed
python -m backend.cli run --test examples\healthlink_interaction.md `
    --var care_email=care.user@example.com --secret "care_password=CareDemo#2026" --headed
```

`--headed` shows the browser for that run; without it runs are headless (`AITEST_BROWSER_HEADLESS`).
Every run prints the outcome per step and the path of an HTML report under `data\artifacts\`.

### Your own test cases

Save the test as a `.txt` or `.md` file (UTF-8 is best; UTF-16 from Notepad / PowerShell and legacy
code pages are detected too, so Japanese and accented text survive). Keep credentials out of the file:

```text
http://your-app.example.com/sign-in
Enter {{user_email}} as Email
Enter {{user_password}} as Password
Click on Login button
...
```

```powershell
python -m backend.cli run --test C:\tests\interaction.txt `
    --var user_email=someone@example.com --secret "user_password=<the password>" --headed
```

Secrets are stored encrypted, typed into the page without ever appearing in logs, reports, screenshots
or the database. A password written inline in a test is detected, moved into the secret store and
masked in everything the platform stores - but prefer `{{placeholders}}` so it never sits in a file.

If a flow leaves the application's own host (single sign-on, an identity provider, a payment page),
allow that host explicitly: `--allow-domain stg-keycloak.example.com` (or `AITEST_ALLOWED_DOMAINS` in
`.env`). Navigation anywhere else is blocked by design.

## 6. API and web UI

```powershell
python -m backend.cli serve                 # API on http://127.0.0.1:8000  (header X-API-Key: dev-api-key)
```

In another window:

```powershell
cd frontend
npm ci
npm run build
npm start                                   # UI on http://localhost:3000
```

The UI talks to `http://localhost:8000` with the key `dev-api-key` by default; override with
`$env:NEXT_PUBLIC_API_URL` / `$env:NEXT_PUBLIC_API_KEY` **before** `npm run build`.

## 7. Optional services

* **PostgreSQL** (shared team database, pgvector similarity search for the memory): set
  `AITEST_DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/aitest`; the server needs the `pgvector`
  extension. Checkpoints follow automatically. Without it, SQLite does everything a single user needs
  (the memory then ranks similar steps in Python).
* **Redis**: only for splitting the API and execution workers into separate processes
  (`AITEST_USE_REDIS=true`, `AITEST_REDIS_URL=...`, `AITEST_RUN_WORKER_IN_API=false`).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (no admin), or skip activation and call `.\.venv\Scripts\python.exe` directly |
| `doctor`: `node ... not on PATH` | Node is not on your *user* PATH, or the terminal predates the change: reopen PowerShell |
| `doctor`: `playwright mcp missing` | `cd browser\playwright-mcp; npm ci` |
| Run fails with `Executable doesn't exist ... chromium` | `cd browser\playwright-mcp; npx playwright install chromium`. If downloads are blocked, point at an installed Chrome/Edge: `AITEST_BROWSER_EXECUTABLE_PATH=C:/Program Files/Google/Chrome/Application/chrome.exe` |
| `connection refused` to PostgreSQL / Redis, `init-db` fails | An old `.env` still points at Docker services. Remove `AITEST_DATABASE_URL` / `AITEST_CHECKPOINT_DATABASE_URL` (SQLite is the default) and set `AITEST_USE_REDIS=false` |
| Rancher Desktop / WSL2 / Docker errors | Not needed for local use; nothing in this guide uses them |
| `database is locked` | Another process holds a long write (e.g. a DB browser with an open transaction). Close it; the platform already uses WAL mode and a 30 s busy timeout |
| `AITEST_SECRET_KEY is not a valid Fernet key` | Leave it empty (a key is created and kept in Windows Credential Manager) or generate one with the command in `.env.example` |
| Secret store errors after moving to a new PC | The encryption key stayed on the old machine. Re-enter the secrets (`--secret`), or copy the key: `python -m backend.cli doctor` shows where it lives |
| `check-llm`: `401` / "check AZURE_OPENAI_API_KEY" | Wrong or rotated key; copy Key 1 or Key 2 again from the portal |
| `check-llm`: `404` / "check AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_VERSION" | The deployment name is not the model name - use the name shown under *Deployments*; or the API version is not available in your region |
| `check-llm`: Responses API unsupported | Nothing to do with `AZURE_OPENAI_API=auto`: it falls back to Chat Completions and remembers it. Force with `AZURE_OPENAI_API=chat` |
| `check-llm`: `429` rate limit | The deployment's tokens-per-minute quota is too low; raise it in Azure AI Foundry or set `AZURE_OPENAI_CLASSIFICATION_DEPLOYMENT` to a separate deployment. Retries with backoff are automatic (`AITEST_LLM_MAX_RETRIES`) |
| Target site answers **HTTP 403** | The site's firewall / bot protection / IP allowlist rejects the automated browser (seen with upload flows). Run `--headed` from the network the site expects (VPN), ask the site owner to allowlist your IP or test user, or use a test environment without the WAF rule. The platform does not try to bypass access controls |
| Step `BLOCKED: navigation to <host> is not allowed` | The flow left the application's host (SSO). Add `--allow-domain <host>` |
| **CAPTCHA / "verify you are human"** | Detected and never solved automatically. The run pauses; complete it in the live browser (UI → *Live browser*, or `--headed` and answer `y` in the terminal), or disable the CAPTCHA for the test account / environment |
| A step was executed twice | The test file contains the instruction twice (common after copy-paste). Every line is executed as written - remove the duplicate |
| Item is on another page of a table | Write the paging step explicitly ("Click on Next page", "Go to page 2") before the step that uses the item |
| Garbled Japanese / accented text in the console | `chcp 65001` and `$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8` (display only - test files are decoded correctly regardless) |
| Port already in use | `serve --port 8001`; demo app: stop the other instance; UI: `npx next start -p 3001` |
| Want to watch every run | `AITEST_BROWSER_HEADLESS=false` in `.env` (or `--headed` per run; a per-run choice always wins) |
