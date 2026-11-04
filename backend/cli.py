"""Command-line entry point: ``python -m backend.cli run --url ... --test login.txt``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from backend.agents.governance.secret_manager import SecretScope


def read_test_file(path: Path) -> str:
    """Test files come from many editors: UTF-8 (with or without BOM), UTF-16 from Notepad/PowerShell,
    legacy Windows code pages. Decode explicitly instead of trusting the OS default (cp1252 on Windows
    would garble Japanese or accented test data)."""

    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


async def check_llm() -> int:
    """One tiny structured call per tier against the configured provider; prints an actionable diagnosis."""

    from pydantic import BaseModel

    from backend.core.config import get_settings
    from backend.core.exceptions import PlatformError
    from backend.harness.model_gateway import ModelGateway, ModelTier

    class Ping(BaseModel):
        ok: bool
        echo: str

    settings = get_settings()
    gateway = ModelGateway(settings)
    print(f"provider: {settings.llm_provider}")
    if settings.llm_provider == "none":
        print("  no model configured - the platform runs fully deterministic (set AITEST_LLM_PROVIDER)")
        return 0
    if settings.llm_provider == "azure_openai":
        print(f"  endpoint: {settings.azure_openai_endpoint or '(missing AZURE_OPENAI_ENDPOINT)'}")
        print(f"  api version: {settings.azure_openai_api_version}  api: {settings.azure_openai_api}")
        print(f"  key: {'set' if settings.azure_openai_api_key else '(missing AZURE_OPENAI_API_KEY)'}")
    failures = 0
    for tier in ModelTier:
        chain = gateway.router.route(tier)
        if not chain:
            print(f"  {tier.value:15} not configured")
            failures += 1
            continue
        _, model = chain[0]
        try:
            result = await gateway.structured(tier=tier, system="Reply with ok=true and echo the word.",
                                              prompt="Word: ping", output_model=Ping, agent="check_llm",
                                              max_tokens=200)
            u = result.usage
            mode = getattr(gateway.providers.get(settings.llm_provider), "_mode", None)
            print(f"  {tier.value:15} OK  model={model} served={u.model} api={mode or 'n/a'} "
                  f"{u.latency_ms:.0f} ms tokens={u.input_tokens}/{u.output_tokens} -> {result.output.echo!r}")
        except PlatformError as exc:
            failures += 1
            print(f"  {tier.value:15} FAILED model={model}: {exc.message} {exc.details or ''}")
    return 1 if failures else 0


def doctor() -> int:
    """Check local prerequisites and print how to fix each problem."""

    import shutil
    import sqlite3

    from backend.core.config import get_settings

    s = get_settings()
    ok = True

    def line(good: bool, label: str, detail: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"[{'ok' if good else '!!'}] {label}: {detail}")

    line(sys.version_info >= (3, 12), "python", sys.version.split()[0] + ("" if sys.version_info >= (3, 12)
                                                                           else " (3.12+ required)"))
    node = shutil.which(s.mcp_node_command)
    line(bool(node), "node", node or f"'{s.mcp_node_command}' not on PATH - install Node.js 20+ (no admin: nvm-windows "
                                     "or the zip build) and reopen the terminal")
    line(s.mcp_server_script.exists(), "playwright mcp",
         str(s.mcp_server_script) if s.mcp_server_script.exists()
         else "missing - run: cd browser/playwright-mcp && npm ci && npx playwright install chromium")
    exe = s.browser_executable_path
    line(not exe or Path(exe).exists(), "browser",
         "Playwright-managed Chromium" if not exe else (exe if Path(exe).exists()
                                                        else f"{exe} does not exist - clear AITEST_BROWSER_EXECUTABLE_PATH"))
    if s.is_sqlite:
        db = Path(s.database_url.split("///", 1)[1])
        try:
            db.parent.mkdir(parents=True, exist_ok=True)
            sqlite3.connect(db).execute("select 1").fetchone()
            line(True, "database", f"SQLite {db}")
        except Exception as exc:  # noqa: BLE001
            line(False, "database", f"cannot open {db}: {exc}")
    else:
        line(True, "database", s.database_url.split("@")[-1])
    line(True, "redis", "enabled" if s.redis_enabled else "not used (in-process events and execution)")
    try:
        from backend.core.security import describe_key_source

        line(True, "secret key", describe_key_source())
    except Exception as exc:  # noqa: BLE001
        line(False, "secret key", str(exc))
    llm = s.llm_provider
    if llm == "azure_openai":
        missing = [n for n, v in (("AZURE_OPENAI_ENDPOINT", s.azure_openai_endpoint),
                                  ("AZURE_OPENAI_API_KEY", s.azure_openai_api_key),
                                  ("AZURE_OPENAI_DEPLOYMENT", s.azure_openai_deployment)) if not v]
        line(not missing, "llm", "azure_openai " + ("configured (run: python -m backend.cli check-llm)" if not missing
                                                    else "missing " + ", ".join(missing)))
    else:
        line(True, "llm", llm)
    line(True, "browser mode", "headless" if s.browser_headless else "headed (AITEST_BROWSER_HEADLESS=false); "
                                                                     "--headed on 'run' overrides per run")
    print("\nall checks passed" if ok else "\nfix the [!!] items above, then run doctor again")
    return 0 if ok else 1


def _pairs(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        key, _, value = item.partition("=")
        out[key.strip()] = value
    return out


async def run_test(args: argparse.Namespace) -> int:
    from backend.bootstrap import bootstrap
    from backend.services.execution_engine import initial_state

    platform = await bootstrap(checkpointer=args.checkpointer, json_logs=False)
    try:
        content = read_test_file(Path(args.test)) if args.test else args.text
        org = args.org
        project = await platform.platform.ensure_project(org, args.project)
        for name, value in _pairs(args.secret).items():
            await platform.secrets.store(SecretScope(org, project.id), name, value)
        variables = _pairs(args.var)
        # run-level data (--var) is execution input, not part of the versioned test definition
        tc, version = await platform.platform.save_test_case(org, project.id, content=content, base_url=args.url,
                                                             name=args.name)
        options: dict[str, Any] = {"use_memory": not args.no_memory, "auto_approve": args.auto_approve,
                                   "headless": not args.headed}
        if args.runner:
            options["runner"] = args.runner
        if args.viewport:
            width, height = args.viewport.lower().split("x")
            options["viewport"] = [int(width), int(height)]
        run = await platform.platform.create_run(org_id=org, tc=tc, version=version.version, options=options)
        state = initial_state(execution_id=run.id, organization_id=org, project_id=project.id,
                              application_id=tc.application_id, test_case_id=tc.id, test_case_version=version.version,
                              # the compiled spec (sensitive literals already vaulted and masked) is what runs;
                              # the raw text never enters the checkpointed graph state
                              spec=version.spec, raw_input={"variables": variables,
                                                            "options": options, "name": tc.name,
                                                            "allowed_domains": args.allow_domain or []})
        print(f"▶ execution {run.id}  test case {tc.id} v{version.version}")
        handle = await platform.engine.run_to_completion(state)
        while handle.status in ("AWAITING_APPROVAL", "AWAITING_HUMAN"):
            print(f"⏸ paused: {json.dumps(handle.interrupt, default=str)[:400]}")
            if not sys.stdin.isatty():
                await platform.engine.cancel(run.id)
                await handle.done.wait()
                break
            answer = input("approve / done (y) or abort (n)? ").strip().lower()
            await platform.engine.resume(run.id, {"approved": answer.startswith("y"), "abort": answer.startswith("n")})
            await asyncio.wait([asyncio.ensure_future(handle.done.wait()),
                                asyncio.ensure_future(handle.paused.wait())], return_when=asyncio.FIRST_COMPLETED)
        final = (handle.final or {}).get("final_result") or {}
        report = (handle.final or {}).get("report") or {}
        print(f"■ outcome: {final.get('outcome')} — {final.get('reason')}")
        for sid, res in sorted(((handle.final or {}).get("step_results") or {}).items(), key=lambda kv: kv[1]["index"]):
            flags = " (recovered)" if res.get("recovered") else ""
            flags += " (memory)" if res.get("memory_used") else ""
            print(f"   {res['index']:>2}. {res['status']:<12} {res['message'][:110]}{flags}")
        if report.get("local_path"):
            print(f"📄 report: {report['local_path']}")
        return 0 if final.get("outcome") == "PASSED" else 1
    finally:
        await platform.engine.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aitest", description="AI-powered end-to-end testing platform")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a natural-language test case")
    run.add_argument("--url", help="application URL (if not in the test text)")
    run.add_argument("--test", help="path to the test case (text/markdown/json/csv)")
    run.add_argument("--text", help="inline test case text")
    run.add_argument("--name")
    run.add_argument("--var", action="append", help="test data variable name=value")
    run.add_argument("--secret", action="append", help="secret name=value (stored encrypted)")
    run.add_argument("--org", default="org_default")
    run.add_argument("--project", default="Default project")
    run.add_argument("--no-memory", action="store_true")
    run.add_argument("--auto-approve", action="store_true", help="auto-approve high-risk actions (test envs only)")
    run.add_argument("--headed", action="store_true")
    run.add_argument("--runner")
    run.add_argument("--viewport", help="e.g. 390x844 for a mobile layout")
    run.add_argument("--allow-domain", action="append")
    run.add_argument("--checkpointer", choices=["auto", "sqlite", "postgres", "memory"], default="auto",
                     help="auto: SQLite locally, PostgreSQL when AITEST_DATABASE_URL points there")
    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the network")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("init-db", help="create or upgrade the database schema")
    sub.add_parser("check-llm", help="verify the configured model provider with one tiny call per tier")
    sub.add_parser("doctor", help="check local prerequisites (node, browser, database, keys, model)")
    args = parser.parse_args(argv)
    from backend.core.exceptions import ConfigurationError

    try:
        return _dispatch(parser, args)
    except ConfigurationError as exc:
        # configuration problems are the user's to fix: say what and how, without a traceback
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.details:
            for key, value in exc.details.items():
                print(f"  {key}: {value}", file=sys.stderr)
        return 2


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "run":
        if not args.test and not args.text:
            parser.error("--test or --text is required")
        return asyncio.run(run_test(args))
    if args.command == "serve":
        import uvicorn

        uvicorn.run("backend.api.main:app", host=args.host, port=args.port, log_level="info")
        return 0
    if args.command == "check-llm":
        return asyncio.run(check_llm())
    if args.command == "doctor":
        return doctor()
    if args.command == "init-db":
        from backend.models.database import init_db

        asyncio.run(init_db())
        print("database ready")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
