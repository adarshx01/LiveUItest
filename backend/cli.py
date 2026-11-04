"""Command-line entry point: ``python -m backend.cli run --url ... --test login.txt``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from backend.agents.governance.secret_manager import SecretScope


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
        content = Path(args.test).read_text() if args.test else args.text
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
                              spec=version.spec, raw_input={"content": content, "variables": variables,
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
    run.add_argument("--checkpointer", choices=["postgres", "memory"], default="postgres")
    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("init-db", help="create the database schema")
    args = parser.parse_args(argv)
    if args.command == "run":
        if not args.test and not args.text:
            parser.error("--test or --text is required")
        return asyncio.run(run_test(args))
    if args.command == "serve":
        import uvicorn

        uvicorn.run("backend.api.main:app", host=args.host, port=args.port, log_level="info")
        return 0
    if args.command == "init-db":
        from backend.models.database import init_db

        asyncio.run(init_db())
        print("database ready")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
