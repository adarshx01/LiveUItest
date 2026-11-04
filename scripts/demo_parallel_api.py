"""Data-driven + parallel execution through the API.

One versioned test case, several data sets: each data set becomes its own execution in an
isolated browser context, and they run concurrently (bounded by AITEST_MAX_CONCURRENT_EXECUTIONS).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000/api/v1"
HEADERS = {"X-API-Key": "dev-api-key"}
DATASETS = [
    {"full_name": "Jane Doe", "email": "jane@example.com", "country": "India"},
    {"full_name": "John Roe", "email": "john@example.com", "country": "Canada"},
    {"full_name": "Mia Chen", "email": "mia@example.com", "country": "United States"},
]


def main() -> int:
    content = (Path(__file__).resolve().parents[1] / "examples" / "profile_data_driven.md").read_text()
    with httpx.Client(timeout=60) as client:
        project = client.post(f"{API}/projects", headers=HEADERS, json={"name": "Data-driven demo"}).json()
        tc = client.post(f"{API}/test-cases", headers=HEADERS, json={"project_id": project["id"], "content": content}).json()
        started = time.time()
        batch = client.post(f"{API}/executions", headers=HEADERS,
                            json={"test_case_id": tc["id"], "datasets": DATASETS}).json()
        ids = [e["id"] for e in batch["executions"]]
        print(f"{len(ids)} executions started for {tc['id']}: {ids}")
        final: dict[str, dict] = {}
        while len(final) < len(ids) and time.time() - started < 600:
            for eid in ids:
                if eid in final:
                    continue
                run = client.get(f"{API}/executions/{eid}", headers=HEADERS).json()
                if run["status"] in ("COMPLETED", "ERROR", "CANCELLED"):
                    final[eid] = run
            time.sleep(2)
        elapsed = time.time() - started
        for eid, data in zip(ids, DATASETS):
            run = final.get(eid, {})
            print(f"  {eid}  {data['full_name']:<10} {data['country']:<14} -> {run.get('outcome')}  "
                  f"({run.get('started_at')} .. {run.get('finished_at')})")
        print(f"wall clock for {len(ids)} runs: {elapsed:.0f}s")
        return 0 if all(final.get(e, {}).get("outcome") == "PASSED" for e in ids) else 1


if __name__ == "__main__":
    sys.exit(main())
