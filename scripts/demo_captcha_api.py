"""Demonstrates CAPTCHA handoff through the public API.

1. the demo app turns on its bot challenge in front of /login;
2. a login test is created and executed via the API;
3. the agent detects the challenge, does NOT operate it, and pauses (AWAITING_HUMAN);
4. an authorized human completes the challenge in the live browser via the API;
5. the execution is resumed and finishes; the report records the pause.
"""

from __future__ import annotations

import sys
import time

import httpx

API = "http://127.0.0.1:8000/api/v1"
DEMO = "http://127.0.0.1:8800"
HEADERS = {"X-API-Key": "dev-api-key"}

TEST = """URL: http://127.0.0.1:8800

Test Case: Login behind a bot challenge

1. Go to http://127.0.0.1:8800/login
2. Enter username as abc.
3. Enter password as abc.
4. Click on the Sign In button.
5. Verify that the user is redirected to the dashboard.
"""


def wait_status(client: httpx.Client, execution_id: str, statuses: set[str], timeout: float = 240) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"{API}/executions/{execution_id}", headers=HEADERS).json()
        if run["status"] in statuses:
            return run
        time.sleep(1.5)
    raise TimeoutError(f"execution did not reach {statuses}")


def main() -> int:
    with httpx.Client(timeout=60) as client:
        client.post(f"{DEMO}/__admin/reset")
        client.post(f"{DEMO}/__admin/config", json={"challenge_enabled": True, "ui_version": 1})
        project = client.post(f"{API}/projects", headers=HEADERS, json={"name": "CAPTCHA demo"}).json()
        tc = client.post(f"{API}/test-cases", headers=HEADERS,
                         json={"project_id": project["id"], "content": TEST}).json()
        print(f"test case {tc['id']} v{tc['version']} with {len(tc['steps'])} steps")
        run = client.post(f"{API}/executions", headers=HEADERS, json={"test_case_id": tc["id"]}).json()
        execution_id = run["id"]
        print(f"execution {execution_id} started")
        paused = wait_status(client, execution_id, {"AWAITING_HUMAN", "COMPLETED", "ERROR"})
        print(f"status: {paused['status']}  pending: {paused.get('pending', {}).get('kind')}")
        if paused["status"] != "AWAITING_HUMAN":
            print("expected the execution to pause for human verification")
            return 1
        observation = client.get(f"{API}/executions/{execution_id}/observation", headers=HEADERS).json()
        print(f"agent sees: {observation['summary']['title']!r}, captcha={observation['page_state']['captcha_detected']}"
              f" evidence={observation['page_state']['captcha_evidence']}")
        checkbox = next(e for e in observation["elements"] if e["role"] == "checkbox" and "robot" in e["name"].lower())
        box = checkbox["bbox"]
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        print(f"human clicks the challenge checkbox at ({x:.0f}, {y:.0f})")
        result = client.post(f"{API}/executions/{execution_id}/browser/human-action", headers=HEADERS,
                             json={"type": "click", "x": x, "y": y}).json()
        print(f"human action ok={result['ok']}")
        time.sleep(1.5)
        client.post(f"{API}/executions/{execution_id}/resume", headers=HEADERS, json={"approved": True})
        final = wait_status(client, execution_id, {"COMPLETED", "ERROR", "CANCELLED"})
        print(f"final: {final['status']} outcome={final['outcome']} — {final['reason']}")
        steps = client.get(f"{API}/executions/{execution_id}/steps", headers=HEADERS).json()
        for step in steps:
            print(f"  {step['index']:>2}. {step['status']:<10} {step['original_text'][:60]:<60} {step['message'][:70]}")
        events = client.get(f"{API}/executions/{execution_id}/events", headers=HEADERS).json()
        print("events:", [e["event"] for e in events if e["event"] in ("HUMAN_APPROVAL_REQUIRED", "EXECUTION_PAUSED",
                                                                      "EXECUTION_RESUMED", "ACTION_BLOCKED")])
        client.post(f"{DEMO}/__admin/config", json={"challenge_enabled": False})
        return 0 if final["outcome"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
