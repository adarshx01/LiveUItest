# End-to-end demonstration results

Generated 2026-11-04 01:25 UTC by `scripts/demo_e2e.py` against the demo application, with no LLM configured (deterministic agents only) and a clean memory store.

| Scenario | Capability | Expected | Outcome | Memory hits | Recovered steps | Wrong element touched | Duration |
|---|---|---|---|---|---|---|---|
| 1. first run (learns) | natural-language login, first run without memory | PASSED | **PASSED** | 0/5 | 0 | none | 25s |
| 2. memory replay, new data | validated memory re-grounded with different values | PASSED | **PASSED** | 4/5 | 0 | none | 25s |
| 3. UI changed (v2 locators) | change detected, stale memory rejected, target re-grounded | PASSED | **PASSED** | 1/5 | 3 | none | 25s |
| 4. replay after healing | healed strategy (new memory version) replayed | PASSED | **PASSED** | 4/5 | 0 | none | 24s |
| 5. wrong password (negative) | real failure reported, never healed into a pass | FAILED | **FAILED** | 1/5 | 3 | none | 31s |

## Evaluation metrics

| Metric | Value |
|---|---|
| scenarios | 5 |
| test completion rate | 1.0 |
| outcome accuracy | 1.0 |
| correct element identification rate | 1.0 |
| incorrect action rate | 0.0 |
| self healing success rate | 1.0 |
| false healing rate | 0.0 |
| memory replay success rate | 1.0 |
| memory hit rate | 0.4 |
| human intervention rate | 0.0 |
| average execution seconds | 26.0 |
| average llm cost usd | 0.0 |
| recoveries | 6 |

Ground truth comes from the demo app's interaction recorder (`data-gt` attributes), so the element-identification metrics reflect which controls were actually operated.
