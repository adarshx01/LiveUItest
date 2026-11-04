# End-to-end demonstration results

Generated 2026-11-04 01:33 UTC by `scripts/demo_e2e.py` against the demo application, with no LLM configured (deterministic agents only) and a clean memory store.

| Scenario | Capability | Expected | Outcome | Memory hits | Recovered steps | Wrong element touched | Duration |
|---|---|---|---|---|---|---|---|
| 1. first run (learns) | natural-language login, first run without memory | PASSED | **PASSED** | 0/5 | 0 | none | 26s |
| 2. memory replay, new data | validated memory re-grounded with different values | PASSED | **PASSED** | 4/5 | 0 | none | 24s |
| 3. UI changed (v2 locators) | change detected, stale memory rejected, target re-grounded | PASSED | **PASSED** | 1/5 | 3 | none | 24s |
| 4. replay after healing | healed strategy (new memory version) replayed | PASSED | **PASSED** | 4/5 | 0 | none | 24s |
| 5. wrong password (negative) | real failure reported, never healed into a pass | FAILED | **FAILED** | 1/5 | 3 | none | 32s |
| complex form + custom dropdowns | dependent native selects, lazy custom combobox, radio, checkbox, switch | PASSED | **PASSED** | 0/12 | 0 | none | 54s |
| registration validation (negative) | conditional field, expected rejection | PASSED | **PASSED** | 0/8 | 0 | none | 42s |
| broken checkout | application error pinned to the failing step | FAILED | **FAILED** | 0/5 | 0 | none | 20s |
| dialogs and tabs | modal, nested alertdialog, native confirm, new tab | PASSED | **PASSED** | 0/8 | 0 | none | 48s |
| upload, download, iframe | file chooser, download event, iframe refs | PASSED | **PASSED** | 0/4 | 0 | none | 29s |
| drag-and-drop + canvas | HTML5 drag, visual colour/shape grounding with coordinate fallback | PASSED | **PASSED** | 0/3 | 0 | none | 26s |
| dynamic content, infinite scroll, icons | late rendering, re-render, lazy loading, unlabeled icon | PASSED | **PASSED** | 0/4 | 0 | none | 50s |
| prompt injection page | instruction-like page text ignored and reported | PASSED | **BLOCKED** | 0/1 | 0 | none | 3s |
| missing required field | native validation failure reported precisely | FAILED | **FAILED** | 2/2 | 0 | none | 12s |

## Evaluation metrics

| Metric | Value |
|---|---|
| scenarios | 14 |
| test completion rate | 1.0 |
| outcome accuracy | 0.929 |
| correct element identification rate | 0.935 |
| incorrect action rate | 0.0 |
| self healing success rate | 1.0 |
| false healing rate | 0.0 |
| memory replay success rate | 1.0 |
| memory hit rate | 0.167 |
| human intervention rate | 0.0 |
| average execution seconds | 29.6 |
| average llm cost usd | 0.0 |
| recoveries | 6 |

Ground truth comes from the demo app's interaction recorder (`data-gt` attributes), so the element-identification metrics reflect which controls were actually operated.
