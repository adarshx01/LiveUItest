# End-to-end demonstration results

Generated 2026-11-04 02:28 UTC by `scripts/demo_e2e.py` against the demo application, with no LLM configured (deterministic agents only) and a clean memory store.

| Scenario | Capability | Expected | Outcome | Memory hits | Recovered steps | Wrong element touched | Duration |
|---|---|---|---|---|---|---|---|
| 1. first run (learns) | natural-language login, first run without memory | PASSED | **PASSED** | 0/5 | 0 | none | 26s |
| 2. memory replay, new data | validated memory re-grounded with different values | PASSED | **PASSED** | 4/5 | 0 | none | 24s |
| 3. UI changed (v2 locators) | change detected, stale memory rejected, target re-grounded | PASSED | **PASSED** | 1/5 | 3 | none | 24s |
| 4. replay after healing | healed strategy (new memory version) replayed | PASSED | **PASSED** | 4/5 | 0 | none | 24s |
| 5. wrong password (negative) | real failure reported, never healed into a pass | FAILED | **FAILED** | 1/5 | 3 | none | 32s |
| complex form + custom dropdowns | dependent native selects, lazy custom combobox, radio, checkbox, switch | PASSED | **PASSED** | 0/12 | 0 | none | 54s |
| registration validation (negative) | conditional field, expected rejection | PASSED | **PASSED** | 0/8 | 0 | none | 43s |
| broken checkout | application error pinned to the failing step | FAILED | **FAILED** | 0/5 | 0 | none | 20s |
| dialogs and tabs | modal, nested alertdialog, native confirm, new tab | PASSED | **PASSED** | 0/8 | 0 | none | 49s |
| upload, download, iframe | file chooser, download event, iframe refs | PASSED | **PASSED** | 0/4 | 0 | none | 30s |
| drag-and-drop + canvas | HTML5 drag, visual colour/shape grounding with coordinate fallback | PASSED | **PASSED** | 0/3 | 0 | none | 27s |
| dynamic content, infinite scroll, icons | late rendering, re-render, lazy loading, unlabeled icon | PASSED | **PASSED** | 0/4 | 0 | none | 51s |
| prompt injection page | instruction-like page text ignored and reported | PASSED | **PASSED** | 0/1 | 0 | none | 14s |
| missing required field | native validation failure reported precisely | FAILED | **FAILED** | 2/2 | 0 | none | 12s |
| autocomplete | debounced suggestions picked by exact label | PASSED | **PASSED** | 0/1 | 0 | none | 15s |
| responsive mobile layout | hamburger menu at 390x844 | PASSED | **PASSED** | 0/2 | 0 | none | 15s |
| transient slow navigation | 2 s server delay absorbed by settle + assertion polling | PASSED | **PASSED** | 0/1 | 0 | none | 13s |
| one-time code from the vault | secret typed into a non-password field, masked in screenshots and reports | PASSED | **PASSED** | 0/2 | 0 | none | 15s |

## Evaluation metrics

| Metric | Value |
|---|---|
| scenarios | 18 |
| test completion rate | 1.0 |
| outcome accuracy | 1.0 |
| correct element identification rate | 1.0 |
| incorrect action rate | 0.0 |
| self healing success rate | 1.0 |
| false healing rate | 0.0 |
| memory replay success rate | 1.0 |
| memory hit rate | 0.154 |
| human intervention rate | 0.0 |
| average execution seconds | 27.2 |
| average llm cost usd | 0.0 |
| recoveries | 6 |

Ground truth comes from the demo app's interaction recorder (`data-gt` attributes), so the element-identification metrics reflect which controls were actually operated.

Sample reports from this run: [1. first run (learns)](sample-reports/login-first-run.html), [3. UI changed (v2 locators)](sample-reports/ui-change-healed.html), [broken checkout](sample-reports/broken-checkout-failed.html), [one-time code from the vault](sample-reports/otp-secret-masked.html).
