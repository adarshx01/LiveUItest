# Architecture

## Layers

```
User / UI (Next.js) ──REST/WebSocket──▶ FastAPI ──▶ ExecutionDispatcher ──▶ ExecutionEngine (LangGraph + AsyncPostgresSaver)
                                                         │ (Redis Streams for separate workers)
                                                         ▼
                                   ExecutionRuntime (one per execution, owns the live browser)
         ┌───────────────┬──────────────┬───────────────┬─────────────┬──────────────┬─────────────┐
     PageObserver   ElementGrounder   PolicyEngine   Interaction-    ActionVerifier  MemoryManager
     (snapshot+DOM)  (+LLM/visual)   (regulator)     Executor        + Assertions    (Postgres/pgvector)
                                                         │
                                         BrowserToolAdapter (allowlist, schemas, budgets, audit)
                                                         │
                                         MCPConnection ──▶ Playwright MCP (isolated context) ──▶ Chromium
```

* **Reasoning is separated from execution.** Agents decide *what* to do; deterministic services
  validate, gate and execute. The model gateway returns schema-validated objects only.
* **State discipline.** LangGraph state holds ids, compact decisions and step summaries; observations,
  screenshots, snapshots and videos live in the runtime cache and object storage (`artifact://` refs).
* **Durability.** Every node is checkpointed in PostgreSQL, so after `execute_action` the graph knows
  the action happened and recovery uses `idempotency_check` instead of blindly repeating it. Approvals
  and human verification use LangGraph interrupts; the paused execution keeps its live browser.

## Execution graph (spec 6.1)

`ingest_test_case → normalize_test_case → analyze_requirements → compile_test_plan →
validate_test_plan → check_permissions → initialize_browser_session → observe_initial_page →
retrieve_application_memory → select_execution_strategy → [prepare_next_action →
retrieve_step_memory → observe_current_page → resolve_target_element → ground_target →
validate_action (→ human_approval) → execute_action → wait_for_state_change → verify_action_result →
learn_pattern → next_test_action]* → final_assertions → evaluate_test_result → persist_test_memory →
generate_report`

Failure branch: `verify_action_result | ground_target | validate_action → classify_failure →
(await_human for CAPTCHA) | recovery_planner → recovery_policy_check → recovery_executor →
observe_current_page …` or `fail_step` when not healable or the budget is spent.

## Grounding

Each element in the inventory is scored with explainable features:

* **identity** — strongest of name similarity (accessible name, label, placeholder, title, text,
  test-id words, nearby text, icon glyphs), purpose compatibility (shared purpose vocabulary for
  instructions and elements), option membership (dropdowns), visual hints; when the instruction
  names a *specific* control, purpose only corroborates (sibling controls share purposes);
* **role** compatibility with the action, **context** (requested form/dialog/section, form
  continuity, submit control of the form filled in previous steps, open modal scoping, header/nav
  look-alikes when a main-content match exists), **state** (visible, enabled, unobstructed), **type**.

Confidence = strength × separation from the best competitor (+ ordinal/position resolution, +
validated memory agreement). Thresholds (configurable): ≥0.95 execute, ≥0.85 execute with enhanced
verification, ≥0.65 refine (re-observe, LLM disambiguation among real candidates, visual
grounding), below → ask a human. High-risk actions need more regardless of confidence.

## Visual and spatial

DOM-first. Visual grounding is used for canvas/spatial surfaces, icon-only or unlabeled controls,
and refinement: a fresh screenshot of the current viewport, colour/shape segmentation inside the
surface's bounding box (or set-of-marks prompting when a vision model is configured), a safe point
inside the shape, then a coordinate action and normal verification. Coordinates are never stored.

## Runners

`memory_replay`, `semantic`, `visual`, `form`, `hybrid` (selected per test, explainable),
`data_driven` (one execution per data set via the API), `parallel` (suite runs in isolated contexts,
bounded by `max_concurrent_executions`), `debug` (screenshots for every phase).

## Test management

Runs keep their data set, so `retry` repeats a run exactly. Plain values are stored as is; values
with credential-like names are sealed with the platform key and opened only when the browser session
starts. `rerun-steps` re-runs selected steps of the same test version: by default the earlier steps
run first to rebuild the page state, and the run stops after the last selected step. Partial runs
follow the normal oracle, so a selection without an assertion is INCONCLUSIVE. Their verified steps
reinforce locator memory, but they never count as validated full runs or record test trajectories.
`compare` lines two executions up step by step (by id, else by text, else by position) and reports
regressions, fixes, changed targets, recovery and memory changes, slowdowns and metric deltas. It
backs the UI's "compare with previous" view.

## Understanding imperfect tests and imperfect pages

**Typos and other words.** Two layers, with a strict line between *finding* and *judging*:

* `planner/spelling.py` repairs the grammar of an instruction before parsing: the verb ("Clik", "Selct",
  "Verfy"), control nouns ("buton", "dropdwon"), predicates ("dispalyed") and field/page words
  ("usrname", "dashbaord"). Data is never rewritten: quoted text, variables, URLs, e-mail addresses,
  values ("as abc"), options ("Select **Indai** from") and expected texts stay byte-identical. Each
  repair is recorded on the step (`spelling: Clik->click`), and the report shows the author's words.
* `grounding/fuzzy.py::match_score` is the forgiving matcher used to find elements, options and
  remembered strategies. It repairs typos with a keyboard-aware Damerau-Levenshtein cost (a
  transposition, a neighbouring key or a dropped letter is cheap; an arbitrary substitution is not),
  maps synonym/abbreviation groups ("sign in" = "log in", "pwd" = "password", "qty" = "quantity"),
  joins split words ("user name" = "username"), keeps numbers exact, and never treats two real words
  as typos of each other ("Edit" ≠ "Exit", "Save" ≠ "Same", "Sign In" ≠ "Sign Up").
* Assertions keep the strict `text_match.similarity`. A typo in an expected value, or in the product,
  fails the check instead of turning into a silent pass.

**Dropdowns and browser-rendered controls** (`execution/dropdown.py`, `forms/option_matcher.py`,
`forms/value_normalizer.py`):

* Native `<select>`: the popup is drawn by the OS, so it is never clicked. Options are read live
  (labels, values, optgroups, disabled flags) and chosen with `select_option`. The engine waits while a
  dependent select is disabled or still loading. Multi-selects take several options. "Click the
  Country dropdown" is a verified no-op, and "Click India" becomes `select_option` on its owner select.
* `<datalist>`: suggestions are browser-rendered, so the matched value is typed.
* Pickers (date, time, datetime-local, month, week, color, range, number) accept one machine format.
  Values such as "12/24/2026", "2:30 PM", "red" or "50%" are normalized, filled without keystrokes,
  and verified against the normalized value.
* Custom dropdowns:
  - The trigger is opened, by typing into it when it is an editable combobox.
  - The options found are the ones that belong to *this* trigger: its `aria-controls` listbox,
    otherwise the nearest visible popup (portals included), otherwise ARIA-less "div soup" lists.
  - When the option is not rendered, the engine filters through the popup's own search box, then
    scrolls virtualized lists.
  - Multi-selects never click an already selected entry. Menu paths ("Export > CSV") walk submenus.
* Option choice order: exact, case, normalized, value code, alias ("USA"), unique containment,
  ordinal ("the second option"), then a unique typo-tolerant best match. A disabled option is
  reported, and nothing is ever a "closest guess".

**Messy markup** (`browser/js_probes.py`, `observation/element_inventory.py`):

* The DOM probe walks open shadow roots. Web-component internals get piercing CSS selectors, and
  page text includes shadow content.
* Unassociated labels are inferred the way a person reads them: preceding text, the table cell to
  the left or the column header, a wrapper's own text, `dt`/`dd`.
* Role-less `div`/`span` controls with click handlers or pointer cursors become candidates.
  Button-like ones rank as buttons, and dropdown-like ones as dropdown triggers ("Select department"
  names the Department field).
* Form flow resolves look-alikes: after filling a field, a same-purpose field in the same group (form,
  table form, dialog, component) is preferred over one elsewhere ("email" in the shipping table, not
  the newsletter box).

**Memory fast path** (`runtime._memory_fastpath`): for the same application page and step, a
validated strategy (from a fully passed run, or repeated verified successes) is replayed without
candidate scoring or a model call. It is used only when it re-grounds on the live page uniquely and
almost perfectly (≥ 0.85), the element is usable for the action, and it still means what the step asks.
Otherwise full grounding takes over: scoring, refinement and, when configured, the model. A UI change
is recorded as such. A remembered element that fails verification is penalised, superseded once a new
strategy is verified, and dropped for the step. Memory can confirm a live match but never overrule a
clearly better one. Memory keys include the words that distinguish steps with the same purpose
("email" vs "newsletter email").
