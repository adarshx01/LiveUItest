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
