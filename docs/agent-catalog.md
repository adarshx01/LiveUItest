# Agent catalogue

The harness registers logical agents (`backend/harness/agent_registry.py`, `GET /api/v1/agents`).
Most are deterministic services; an LLM is consulted only where evidence is insufficient, never for
every step, and always through a closed, validated output schema.

| Agent | Kind | Graph nodes | Responsibility |
|---|---|---|---|
| supervisor_agent | deterministic | select_execution_strategy, prepare_next_action, next_test_action | lifecycle, runner selection, budgets, conditional branches |
| requirement_analyzer_agent | hybrid | normalize_test_case, analyze_requirements | NL → DSL; LLM only for low-confidence lines |
| scenario_planner_agent | deterministic | compile_test_plan | implicit navigation, form groups, page openers, risk hints, postconditions |
| plan_validator | deterministic | validate_test_plan | targets, values, variables, secrets, URLs, verifiable outcome |
| policy_guard_agent | deterministic | check_permissions, validate_action, recovery_policy_check | regulator (see security.md) |
| page_state_analyzer_agent | deterministic | observe_current_page | loading, error pages/banners, CAPTCHA, overlays, validation, injection |
| dom_intelligence_agent | hybrid | resolve_target_element, ground_target | inventory, classification, contextual grounding; LLM picks among real candidates when ambiguous |
| visual_grounding_agent | hybrid | ground_target | canvas colour/shape segmentation; set-of-marks vision prompting |
| logical_action_agent | deterministic | execute_action | logical step → concrete interaction for the actual element |
| locator_strategy_agent | deterministic | resolve_target_element, learn_pattern | ref-free strategies + selector validation |
| form_analyzer_agent | deterministic | execute_action | semantic field mapping, generated non-sensitive data |
| assertion_agent | hybrid | execute_action, verify_action_result | deterministic assertions; LLM may only quote verifiable page text |
| verification_agent | deterministic | verify_action_result | postconditions per action |
| failure_classifier_agent / root_cause_analyzer_agent | deterministic | classify_failure | taxonomy, product vs automation failure |
| healing_agent / recovery_agent | hybrid | recovery_planner, recovery_executor | bounded strategies, idempotency checks |
| memory_agent / learning_agent | deterministic | retrieve_*_memory, learn_pattern, persist_test_memory | validated retrieval, versioned learning |
| test_result_evaluator_agent | deterministic | evaluate_test_result | independent oracle |
| report_generator_agent | deterministic | generate_report | HTML/JSON evidence reports |

The remaining items of the original 130-agent inventory map onto these services (spec §26.2):
inventory construction, locator validation, tool dispatch, field mapping, checkbox/radio/dropdown
handling, session recovery, screenshot collection, assertion evaluation, timeouts, audit logging,
scheduling, memory indexing and invalidation, confidence calculation.
