export type Outcome = "PASSED" | "FAILED" | "BLOCKED" | "INCONCLUSIVE";

export interface Project { id: string; name: string; description: string }

export interface TestCaseSummary {
  id: string; name: string; project_id: string; application_id: string; tags: string[];
  current_version: number; updated_at: string;
}

export interface ParsedStep { id: string; index: number; text: string; action: string; summary: string; confidence: number }

export interface TestCaseDetail extends TestCaseSummary {
  version: number; source_text: string; source_format: string; issues: { line: string; reason: string; severity: string }[];
  steps: ParsedStep[];
}

export interface Execution {
  id: string; test_case_id: string; test_case_version: number; application_id: string; status: string;
  outcome: Outcome | null; reason: string; runner: string | null; metrics: Record<string, number>;
  created_at: string; started_at: string | null; finished_at: string | null; report: string | null;
  pending?: Record<string, unknown>; approvals?: Approval[];
  retry_of?: string | null; scope?: { step_ids: string[]; with_prerequisites: boolean; rerun_of: string | null } | null;
}

export interface ComparedStep {
  status: string; message: string; duration_ms: number; attempts: number; recovered: boolean; memory_used: boolean;
  failure_category: string | null; target: { display: string | null; method: string | null; confidence: number | null } | null;
}

export interface Comparison {
  base: Partial<Execution>; other: Partial<Execution>; same_test_case: boolean; same_version: boolean; summary: string[];
  metric_deltas: Record<string, { base: number | null; other: number | null }>;
  counts: { regressions: number; fixes: number; target_changes: number; changed_steps: number };
  steps: { index: number; text: string; action: string; base: ComparedStep | null; other: ComparedStep | null; changes: string[] }[];
}

export interface Approval { id: string; kind: string; status: string; step_id: string | null; request: Record<string, unknown> }

export interface Candidate { display: string; score: number; features?: Record<string, number>; evidence?: string[] }

export interface Grounding {
  display: string; confidence: number; decision: string; method: string; evidence: string[];
  candidates: Candidate[]; locator_strategy: Record<string, unknown>; memory_id: string | null;
  bbox: { x: number; y: number; width: number; height: number } | null;
}

export interface ActionRecord {
  attempt: number; action: string; tool: string | null; ok: boolean; error: string | null; failure_category: string | null;
  grounding: Grounding | null; playwright_code: string | null; screenshot_before: string | null;
  screenshot_after: string | null; verification: { status: string; message: string; checks: { check: string; ok: boolean }[] };
}

export interface AssertionResult { assertion_id: string; type: string; description: string; expected: unknown; actual: unknown; status: string; message: string }

export interface Recovery { attempt: number; failure_category: string; strategy: string; rationale: string; outcome: string; details: Record<string, unknown> }

export interface StepResult {
  step_id: string; index: number; original_text: string; action: string; summary: string; status: string; message: string;
  attempts: number; actions: ActionRecord[]; assertions: AssertionResult[]; recovery: Recovery[]; recovered: boolean;
  memory_used: boolean; duration_ms: number; failure_category: string | null;
}

export interface ExecutionEvent { event: string; seq: number; ts: number; step_id: string | null; message: string; payload: Record<string, unknown> }

export interface ObservationElement {
  element_id: string; role: string; name: string; label: string | null; purpose: string | null; visible: boolean; enabled: boolean;
  checked: boolean | null; bbox: { x: number; y: number; width: number; height: number } | null; form: string | null;
  dialog: string | null; source: string; test_id: string | null;
}

export interface Observation {
  summary: Record<string, unknown>; page_state: Record<string, unknown>; forms: Record<string, unknown>[];
  dialogs: Record<string, unknown>[]; headings: string[]; elements: ObservationElement[]; snapshot: string;
}

export interface LocatorMemory {
  memory_id: string; route_pattern: string; action_signature: string; intent: string; strategy: Record<string, unknown>;
  successes: number; failures: number; validated_runs: number; confidence: number; status: string; version: number;
  last_verified_at: string;
}
