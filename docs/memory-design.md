# Memory design

Memory is **validated execution knowledge**, not truth.

## Stores (PostgreSQL, tenant-scoped)

| Table | Content |
|---|---|
| `application_memory` | facts (e.g. authentication route verified by a passed run) |
| `page_memory`, `page_fingerprints` | route pattern, structural hash and signature, page kind, form fingerprints, pgvector embedding |
| `locator_memory` | action signature (`fill:username_or_email`), route, page fingerprint, semantic `LocatorStrategy` (role, name, label, placeholder, test id, ids, scoped CSS, form/dialog/section context), success/failure counts, validated runs, confidence, status, version, embedding |
| `test_execution_memory` | the last successful trajectory per test case version |
| `failure_memory` | failure category and signature, failed strategies, successful alternative, root cause |
| `interaction_patterns` | reusable component behaviours |
| `memory_versions` | history: every creation, failure and supersession |

Never stored: snapshot refs, coordinates, raw field values, secrets.

## Lookup

Key = organization + project + application (+ environment) + normalized route + action signature
(+ test case + page fingerprint). Retrieval order: exact test case → exact route → pgvector similarity
(same app, same action, cosine ≥ 0.78). Similarity only *proposes* a memory.

## Validation before replay

1. Replay requires a fully passed run (`validated_run_count ≥ 1`) or repeated verified successes.
2. The strategy is re-grounded against the live observation (`resolve_strategy`): quality ≥ 0.7 and
   unambiguous, and the element must still semantically match the instruction (identity ≥ 0.5).
3. Only then does it raise the grounding confidence; the action is still verified afterwards.

## Change detection and versioning

If validated memory exists but does not re-ground, the UI changed: the step records a
`PAGE_STRUCTURE_CHANGED` recovery (`reground_without_memory`), the stale strategy's failure count
increases, the target is grounded semantically on the live page, and — once the action is verified —
the new strategy becomes version N+1 while the old one is marked `superseded` (history kept). A
strategy can also be invalidated manually (`POST /api/v1/memory/{id}/invalidate`).

Confidence = (successes + 1) / (successes + failures + 2), gently decayed with age.
