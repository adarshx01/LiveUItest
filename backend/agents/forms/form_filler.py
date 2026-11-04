"""Form intelligence (spec section 11): semantic field mapping and filling.

Test data keys are mapped to fields by *purpose* (label / accessible name /
input type / autocomplete), never by position. Missing non-sensitive values can
be generated; credentials, payment data and other sensitive fields are never
invented — they must come from test data or the secret store.
"""

from __future__ import annotations

from typing import Any

from backend.agents.grounding.element_grounder import ElementGrounder
from backend.agents.grounding.fuzzy import match_score as similarity
from backend.agents.grounding.text_match import split_identifier
from backend.agents.planner.lexicon import SENSITIVE_PURPOSES, classify_phrase
from backend.agents.planner.variable_resolver import VariableResolver
from backend.dsl.observation import FormInfo, InventoryElement, Observation
from backend.dsl.results import FailureCategory
from backend.execution.action_executor import ActionOutcome, InteractionExecutor

GENERATORS = {"full_name": "generated_name", "first_name": "generated_name", "email": "generated_email",
              "username_or_email": "generated_email", "phone": "generated_phone"}


def map_data_to_fields(form: FormInfo | None, observation: Observation,
                       data: dict[str, str]) -> list[tuple[InventoryElement, str, str]]:
    """Return (field, data_key, reason) pairs using purpose and label similarity."""

    fields = [observation.element(i) for i in (form.field_ids if form else [])] if form else [
        e for e in observation.elements if e.role in ("textbox", "searchbox", "combobox", "spinbutton")]
    fields = [f for f in fields if f is not None and f.visible and f.enabled]
    used: set[str] = set()
    mapping: list[tuple[InventoryElement, str, str]] = []
    for key, _value in data.items():
        key_words = split_identifier(key)
        key_purpose = classify_phrase(key_words)
        best: tuple[float, InventoryElement | None, str] = (0.0, None, "")
        for field in fields:
            if field.element_id in used:
                continue
            score = 0.0
            reason = ""
            if key_purpose and field.semantic_purpose and key_purpose.purpose == field.semantic_purpose:
                score = 0.9 * key_purpose.score + 0.1
                reason = f"purpose {field.semantic_purpose}"
            texts = [t for t in (field.name, field.label or "", field.placeholder or "",
                                 split_identifier(field.name_attr), split_identifier(field.dom_id)) if t]
            label_sim = max((similarity(key_words, t) for t in texts), default=0.0)
            if label_sim > score:
                score, reason = label_sim, f"label similarity {label_sim:.2f}"
            if score > best[0]:
                best = (score, field, reason)
        if best[1] is not None and best[0] >= 0.6:
            used.add(best[1].element_id)
            mapping.append((best[1], key, best[2]))
    return mapping


class FormFiller:
    name = "form_filler_agent"

    def __init__(self, executor: InteractionExecutor, resolver: VariableResolver, grounder: ElementGrounder) -> None:
        self.executor = executor
        self.resolver = resolver
        self.grounder = grounder

    async def fill(self, observation: Observation, variables: dict[str, str]) -> tuple[ActionOutcome, dict[str, Any]]:
        form = max(observation.forms, key=lambda f: len(f.field_ids), default=None)
        data = dict(variables)
        mapping = map_data_to_fields(form, observation, data)
        mapped_ids = {f.element_id for f, _, _ in mapping}
        # generate values for required/obvious personal-data fields that the test data does not cover
        candidates = [observation.element(i) for i in (form.field_ids if form else [])]
        for field in candidates:
            if field is None or field.element_id in mapped_ids or field.role not in ("textbox",):
                continue
            purpose = field.semantic_purpose
            if purpose in SENSITIVE_PURPOSES or purpose not in GENERATORS:
                continue
            key = f"__generated_{purpose}"
            data[key] = self.resolver.generated(GENERATORS[purpose])
            mapping.append((field, key, f"generated value for {purpose}"))
        if not mapping:
            return ActionOutcome(False, error="no form fields could be mapped to the provided test data",
                                 failure_category=FailureCategory.ELEMENT_NOT_FOUND), {}
        filled: dict[str, Any] = {}
        for field, key, reason in mapping:
            value = data[key]
            if field.semantic_purpose in SENSITIVE_PURPOSES:
                filled[field.display()] = "skipped: sensitive field requires a secret reference"
                continue
            if field.role == "combobox" and (field.options or field.tag == "select"):
                outcome = await self.executor.select(field, value, observation)
            else:
                outcome = await self.executor.fill(field, value, sensitive=False)
            filled[field.display()] = {"from": key if not key.startswith("__") else "generated", "value": value,
                                       "reason": reason, "ok": outcome.ok}
            if not outcome.ok:
                return ActionOutcome(False, error=f"could not fill {field.display()}: {outcome.error}",
                                     failure_category=outcome.failure_category), filled
        return ActionOutcome(True, tool="browser_type", notes=[f"filled {len(filled)} fields by semantic mapping"],
                             expected={"form_fields": len(filled)}), filled
