"""Requirement analyzer agent (spec section 7.2).

Deterministic parsing first; the LLM (when configured) is consulted only for
instructions the parser could not understand confidently. The LLM may only
propose values from a closed schema which are then rebuilt into DSL steps by
the same deterministic helpers, so it can never inject executable content.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from backend.agents.planner.lexicon import SENSITIVE_PURPOSES, classify_phrase
from backend.agents.planner.nl_parser import (
    ParseContext,
    StepParser,
    build_target,
    parse_assertion,
    parse_condition,
    parse_value,
)
from backend.agents.planner.test_ingestion import RawLine, RawTestCase
from backend.core.exceptions import ModelError, ModelUnavailable
from backend.core.logging import get_logger
from backend.dsl.models import (
    ActionType,
    Ambiguity,
    AssertionSpec,
    AssertionType,
    Step,
    TargetSpec,
    TestSpec,
    ValueSpec,
)
from backend.harness.model_gateway import ModelGateway, ModelTier

log = get_logger(__name__)

LLM_ACTIONS = Literal["navigate", "open_page", "click", "double_click", "right_click", "hover", "fill", "type", "clear",
                      "select", "check", "uncheck", "toggle", "press_key", "scroll", "drag", "upload", "download", "wait",
                      "switch_tab", "accept_dialog", "dismiss_dialog", "assert", "capture_screenshot", "go_back"]
LLM_ASSERTIONS = Literal["URL_CONTAINS", "URL_EQUALS", "PAGE_TITLE_CONTAINS", "TEXT_VISIBLE", "TEXT_NOT_VISIBLE",
                         "ELEMENT_VISIBLE", "ELEMENT_NOT_VISIBLE", "ELEMENT_ENABLED", "ELEMENT_DISABLED",
                         "CHECKBOX_CHECKED", "DROPDOWN_SELECTED", "INPUT_VALUE_EQUALS", "ELEMENT_COUNT",
                         "DIALOG_VISIBLE", "DOWNLOAD_STARTED"]


class LLMStepProposal(BaseModel):
    action: LLM_ACTIONS
    target_description: str | None = Field(default=None, description="Semantic description of the element")
    target_exact_text: str | None = Field(default=None, description="Exact visible label when the instruction quotes it")
    element_kind: str | None = Field(default=None, description="button, link, field, dropdown, checkbox, radio, tab...")
    value: str | None = Field(default=None, description="Literal value or {{variable}} reference to enter/select")
    destination_description: str | None = None
    key: str | None = None
    url: str | None = None
    assertion_type: LLM_ASSERTIONS | None = None
    assertion_expected: str | None = None
    wait_seconds: float | None = None


class LLMParseOutput(BaseModel):
    steps: list[LLMStepProposal]
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity: str | None = Field(default=None, description="What is unclear, if anything")
    missing_information: list[str] = Field(default_factory=list)


ANALYZER_SYSTEM = (
    "You convert one natural-language end-to-end test instruction into structured browser test steps. "
    "Use only the allowed action and assertion vocabularies. Describe elements semantically (what they are for and "
    "their visible label); never output CSS/XPath selectors, coordinates or code. Never invent credentials, payment "
    "data or other values that the instruction does not provide: report them in missing_information instead. "
    "Assertions must keep the expected value exactly as the test states it."
)


class RequirementAnalyzer:
    name = "requirement_analyzer_agent"

    def __init__(self, gateway: ModelGateway | None = None, *, llm_threshold: float = 0.7) -> None:
        self.parser = StepParser()
        self.gateway = gateway
        self.llm_threshold = llm_threshold
        self.llm_calls = 0

    async def analyze(self, raw: RawTestCase, *, secret_names: set[str] | None = None) -> TestSpec:
        ctx = ParseContext(base_url=raw.base_url, variables=dict(raw.variables), secret_names=set(secret_names or ()))
        ambiguities: list[Ambiguity] = []
        missing: list[str] = []
        if raw.structured_steps is not None:
            steps = self._structured(raw, ambiguities)
            source_format = "structured"
        else:
            steps = []
            for line in raw.lines:
                steps.extend(await self._analyze_line(line, ctx, ambiguities, missing))
            for expected in raw.expected_results:
                assertion, confidence = parse_assertion(_strip_verify(expected), ctx)
                if assertion is not None:
                    ctx.step_counter += 1
                    steps.append(Step(id=f"step_{ctx.step_counter:03d}", index=ctx.step_counter,
                                      original_text=f"Expected: {expected}", action=ActionType.ASSERT,
                                      assertion=assertion, confidence=confidence, risk_hint="read"))
            source_format = raw.source_format
        variables_used = _collect_references(steps)
        for ref in sorted(variables_used):
            if ref not in raw.variables and ref not in ctx.secret_names and not ref.startswith(("generated_", "random_")) \
                    and ref not in ("current_date", "timestamp", "today", "uuid", "run_id") \
                    and not any(s.capture_as == ref for s in _flatten(steps)):
                missing.append(f"value for {{{{{ref}}}}}")
        if not raw.base_url and not any(s.action == ActionType.NAVIGATE for s in _flatten(steps)):
            missing.append("application URL")
        return TestSpec(
            test_name=raw.name,
            base_url=raw.base_url,
            description=raw.description,
            preconditions=raw.preconditions,
            variables=raw.variables,
            secret_refs=sorted({s.value.reference for s in _flatten(steps)
                                if s.value and s.value.source == "secret" and s.value.reference}),
            steps=steps,
            expected_results=raw.expected_results,
            tags=raw.tags,
            source_format=source_format,  # type: ignore[arg-type]
            ambiguities=ambiguities,
            missing_information=sorted(set(missing)),
            analyzer={"engine": "deterministic+llm" if self.llm_calls else "deterministic",
                      "llm_calls": self.llm_calls},
        )

    # ------------------------------------------------------------------ lines
    async def _analyze_line(self, line: RawLine, ctx: ParseContext, ambiguities: list[Ambiguity],
                            missing: list[str]) -> list[Step]:
        if line.condition is not None:
            condition, cond_conf = parse_condition(line.condition, ctx)
            then_steps: list[Step] = []
            else_steps: list[Step] = []
            for child in line.then_lines:
                then_steps.extend(await self._analyze_line(child, ctx, ambiguities, missing))
            for child in line.else_lines:
                else_steps.extend(await self._analyze_line(child, ctx, ambiguities, missing))
            ctx.step_counter += 1
            return [Step(id=f"step_{ctx.step_counter:03d}", index=ctx.step_counter, original_text=line.text,
                         action=ActionType.CONDITIONAL, condition=condition, then_steps=then_steps,
                         else_steps=else_steps, confidence=cond_conf)]
        parsed = self.parser.parse(line.text, ctx)
        steps, confidence = parsed.steps, parsed.confidence
        if (not steps or confidence < self.llm_threshold) and self.gateway is not None and self.gateway.available:
            llm_steps, llm_conf, ambiguity, llm_missing = await self._llm_parse(line.text, ctx)
            if llm_steps and llm_conf > confidence:
                steps, confidence = llm_steps, llm_conf
            if ambiguity:
                ambiguities.append(Ambiguity(line=line.text, reason=ambiguity, severity="warning"))
            missing.extend(llm_missing)
        if not steps:
            ambiguities.append(Ambiguity(line=line.text, reason=f"instruction not understood ({parsed.reason})",
                                         severity="blocking"))
            return []
        if confidence < 0.6:
            ambiguities.append(Ambiguity(line=line.text, reason=f"low interpretation confidence {confidence:.2f}",
                                         step_index=steps[0].index, severity="warning"))
        return steps

    async def _llm_parse(self, text: str, ctx: ParseContext) -> tuple[list[Step], float, str | None, list[str]]:
        assert self.gateway is not None
        prompt = (f"Application base URL: {ctx.base_url or 'unknown'}\n"
                  f"Known test variables: {', '.join(sorted(ctx.variables)) or 'none'}\n"
                  f"Instruction: {text}")
        try:
            result = await self.gateway.structured(tier=ModelTier.CLASSIFICATION, system=ANALYZER_SYSTEM,
                                                   prompt=prompt, output_model=LLMParseOutput, agent=self.name,
                                                   max_tokens=2000)
        except (ModelError, ModelUnavailable) as exc:
            log.warning("LLM step analysis unavailable", error=str(exc)[:200])
            return [], 0.0, None, []
        self.llm_calls += 1
        out = result.output
        steps = [s for s in (self._from_proposal(p, text, ctx) for p in out.steps) if s is not None]
        return steps, out.confidence * 0.95, out.ambiguity, out.missing_information

    def _from_proposal(self, p: LLMStepProposal, text: str, ctx: ParseContext) -> Step | None:
        action = ActionType(p.action)
        ctx.step_counter += 1
        step = Step(id=f"step_{ctx.step_counter:03d}", index=ctx.step_counter, original_text=text, action=action,
                    parse_source="llm", confidence=0.85)
        phrase = p.target_exact_text and f'"{p.target_exact_text}" {p.element_kind or ""}' or \
            f"{p.target_description or ''} {p.element_kind or ''}"
        if action in (ActionType.NAVIGATE,):
            if not p.url:
                return None
            step.target = TargetSpec(intent="url", url=p.url)
        elif action == ActionType.ASSERT:
            if not p.assertion_type:
                return None
            target = build_target(phrase, "hover") if phrase.strip() else None
            step.assertion = AssertionSpec(type=AssertionType(p.assertion_type), expected=p.assertion_expected,
                                           target=target, description=text, match="case_insensitive")
        elif action == ActionType.PRESS_KEY:
            step.key = p.key or "Enter"
        elif action == ActionType.WAIT:
            from backend.dsl.models import WaitSpec

            step.wait = WaitSpec(seconds=p.wait_seconds or 1.0)
        elif phrase.strip():
            step.target = build_target(phrase.strip(), action.value)
        if p.value is not None and action in (ActionType.FILL, ActionType.TYPE, ActionType.SELECT, ActionType.UPLOAD):
            step.value = parse_value(p.value, ctx, purpose=step.target.purpose if step.target else None)
            if action == ActionType.SELECT and step.target:
                step.target.option_text = step.value.value
            if step.target and step.target.purpose in SENSITIVE_PURPOSES:
                step.value.sensitive = True
                step.sensitive = True
        if action == ActionType.DRAG and p.destination_description:
            step.destination = build_target(p.destination_description, "drop")
        return step

    # ------------------------------------------------------------- structured
    def _structured(self, raw: RawTestCase, ambiguities: list[Ambiguity]) -> list[Step]:
        steps: list[Step] = []
        for idx, data in enumerate(raw.structured_steps or [], start=1):
            payload = {"id": f"step_{idx:03d}", "index": idx, "original_text": data.get("description")
                       or data.get("original_text") or f"{data.get('action')} step", **data}
            payload.pop("description", None)
            try:
                step = Step.model_validate(payload)
                step.parse_source = "structured"
                if step.target and step.target.purpose is None and (step.target.intent or step.target.exact_name):
                    match = classify_phrase(step.target.exact_name or step.target.intent)
                    if match and match.score >= 0.7:
                        step.target.purpose = match.purpose
                steps.append(step)
            except Exception as exc:  # pydantic ValidationError
                ambiguities.append(Ambiguity(line=str(data)[:200], reason=f"invalid structured step: {exc}"[:300],
                                             severity="blocking"))
        return steps


def _strip_verify(text: str) -> str:
    return re.sub(r"^(?:verify|validate|ensure|confirm|check)\s+(?:that\s+)?", "", text.strip(), flags=re.IGNORECASE)


def _flatten(steps: list[Step]) -> list[Step]:
    out: list[Step] = []
    for step in steps:
        out.extend(step.iter_flat())
    return out


def _collect_references(steps: list[Step]) -> set[str]:
    refs: set[str] = set()
    pattern = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*\}\}")

    def scan_value(value: ValueSpec | None) -> None:
        if value is None:
            return
        if value.source in ("variable", "secret") and value.reference:
            refs.add(value.reference)
        if value.value:
            refs.update(pattern.findall(value.value))

    for step in _flatten(steps):
        scan_value(step.value)
        if step.target and step.target.url:
            refs.update(pattern.findall(step.target.url))
        if step.assertion:
            scan_value(step.assertion.expected_ref)
            if isinstance(step.assertion.expected, str):
                refs.update(pattern.findall(step.assertion.expected))
        if step.condition and step.condition.kind == "variable_equals" and step.condition.variable:
            refs.add(step.condition.variable)
    return refs
