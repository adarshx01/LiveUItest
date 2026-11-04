"""Deterministic assertion engine with evidence (spec section 18).

Expected and actual values are kept separate and the expected value is never
rewritten. An LLM may help interpret a *semantic* expectation, but only by
quoting text that is verifiably present on the page; it can never declare a
pass on its own.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from backend.agents.grounding.element_grounder import ElementGrounder, GroundingContext
from backend.agents.grounding.text_match import contains_text, similarity
from backend.agents.observation.page_observer import PageObserver
from backend.agents.planner.variable_resolver import VariableResolver
from backend.browser.js_probes import ELEMENT_STATE, VISIBLE_TEXT
from backend.browser.tool_adapter import BrowserToolAdapter
from backend.core.exceptions import ModelError, ModelUnavailable
from backend.core.ids import new_id
from backend.dsl.models import ActionType, AssertionSpec, AssertionType, Step, TargetSpec
from backend.dsl.observation import InventoryElement, Observation
from backend.dsl.results import AssertionResult, Evidence
from backend.harness.model_gateway import ModelGateway, ModelTier

RETRYABLE = {AssertionType.TEXT_VISIBLE, AssertionType.TEXT_NOT_VISIBLE, AssertionType.ELEMENT_VISIBLE,
             AssertionType.ELEMENT_NOT_VISIBLE, AssertionType.URL_CONTAINS, AssertionType.URL_EQUALS,
             AssertionType.PAGE_TITLE_CONTAINS, AssertionType.PAGE_TITLE_EQUALS, AssertionType.ANY_OF,
             AssertionType.ALL_OF, AssertionType.DIALOG_VISIBLE, AssertionType.ELEMENT_COUNT,
             AssertionType.DROPDOWN_SELECTED, AssertionType.CHECKBOX_CHECKED, AssertionType.INPUT_VALUE_EQUALS,
             AssertionType.NAVIGATION_OCCURRED, AssertionType.DOWNLOAD_STARTED, AssertionType.ELEMENT_ENABLED,
             AssertionType.ELEMENT_DISABLED, AssertionType.PAGE_LOADED, AssertionType.CHECKBOX_UNCHECKED}


class SemanticJudgement(BaseModel):
    satisfied: bool
    evidence_quote: str = Field(description="Exact text copied from the page that proves the expectation, or ''")
    confidence: float = Field(ge=0, le=1)


class AssertionContext:
    def __init__(self, *, url_at_step_start: str | None = None, downloads: list[dict[str, str]] | None = None,
                 network_log: str | None = None) -> None:
        self.url_at_step_start = url_at_step_start
        self.downloads = downloads or []
        self.network_log = network_log


class AssertionEngine:
    name = "assertion_agent"

    def __init__(self, adapter: BrowserToolAdapter, observer: PageObserver, grounder: ElementGrounder,
                 resolver: VariableResolver, gateway: ModelGateway | None = None) -> None:
        self.adapter = adapter
        self.observer = observer
        self.grounder = grounder
        self.resolver = resolver
        self.gateway = gateway
        self.llm_calls = 0

    async def evaluate(self, spec: AssertionSpec, step: Step, ctx: AssertionContext,
                       first_observation: Observation | None = None) -> tuple[AssertionResult, Observation | None]:
        deadline = time.monotonic() + spec.timeout_ms / 1000
        observation = first_observation
        result: AssertionResult | None = None
        while True:
            if observation is None:
                observation = await self.observer.observe(with_tabs=False)
            result = await self._once(spec, step, observation, ctx)
            if result.status == "PASSED" or spec.type not in RETRYABLE or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.5)
            observation = None
        if result.status == "FAILED" and spec.match == "semantic" and self.gateway is not None and self.gateway.available:
            result = await self._semantic_check(spec, result, observation)
        if result.status == "FAILED" and spec.derived:
            result.status = "INCONCLUSIVE"
            result.message = ("Expected value was inferred from earlier test data, not stated explicitly; "
                              + result.message + " Provide the expected value in test data to make this strict.")
        return result, observation

    # ------------------------------------------------------------------ single evaluation
    async def _expected(self, spec: AssertionSpec) -> Any:
        if spec.expected_ref is not None:
            resolved = await self.resolver.resolve(spec.expected_ref)
            return resolved.text
        if isinstance(spec.expected, str) and "{{" in spec.expected:
            return await self.resolver.interpolate(spec.expected)
        return spec.expected

    async def _once(self, spec: AssertionSpec, step: Step, obs: Observation,
                    ctx: AssertionContext) -> AssertionResult:
        expected = await self._expected(spec)
        aid = new_id("assert")
        base = {"assertion_id": aid, "type": spec.type.value, "description": spec.description or step.original_text,
                "expected": expected, "derived_expectation": spec.derived}
        ev = [Evidence(source="accessibility_snapshot", observation_id=obs.observation_id,
                       detail={"url": obs.url, "title": obs.title})]
        t = spec.type

        def done(ok: bool, actual: Any, message: str = "", extra: list[Evidence] | None = None) -> AssertionResult:
            passed = ok != spec.negate
            return AssertionResult(**base, actual=actual, status="PASSED" if passed else "FAILED",
                                   message=message, evidence=ev + (extra or []))

        if t in (AssertionType.ANY_OF, AssertionType.ALL_OF):
            children = [await self._once(child, step, obs, ctx) for child in spec.children]
            passed = [c for c in children if c.status == "PASSED"]
            ok = bool(passed) if t == AssertionType.ANY_OF else len(passed) == len(children)
            actual = {c.type: {"status": c.status, "actual": c.actual} for c in children}
            message = "; ".join(f"{c.description or c.type}: {c.status}" for c in children)
            return done(ok, actual, message, [e for c in children for e in c.evidence[1:]][:6])
        if t == AssertionType.URL_EQUALS:
            ok = _url_equals(obs.url, str(expected))
            return done(ok, obs.url, "" if ok else f"URL is {obs.url}")
        if t == AssertionType.URL_CONTAINS:
            ok = str(expected).lower() in obs.url.lower()
            return done(ok, obs.url)
        if t == AssertionType.PAGE_TITLE_EQUALS:
            return done(obs.title.strip() == str(expected).strip(), obs.title)
        if t == AssertionType.PAGE_TITLE_CONTAINS:
            return done(str(expected).lower() in obs.title.lower(), obs.title)
        if t == AssertionType.PAGE_LOADED:
            ok = not obs.page_state.loading and not obs.page_state.error_page and bool(obs.elements)
            return done(ok, {"loading": obs.page_state.loading, "error_page": obs.page_state.error_page,
                             "http_status": obs.page_state.http_status})
        if t == AssertionType.NAVIGATION_OCCURRED:
            ok = bool(ctx.url_at_step_start) and obs.url != ctx.url_at_step_start
            return done(ok, obs.url, f"started at {ctx.url_at_step_start}")
        if t in (AssertionType.DOWNLOAD_STARTED, AssertionType.FILE_EXISTS):
            names = [d.get("name") for d in ctx.downloads]
            ok = bool(names) if not expected else any(str(expected).lower() in (n or "").lower() for n in names)
            return done(ok, names, extra=[Evidence(source="mcp_response", detail={"downloads": ctx.downloads})])
        if t == AssertionType.DIALOG_VISIBLE:
            texts = []
            if obs.modal_state:
                texts.append(obs.modal_state.message)
            texts.extend(d.name for d in obs.dialogs)
            ok = bool(texts) and (not expected or any(contains_text(x, str(expected)) for x in texts))
            return done(ok, texts)
        if t in (AssertionType.TEXT_VISIBLE, AssertionType.TEXT_NOT_VISIBLE):
            return await self._text(spec, base, obs, ev, str(expected or ""), negative=t == AssertionType.TEXT_NOT_VISIBLE)
        if t in (AssertionType.ELEMENT_VISIBLE, AssertionType.ELEMENT_NOT_VISIBLE):
            el, conf = self._find(spec.target, obs, step, "hover")
            visible = el is not None and conf >= 0.62 and el.visible
            want_visible = t == AssertionType.ELEMENT_VISIBLE
            ok = visible if want_visible else not visible
            actual = el.display() if el is not None and visible else None
            msg = f"matched {el.display()} (confidence {conf:.2f})" if el is not None else "no matching element"
            return done(ok, actual, msg, [Evidence(source="grounding", observation_id=obs.observation_id,
                                                   detail={"confidence": round(conf, 3)})])
        if t in (AssertionType.ELEMENT_ENABLED, AssertionType.ELEMENT_DISABLED):
            el, conf = self._find(spec.target, obs, step, "click")
            if el is None or conf < 0.62:
                return AssertionResult(**base, actual=None, status="FAILED", message="element not found", evidence=ev)
            want = t == AssertionType.ELEMENT_ENABLED
            return done(el.enabled == want, "enabled" if el.enabled else "disabled", el.display())
        if t in (AssertionType.CHECKBOX_CHECKED, AssertionType.CHECKBOX_UNCHECKED, AssertionType.RADIO_SELECTED):
            el, conf = self._find(spec.target, obs, step, "check")
            if el is None or conf < 0.62:
                return AssertionResult(**base, actual=None, status="FAILED", message="control not found", evidence=ev)
            want = t != AssertionType.CHECKBOX_UNCHECKED
            return done(bool(el.checked) == want, "checked" if el.checked else "unchecked", el.display())
        if t == AssertionType.DROPDOWN_SELECTED:
            el, conf = self._find(spec.target, obs, step, "select")
            if el is None:
                return AssertionResult(**base, actual=None, status="FAILED", message="dropdown not found", evidence=ev)
            actual = el.selected_options or ([el.value_preview] if el.value_preview else []) or [el.name or el.text]
            if el.role == "combobox" and not el.options:
                actual = [el.text or el.name]
            ok = any(similarity(str(expected), a or "") >= 0.95 or (a or "").strip().lower() == str(expected).lower()
                     for a in actual)
            return done(ok, actual, el.display())
        if t == AssertionType.INPUT_VALUE_EQUALS:
            el, conf = self._find(spec.target, obs, step, "fill")
            if el is None or not el.ref:
                return AssertionResult(**base, actual=None, status="FAILED", message="field not found", evidence=ev)
            state = await self.adapter.probe(ELEMENT_STATE, el.ref, el.display()) or {}
            actual = state.get("value")
            if state.get("sensitive"):
                return AssertionResult(**base, actual="••••••", status="INCONCLUSIVE",
                                       message="password fields are not read back", evidence=ev)
            ok = (actual or "") == str(expected) if spec.match == "exact" else contains_text(actual or "", str(expected))
            return done(ok, actual, el.display(), [Evidence(source="dom_probe", detail={"element": el.display()})])
        if t == AssertionType.ELEMENT_COUNT:
            count, what = _count(spec.target, obs)
            ok = count == int(expected)
            return done(ok, count, f"counted {count} {what}")
        if t == AssertionType.NETWORK_RESPONSE:
            log = (await self.adapter.network_requests()).text
            ok = str(expected).lower() in log.lower()
            return done(ok, log[:300])
        if t in (AssertionType.APPLICATION_STATE, AssertionType.VISUAL_STATE):
            return AssertionResult(**base, actual=None, status="INCONCLUSIVE",
                                   message=f"{t.value} needs an explicit observable criterion", evidence=ev)
        return AssertionResult(**base, actual=None, status="ERROR", message=f"unsupported assertion {t}", evidence=ev)

    async def _text(self, spec: AssertionSpec, base: dict[str, Any], obs: Observation, ev: list[Evidence],
                    expected: str, *, negative: bool) -> AssertionResult:
        scope_text = None
        scope_display = None
        if spec.target is not None:
            el, conf = self._find(spec.target, obs, None, "hover")
            # a heading names a section, it does not contain it: headings scope to the page
            if el is not None and conf >= 0.55 and el.role != "heading":
                scope_text = " ".join(filter(None, [el.name, el.text, el.value_preview]))
                scope_display = el.display()
        if scope_text is None:
            probe = await self.adapter.probe(VISIBLE_TEXT) or {}
            page_text = probe.get("text") or ""
            # iframe content is not part of the main document's innerText; add snapshot text
            frame_text = " ".join(f"{e.name} {e.text}" for e in obs.elements if e.context.frame)
            scope_text = f"{page_text}\n{frame_text}\n" + " ".join(obs.visible_text)
        mode = spec.match if spec.match != "semantic" else "semantic"
        found = contains_text(scope_text, expected, mode=mode) if expected else False
        snippet = _snippet(scope_text, expected) if found else None
        ok = (not found) if negative else found
        actual = snippet if found else (f"'{expected}' not present" if not negative else None)
        detail = {"scope": scope_display or "page", "match": mode, "snippet": snippet}
        return AssertionResult(**base, actual=actual, status="PASSED" if ok else "FAILED",
                               message="" if ok else (f"'{expected}' is visible" if negative else
                                                      f"'{expected}' was not found ({mode} match)"),
                               evidence=ev + [Evidence(source="visible_text", observation_id=obs.observation_id,
                                                       detail=detail)])

    def _find(self, target: TargetSpec | None, obs: Observation, step: Step | None,
              action: str) -> tuple[InventoryElement | None, float]:
        if target is None:
            return None, 0.0
        probe_step = Step(id="assert_probe", index=0, original_text="assertion target",
                          action=ActionType.ASSERT, target=target)
        decision = self.grounder.ground(probe_step, obs, GroundingContext(), action_override=action)
        if decision.element_id is None:
            return None, 0.0
        el = obs.element(decision.element_id)
        identity = decision.candidates[0].features.get("identity", 0) if decision.candidates else 0
        return el, max(decision.confidence, identity * 0.95)

    async def _semantic_check(self, spec: AssertionSpec, result: AssertionResult,
                              obs: Observation | None) -> AssertionResult:
        probe = await self.adapter.probe(VISIBLE_TEXT) or {}
        text = (probe.get("text") or "")[:12000]
        try:
            judgement = await self.gateway.structured(  # type: ignore[union-attr]
                tier=ModelTier.CLASSIFICATION, agent="test_oracle_agent", output_model=SemanticJudgement,
                system="You check whether a web page satisfies a test expectation. Quote page text exactly.",
                prompt=f"Expectation: {spec.description or spec.expected}", untrusted=text, max_tokens=1000)
        except (ModelError, ModelUnavailable):
            return result
        self.llm_calls += 1
        out = judgement.output
        quote = out.evidence_quote.strip()
        if out.satisfied and quote and len(quote) >= 4 and quote in text and out.confidence >= 0.8:
            result.status = "PASSED"
            result.actual = quote
            result.message = "semantic expectation satisfied; quoted evidence verified on the page"
            result.evidence.append(Evidence(source="llm_semantic_judgement", detail={"quote": quote,
                                                                                     "confidence": out.confidence}))
        else:
            result.evidence.append(Evidence(source="llm_semantic_judgement",
                                            detail={"satisfied": out.satisfied, "quote_verified": quote in text}))
        return result


def _url_equals(actual: str, expected: str) -> bool:
    if expected.startswith("/"):
        from urllib.parse import urlparse

        path = urlparse(actual).path or "/"
        return path.rstrip("/") == expected.rstrip("/")
    return actual.rstrip("/") == expected.rstrip("/")


def _snippet(text: str, needle: str, width: int = 60) -> str:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        tokens = [t for t in re.split(r"\W+", needle) if len(t) > 3]
        idx = min((text.lower().find(t.lower()) for t in tokens if text.lower().find(t.lower()) >= 0), default=0)
    start = max(0, idx - width // 2)
    return re.sub(r"\s+", " ", text[start:start + len(needle) + width]).strip()


def _count(target: TargetSpec | None, obs: Observation) -> tuple[int, str]:
    rows = [e for e in obs.elements if e.role == "row" and e.visible and "columnheader" not in (e.text or "")
            and e.context.landmark in (None, "main", "region")]
    header_rows = [r for r in rows if r.snapshot_index is not None and any(
        c.role == "columnheader" and c.context.row_text == r.text for c in obs.elements)]
    data_rows = [r for r in rows if r not in header_rows]
    items = [e for e in obs.elements if e.role == "listitem" and e.visible and e.context.landmark == "main"]
    if target and target.intent and re.search(r"item|entr|notification|card", target.intent, re.IGNORECASE) and items:
        return len(items), "list items"
    if data_rows:
        # subtract the header row (rows inside thead are reported by role=row too)
        thead = 1 if any(e.role == "columnheader" for e in obs.elements) else 0
        return max(0, len(data_rows) - thead), "table rows"
    return len(items), "list items"
