"""Deterministic natural-language step parser.

Turns one instruction ("Enter username as abc", "Select India from the country
dropdown", "Verify that the user is redirected to the dashboard") into typed
DSL steps. It distinguishes interaction types (click vs fill vs select vs check),
extracts semantic targets rather than selectors, detects sensitive values and
reports its own confidence so low-confidence lines can be escalated to an LLM
or to a human instead of being guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.agents.planner.lexicon import (
    COLOR_WORDS,
    GAZETTEER,
    KIND_NOUN_RE,
    KIND_ROLES,
    ORDINALS,
    PURPOSES,
    SENSITIVE_PURPOSES,
    SHAPE_WORDS,
    classify_phrase,
    normalize_text,
    risk_for_text,
    synonyms_for,
)
from backend.agents.planner.spelling import correct_instruction
from backend.dsl.models import (
    ActionType,
    AssertionSpec,
    AssertionType,
    ConditionSpec,
    ScrollSpec,
    Step,
    TabSpec,
    TargetSpec,
    ValueSpec,
    WaitSpec,
)

QUOTE_RE = re.compile(r'"([^"]*)"|(?<![A-Za-z0-9])\'([^\']+)\'(?![A-Za-z0-9])|`([^`]*)`')
VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*\}\}")
URL_RE = re.compile(r"(https?://[^\s\"'<>]+|www\.[^\s\"'<>]+)", re.IGNORECASE)
LEADING_ENUM_RE = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]|\d+\s*[.)\]:-]|[-*•+]|\(\w\)|[a-z]\))\s*", re.IGNORECASE)
DETERMINERS_RE = re.compile(r"^(?:the|a|an|your|my|our|this|that|its|their)\s+", re.IGNORECASE)

SECRET_NAME_RE = re.compile(r"(pass(word|code)?|secret|token|api[_-]?key|pin|otp|cvv|card)", re.IGNORECASE)

KEY_ALIASES = {
    "enter": "Enter", "return": "Enter", "tab": "Tab", "escape": "Escape", "esc": "Escape",
    "backspace": "Backspace", "delete": "Delete", "del": "Delete", "space": " ", "spacebar": " ",
    "arrow up": "ArrowUp", "up arrow": "ArrowUp", "arrowup": "ArrowUp", "arrow down": "ArrowDown",
    "down arrow": "ArrowDown", "arrowdown": "ArrowDown", "arrow left": "ArrowLeft", "left arrow": "ArrowLeft",
    "arrowleft": "ArrowLeft", "arrow right": "ArrowRight", "right arrow": "ArrowRight", "arrowright": "ArrowRight",
    "page up": "PageUp", "pageup": "PageUp", "page down": "PageDown", "pagedown": "PageDown", "home": "Home",
    "end": "End",
}
MODIFIER_ALIASES = {"ctrl": "Control", "control": "Control", "cmd": "Meta", "command": "Meta", "meta": "Meta",
                    "shift": "Shift", "alt": "Alt", "option": "Alt"}

STATE_VERBS_RE = re.compile(
    r"\b(is|are|was|were|should|shows?|displays?|contains?|appears?|has|have|exists?|redirect\w*|"
    r"navigat\w*|visible|displayed|present|enabled|disabled|checked|selected|loaded|equals?|matches|"
    r"reads?|says|lands?)\b",
    re.IGNORECASE,
)

PAGE_NOUNS = {"dashboard", "home", "homepage", "home page", "settings", "profile", "account", "cart", "basket",
              "checkout", "login", "sign in", "signup", "sign up", "register", "registration", "contact", "about",
              "search", "inbox", "admin", "products", "orders", "reports", "help", "faq", "pricing", "blog",
              "landing"}


@dataclass
class ParseContext:
    base_url: str | None = None
    variables: dict[str, str] = field(default_factory=dict)
    secret_names: set[str] = field(default_factory=set)
    frame_hint: str | None = None
    last_values: dict[str, ValueSpec] = field(default_factory=dict)  # purpose -> value used
    step_counter: int = 0
    has_navigated: bool = False
    chosen_options: list[str] = field(default_factory=list)  # options picked earlier in this test, in order


@dataclass
class ParsedLine:
    steps: list[Step]
    confidence: float
    reason: str = ""


class StepParser:
    """Rule-based parser. Rules are ordered from most to least specific."""

    def __init__(self) -> None:
        self._rules = [
            self._rule_conditional,
            self._rule_login_macro,
            self._rule_assert,
            self._rule_expected_line,
            self._rule_navigate_url,
            self._rule_back_reload,
            self._rule_new_tab,
            self._rule_switch_tab,
            self._rule_close_tab,
            self._rule_switch_frame,
            self._rule_dialog,
            self._rule_press_key,
            self._rule_wait,
            self._rule_scroll,
            self._rule_screenshot,
            self._rule_capture_value,
            self._rule_upload,
            self._rule_download,
            self._rule_hover,
            self._rule_drag,
            self._rule_double_right_click,
            self._rule_clear,
            self._rule_uncheck,
            self._rule_accept_terms,
            self._rule_check,
            self._rule_toggle,
            self._rule_search_for,
            self._rule_fill_form,
            self._rule_fill,
            self._rule_named,
            self._rule_select,
            self._rule_open_page,
            self._rule_logout_login_bare,
            self._rule_submit,
            self._rule_click,
            self._rule_open_click,
            self._rule_type_bare,
        ]

    # ------------------------------------------------------------------ public
    def parse(self, raw_line: str, ctx: ParseContext) -> ParsedLine:
        line = clean_line(raw_line)
        if not line:
            return ParsedLine([], 0.0, "empty line")
        spelling = correct_instruction(line)
        candidates = [spelling.text, line] if spelling.changed else [line]
        for attempt in candidates:
            result = self._parse_line(attempt, raw_line, ctx)
            if result.steps:
                if attempt is not line and spelling.changed:
                    for step in result.steps:
                        step.notes.append(spelling.note())
                        step.original_text = line  # the author's words, not the repaired ones
                return result
        return ParsedLine([], 0.0, "no rule matched")

    def _parse_line(self, line: str, raw_line: str, ctx: ParseContext) -> ParsedLine:
        # phrasings that contain "and" but describe ONE interaction run before compound splitting
        for rule in (self._rule_click_and_select, self._rule_time_fields, self._rule_check_on_tab,
                     self._rule_range_cell, self._rule_verify_else, self._rule_click_fill, self._rule_close_popup):
            result = rule(line, ctx)
            if result is not None:
                return result
        if not re.match(r"^if\b", line, re.IGNORECASE):
            compound = self._parse_compound(line, raw_line, ctx)
            if compound is not None:
                return compound
        for rule in self._rules:
            result = rule(line, ctx)
            if result is not None:
                return result
        return ParsedLine([], 0.0, "no rule matched")

    def _parse_compound(self, line: str, raw_line: str, ctx: ParseContext) -> ParsedLine | None:
        parts = split_compound(line)
        if len(parts) < 2:
            return None
        snapshot = (ctx.step_counter, dict(ctx.last_values), ctx.frame_hint)
        steps: list[Step] = []
        confidences: list[float] = []
        inherited_verb = leading_verb(parts[0])
        for part in parts:
            candidate = part
            if inherited_verb and not leading_verb(part):
                candidate = f"{inherited_verb} {part}"
            sub = None
            for rule in self._rules:
                sub = rule(clean_line(candidate), ctx)
                if sub is not None:
                    break
            if sub is None or not sub.steps:
                ctx.step_counter, ctx.last_values, ctx.frame_hint = snapshot[0], snapshot[1], snapshot[2]
                return None
            steps.extend(sub.steps)
            confidences.append(sub.confidence)
        for step in steps:
            step.original_text = raw_line.strip()
        return ParsedLine(steps, min(confidences) * 0.97, "compound instruction")

    # ------------------------------------------------------------------ one-interaction phrasings
    def _rule_click_and_select(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """"Click on Theme Box and select Initial Call" / "Open the Status dropdown and choose Not reachable"."""

        m = re.match(r"^(?:click|tap|press|open|expand)\s+(?:on\s+)?(?:the\s+)?(?P<target>.+?)\s+and\s+"
                     r"(?:then\s+)?(?:select|choose|pick|set)\s+(?:the\s+)?(?:option\s+)?(?:to\s+)?(?P<opt>.+)$",
                     line, re.IGNORECASE)
        if not m:
            return None
        option = clean_option(m.group("opt"))
        said = strip_quotes(m.group("target")).strip()
        label = strip_quotes(re.sub(r"\s+(?:box|field|selection|selector|picker|dropdown|drop-down|drop down|list)$", "",
                                    said, flags=re.IGNORECASE).strip()) or said
        target = build_target(label, "select")
        # the control is labelled "Theme", not "Theme Box": the label leads, the full wording stays a hint
        target.intent = label
        target.semantic_hints = list(dict.fromkeys([label, said, *target.semantic_hints]))[:14]
        # a "box"/"selection" that one opens and picks from is a list or a picker, whatever its markup
        target.preferred_roles = ["combobox", "listbox", "button", "textbox", "generic"]
        target.option_text = option
        step = self._step(ctx, line, ActionType.SELECT, 0.93, target=target,
                          value=parse_value(clean_option(m.group("opt"), keep_quotes=True), ctx),
                          risk_hint="input", expected_postcondition="selected option matches",
                          notes=["open_then_choose"])
        ctx.chosen_options.append(option)
        return ParsedLine([step], 0.93, "open control and choose option")

    def _rule_time_fields(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """"Select Start Time and set the first field as 23 HH and second field as 15 MM" -> time 23:15."""

        m = re.match(r"^(?:select|set|enter|choose|open|click(?:\s+on)?)\s+(?:the\s+)?(?P<target>.+?)\s*(?:,|and)?\s+"
                     r"(?:then\s+)?set\s+(?:the\s+)?(?:first|1st|hour|hours)\s*(?:field|box|input)?\s+(?:as|to)\s+"
                     r"(?P<h>\d{1,2})\s*(?:hh|hours?|hrs?|h)?\s*(?:,|and)?\s*(?:the\s+)?(?:second|2nd|minute|minutes)\s*"
                     r"(?:field|box|input)?\s+(?:as|to)\s+(?P<m>\d{1,2})\s*(?:mm|minutes?|mins?|m)?"
                     r"(?:\s*(?:,|and)?\s*(?:the\s+)?(?:third|3rd|am\s*/\s*pm|meridiem)\s*(?:field)?\s+(?:as|to)\s+"
                     r"(?P<ampm>am|pm))?$", line, re.IGNORECASE)
        if not m:
            return None
        hour, minute = int(m.group("h")), int(m.group("m"))
        if m.group("ampm"):
            hour = hour % 12 + (12 if m.group("ampm").lower() == "pm" else 0)
        if hour > 23 or minute > 59:
            return None
        target = build_target(m.group("target"), "fill")
        target.preferred_roles = ["textbox", "combobox", "button", "spinbutton", "generic"]
        value = ValueSpec(source="literal", value=f"{hour:02d}:{minute:02d}")
        step = self._step(ctx, line, ActionType.FILL, 0.93, target=target, value=value, risk_hint="input",
                          expected_postcondition="time is set", notes=["time_parts"])
        return ParsedLine([step], 0.93, "time entered field by field")

    def _rule_check_on_tab(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """"Check whether new Interaction for patient Abc is added on the Interaction Logs tab": open the tab,
        then verify the entity is listed together with what this test recorded."""

        m = re.match(r"^(?:check|verify|ensure|confirm|validate)\s+(?:whether|that|if)\s+(?P<claim>.+?)\s+"
                     r"(?:is|are|was|were|has\s+been|have\s+been|gets?|got)\s+(?:successfully\s+)?"
                     r"(?P<state>added|created|listed|shown|displayed|present|visible|recorded|saved|logged|updated)\s+"
                     r"(?:on|in|under|to)\s+(?:the\s+)?(?P<where>.+?)\s+(?P<kind>tab|section|page|list|table|grid|panel)$",
                     line, re.IGNORECASE)
        if not m:
            return None
        # "Percentage is displayed in the Range Type field in the Vertical Axis section" is a value
        # check on a control that is already on screen. Treating the section as a page to open clicks
        # the wrong element and then blocks every later step.
        if re.search(r"\b(?:field|dropdown|combobox|textbox|input)\b", line, re.IGNORECASE):
            return None
        if m.group("kind").lower() in ("section", "panel") and m.group("state").lower() in (
                "displayed", "shown", "visible", "present"):
            return None
        where, kind = strip_quotes(m.group("where")).strip(), m.group("kind").lower()
        claim = m.group("claim").strip()
        steps: list[Step] = []
        if kind in ("tab", "section", "panel", "page"):
            nav = TargetSpec(intent=where, preferred_roles=["tab", "link", "button", "menuitem"],
                             semantic_hints=[where, f"{where} {kind}"], element_kind=kind if kind == "tab" else None)
            steps.append(self._step(ctx, line, ActionType.CLICK, 0.9, target=nav, risk_hint="navigation",
                                    notes=["open_view_for_check"]))
        entity = _entity_of(claim)
        checks: list[AssertionSpec] = []
        if entity:
            checks.append(AssertionSpec(type=AssertionType.TEXT_VISIBLE, expected=entity, match="case_insensitive",
                                        description=f"'{entity}' is listed in {where}"))
        recorded = [o for o in ctx.chosen_options if not re.search(r"\d{2,}", o)][:3]
        if recorded:
            # the new record should show what this test entered (any one of the chosen values)
            checks.append(AssertionSpec(type=AssertionType.ANY_OF, description="the new entry shows values chosen "
                                        "earlier in this test", children=[
                                            AssertionSpec(type=AssertionType.TEXT_VISIBLE, expected=o,
                                                          match="case_insensitive") for o in recorded], derived=True))
        if not checks:
            return self._assertion_steps(line, f"{claim} is {m.group('state')}", ctx)
        assertion = AssertionSpec(type=AssertionType.ALL_OF, description=f"{claim} is {m.group('state')} in {where}",
                                  children=checks, timeout_ms=8000)
        steps.append(self._step(ctx, line, ActionType.ASSERT, 0.88, assertion=assertion,
                                notes=[f"semantic_claim:{claim} is {m.group('state')}"]))
        return ParsedLine(steps, 0.88, "open view and verify the new entry")

    def _rule_named(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """"Select the Patient named Abc Xy" -> click the item whose text is exactly "Abc Xy"."""

        m = re.match(r"^(?:select|choose|pick|open|click(?:\s+on)?|tap(?:\s+on)?)\s+(?:the\s+|a\s+|an\s+)?"
                     r"(?P<kind>[A-Za-z][\w -]{1,30}?)\s+(?:named|called|titled|labell?ed|with\s+(?:the\s+)?name)\s+"
                     r"(?P<name>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        name, kind = strip_quotes(m.group("name")).strip(), m.group("kind").strip()
        target = TargetSpec(intent=name, exact_name=name, literal=True, semantic_hints=[name, f"{kind} {name}"],
                            preferred_roles=["link", "button", "row", "cell", "listitem", "option", "generic"],
                            context_hint=None)
        step = self._step(ctx, line, ActionType.CLICK, 0.92, target=target, risk_hint="navigation",
                          notes=[f"entity_kind:{kind.lower()}"])
        return ParsedLine([step], 0.92, "open named item")

    # ------------------------------------------------------------------ helpers
    def _step(self, ctx: ParseContext, line: str, action: ActionType, confidence: float, **kwargs: Any) -> Step:
        ctx.step_counter += 1
        step = Step(id=f"step_{ctx.step_counter:03d}", index=ctx.step_counter, original_text=line, action=action,
                    confidence=confidence, **kwargs)
        if step.target and ctx.frame_hint and not step.target.frame_hint:
            step.target.frame_hint = ctx.frame_hint
        return step

    # ------------------------------------------------------------------ rules
    def _rule_conditional(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^if\s+(?P<cond>.+?)(?:\s*,\s*then\s+|\s*,\s*|\s+then\s+)(?P<then>.+?)"
                     r"(?:\s*(?:[,;.]\s*)?(?:otherwise|else)\s*,?\s*(?P<else>.+))?$", line, re.IGNORECASE)
        if not m:
            return None
        condition = parse_condition(m.group("cond"), ctx)
        then_parsed = self.parse(m.group("then"), ctx)
        else_parsed = self.parse(m.group("else"), ctx) if m.group("else") else ParsedLine([], 1.0)
        if not then_parsed.steps:
            return None
        step = self._step(ctx, line, ActionType.CONDITIONAL, min(then_parsed.confidence, else_parsed.confidence or 1.0)
                          * condition[1], condition=condition[0], then_steps=then_parsed.steps,
                          else_steps=else_parsed.steps)
        return ParsedLine([step], step.confidence, "inline conditional")

    def _observe(self, line: str, cond: str, ctx: ParseContext) -> ParsedLine | None:
        """ "Observe the Vertical Range (%) inputs; start entering from left to right" names something to look
        at, then talks to the tester. The first clause becomes a visibility check; a later clause is kept only
        when it is itself a checkable claim ("the values below 01 and 02 are already 0.000 and 100.000")."""

        clauses = [c.strip() for c in re.split(r"\s*;\s*", cond) if c.strip()]
        first = clauses[0]
        if not STATE_VERBS_RE.search(first) and not re.search(r"\b(is|are)\b", first, re.IGNORECASE):
            first = f"{first} is displayed"
        parsed = self._assertion_steps(line, first, ctx)
        if parsed is None:
            return None
        steps = list(parsed.steps)
        for clause in clauses[1:]:
            spec, confidence = parse_assertion(clause, ctx)
            if spec is not None and confidence >= 0.85 and not (spec.match == "semantic" and confidence < 0.9):
                steps.append(self._step(ctx, line, ActionType.ASSERT, confidence, assertion=spec, risk_hint="read",
                                        notes=[f"clause: {clause[:120]}"]))
            elif steps:
                steps[0].notes.append(f"tester guidance (not checked): {clause[:120]}")
        return ParsedLine(steps, min(st.confidence for st in steps), "observe")

    def _rule_click_fill(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """ "click on the field "Quintile Title", clear the existing text, and enter "X"" is ONE fill: fill
        replaces the content, so the click and the clearing are implied (split apart, the "clear" became a
        click on a random block of text)."""

        quoted = r"(?:\"[^\"]*\"|'[^']*'|\{\{\s*\w+\s*\}\})"
        m = re.match(rf"^(?:click|tap|go)\s+(?:on\s+|in(?:to)?\s+|to\s+)?(?:the\s+)?(?:(?:text\s+)?(?:field|input(?:\s+field)?|"
                     rf"textbox|text\s+box|box|text\s+area|textarea|cell)\s+)?(?P<target>{quoted}|[^,\"]+?)"
                     rf"(?:\s+(?:field|input|textbox|box))?(?:\s*,\s*|\s+and\s+|\s+then\s+)"
                     rf"(?:(?:clear|remove|delete|erase)\s+(?:the\s+|its\s+|all\s+)?"
                     rf"(?:existing\s+|current\s+|old\s+|previous\s+|default\s+)?(?:text|value|content|data|entry)?"
                     rf"(?:\s*,\s*|\s+and\s+|\s+then\s+)+)?(?:and\s+)?(?:then\s+)?"
                     rf"(?:enter|type|input|write|key\s+in|fill(?:\s+in)?)\s+(?P<val>{quoted}|.+?)\s*$",
                     line, re.IGNORECASE)
        if not m or re.search(r"\b(below|under|above|next\s+to)\b", line, re.IGNORECASE) and \
                re.search(r"\b(range|column|row)\b", line, re.IGNORECASE):
            return None  # Range (%) grid cells are positional: _rule_range_cell
        target_text = m.group("target").strip()
        if not QUOTE_RE.fullmatch(target_text) and not looks_like_field(target_text + " field"):
            return None
        target = build_target(f"{target_text} field" if not QUOTE_RE.fullmatch(target_text) else target_text, "fill")
        target.preferred_roles = target.preferred_roles or ["textbox", "searchbox", "spinbutton", "combobox"]
        value = parse_value(m.group("val"), ctx, purpose=target.purpose)
        step = self._step(ctx, line, ActionType.FILL, 0.93, target=target, value=value, risk_hint="input",
                          notes=["replace_content"])
        return ParsedLine([step], 0.93, "click, clear and enter")

    def _rule_close_popup(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """ "Click on the right sidebar to close the Cycle Number dropdown": the goal is to close the popup; the
        place to click is incidental. Escape closes it; a click on an empty spot when Escape does not."""

        m = re.match(r"^(?:click|tap|press)\s+(?:on\s+|in\s+)?(?:the\s+|an?\s+)?(?P<where>.+?)\s+(?:in\s+order\s+)?to\s+"
                     r"(?:close|dismiss|collapse|hide)\s+(?:the\s+)?(?P<what>.+?)$", line, re.IGNORECASE)
        if not m:
            return None
        if not re.search(r"\b(dropdown|drop-down|list|popup|pop-up|menu|calendar|picker|suggestions?|overlay|tooltip)\b",
                         m.group("what"), re.IGNORECASE):
            return None
        step = self._step(ctx, line, ActionType.PRESS_KEY, 0.9, key="Escape", risk_hint="read",
                          notes=["close_popup", f"click_area:{m.group('where').strip()[:60]}"])
        return ParsedLine([step], 0.9, "close popup")

    def _rule_range_cell(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """A Range (%) grid is two rows. The letter or number is the column header; the value goes in the box under it.

        "second row input field under column H" and "input field below 01" are that lower box.
        "input field 1 ... enter 01 below the Horizontal dropdown" is the header box itself (it currently
        shows 1, and the next steps refer to it as 01).
        """

        quoted = r"(?:\"[^\"]*\"|'[^']*'|\{\{\s*\w+\s*\}\})"
        ordinals = r"first|second|third|fourth|fifth|last|final|1st|2nd|3rd|4th|5th"
        under_col = re.match(
            rf"^(?:click|tap)\s+on\s+the\s+(?P<ord>{ordinals})\s+row\s+input\s+field\s+under\s+column\s+"
            rf"(?P<col>{quoted})\s+and\s+enter\s+(?P<val>{quoted})\s*$", line, re.IGNORECASE)
        if under_col:
            return self._grid_fill(ctx, line, column=strip_quotes(under_col.group("col")),
                                   value_text=under_col.group("val"), hint="column_row",
                                   row=ORDINALS[under_col.group("ord").lower()])
        below = re.match(
            rf"^(?:click|tap)\s+on\s+the\s+input\s+field\s+below\s+(?P<col>{quoted})\s+and\s+enter\s+"
            rf"(?P<val>{quoted})\s*$", line, re.IGNORECASE)
        if below:
            return self._grid_fill(ctx, line, column=strip_quotes(below.group("col")),
                                   value_text=below.group("val"), hint="below_label")
        rename = re.match(
            rf"^(?:click|tap)\s+on\s+the\s+input\s+field\s+(?P<col>{quoted})\s*,\s*clear\s+the\s+existing\s+"
            rf"value\s*,\s*and\s+enter\s+(?P<val>{quoted})\s+below\s+(?P<ctx>.+?)\s*$", line, re.IGNORECASE)
        if rename:
            where = rename.group("ctx").strip()
            named = QUOTE_RE.search(where)
            context = strip_quotes(named.group(0)) if named else where
            return self._grid_fill(ctx, line, column=strip_quotes(rename.group("col")),
                                   value_text=rename.group("val"), hint="labeled_below", context=context)
        return None

    def _grid_fill(self, ctx: ParseContext, line: str, *, column: str, value_text: str, hint: str,
                   row: int | None = None, context: str | None = None) -> ParsedLine:
        value = parse_value(value_text, ctx)
        target = TargetSpec(
            intent=f"range input {column}",
            exact_name=column,
            preferred_roles=["textbox", "spinbutton"],
            element_kind="input field",
            context_hint=context,
            ordinal=row,
            position_hint=hint,
            semantic_hints=[column, "range"],
        )
        step = self._step(ctx, line, ActionType.FILL, 0.93, target=target, value=value, risk_hint="input",
                          notes=[f"grid:{hint}"])
        return ParsedLine([step], 0.93, "range grid cell")

    def _rule_verify_else(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        """\"Verify that X else do Y\" runs Y only when X is not already true."""

        m = re.match(r"^(?P<check>(?:verify|validate|ensure|confirm|assert|observe|check)\s+.+?)\s+"
                     r"else\s+(?P<fallback>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        checked = self._rule_assert(m.group("check").strip(), ctx)
        if checked is None or not checked.steps or checked.steps[0].assertion is None:
            return None
        assertion = checked.steps[0].assertion
        fallback = self.parse(m.group("fallback").strip(), ctx)
        if not fallback.steps:
            return None
        # the check is evaluated as the assertion it is (a selected option, a count, a checked box), not as
        # a search for its wording; a short timeout: the else branch exists to repair it
        cond = ConditionSpec(kind="assertion", assertion=assertion.model_copy(update={"timeout_ms": 3000}),
                             expected=None if assertion.expected is None else str(assertion.expected),
                             description=m.group("check").strip())
        step = self._step(ctx, line, ActionType.CONDITIONAL, 0.86, condition=cond, then_steps=[],
                          else_steps=fallback.steps)
        return ParsedLine([step], 0.86, "verify else recover")

    def _rule_login_macro(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(
            r"^(?:log\s*in|sign\s*in|login|authenticate)\s+(?:as|with|using)\s+(?:the\s+)?"
            r"(?:(?:username|user\s*name|email(?:\s+address)?|user|account)\s+)?(?P<user>\S+)\s+"
            r"(?:and|with|,)\s+(?:the\s+)?(?:password\s+)?(?P<pw>\S+)$", line, re.IGNORECASE)
        if not m:
            return None
        user_value = parse_value(m.group("user"), ctx, purpose="username_or_email")
        pw_value = parse_value(m.group("pw"), ctx, purpose="password")
        steps = [
            self._step(ctx, line, ActionType.FILL, 0.9, target=target_for_purpose("username_or_email"),
                       value=user_value, risk_hint="input"),
            self._step(ctx, line, ActionType.FILL, 0.9, target=target_for_purpose("password"), value=pw_value,
                       sensitive=True, risk_hint="input"),
            self._step(ctx, line, ActionType.CLICK, 0.88, target=target_for_purpose("submit_login"),
                       expects_navigation=True, risk_hint="submit"),
        ]
        ctx.last_values["username_or_email"] = user_value
        return ParsedLine(steps, 0.9, "login macro")

    def _rule_expected_line(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:expected(?:\s+result)?s?|then)\s*[:\-]\s*(?P<cond>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        return self._assertion_steps(line, m.group("cond"), ctx)

    def _rule_assert(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:verify|validate|ensure|confirm|assert|make\s+sure|expect|observe|check)\s+"
                     r"(?:that\s+|if\s+|whether\s+)?(?P<cond>.+)$", line, re.IGNORECASE)
        if m:
            verb = line.split()[0].lower()
            cond = m.group("cond")
            if verb == "check" and not re.match(r"^check\s+(that|if|whether)\b", line, re.IGNORECASE):
                if not STATE_VERBS_RE.search(cond) or re.search(r"\b(checkbox|check box|tick box|toggle|switch)\b",
                                                               cond, re.IGNORECASE):
                    return None  # "Check the remember me checkbox" -> CHECK action
            if verb == "confirm" and re.match(r"^confirm\s+(?:the\s+)?(?:alert|dialog|popup|prompt|order|"
                                              r"purchase|payment|booking)\b", line, re.IGNORECASE):
                return None
            if verb == "observe":
                return self._observe(line, cond, ctx)
            return self._assertion_steps(line, cond, ctx)
        m = re.match(r"^(?:the\s+)?(?P<subject>.+?)\s+(?:should|must|will)\s+(?:now\s+)?(?P<pred>.+)$", line,
                     re.IGNORECASE)
        if m:
            subject = m.group("subject").strip()
            pred = m.group("pred").strip()
            see = re.match(r"^(?:be\s+able\s+to\s+)?(?:see|view|notice|observe|get)\s+(?:a\s+|an\s+|the\s+)?(?P<obj>.+)$",
                           pred, re.IGNORECASE)
            if see and re.match(r"^(?:the\s+)?(?:user|i|customer|visitor|admin|they|he|she)$", subject, re.IGNORECASE):
                return self._assertion_steps(line, f"{see.group('obj')} is visible", ctx)
            cond = f"{subject} {normalize_should(pred)}"
            return self._assertion_steps(line, cond, ctx)
        m = re.match(r"^(?:i|user|the user)\s+(?:can\s+)?(?:see|sees)\s+(?P<what>.+)$", line, re.IGNORECASE)
        if m:
            return self._assertion_steps(line, f"{m.group('what')} is visible", ctx)
        return None

    def _assertion_steps(self, line: str, cond: str, ctx: ParseContext) -> ParsedLine | None:
        parts = re.split(r"\s+and\s+(?=(?:the\s+)?[\w'\"]+(?:\s+[\w'\"]+){0,4}\s+(?:is|are|has|shows|contains|"
                         r"displays|should)\b)", cond, flags=re.IGNORECASE)
        steps: list[Step] = []
        confidences: list[float] = []
        for part in parts:
            assertion, confidence = parse_assertion(part, ctx)
            if assertion is None:
                return None
            steps.append(self._step(ctx, line, ActionType.ASSERT, confidence, assertion=assertion, risk_hint="read"))
            confidences.append(confidence)
        return ParsedLine(steps, min(confidences), "assertion")

    def _rule_navigate_url(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:open|go\s+to|navigate\s+to|visit|load|browse\s+to|launch|access|start\s+at)\s+"
                     r"(?:the\s+)?(?:url\s+|page\s+|website\s+|site\s+|application\s+|app\s+)?(?:at\s+)?"
                     r"(?P<url>(?:https?://|www\.)\S+|/[^\s]*|\{\{\s*\w+\s*\}\})(?P<rest>.*)$", line, re.IGNORECASE)
        if not m:
            return None
        url = m.group("url").rstrip(".,;")
        if url.lower().startswith("www."):
            url = "https://" + url
        var = VAR_RE.fullmatch(url)
        if var:
            target = TargetSpec(intent="url", url="{{" + var.group(1) + "}}")
        else:
            target = TargetSpec(intent="url", url=url)
        ctx.has_navigated = True
        step = self._step(ctx, line, ActionType.NAVIGATE, 0.98, target=target, risk_hint="navigation",
                          expects_navigation=True)
        return ParsedLine([step], 0.98, "navigate url")

    def _rule_back_reload(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        if re.match(r"^(?:go|navigate|click)\s+back(?:\s+to\s+the\s+previous\s+page)?$|^press\s+(?:the\s+)?back"
                    r"(?:\s+button)?$", line, re.IGNORECASE):
            return ParsedLine([self._step(ctx, line, ActionType.GO_BACK, 0.95, risk_hint="navigation",
                                          expects_navigation=True)], 0.95, "back")
        if re.match(r"^(?:refresh|reload)(?:\s+the)?(?:\s+(?:current\s+)?page)?$", line, re.IGNORECASE):
            return ParsedLine([self._step(ctx, line, ActionType.RELOAD, 0.95, risk_hint="navigation")], 0.95,
                              "reload")
        return None

    def _rule_new_tab(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^open\s+(?:a\s+)?new\s+(?:browser\s+)?(?:tab|window)(?:\s+(?:with|at|to|for)\s+(?P<url>\S+))?$",
                     line, re.IGNORECASE)
        if not m:
            return None
        target = TargetSpec(intent="url", url=m.group("url")) if m.group("url") else None
        step = self._step(ctx, line, ActionType.SWITCH_TAB, 0.95, tab=TabSpec(which="new"), target=target)
        return ParsedLine([step], 0.95, "new tab")

    def _rule_switch_tab(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:switch|go|move|change|return|navigate)\s+(?:back\s+)?to\s+(?:the\s+)?"
                     r"(?P<which>newly\s+opened|new|second|last|latest|next|previous|first|original|main|other|"
                     r"parent|initial)\s+(?:browser\s+)?(?:tab|window)", line, re.IGNORECASE)
        if m:
            which_word = m.group("which").lower()
            mapping = {"new": "latest", "newly opened": "latest", "latest": "latest", "last": "latest",
                       "second": "index", "next": "next", "previous": "previous", "first": "first",
                       "original": "first", "main": "first", "parent": "first", "initial": "first", "other": "next"}
            which = mapping.get(re.sub(r"\s+", " ", which_word), "latest")
            spec = TabSpec(which=which, index=1 if which_word == "second" else None)  # type: ignore[arg-type]
            return ParsedLine([self._step(ctx, line, ActionType.SWITCH_TAB, 0.95, tab=spec)], 0.95, "switch tab")
        m = re.match(r"^switch\s+to\s+(?:the\s+)?(?:tab|window)\s+(?:titled|named|called|with\s+title)\s+(?P<t>.+)$",
                     line, re.IGNORECASE)
        if m:
            spec = TabSpec(which="title", title=strip_quotes(m.group("t")))
            return ParsedLine([self._step(ctx, line, ActionType.SWITCH_TAB, 0.93, tab=spec)], 0.93, "switch tab")
        return None

    def _rule_close_tab(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        if re.match(r"^close\s+(?:the\s+)?(?:current\s+|new\s+|this\s+)?(?:browser\s+)?(?:tab|window)$", line,
                    re.IGNORECASE):
            return ParsedLine([self._step(ctx, line, ActionType.SWITCH_TAB, 0.95, tab=TabSpec(which="close_current"))],
                              0.95, "close tab")
        return None

    def _rule_switch_frame(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        if re.match(r"^switch\s+(?:back\s+)?to\s+(?:the\s+)?(?:main|default|top|parent)\s+(?:page|frame|content|"
                    r"document|window)$", line, re.IGNORECASE):
            ctx.frame_hint = None
            return ParsedLine([self._step(ctx, line, ActionType.SWITCH_FRAME, 0.95,
                                          target=TargetSpec(intent="main document"))], 0.95, "frame reset")
        m = re.match(r"^switch\s+(?:in)?to\s+(?:the\s+)?(?:(?:iframe|frame)\s+(?:named|titled|called)?\s*(?P<a>.+)|"
                     r"(?P<b>.+?)\s+(?:iframe|frame))$", line, re.IGNORECASE)
        if m:
            name = strip_quotes((m.group("a") or m.group("b") or "").strip())
            ctx.frame_hint = name
            return ParsedLine([self._step(ctx, line, ActionType.SWITCH_FRAME, 0.9,
                                          target=TargetSpec(intent=name, preferred_roles=["iframe"],
                                                            frame_hint=name))], 0.9, "frame switch")
        return None

    def _rule_dialog(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:accept|ok|approve|confirm)\s+(?:the\s+)?(?:browser\s+|javascript\s+|js\s+|native\s+)?"
                     r"(?P<kind>alert|dialog|confirmation(?:\s+dialog|\s+box)?|confirm(?:\s+dialog|\s+box)?|prompt|"
                     r"pop-?up)(?:\s+with\s+(?:the\s+)?(?:text|value|input)?\s*(?P<text>.+))?$", line, re.IGNORECASE)
        if m:
            value = ValueSpec(source="literal", value=strip_quotes(m.group("text"))) if m.group("text") else None
            step = self._step(ctx, line, ActionType.ACCEPT_DIALOG, 0.95, value=value,
                              target=TargetSpec(intent=m.group("kind").lower(), element_kind="dialog"))
            return ParsedLine([step], 0.95, "accept dialog")
        m = re.match(r"^(?:dismiss|cancel|reject|decline|close)\s+(?:the\s+)?(?:browser\s+|javascript\s+)?"
                     r"(?P<kind>alert|dialog|confirmation(?:\s+dialog|\s+box)?|confirm(?:\s+dialog|\s+box)?|prompt|"
                     r"pop-?up|modal|overlay)$", line, re.IGNORECASE)
        if m:
            step = self._step(ctx, line, ActionType.DISMISS_DIALOG, 0.93,
                              target=TargetSpec(intent=m.group("kind").lower(), element_kind="dialog",
                                                preferred_roles=["dialog", "alertdialog"]))
            return ParsedLine([step], 0.93, "dismiss dialog")
        return None

    def _rule_press_key(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:press|hit|tap|type|push)\s+(?:the\s+)?(?:key\s+)?(?P<key>"
                     r"(?:(?:ctrl|control|cmd|command|meta|shift|alt|option)\s*\+\s*)+\S+|enter|return|tab|escape|"
                     r"esc|backspace|delete|del|space(?:bar)?|arrow\s*(?:up|down|left|right)|(?:up|down|left|right)"
                     r"\s+arrow|page\s*(?:up|down)|home|end|f\d{1,2})(?:\s+key)?(?:\s+(?:in|on|inside)\s+(?:the\s+)?"
                     r"(?P<target>.+))?$", line, re.IGNORECASE)
        if not m:
            return None
        key = normalize_key(m.group("key"))
        target = build_target(m.group("target"), "type") if m.group("target") else None
        step = self._step(ctx, line, ActionType.PRESS_KEY, 0.95, key=key, target=target,
                          risk_hint="submit" if key == "Enter" else "input")
        return ParsedLine([step], 0.95, "press key")

    def _rule_wait(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^wait\s+(?:for\s+)?(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|s|milliseconds?|ms|"
                     r"minutes?|mins?)\b", line, re.IGNORECASE)
        if m:
            n = float(m.group("n"))
            unit = m.group("unit").lower()
            seconds = n / 1000 if unit.startswith("m") and unit not in ("min", "mins", "minute", "minutes") else n
            if unit in ("min", "mins", "minute", "minutes"):
                seconds = n * 60
            return ParsedLine([self._step(ctx, line, ActionType.WAIT, 0.98, wait=WaitSpec(seconds=seconds),
                                          risk_hint="read")], 0.98, "wait time")
        # "wait for sometime", "wait a moment", "wait for a few seconds": a pause, not text to look for
        m = re.match(r"^wait\s+(?:for\s+)?(?:a\s+|some\s+)?(?P<what>some\s*time|sometime|while|moment|bit|little(?:\s+while)?|"
                     r"second|few\s+(?:more\s+)?seconds?|couple\s+of\s+seconds|some\s+seconds|few\s+moments?)"
                     r"(?:\s+(?:more|longer))?$", line, re.IGNORECASE)
        if m:
            what = m.group("what").lower()
            seconds = 5.0 if "while" in what else 1.0 if what == "second" else 3.0
            return ParsedLine([self._step(ctx, line, ActionType.WAIT, 0.97, wait=WaitSpec(seconds=seconds),
                                          risk_hint="read", notes=["pause"])], 0.97, "pause")
        # "wait until (the) results are loaded properly", "wait for the page to load", "wait till the grid refreshes"
        m = re.match(r"^wait\s+(?:for|until|till|untill)\s+(?:all\s+)?(?:the\s+)?(?P<what>page|screen|results?|data|content|"
                     r"grid|table|list|report|records?|rows?|search\s+results?|details?|form|everything|it|values?|"
                     r"options?|dropdowns?|chart|graph)\s+(?:to\s+)?(?:is\s+|are\s+|has\s+|have\s+|gets?\s+|be\s+)?"
                     r"(?:been\s+)?(?:fully\s+|properly\s+|completely\s+|correctly\s+)?(?:loads?|loaded|loading|displayed|ready|"
                     r"shown|rendered|refreshe?[sd]?|updated?|populated|finished|done|complete[sd]?|settles?|settled)"
                     r"(?:\s+(?:properly|completely|fully|correctly))?$", line, re.IGNORECASE)
        if m:
            return ParsedLine([self._step(ctx, line, ActionType.WAIT, 0.95, wait=WaitSpec(until_stable=True),
                                          risk_hint="read", notes=["page_load"])], 0.95, "wait until settled")
        m = re.match(r"^wait\s+(?:for|until)\s+(?:the\s+)?(?P<what>.+?)\s+(?:to\s+)?(?P<state>appear|appears|be\s+visible|"
                     r"is\s+visible|be\s+displayed|is\s+displayed|load|loads|be\s+loaded|show\s+up|shows\s+up|"
                     r"disappear|disappears|vanish|vanishes|be\s+hidden|is\s+hidden|is\s+gone|go\s+away|goes\s+away)$",
                     line, re.IGNORECASE)
        if m:
            what = strip_quotes(m.group("what"))
            gone = any(w in m.group("state").lower() for w in ("disappear", "vanish", "hidden", "gone", "away"))
            spec = WaitSpec(until_text_gone=what) if gone else WaitSpec(until_text=what)
            return ParsedLine([self._step(ctx, line, ActionType.WAIT, 0.9, wait=spec, risk_hint="read")], 0.9,
                              "wait for")
        m = re.match(r"^wait\s+(?:for|until)\s+(?:the\s+)?(?P<what>.+)$", line, re.IGNORECASE)
        if m:
            spec = WaitSpec(until_text=strip_quotes(m.group("what")))
            return ParsedLine([self._step(ctx, line, ActionType.WAIT, 0.8, wait=spec, risk_hint="read")], 0.8,
                              "wait for (implicit)")
        return None

    def _rule_scroll(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^scroll\s+(?:the\s+page\s+)?(?:all\s+the\s+way\s+)?(?:down\s+|up\s+)?to\s+(?:the\s+)?"
                     r"(?P<edge>bottom|top|end|very\s+bottom|very\s+top)(?:\s+of\s+the\s+(?:page|list))?$", line,
                     re.IGNORECASE)
        if m:
            edge = "top" if "top" in m.group("edge").lower() else "bottom"
            spec = ScrollSpec(direction="up" if edge == "top" else "down", to=edge)  # type: ignore[arg-type]
            return ParsedLine([self._step(ctx, line, ActionType.SCROLL, 0.95, scroll=spec, risk_hint="read")], 0.95,
                              "scroll edge")
        m = re.match(r"^scroll\s+(?:the\s+page\s+)?(?P<dir>down|up|left|right)(?:\s+by\s+(?P<px>\d+)\s*(?:px|pixels)?)?"
                     r"(?:\s+(?P<times>\d+)\s+times)?$", line, re.IGNORECASE)
        if m:
            spec = ScrollSpec(direction=m.group("dir").lower(),  # type: ignore[arg-type]
                              amount_px=int(m.group("px")) if m.group("px") else None)
            return ParsedLine([self._step(ctx, line, ActionType.SCROLL, 0.95, scroll=spec, risk_hint="read")], 0.95,
                              "scroll")
        m = re.match(r"^scroll\s+(?:down\s+|up\s+)?(?:to|until)\s+(?:you\s+see\s+|it\s+shows\s+)?(?:the\s+)?(?P<text>.+?)"
                     r"(?:\s+(?:is|becomes)\s+visible|\s+appears|\s+is\s+shown)?$", line, re.IGNORECASE)
        if m:
            spec = ScrollSpec(direction="down", to="until_text", text=strip_quotes(m.group("text")))
            return ParsedLine([self._step(ctx, line, ActionType.SCROLL, 0.88, scroll=spec, risk_hint="read")], 0.88,
                              "scroll until")
        return None

    def _rule_screenshot(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:take|capture|grab|save)\s+(?:a\s+)?(?:full[- ]page\s+)?screen\s*shot(?:\s+of\s+(?:the\s+)?"
                     r"(?P<target>.+))?$", line, re.IGNORECASE)
        if not m:
            return None
        target = build_target(m.group("target"), "hover") if m.group("target") else None
        step = self._step(ctx, line, ActionType.CAPTURE_SCREENSHOT, 0.97, target=target, risk_hint="read",
                          notes=["full_page"] if "full" in line.lower() else [])
        return ParsedLine([step], 0.97, "screenshot")

    def _rule_capture_value(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:store|save|remember|capture|note|record|read)\s+(?:the\s+)?(?:value\s+|text\s+)?(?:of\s+)?"
                     r"(?:the\s+)?(?P<target>.+?)\s+(?:as|into|in|to)\s+(?:a\s+)?(?:variable\s+)?(?:\{\{)?\s*"
                     r"(?P<var>[A-Za-z_]\w*)\s*(?:\}\})?$", line, re.IGNORECASE)
        if not m:
            return None
        step = self._step(ctx, line, ActionType.CAPTURE_VALUE, 0.9, target=build_target(m.group("target"), "hover"),
                          capture_as=m.group("var"), risk_hint="read")
        ctx.variables.setdefault(m.group("var"), "")
        return ParsedLine([step], 0.9, "capture value")

    def _rule_upload(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:upload|attach|choose|select)\s+(?:the\s+)?(?:a\s+)?(?:file\s+)?(?P<file>\"[^\"]+\"|'[^']+'|"
                     r"\{\{\s*\w+\s*\}\}|[\w./\\-]+\.[A-Za-z0-9]{1,5})(?:\s+(?:file\s+)?(?:to|into|in|using|via|with|on|as|"
                     r"for)\s+(?:the\s+)?(?P<target>.+))?$", line, re.IGNORECASE)
        if not m:
            return None
        file_value = parse_value(m.group("file"), ctx)
        target = build_target(m.group("target") or "file upload", "upload")
        if not target.purpose:
            target.purpose = "upload"
        step = self._step(ctx, line, ActionType.UPLOAD, 0.93, target=target, value=file_value, risk_hint="input")
        return ParsedLine([step], 0.93, "upload")

    def _rule_download(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:download|click\s+(?:on\s+)?(?:the\s+)?(?P<t2>.+?)\s+to\s+download)\s*(?:the\s+)?(?P<target>.*)$",
                     line, re.IGNORECASE)
        if not m:
            return None
        phrase = (m.group("t2") or m.group("target") or "download").strip()
        target = build_target(phrase, "download")
        if not target.preferred_roles:
            target.preferred_roles = ["link", "button"]
        if "download" not in [h.lower() for h in target.semantic_hints]:
            target.semantic_hints.append("download")
        step = self._step(ctx, line, ActionType.DOWNLOAD, 0.92, target=target, expects_download=True,
                          risk_hint="read")
        return ParsedLine([step], 0.92, "download")

    def _rule_hover(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:hover|mouse\s*over|mouseover|move\s+(?:the\s+)?(?:mouse|cursor|pointer)\s+(?:over|to|onto|on))"
                     r"\s+(?:over\s+|on\s+|onto\s+)?(?:the\s+)?(?P<target>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        step = self._step(ctx, line, ActionType.HOVER, 0.93, target=build_target(m.group("target"), "hover"),
                          risk_hint="read")
        return ParsedLine([step], 0.93, "hover")

    def _rule_drag(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:drag(?:\s+and\s+drop)?|move)\s+(?:the\s+)?(?P<src>.+?)\s+(?:to|onto|into|over|on\s+to|"
                     r"in\s+to|on)\s+(?:the\s+)?(?P<dst>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        if re.match(r"^move\s+(?:the\s+)?(?:mouse|cursor|pointer)\b", line, re.IGNORECASE):
            return None
        step = self._step(ctx, line, ActionType.DRAG, 0.9, target=build_target(m.group("src"), "drag"),
                          destination=build_target(m.group("dst"), "drop"), risk_hint="input")
        return ParsedLine([step], 0.9, "drag")

    def _rule_double_right_click(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^double[- ]?click\s+(?:on\s+)?(?:the\s+)?(?P<target>.+)$", line, re.IGNORECASE)
        if m:
            target = build_target(m.group("target"), "click")
            return ParsedLine([self._step(ctx, line, ActionType.DOUBLE_CLICK, 0.94, target=target,
                                          risk_hint=risk_for_text(m.group("target")) or "input")], 0.94,
                              "double click")
        m = re.match(r"^(?:right[- ]?click|context[- ]?click)\s+(?:on\s+)?(?:the\s+)?(?P<target>.+)$", line,
                     re.IGNORECASE)
        if m:
            target = build_target(m.group("target"), "click")
            return ParsedLine([self._step(ctx, line, ActionType.RIGHT_CLICK, 0.94, target=target, risk_hint="read")],
                              0.94, "right click")
        return None

    def _rule_clear(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:clear|empty|erase)\s+(?:out\s+)?(?:the\s+)?(?P<target>.+?)(?:\s+(?:field|input|box|"
                     r"textbox|text\s+box))?$", line, re.IGNORECASE)
        if not m:
            return None
        phrase = m.group("target")
        if re.search(r"\b(checkbox|check box|toggle|switch|selection)\b", phrase, re.IGNORECASE):
            return None
        target = build_target(phrase + (" field" if not KIND_NOUN_RE.search(phrase) else ""), "fill")
        return ParsedLine([self._step(ctx, line, ActionType.CLEAR, 0.9, target=target, risk_hint="input")], 0.9,
                          "clear")

    def _rule_uncheck(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:uncheck|untick|un-check|un-tick|deselect|unselect|clear|disable|turn\s+off|switch\s+off|"
                     r"opt\s+out\s+of)\s+(?:the\s+)?(?P<target>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        target = build_target(m.group("target"), "uncheck")
        if not target.preferred_roles:
            target.preferred_roles = ["checkbox", "switch"]
        return ParsedLine([self._step(ctx, line, ActionType.UNCHECK, 0.93, target=target, risk_hint="input")], 0.93,
                          "uncheck")

    def _rule_accept_terms(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:accept|agree\s+to|agree\s+with|consent\s+to)\s+(?:the\s+|all\s+)?(?P<target>.*(?:terms|"
                     r"conditions|policy|privacy|agreement|eula|license|tos).*)$", line, re.IGNORECASE)
        if not m:
            return None
        target = build_target(m.group("target"), "check")
        target.preferred_roles = ["checkbox", "switch"]
        target.purpose = target.purpose or "terms"
        return ParsedLine([self._step(ctx, line, ActionType.CHECK, 0.9, target=target, risk_hint="input")], 0.9,
                          "accept terms")

    def _rule_check(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:check|tick|enable|turn\s+on|switch\s+on|opt\s+in\s+to|opt\s+into|mark)\s+(?:the\s+)?"
                     r"(?P<target>.+)$", line, re.IGNORECASE)
        if not m:
            m2 = re.match(r"^(?:select|choose|pick|click(?:\s+on)?)\s+(?:the\s+)?(?P<target>.+?\s+(?:checkbox|check\s+box|"
                          r"radio(?:\s+button)?|option\s+button|toggle|switch))$", line, re.IGNORECASE)
            if not m2:
                return None
            phrase = m2.group("target")
        else:
            phrase = m.group("target")
        target = build_target(phrase, "check")
        if not target.preferred_roles:
            target.preferred_roles = ["checkbox", "switch", "radio"]
        return ParsedLine([self._step(ctx, line, ActionType.CHECK, 0.93, target=target, risk_hint="input")], 0.93,
                          "check")

    def _rule_toggle(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^toggle\s+(?:the\s+)?(?P<target>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        target = build_target(m.group("target"), "toggle")
        return ParsedLine([self._step(ctx, line, ActionType.TOGGLE, 0.9, target=target, risk_hint="input")], 0.9,
                          "toggle")

    def _rule_search_for(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^search\s+(?:for\s+)?(?P<value>\"[^\"]+\"|'[^']+'|.+?)(?:\s+(?:in|using)\s+(?:the\s+)?"
                     r"(?P<target>.+))?$", line, re.IGNORECASE)
        if not m:
            return None
        value = parse_value(m.group("value"), ctx)
        target = build_target(m.group("target"), "fill") if m.group("target") else target_for_purpose("search")
        if not target.purpose:
            target.purpose = "search"
        fill = self._step(ctx, line, ActionType.FILL, 0.9, target=target, value=value, risk_hint="input")
        press = self._step(ctx, line, ActionType.PRESS_KEY, 0.9, key="Enter", target=target.model_copy(deep=True),
                           risk_hint="submit", notes=["submit_search"])
        return ParsedLine([fill, press], 0.9, "search")

    def _rule_fill_form(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:enter|fill\s+in|fill\s+out|fill|provide|complete|populate|input)\s+(?:the\s+|all\s+(?:the\s+)?|"
                     r"your\s+|his\s+|her\s+|their\s+)?(?P<what>.*?)\s*(?:details|information|info|data|fields|form)"
                     r"(?:\s+(?:with|using)\s+(?P<source>.+))?$", line, re.IGNORECASE)
        if not m:
            return None
        what = (m.group("what") or "").strip()
        step = self._step(ctx, line, ActionType.FILL_FORM, 0.75, target=TargetSpec(intent=f"{what} form".strip(),
                                                                                   preferred_roles=["form"]),
                          risk_hint="input", notes=[f"data_source:{m.group('source') or 'test_data_or_generated'}"])
        return ParsedLine([step], 0.75, "semantic form fill")

    def _rule_fill(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        verbs = r"(?:enter|type|input|fill\s+in|fill(?:\s+out)?|key\s+in|write|provide|put|insert|supply|add|set|use)"
        # value-first with an explicit preposition: "Type abc into the username field"
        m = re.match(rf"^{verbs}\s+(?P<value>\"[^\"]*\"|'[^']*'|\{{\{{\s*\w+\s*\}}\}}|\S+(?:\s+\S+){{0,3}}?)\s+"
                     rf"(?:in|into|in\s+to|inside|within|on)\s+(?:the\s+|a\s+|an\s+|your\s+)?(?P<target>.+)$",
                     line, re.IGNORECASE)
        if m and (is_explicit_value(m.group("value")) or looks_like_field(m.group("target"))):
            return self._fill_step(line, ctx, m.group("target"), m.group("value"), 0.95)
        # "Enter abc as the username"
        m = re.match(rf"^{verbs}\s+(?P<value>\"[^\"]*\"|'[^']*'|\{{\{{\s*\w+\s*\}}\}}|\S+)\s+as\s+(?:the|your|my)\s+"
                     rf"(?P<target>.+)$", line, re.IGNORECASE)
        if m and looks_like_field(m.group("target")):
            return self._fill_step(line, ctx, m.group("target"), m.group("value"), 0.92)
        # "Enter username as abc" / "Fill username with abc" / "Set email to x" / "Enter username: abc"
        m = re.match(rf"^{verbs}\s+(?:the\s+|a\s+|an\s+|your\s+|my\s+)?(?P<target>.+?)\s*(?:\s(?:as|with|to|=)\s+|:\s*)"
                     rf"(?P<value>.+)$", line, re.IGNORECASE)
        if m:
            target_phrase, value_phrase = m.group("target"), m.group("value")
            verb = line.split()[0].lower()
            if " as " in f" {line.lower()} " and _looks_like_data(target_phrase) and looks_like_field(value_phrase):
                # "Enterjane.doe@example.com as Email", "Enter S3cret!pass as Password": value first, field last
                target_phrase, value_phrase = value_phrase, target_phrase
            if verb == "set" and looks_like_choice(target_phrase) and not is_explicit_value(value_phrase):
                return None  # "Set country to India" -> select rule
            if verb in ("add", "use", "put") and not looks_like_field(target_phrase):
                return None
            confidence = 0.95 if (is_explicit_value(value_phrase) or looks_like_field(target_phrase)) else 0.8
            return self._fill_step(line, ctx, target_phrase, value_phrase, confidence)
        # "Enter password {{test_password}}" / "Enter email test@example.com"
        m = re.match(rf"^{verbs}\s+(?:the\s+|a\s+|an\s+|your\s+)?(?P<target>[A-Za-z][\w\s'-]*?)\s+(?P<value>\"[^\"]*\"|"
                     rf"'[^']*'|\{{\{{\s*\w+\s*\}}\}}|\S+@\S+\.\S+|\+?\d[\d\s()-]{{2,}})$", line, re.IGNORECASE)
        if m:
            return self._fill_step(line, ctx, m.group("target"), m.group("value"), 0.9)
        return None

    def _fill_step(self, line: str, ctx: ParseContext, target_phrase: str, value_phrase: str,
                   confidence: float) -> ParsedLine:
        target = build_target(target_phrase, "fill")
        value = parse_value(value_phrase, ctx, purpose=target.purpose)
        sensitive = value.sensitive or (target.purpose in SENSITIVE_PURPOSES)
        value.sensitive = sensitive
        if target.purpose:
            ctx.last_values[target.purpose] = value
        if target.input_type is None and target.purpose == "password":
            target.input_type = "password"
        step = self._step(ctx, line, ActionType.FILL, confidence, target=target, value=value, sensitive=sensitive,
                          risk_hint="input", expected_postcondition="field contains the entered value")
        if target.purpose is None and not looks_like_field(target_phrase):
            step.notes.append("target_purpose_unknown")
            step.confidence = min(step.confidence, 0.8)
        return ParsedLine([step], step.confidence, "fill")

    def _rule_select(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        # "Select India from the country dropdown"
        m = re.match(r"^(?:select|choose|pick)\s+(?:the\s+)?(?:option\s+)?(?P<opt>\"[^\"]+\"|'[^']+'|.+?)"
                     r"(?:\s+(?:option|value|item|entry|choice))?\s+"
                     r"(?:from|in|on|under|within|for)\s+(?:the\s+)?(?P<target>.+)$", line, re.IGNORECASE)
        if m and not re.search(r"\b(page|tab|window|menu bar)\b", m.group("target"), re.IGNORECASE):
            choices = quoted_choices(m.group("opt"))
            option = ", ".join(choices) if len(choices) > 1 else clean_option(m.group("opt"))
            target = build_target(m.group("target"), "select")
            if not target.preferred_roles:
                target.preferred_roles = ["combobox", "listbox", "radiogroup"]
            target.option_text = option
            value = ValueSpec(source="literal", value=option, values=choices) if len(choices) > 1 else \
                parse_value(clean_option(m.group("opt"), keep_quotes=True), ctx)
            ctx.chosen_options.append(option)
            step = self._step(ctx, line, ActionType.SELECT, 0.95, target=target, value=value, risk_hint="input",
                              expected_postcondition="selected option matches")
            return ParsedLine([step], 0.95, "select option from target")
        # "Select country as India" / "Set the country to India" / "Choose Gender: Female"
        m = re.match(r"^(?:select|choose|pick|set)\s+(?:the\s+)?(?P<target>.+?)\s*(?:\s(?:as|to)\s+|:\s*)(?P<opt>.+)$",
                     line, re.IGNORECASE)
        if m and (looks_like_choice(m.group("target")) or line.lower().startswith(("select", "choose", "pick"))):
            target = build_target(m.group("target"), "select")
            if not target.preferred_roles:
                target.preferred_roles = ["combobox", "listbox", "radiogroup"]
            option = strip_quotes(m.group("opt"))
            target.option_text = option
            step = self._step(ctx, line, ActionType.SELECT, 0.92, target=target, value=parse_value(m.group("opt"), ctx),
                              risk_hint="input")
            return ParsedLine([step], 0.92, "select target as option")
        # "Select India, Kerala and Kochi" -> map options onto matching dropdowns at runtime
        m = re.match(r"^(?:select|choose|pick)\s+(?P<opts>.+)$", line, re.IGNORECASE)
        if m:
            phrase = m.group("opts")
            if re.search(r"\b(button|link|tab|menu|icon)\b", phrase, re.IGNORECASE):
                return None  # click rule
            options = split_list(phrase)
            steps = []
            for option in options:
                category = GAZETTEER.get(option.lower())
                target = TargetSpec(intent=f"{category or 'choice'} containing {option}", option_text=option,
                                    preferred_roles=["combobox", "listbox", "radio", "option", "tab", "button"],
                                    purpose=category, semantic_hints=synonyms_for(category) if category else [])
                step = self._step(ctx, line, ActionType.SELECT, 0.85 if category else 0.78, target=target,
                                  value=ValueSpec(source="literal", value=option), risk_hint="input",
                                  notes=["flexible_choice"])
                steps.append(step)
            return ParsedLine(steps, min(s.confidence for s in steps), "select options by mapping")
        return None

    def _rule_open_page(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:open|go\s+to|navigate\s+to|visit|access|land\s+on|load|be\s+on)\s+(?:the\s+)?(?P<page>.+?)"
                     r"\s+(?:page|screen|view|section|area)(?:\s+of\s+the\s+(?:application|app|site|website))?$", line,
                     re.IGNORECASE)
        if not m:
            m = re.match(r"^(?:go\s+to|navigate\s+to|open|visit)\s+(?:the\s+)?(?P<page>[\w\s]+)$", line, re.IGNORECASE)
            if not m or normalize_text(m.group("page")) not in PAGE_NOUNS:
                return None
        page = strip_quotes(m.group("page").strip())
        purpose_match = classify_phrase(page, kinds=("action",))
        target = TargetSpec(intent=f"{page} page", exact_name=None, preferred_roles=["link", "button", "menuitem", "tab"],
                            semantic_hints=[page] + (synonyms_for(purpose_match.purpose) if purpose_match else []),
                            purpose=purpose_match.purpose if purpose_match else None)
        step = self._step(ctx, line, ActionType.OPEN_PAGE, 0.88, target=target, risk_hint="navigation")
        return ParsedLine([step], 0.88, "open semantic page")

    def _rule_logout_login_bare(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?P<what>log\s*out|sign\s*out|logout|log\s*off|log\s*in|sign\s*in|login|sign\s*up|register)"
                     r"(?:\s+of\s+the\s+(?:application|app|site))?$", line, re.IGNORECASE)
        if not m:
            return None
        purpose = classify_phrase(m.group("what"), kinds=("action",))
        target = target_for_purpose(purpose.purpose if purpose else "submit_generic")
        step = self._step(ctx, line, ActionType.CLICK, 0.9, target=target, expects_navigation=True, risk_hint="submit")
        return ParsedLine([step], 0.9, "bare auth action")

    def _rule_submit(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^submit\s+(?:the\s+)?(?P<what>.*?)\s*(?:form)?$", line, re.IGNORECASE)
        if not m:
            return None
        what = (m.group("what") or "").strip()
        target = TargetSpec(intent=f"submit {what} form".replace("  ", " ").strip(), preferred_roles=["button"],
                            purpose="submit_generic", element_kind="button",
                            semantic_hints=synonyms_for("submit_generic") + synonyms_for("submit_login")[:3],
                            context_hint=f"{what} form" if what else None)
        step = self._step(ctx, line, ActionType.CLICK, 0.88, target=target, expects_navigation=True,
                          risk_hint="submit", notes=["submit_form"])
        return ParsedLine([step], 0.88, "submit form")

    def _rule_click(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:click|tap|press|hit|push|select|choose|activate|follow)\s+(?:on\s+|the\s+link\s+)?"
                     r"(?:the\s+)?(?P<target>.+)$", line, re.IGNORECASE)
        if not m:
            return None
        phrase = m.group("target")
        expects_download = bool(re.search(r"\bto\s+download\b", phrase, re.IGNORECASE))
        phrase = re.sub(r"\s+to\s+download\b.*$", "", phrase, flags=re.IGNORECASE)
        # "Click on Record Interaction to open sidebar": the clause says why, not what to click
        phrase = re.sub(r"\s+(?:in\s+order\s+)?to\s+(?:open|close|show|display|view|expand|collapse|reveal|launch|"
                        r"start|bring\s+up|go\s+to|navigate\s+to|see|load)\b.*$", "", phrase, flags=re.IGNORECASE) or phrase
        target = build_target(phrase, "click")
        risk = risk_for_text(target.exact_name or phrase) or "input"
        expects_nav = bool(target.purpose in ("submit_login", "submit_register", "logout", "checkout")
                           or "link" in target.preferred_roles[:1])
        confidence = 0.95 if (target.exact_name or target.element_kind) else 0.9
        step = self._step(ctx, line, ActionType.CLICK, confidence, target=target, risk_hint=risk,  # type: ignore[arg-type]
                          expects_navigation=expects_nav, expects_download=expects_download)
        return ParsedLine([step], confidence, "click")

    def _rule_open_click(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:open|expand|collapse|close|show|reveal|launch)\s+(?:the\s+)?(?P<target>.+)$", line,
                     re.IGNORECASE)
        if not m:
            return None
        verb = line.split()[0].lower()
        phrase = m.group("target")
        target = build_target(phrase, "click")
        if verb == "close" and not target.exact_name:
            target.context_hint = target.context_hint or phrase
            target.semantic_hints = list(dict.fromkeys(["close", "dismiss", "×", "x", "cancel", *target.semantic_hints]))
            target.intent = f"close {phrase}"
            target.purpose = "close"
            target.preferred_roles = ["button"]
        step = self._step(ctx, line, ActionType.CLICK, 0.82, target=target, risk_hint="input",
                          notes=[f"verb:{verb}"])
        return ParsedLine([step], 0.82, "open/expand/close via click")

    def _rule_type_bare(self, line: str, ctx: ParseContext) -> ParsedLine | None:
        m = re.match(r"^(?:type|enter|input)\s+(?P<value>\"[^\"]*\"|'[^']*'|\{\{\s*\w+\s*\}\})$", line, re.IGNORECASE)
        if not m:
            return None
        value = parse_value(m.group("value"), ctx)
        step = self._step(ctx, line, ActionType.TYPE, 0.8, target=TargetSpec(intent="focused element"), value=value,
                          risk_hint="input", notes=["focused_element"])
        return ParsedLine([step], 0.8, "type into focused element")


# ---------------------------------------------------------------------------
# reusable parsing helpers
# ---------------------------------------------------------------------------

def clean_line(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = LEADING_ENUM_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    # strip a single trailing sentence terminator, but never from inside a URL/email/quote
    if text.endswith((".", ";", "!")) and not re.search(r"(https?://\S+|\S+@\S+)\.$", text[:-1] + "x"):
        text = text[:-1].rstrip()
    elif text.endswith(".") and URL_RE.search(text):
        text = text[:-1]
    return text


def leading_verb(text: str) -> str | None:
    m = re.match(r"^(enter|type|input|fill(?:\s+in)?|click(?:\s+on)?|select|choose|check|uncheck|verify|ensure|"
                 r"press|tap|hover\s+over|open|navigate\s+to|go\s+to|set)\b", text.strip(), re.IGNORECASE)
    return m.group(1) if m else None


_COMPOUND_VERBS = (r"(?:click|press|enter|type|fill|select|choose|pick|check|uncheck|tick|verify|ensure|confirm|"
                   r"validate|assert|tap|hover|wait|submit|go|open|navigate|scroll|upload|drag|accept|dismiss|close|"
                   r"clear|switch|take|log\s*in|log\s*out|sign\s*in|sign\s*out|search|download|expand|store|save)")


def split_compound(line: str) -> list[str]:
    protected = QUOTE_RE.sub(lambda m: m.group(0).replace(" and ", " \x00and\x00 ").replace(",", "\x01")
                             .replace(" then ", " \x00then\x00 "), line)
    # a verb only starts a new instruction when it has an object: "Click Yes, save" is one button label
    parts = re.split(rf"\s*(?:,?\s+and\s+then\s+|,\s*then\s+|;\s*|\s+then\s+(?={_COMPOUND_VERBS}\s+\S)|"
                     rf",?\s+and\s+(?={_COMPOUND_VERBS}\s+\S)|,\s*(?={_COMPOUND_VERBS}\s+\S))", protected,
                     flags=re.IGNORECASE)
    if len(parts) == 1:
        # "Enter username as abc and password as xyz"
        parts = re.split(r"\s+and\s+(?=(?:the\s+|your\s+)?[\w\s]{2,30}?\s+(?:as|with|to)\s+)", protected,
                         flags=re.IGNORECASE)
        if len(parts) > 1 and not re.match(r"^(?:enter|type|input|fill|set|select|choose|provide)\b", line,
                                           re.IGNORECASE):
            parts = [protected]
    restored = [p.replace(" \x00and\x00 ", " and ").replace(" \x00then\x00 ", " then ").replace("\x01", ",").strip()
                for p in parts if p and p.strip()]
    return restored if len(restored) > 1 else [line]


def strip_quotes(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip().rstrip(".,;")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        return text[1:-1]
    return text


COLOR_NAME_RE = (r"(?:light\s+|dark\s+|pale\s+|bright\s+|sky\s+|navy\s+|royal\s+|lime\s+|forest\s+|olive\s+|"
                 r"hot\s+|deep\s+)?(?:red|green|blue|yellow|orange|purple|violet|pink|grey|gray|black|white|brown|cyan|"
                 r"magenta|teal|maroon|navy|olive|lime|gold|silver|beige|turquoise|indigo|crimson|amber)")


def grid_target(column: str, hint: str, row: int | None = None) -> TargetSpec:
    """The input cell of a header-and-values grid (Range (%)): resolved by geometry, not by its name."""

    return TargetSpec(intent=f"range input {column}", exact_name=column, preferred_roles=["textbox", "spinbutton"],
                      element_kind="input field", ordinal=row, position_hint=hint, semantic_hints=[column, "range"])


def column_range(first: str, last: str) -> list[str]:
    """'A'..'K' -> A, B, ... K; '1'..'11' -> 1 ... 11 (a run of column headers)."""

    if first.isdigit() and last.isdigit():
        lo, hi = int(first), int(last)
        return [str(n) for n in range(lo, hi + 1)] if 0 < hi - lo < 60 else []
    if len(first) == 1 and len(last) == 1 and first.isalpha() and last.isalpha():
        lo, hi = ord(first.upper()), ord(last.upper())
        return [chr(c) for c in range(lo, hi + 1)] if 0 < hi - lo < 26 else []
    return []


def clean_option(text: str, *, keep_quotes: bool = False) -> str:
    """'"10" option' -> '10'; "the option Yes" -> "Yes": the option's label without the words around it."""

    raw = re.sub(r"^(?:the\s+)?(?:option\s+)?", "", (text or "").strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s+(?:option|value|item|entry|choice)$", "", raw, flags=re.IGNORECASE).strip()
    return raw if keep_quotes else strip_quotes(raw)


def quoted_choices(text: str) -> list[str]:
    """'"202404" and "202409"' -> ['202404', '202409'] (two or more separately quoted choices)."""

    items = [next(g for g in m.groups() if g is not None) for m in QUOTE_RE.finditer(text or "")]
    rest = QUOTE_RE.sub("", text or "")
    if len(items) >= 2 and re.fullmatch(r"[\s,;/&]*(?:(?:and|or|to|&|,)\s*)*[\s,;/&]*", rest, re.IGNORECASE):
        return items
    return []


def outer_quoted(text: str) -> str | None:
    """'the checkbox "Put a "top" on the axis range"' -> 'Put a "top" on the axis range' (nested quotes).

    Only when the quotes are nested inside one name: separate quoted items ('"A" and "B"') are not merged."""

    m = re.search(r'"(.+)"', text or "")
    if m is None or m.group(1).count('"') != 2:
        return None
    inner = m.group(1)
    before, middle, after = inner.split('"')
    # nested: 'Put a "top" on the axis' - the inner word sits flush against its quotes inside spaced text.
    # separate items: '"Add more" link next to "Upload Documents"' - spaced connector text between two quotes.
    nested = (bool(middle) and middle == middle.strip() and (before == "" or before.endswith(" "))
              and (after == "" or after.startswith(" ")) and bool(before.strip() or after.strip()))
    return inner if nested else None


def is_explicit_value(text: str | None) -> bool:
    if not text:
        return False
    text = text.strip()
    return bool(QUOTE_RE.fullmatch(text) or VAR_RE.fullmatch(text) or re.fullmatch(r"\S+@\S+\.\S+", text)
                or re.fullmatch(r"\+?\d[\d\s()-]*", text))


def looks_like_field(phrase: str | None) -> bool:
    if not phrase:
        return False
    norm = normalize_text(phrase)
    if re.search(r"\b(field|input|box|textbox|text box|textarea|text area|area)\b", norm):
        return True
    match = classify_phrase(phrase)
    return bool(match and match.score >= 0.75 and PURPOSES[match.purpose].kind == "field")


def looks_like_choice(phrase: str | None) -> bool:
    if not phrase:
        return False
    norm = normalize_text(phrase)
    if re.search(r"\b(dropdown|drop down|select|combo ?box|list|picker|menu|option|radio)\b", norm):
        return True
    if re.search(r"\b(country|state|province|city|gender|type|category|size|color|colour|language|month|year|day|"
                 r"title|currency|plan|role|status|priority|department|level|sort|time ?zone|region)\b", norm):
        return True
    return False


def split_list(phrase: str) -> list[str]:
    phrase = phrase.strip().rstrip(".")
    quoted = [next(g for g in m.groups() if g is not None) for m in QUOTE_RE.finditer(phrase)]
    if len(quoted) >= 2:
        return quoted
    items = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", phrase)
    return [strip_quotes(i) for i in items if i.strip()]


def normalize_key(raw: str) -> str:
    raw = raw.strip()
    if "+" in raw:
        parts = [p.strip() for p in raw.split("+")]
        mods = [MODIFIER_ALIASES.get(p.lower(), p) for p in parts[:-1]]
        last = parts[-1]
        last = KEY_ALIASES.get(last.lower(), last if len(last) > 1 else last.lower())
        return "+".join([*mods, last])
    norm = re.sub(r"\s+", " ", raw.lower())
    if norm in KEY_ALIASES:
        return KEY_ALIASES[norm]
    if re.fullmatch(r"f\d{1,2}", norm):
        return norm.upper()
    return raw


def normalize_should(pred: str) -> str:
    pred = re.sub(r"^be\s+", "is ", pred, flags=re.IGNORECASE)
    pred = re.sub(r"^not\s+be\s+", "is not ", pred, flags=re.IGNORECASE)
    pred = re.sub(r"^(see|show|display|contain|have)\s+", lambda m: m.group(1) + "s ", pred, flags=re.IGNORECASE)
    pred = re.sub(r"^(appear|disappear|load)\b", lambda m: m.group(1) + "s", pred, flags=re.IGNORECASE)
    return pred


def parse_value(raw: str | None, ctx: ParseContext, *, purpose: str | None = None) -> ValueSpec:
    text = (raw or "").strip().rstrip(".,;")
    sensitive_purpose = purpose in SENSITIVE_PURPOSES if purpose else False
    var = VAR_RE.fullmatch(text)
    if var:
        name = var.group(1)
        is_secret = name in ctx.secret_names or bool(SECRET_NAME_RE.search(name)) or sensitive_purpose
        if name.startswith(("generated_", "random_")) or name in ("current_date", "timestamp", "today", "uuid"):
            return ValueSpec(source="generated", reference=name, generator=name)
        return ValueSpec(source="secret" if is_secret else "variable", reference=name,
                         sensitive=is_secret or sensitive_purpose)
    if VAR_RE.search(text):
        # template mixing literals and variables, e.g. "user_{{run_id}}@example.com"
        return ValueSpec(source="literal", value=text, sensitive=sensitive_purpose, generator="template")
    literal = strip_quotes(text)
    return ValueSpec(source="literal", value=literal, sensitive=sensitive_purpose)


def target_for_purpose(purpose: str) -> TargetSpec:
    spec = PURPOSES[purpose]
    return TargetSpec(intent=spec.phrases[0], preferred_roles=list(spec.roles), semantic_hints=list(spec.phrases),
                      purpose=purpose, input_type=spec.input_types[0] if purpose in ("password", "confirm_password",
                                                                                      "email") else None,
                      element_kind="button" if spec.kind == "action" else None)


def build_target(phrase: str | None, action: str) -> TargetSpec:
    """Convert a target noun phrase into a semantic TargetSpec."""

    raw = (phrase or "").strip().rstrip(".,;")
    exact_name: str | None = None
    nested = outer_quoted(raw)
    if nested is not None:
        # 'Put a "top" on the horizontal axis range' is one name with quotes inside it
        exact_name = nested
        raw = raw.replace(f'"{nested}"', ' "__NESTED__" ')
    quoted = [next(g for g in m.groups() if g is not None) for m in QUOTE_RE.finditer(raw)]
    if quoted and exact_name is None:
        exact_name = quoted[0]
    work = QUOTE_RE.sub(lambda m: " __Q__ ", raw) if quoted else raw

    context_hint = None
    ctx_m = re.search(r"\s+(?:in|inside|within|on|under|from|of)\s+(?:the\s+)?(?P<ctx>[\w\s'-]+?\s+(?:form|section|"
                      r"dialog|modal|popup|pop-up|panel|header|footer|menu|navigation|nav|navbar|sidebar|card|row|"
                      r"table|frame|iframe|list|column|toolbar|banner|page|area|region|drawer|tab|canvas|map|chart|board|editor|"
                      r"diagram|graph|grid|carousel))$", work, re.IGNORECASE)
    relation: str | None = None
    anchor: str | None = None
    if ctx_m:
        context_hint = DETERMINERS_RE.sub("", ctx_m.group("ctx").strip())
        work = work[: ctx_m.start()]
    from backend.agents.grounding.spatial import split_relation

    split = split_relation(work)
    if split is not None and not ctx_m:
        head, relation, anchor_text = split
        if "__Q__" in anchor_text:
            index = head.count("__Q__")
            anchor_text = anchor_text.replace("__Q__", quoted[index] if index < len(quoted) else "", 1)
        anchor = strip_quotes(re.sub(r"\s+", " ", anchor_text).strip()) or None
        if anchor:
            work = head
            if "__Q__" not in head and nested is None:
                exact_name = None  # 'the input field below "01"': the quote names the anchor, not the target
            elif "__Q__" in head:
                exact_name = quoted[0]
        else:
            relation = None
    if relation is None:
        ctx_m = re.search(r"\s+(?:for)\s+(?:the\s+)?(?P<ctx>.+)$", work, re.IGNORECASE)
        if ctx_m and action in ("click", "check", "uncheck", "hover", "toggle"):
            context_hint = strip_quotes(ctx_m.group("ctx").strip())
            work = work[: ctx_m.start()]

    position_hint = None
    pos_m = re.search(r"\s+(?:at|in|on)\s+the\s+(?P<pos>top|bottom)?\s*(?P<side>left|right|center|middle)?"
                      r"(?:\s+corner)?(?:\s+of\s+the\s+(?:page|screen))?$", work, re.IGNORECASE)
    if pos_m and (pos_m.group("pos") or pos_m.group("side")):
        position_hint = " ".join(p for p in (pos_m.group("pos"), pos_m.group("side")) if p).lower()
        work = work[: pos_m.start()]

    work = DETERMINERS_RE.sub("", work.strip())
    ordinal = None
    ord_m = re.match(r"^(?P<ord>first|second|third|fourth|fifth|last|final|1st|2nd|3rd|4th|5th)\s+", work, re.IGNORECASE)
    if ord_m:
        ordinal = ORDINALS[ord_m.group("ord").lower()]
        work = work[ord_m.end():]

    element_kind = None
    preferred_roles: list[str] = []
    kinds = list(KIND_NOUN_RE.finditer(work))
    if kinds:
        kind_m = kinds[-1]
        tail = work[kind_m.end():].strip()
        if not tail or tail in ("__Q__",) or len(tail.split()) <= 2:
            element_kind = kind_m.group(1).lower()
            preferred_roles = list(KIND_ROLES.get(element_kind, ()))
            remainder = (work[: kind_m.start()] + " " + tail).strip()
            # "the link 'Forgot password'" or "the Sign In button"
            if remainder and remainder != "__Q__":
                work = remainder
            elif exact_name:
                work = exact_name
            else:
                work = remainder or element_kind
    work = work.replace("__Q__", exact_name or "").strip()
    work = re.sub(r"\s+", " ", work).strip(" -:")

    visual_tokens = [w for w in normalize_text(work).split()
                     if (w in COLOR_WORDS or w in SHAPE_WORDS) and w != (element_kind or "")]
    visual_hint = " ".join(visual_tokens) if visual_tokens else None
    if element_kind == "icon":
        visual_hint = work.strip() or "icon"

    name_text = exact_name or work
    if action in ("drag", "drop"):
        name_text = ""  # draggable items and drop zones are content, not form controls with purposes
    preferred_kinds: tuple[str, ...] | None = None
    if element_kind in ("field", "input", "textbox", "text field", "text box", "input field", "textarea"):
        preferred_kinds = ("field",)
    elif element_kind in ("button", "link", "icon", "menu", "menu item", "tab"):
        preferred_kinds = ("action",)
    elif element_kind in ("checkbox", "check box", "toggle", "switch", "radio", "radio button"):
        preferred_kinds = ("toggle",)
    elif action in ("fill", "type", "clear", "upload"):
        preferred_kinds = ("field",)
    elif action in ("click", "double_click", "download"):
        preferred_kinds = ("action",)
    elif action in ("check", "uncheck", "toggle"):
        preferred_kinds = ("toggle",)
    purpose = classify_phrase(name_text, kinds=preferred_kinds) if preferred_kinds else None
    if purpose is None or purpose.score < 0.6:
        fallback = classify_phrase(name_text)
        # a cross-kind purpose is only accepted for read-only uses, or for exact phrase matches when the
        # action can operate that kind of control ("Click Terms of Service" is a link, not the terms checkbox)
        cross_kind_ok = action not in ("click", "double_click", "download") or \
            (fallback is not None and PURPOSES[fallback.purpose].kind in ("action", "page"))
        if fallback and (preferred_kinds is None or (fallback.score >= 0.99 and cross_kind_ok)):
            purpose = fallback
    purpose_name = purpose.purpose if purpose and purpose.score >= 0.6 else None

    if not preferred_roles and purpose_name:
        if action in ("fill", "type", "clear"):
            preferred_roles = [r for r in PURPOSES[purpose_name].roles if r in ("textbox", "searchbox", "combobox",
                                                                                 "spinbutton")] or ["textbox"]
        elif action in ("click",) and PURPOSES[purpose_name].kind == "action":
            preferred_roles = list(PURPOSES[purpose_name].roles)

    if action == "upload":
        preferred_roles = ["button", "textbox", "generic"]
    if context_hint and re.search(r"\b(canvas|map|chart|graph|diagram|board|editor)\b", context_hint, re.IGNORECASE):
        preferred_roles = ["canvas", "img", "generic", *[r for r in preferred_roles if r not in ("canvas", "img",
                                                                                                 "generic")]]
        visual_hint = visual_hint or work
    if not preferred_roles and action in ("fill", "type", "clear"):
        preferred_roles = ["textbox", "searchbox", "combobox", "spinbutton"]
    if not preferred_roles and action == "click":
        preferred_roles = ["button", "link", "menuitem", "tab"]

    input_type = None
    if purpose_name in ("password", "confirm_password"):
        input_type = "password"
    elif purpose_name == "email" and action in ("fill", "type"):
        input_type = "email"

    hints = [h for h in dict.fromkeys([work] + (synonyms_for(purpose_name) if purpose_name else [])) if h]
    return TargetSpec(
        intent=(work or exact_name or raw).strip(),
        exact_name=exact_name,
        preferred_roles=preferred_roles,
        semantic_hints=hints[:14],
        element_kind=element_kind,
        input_type=input_type,
        purpose=purpose_name,
        context_hint=context_hint,
        ordinal=ordinal,
        position_hint=position_hint,
        visual_hint=visual_hint,
        relation=relation,
        anchor=anchor,
    )


# ---------------------------------------------------------------------------
# assertions & conditions
# ---------------------------------------------------------------------------

def _text_target(text: str) -> TargetSpec:
    return TargetSpec(intent=text, exact_name=text)


def parse_assertion(cond: str, ctx: ParseContext) -> tuple[AssertionSpec | None, float]:
    text = cond.strip().rstrip(".")
    low = text.lower()
    quoted = [next(g for g in m.groups() if g is not None) for m in QUOTE_RE.finditer(text)]

    def a(type_: AssertionType, confidence: float, **kw: Any) -> tuple[AssertionSpec, float]:
        return AssertionSpec(type=type_, description=text, **kw), confidence

    # Range (%) grids: the value / state of the input cell under a column header (resolved spatially)
    q = r"(?:\"[^\"]+\"|'[^']+')"
    m = re.match(rf"^(?:the\s+)?values?\s+(?:below|under)\s+(?:columns?\s+)?(?P<cols>{q}(?:\s*(?:,|and)\s*{q})*)\s+(?:is|are)\s+"
                 rf"(?:already\s+|still\s+|now\s+)?(?P<vals>{q}(?:\s*(?:,|and)\s*{q})*)$", text, re.IGNORECASE)
    if m:
        cols = [strip_quotes(x.group(0)) for x in QUOTE_RE.finditer(m.group("cols"))]
        vals = [strip_quotes(x.group(0)) for x in QUOTE_RE.finditer(m.group("vals"))]
        if len(cols) == len(vals):
            return a(AssertionType.ALL_OF, 0.9, expected=dict(zip(cols, vals, strict=True)), children=[
                AssertionSpec(type=AssertionType.INPUT_VALUE_EQUALS, description=f"value below {c} is {v}", expected=v,
                              target=grid_target(c, "below_label")) for c, v in zip(cols, vals, strict=True)])
    m = re.match(rf"^(?:the\s+)?(?:value\s+)?(?P<v>{q})\s+is\s+(?:displayed|shown|entered|present)\s+in\s+(?:the\s+)?.*?"
                 rf"(?:input|field|box|cell)\s+(?:under|below)\s+column\s+(?P<c>{q})$", text, re.IGNORECASE)
    if m:
        return a(AssertionType.INPUT_VALUE_EQUALS, 0.9, expected=strip_quotes(m.group("v")),
                 target=grid_target(strip_quotes(m.group("c")), "below_label"))
    m = re.match(rf"^(?:the\s+)?.*?(?:input|field|box|cell)\s+(?:under|below)\s+column\s+(?P<c>{q})\s+is\s+"
                 r"(?P<state>disabled|enabled|editable|read-?only|not\s+editable)(?:\s+and\s+(?:can|cannot|can't|can\s+not)"
                 r"\s+be\s+edited)?$", text, re.IGNORECASE)
    if m:
        state = m.group("state").lower()
        t = AssertionType.ELEMENT_ENABLED if state in ("enabled", "editable") else AssertionType.ELEMENT_DISABLED
        return a(t, 0.9, target=grid_target(strip_quotes(m.group("c")), "below_label"))
    m = re.match(r"^(?:each|every|all(?:\s+the)?)\s+(?:range\s+|input\s+|value\s+)?(?:fields?|inputs?|boxes|box|cells?)\s+"
                 r"(?:corresponding\s+to|for|under|below|of)\s+(?:columns?\s+)?(?P<a>[A-Za-z]|\d{1,2})\s+(?:through|to|-|~|until)\s+"
                 r"(?P<b>[A-Za-z]|\d{1,2})\s+(?:is|are)\s+(?P<state>enabled(?:\s+and\s+editable)?|editable|disabled|read-?only)$",
                 text, re.IGNORECASE)
    if m:
        cols = column_range(m.group("a"), m.group("b"))
        state = m.group("state").lower()
        t = AssertionType.ELEMENT_DISABLED if state in ("disabled", "readonly", "read-only") else AssertionType.ELEMENT_ENABLED
        if cols:
            return a(AssertionType.ALL_OF, 0.88, expected=f"{cols[0]}..{cols[-1]} {state}", children=[
                AssertionSpec(type=t, description=f"field under {c} is {state}", target=grid_target(c, "below_label"))
                for c in cols])

    # "1,2,3 are in the sky blue color" / "the Total cell is highlighted in red": a colour on screen
    m = re.match(r"^(?:the\s+)?(?:values?\s+|cells?\s+|boxes\s+|numbers?\s+)?(?P<items>.+?)\s+(?:is|are)\s+(?:shown\s+|displayed\s+|"
                 r"highlighted\s+|marked\s+|coloured\s+|colored\s+|painted\s+)?(?:in\s+)?(?:the\s+|a\s+)?(?P<color>"
                 + COLOR_NAME_RE + r")\s*(?:colou?r(?:ed)?|background|text|font)?$", low)
    if m:
        items = [i for i in (strip_quotes(x) for x in re.split(r"\s*(?:,|\band\b|&)\s*", text[m.start("items"):m.end("items")]))
                 if i]
        return a(AssertionType.VISUAL_STATE, 0.9, expected={"items": items, "color": m.group("color").strip()})
    # "Data Perid is 202404 to 202409", "the Data period field shows 202404 ~ 202409": a from-to range in a field
    m = re.match(r"^(?:the\s+)?(?P<field>\"[^\"]+\"|'[^']+'|.+?)\s+(?:field\s+|dropdowns?\s+)?(?:is|shows|displays|"
                 r"is set to|has)\s+(?:the\s+range\s+|from\s+)?(?P<a>\"[^\"]+\"|'[^']+'|\S+)\s+(?:to|~|～|-|through|until)\s+"
                 r"(?P<b>\"[^\"]+\"|'[^']+'|\S+)$", text, re.IGNORECASE)
    if m and not re.search(r"\b(page|url|title)\b", m.group("field"), re.IGNORECASE):
        field = strip_quotes(m.group("field"))
        first, second = strip_quotes(m.group("a")), strip_quotes(m.group("b"))
        children = [AssertionSpec(type=AssertionType.TEXT_VISIBLE, description=f"{field} shows {v}", expected=v,
                                  target=build_target(f"{field} field", "select"), match="case_insensitive")
                    for v in (first, second)]
        return a(AssertionType.ALL_OF, 0.88, expected=f"{first} to {second}", children=children)
    # "the total number of range columns displayed is 11", "there are 5 rows in the results table"
    m = re.match(r"^(?:the\s+)?(?:total\s+)?(?:number|count|no\.?)\s+of\s+(?P<what>.+?)\s+(?:displayed\s+|shown\s+|visible\s+|"
                 r"present\s+|on\s+the\s+page\s+)?(?:is|are|equals|=)\s+(?P<n>\d+)$|^there\s+(?:is|are)\s+(?:exactly\s+)?"
                 r"(?P<n2>\d+)\s+(?P<what2>.+?)(?:\s+(?:displayed|shown|visible|present))?$", low)
    if m:
        what = (m.group("what") or m.group("what2") or "").strip()
        count = int(m.group("n") or m.group("n2"))
        return a(AssertionType.ELEMENT_COUNT, 0.86, expected=count, target=build_target(what, "hover"))

    # "Verify that the page is displayed" names no destination. It must not become
    # "URL contains 'the'" / "heading 'the' visible".
    if re.fullmatch(r"(?:the\s+)?(?:page|screen)\s+(?:is\s+)?(?:fully\s+)?"
                    r"(?:loaded|loads|displayed|rendered|shown|visible|opened|opens)", low):
        return a(AssertionType.PAGE_LOADED, 0.95)

    # derived identity expectation: "the dashboard displays the logged-in user's name"
    if re.search(r"(logged[- ]in|current|signed[- ]in)\s+user'?s?\s+(?:user)?name|user'?s\s+(?:user)?name|"
                 r"name of the (?:logged[- ]in )?user|welcome message with (?:the )?(?:user'?s )?name", low):
        source = ctx.last_values.get("full_name") or ctx.last_values.get("username_or_email") or \
            ctx.last_values.get("email")
        if source is not None and not source.sensitive:
            return a(AssertionType.TEXT_VISIBLE, 0.7, expected_ref=source.model_copy(), derived=True,
                     match="case_insensitive")
        return a(AssertionType.TEXT_VISIBLE, 0.5, expected_ref=ValueSpec(source="variable", reference="display_name"),
                 derived=True, match="case_insensitive")

    m = re.match(r"^(?:the\s+)?(?:user\s+|page\s+|browser\s+|application\s+|app\s+)?(?:is|gets|was|has been|should be)?"
                 r"\s*(?:redirected|navigated|taken|sent|brought|forwarded|routed)\s+(?:back\s+)?to\s+(?:the\s+)?"
                 r"(?P<page>.+?)(?:\s+page|\s+screen)?$", low)
    if m or re.match(r"^(?:the\s+)?(?P<page>[\w\s]+?)\s+(?:page|screen)\s+(?:is\s+)?(?:opens?|is\s+opened|loads?|is\s+"
                     r"loaded|is\s+displayed|appears?|is\s+shown|is\s+visible)$", low):
        page = (m.group("page") if m else re.match(r"^(?:the\s+)?(?P<page>[\w\s]+?)\s+(?:page|screen)", low).group("page"))  # type: ignore[union-attr]
        page = page.strip()
        if page.lower() in {"the", "a", "an", "this", "that", "page", "screen", "it"}:
            return a(AssertionType.PAGE_LOADED, 0.95)
        if URL_RE.match(page) or page.startswith("/"):
            return a(AssertionType.URL_CONTAINS, 0.95, expected=page)
        slug = re.sub(r"\s+", "-", page)
        children = [
            AssertionSpec(type=AssertionType.URL_CONTAINS, expected=slug, description=f"URL contains '{slug}'"),
            AssertionSpec(type=AssertionType.PAGE_TITLE_CONTAINS, expected=page, match="case_insensitive",
                          description=f"title contains '{page}'"),
            AssertionSpec(type=AssertionType.ELEMENT_VISIBLE, description=f"heading '{page}' visible",
                          target=TargetSpec(intent=page, exact_name=page, preferred_roles=["heading"])),
        ]
        return a(AssertionType.ANY_OF, 0.88, children=children, expected=page)

    m = re.match(r"^(?:the\s+)?(?:current\s+|page\s+)?url\s+(?:is|equals|=|should be|matches)\s+(?P<v>.+)$", low)
    if m:
        return a(AssertionType.URL_EQUALS, 0.95, expected=strip_quotes(text[m.start("v"):]))
    m = re.match(r"^(?:the\s+)?(?:current\s+|page\s+)?url\s+(?:contains|includes|has|ends with)\s+(?P<v>.+)$", low)
    if m:
        return a(AssertionType.URL_CONTAINS, 0.95, expected=strip_quotes(text[m.start("v"):]))
    m = re.match(r"^(?:the\s+)?(?:page\s+|browser\s+|tab\s+)?title\s+(?:is|equals|=|should be)\s+(?P<v>.+)$", low)
    if m:
        return a(AssertionType.PAGE_TITLE_EQUALS, 0.95, expected=strip_quotes(text[m.start("v"):]), match="exact")
    m = re.match(r"^(?:the\s+)?(?:page\s+|browser\s+|tab\s+)?title\s+(?:contains|includes|has)\s+(?P<v>.+)$", low)
    if m:
        return a(AssertionType.PAGE_TITLE_CONTAINS, 0.95, expected=strip_quotes(text[m.start("v"):]),
                 match="case_insensitive")
    if re.match(r"^(?:the\s+)?(?:file\s+)?(?:download|file)\s+(?:starts|started|begins|is\s+downloaded|is\s+saved|"
                r"completes)", low) or re.match(r"^(?:a\s+)?file\s+(?:is\s+)?downloaded", low):
        return a(AssertionType.DOWNLOAD_STARTED, 0.92)
    m = re.match(r"^(?:the\s+)?(?:user\s+is\s+)?(?:successfully\s+)?(?:logged|signed)\s+(?P<state>in|out)$", low)
    if m:
        logged_in = m.group("state") == "in"
        children = [
            AssertionSpec(type=AssertionType.ELEMENT_VISIBLE, description="logout control visible",
                          target=TargetSpec(intent="logout", purpose="logout", semantic_hints=synonyms_for("logout"),
                                            preferred_roles=["button", "link", "menuitem"]), negate=not logged_in),
            AssertionSpec(type=AssertionType.ELEMENT_NOT_VISIBLE, description="password field absent",
                          target=TargetSpec(intent="password", purpose="password", preferred_roles=["textbox"],
                                            input_type="password"), negate=not logged_in),
        ]
        return a(AssertionType.ALL_OF if logged_in else AssertionType.ANY_OF, 0.78, children=children,
                 expected="logged in" if logged_in else "logged out")
    m = re.match(r"^(?:a|an|the)?\s*(?P<kind>dialog|modal|popup|pop-up|alert|confirmation)\s+(?:box\s+)?(?:is\s+)?"
                 r"(?:appears|appear|shown|displayed|visible|opens|is\s+shown|is\s+displayed)"
                 r"(?:\s+with\s+(?:the\s+)?(?:text|message)?\s*(?P<text>.+))?$", low)
    if m:
        expected = strip_quotes(text[m.start("text"):]) if m.group("text") else None
        return a(AssertionType.DIALOG_VISIBLE, 0.9, expected=expected)
    m = re.match(r"^(?P<n>\d+)\s+(?P<what>[\w\s]+?)\s+(?:are|is)?\s*(?:displayed|shown|listed|visible|present|returned|"
                 r"found)$", low) or re.match(r"^(?:the\s+)?(?P<what>[\w\s]+?)\s+(?:has|have|contains|shows|lists)\s+"
                                         r"(?:exactly\s+)?(?P<n>\d+)\s+(?:rows|items|results|entries|records|products|"
                                         r"options|cards)$", low)
    if m:
        what = m.group("what")
        return a(AssertionType.ELEMENT_COUNT, 0.85, expected=int(m.group("n")),
                 target=TargetSpec(intent=what, preferred_roles=["row", "listitem", "article", "option"]))
    # "the DCF/DSF 7 digits checkbox is enabled" means it can be used (no longer disabled), not that it is ticked
    m = re.match(r"^(?:the\s+)?(?P<target>.+?)\s+(?:checkbox|check box|toggle|switch|radio(?:\s+button)?)\s+"
                 r"(?:is|should be)\s+(?P<state>enabled|disabled|editable|greyed out|grayed out)$", low)
    if m:
        target = build_target(text[: m.end("target")] + " " + low[m.end("target"):m.start("state")].split(" is")[0], "check")
        target.preferred_roles = ["checkbox", "switch", "radio"]
        state = m.group("state")
        t = AssertionType.ELEMENT_DISABLED if state in ("disabled", "greyed out", "grayed out") else AssertionType.ELEMENT_ENABLED
        return a(t, 0.9, target=target)
    m = re.match(r"^(?:the\s+)?(?P<target>.+?)\s+(?:checkbox|check box|toggle|switch)?\s*(?:is|should be)\s+"
                 r"(?P<neg>not\s+|un)?(?:checked|ticked|selected|on)$", low)
    if m and re.search(r"(checkbox|check box|toggle|switch|radio|remember|agree|terms|subscribe|newsletter|option)", low):
        negate = bool(m.group("neg"))
        target = build_target(text[: m.end("target")], "check")
        if re.search(r"\bradio\b", low):
            # "the radio button X is selected" asks whether that radio is on, not whether a dropdown
            # option named "selected" is chosen.
            target.preferred_roles = ["radio"]
            return a(AssertionType.CHECKBOX_UNCHECKED if negate else AssertionType.RADIO_SELECTED, 0.9, target=target)
        target.preferred_roles = target.preferred_roles or ["checkbox", "switch", "radio"]
        return a(AssertionType.CHECKBOX_UNCHECKED if negate else AssertionType.CHECKBOX_CHECKED, 0.9, target=target)
    m = re.match(r"^(?:the\s+)?(?P<target>.+?)\s+(?:button|link|field|input|control)?\s*(?:is|should be)\s+"
                 r"(?P<state>enabled|disabled|clickable|greyed out|grayed out|inactive)$", low)
    if m:
        disabled = m.group("state") in ("disabled", "greyed out", "grayed out", "inactive")
        target = build_target(text[: m.end("target")] + (" " + text[m.end("target"):].split()[0]
                                                        if text[m.end("target"):].split()[0].lower() in
                                                        ("button", "link", "field") else ""), "click")
        return a(AssertionType.ELEMENT_DISABLED if disabled else AssertionType.ELEMENT_ENABLED, 0.9, target=target)
    m = re.match(r"^(?P<opt>\"[^\"]+\"|'[^']+'|[\w\s]+?)\s+is\s+selected\s+(?:in|from|as)\s+(?:the\s+)?(?P<target>.+)$",
                 text, re.IGNORECASE)
    if m:
        return a(AssertionType.DROPDOWN_SELECTED, 0.9, expected=strip_quotes(m.group("opt")),
                 target=build_target(m.group("target"), "select"))
    # "Verify that \"Same Period Last Year\" is selected" names a radio, not a dropdown option called "selected".
    m = re.match(r"^(?:the\s+)?(?P<name>.+?)\s+is\s+selected$", text, re.IGNORECASE)
    if m and not re.search(r"\b(dropdown|combobox|list|menu|field)\b", low):
        name = strip_quotes(m.group("name"))
        target = TargetSpec(intent=name, exact_name=name, preferred_roles=["radio", "checkbox", "switch"],
                            element_kind="radio")
        return a(AssertionType.RADIO_SELECTED, 0.9, target=target)
    m = re.match(r"^(?:the\s+)?(?P<target>.+?)\s+(?:dropdown|select|combobox|field)?\s*(?:shows|has|is set to|displays|"
                 r"has the value|value is|contains the value|equals|is)\s+(?P<v>\"[^\"]+\"|'[^']+'|\{\{\s*\w+\s*\}\}|.+)$",
                 text, re.IGNORECASE)
    if m and looks_like_choice(m.group("target")) and not re.search(r"\b(visible|displayed|shown|present)\b", low):
        return a(AssertionType.DROPDOWN_SELECTED, 0.85, expected=strip_quotes(m.group("v")),
                 target=build_target(m.group("target"), "select"))
    m = re.match(r"^(?:the\s+)?(?P<target>.+?)\s+(?:field|input|textbox|box)\s+(?:contains|has|shows|has the value|"
                 r"value is|equals|is)\s+(?P<v>.+)$", text, re.IGNORECASE)
    if m:
        value = parse_value(m.group("v"), ctx)
        return a(AssertionType.INPUT_VALUE_EQUALS, 0.9, expected=value.value, expected_ref=value
                 if value.source != "literal" else None, target=build_target(m.group("target") + " field", "fill"),
                 match="exact")
    # error / success / validation message assertions
    m = re.match(r"^(?:an?\s+|the\s+)?(?P<kind>error|validation|success|warning|confirmation|info|information)\s+"
                 r"(?:message|text|banner|notification|toast|alert)?\s*(?P<text>\"[^\"]+\"|'[^']+')?\s*(?:saying|stating|"
                 r"with(?:\s+the)?(?:\s+text)?|that says|reading)?\s*(?P<text2>\"[^\"]+\"|'[^']+'|[^\"']+?)?\s*"
                 r"(?:is|are)?\s*(?:displayed|shown|visible|appears|appear|present|is\s+shown|is\s+displayed)?$", text,
                 re.IGNORECASE)
    if m and (m.group("text") or m.group("text2") or re.search(r"\b(displayed|shown|visible|appears|present)\b", low)):
        expected = strip_quotes(m.group("text") or m.group("text2") or "")
        expected = re.sub(r"\s+(?:is|are)\s+(?:displayed|shown|visible|present)$", "", expected, flags=re.IGNORECASE)
        kind = m.group("kind").lower()
        if expected:
            return a(AssertionType.TEXT_VISIBLE, 0.9, expected=expected, match="case_insensitive")
        return a(AssertionType.ELEMENT_VISIBLE, 0.75, target=TargetSpec(intent=f"{kind} message",
                                                                         preferred_roles=["alert", "status"],
                                                                         semantic_hints=[kind, "message"]))
    # text visibility (negative first)
    m = re.match(r"^(?P<subj>.+?)\s+(?:is|are)\s+(?:no\s+longer|not)\s+(?:displayed|visible|shown|present|on the page)$|"
                 r"^(?P<subj2>.+?)\s+(?:disappears|vanishes|is\s+gone|is\s+hidden|are\s+hidden)$|"
                 r"^(?:there\s+is\s+no|no)\s+(?P<subj3>.+?)(?:\s+(?:is|are)\s+(?:displayed|shown|visible))?$", text,
                 re.IGNORECASE)
    if m:
        subject = strip_quotes(m.group("subj") or m.group("subj2") or m.group("subj3"))
        return text_or_element_assertion(subject, negate=True, description=text)
    shown_in_field = re.match(
        r"^(?:the\s+)?(?:truncated\s+text\s+|value\s+|text\s+)?(?P<value>\"[^\"]+\"|'[^']+'|.+?)\s+"
        r"(?:is|are)\s+displayed\s+in\s+"
        r"(?:the\s+)?(?P<field>.+?)\s+(?:field|dropdown|combobox|textbox|input|box)"
        r"(?:\s+in\s+(?:the\s+)?(?P<section>.+?))?"
        r"(?:\s+under\s+(?:the\s+)?(?P<under>.+?))?(?:\s+as\s+.+)?$", text, re.IGNORECASE)
    if shown_in_field:
        target = build_target(f"{strip_quotes(shown_in_field.group('field'))} field", "select")
        section = shown_in_field.group("section") or shown_in_field.group("under")
        if section:
            target.context_hint = section.strip()
        return a(AssertionType.TEXT_VISIBLE, 0.9, expected=strip_quotes(shown_in_field.group("value")),
                 target=target, match="case_insensitive")
    m = re.match(r"^(?P<subj>.+?)\s+(?:is|are)\s+(?:now\s+)?(?:displayed|visible|shown|present|rendered|on\s+the\s+page|"
                 r"on\s+screen|seen)(?:\s+(?:on|in)\s+(?:the\s+)?(?P<where>.+))?$|^(?P<subj2>.+?)\s+(?:appears|appear|"
                 r"shows\s+up|is\s+shown)(?:\s+(?:on|in)\s+(?:the\s+)?(?P<where2>.+))?$", text, re.IGNORECASE)
    if m:
        subject = strip_quotes(m.group("subj") or m.group("subj2"))
        where = m.group("where") or m.group("where2")
        return text_or_element_assertion(subject, negate=False, description=text, where=where)
    m = re.match(r"^(?:the\s+)?(?P<subj>.+?)\s+(?:displays|shows|contains|says|reads|includes|has)\s+(?:the\s+)?"
                 r"(?:text\s+|message\s+)?(?P<v>.+)$", text, re.IGNORECASE)
    if m:
        expected = strip_quotes(m.group("v"))
        subj = m.group("subj")
        target = None
        subject_norm = normalize_text(re.sub(r"\s+(page|screen|view)$", "", subj.strip(), flags=re.IGNORECASE))
        if not re.match(r"^(?:page|screen|site|application|app|browser|window)$", subj.strip(), re.IGNORECASE) \
                and subject_norm not in PAGE_NOUNS:
            target = build_target(subj, "hover")
        return a(AssertionType.TEXT_VISIBLE, 0.82, expected=expected, target=target, match="case_insensitive")
    if quoted:
        return a(AssertionType.TEXT_VISIBLE, 0.8, expected=quoted[0], match="case_insensitive")
    # fall back: semantic assertion of the whole phrase (low confidence, LLM/human may refine)
    phrase = re.sub(r"^(?:the\s+)?", "", text)
    return a(AssertionType.TEXT_VISIBLE, 0.45, expected=phrase, match="semantic")


def text_or_element_assertion(subject: str, *, negate: bool, description: str,
                              where: str | None = None) -> tuple[AssertionSpec, float]:
    # "the checkbox \"HP\" is displayed" names a control. Searching the page for the words
    # "checkbox HP" fails even when that checkbox is on screen.
    named = re.match(r"^(?:the\s+|a\s+|an\s+)?(?P<kind>checkbox|check\s*box|radio(?:\s+button)?|button|link|"
                     r"dropdown|combobox|switch|toggle|tab)\s+(?P<name>.+)$", subject.strip(), re.IGNORECASE)
    if named and not where:
        target = build_target(subject, "hover")
        t = AssertionType.ELEMENT_NOT_VISIBLE if negate else AssertionType.ELEMENT_VISIBLE
        return AssertionSpec(type=t, description=description, target=target), 0.9
    kinds = list(KIND_NOUN_RE.finditer(subject))
    clean = DETERMINERS_RE.sub("", subject.strip())
    if kinds and kinds[-1].end() >= len(subject.rstrip()) - 1 and kinds[-1].group(1).lower() not in ("text", "message"):
        target = build_target(subject, "hover")
        t = AssertionType.ELEMENT_NOT_VISIBLE if negate else AssertionType.ELEMENT_VISIBLE
        return AssertionSpec(type=t, description=description, target=target), 0.88
    quoted_in_subject = [next(g for g in m.groups() if g is not None) for m in QUOTE_RE.finditer(subject)]
    expected = quoted_in_subject[0] if quoted_in_subject else strip_quotes(clean)
    expected = re.sub(r"^(?:the\s+)?(?:value|text|message|heading|label|truncated\s+text)\s+", "", expected,
                      flags=re.IGNORECASE)
    qualifier = re.match(r"^(?P<q>.+?)\s+(?:message|text|notification|toast|banner|label|greeting)$", expected,
                         re.IGNORECASE)
    if qualifier and not QUOTE_RE.search(subject):
        expected = qualifier.group("q")
    t = AssertionType.TEXT_NOT_VISIBLE if negate else AssertionType.TEXT_VISIBLE
    target = build_target(where, "hover") if where else None
    quoted_subject = bool(QUOTE_RE.search(subject))
    match = "case_insensitive" if quoted_subject or len(expected.split()) <= 2 else "semantic"
    return AssertionSpec(type=t, description=description, expected=expected, target=target,
                         match=match), 0.88 if match != "semantic" else 0.8


STATE_CONDITIONS = {AssertionType.CHECKBOX_CHECKED, AssertionType.CHECKBOX_UNCHECKED, AssertionType.RADIO_SELECTED,
                    AssertionType.DROPDOWN_SELECTED, AssertionType.INPUT_VALUE_EQUALS, AssertionType.ELEMENT_ENABLED,
                    AssertionType.ELEMENT_DISABLED, AssertionType.ELEMENT_COUNT, AssertionType.PAGE_LOADED}


def parse_condition(text: str, ctx: ParseContext) -> tuple[ConditionSpec, float]:
    raw = text.strip().rstrip(",")
    m = re.match(r"^(?:the\s+)?(?P<var>[\w\s]+?)\s+(?:is|equals|==|=)\s+(?P<val>\"[^\"]+\"|'[^']+'|[\w\s-]+)$", raw,
                 re.IGNORECASE)
    if m:
        var_name = re.sub(r"\s+", "_", m.group("var").strip().lower())
        candidates = [var_name, var_name.replace("the_", "")]
        for name in candidates:
            if name in ctx.variables:
                return ConditionSpec(kind="variable_equals", variable=name, expected=strip_quotes(m.group("val")),
                                     description=raw), 1.0
    # "the checkbox HP is checked", "Country shows India": the state of a control on the page, checked by the
    # assertion engine. (Before this, such conditions were read as a test variable "checkbox_hp" == "checked".)
    spec, confidence = parse_assertion(raw, ctx)
    if spec is not None and spec.type in STATE_CONDITIONS and confidence >= 0.85:
        return ConditionSpec(kind="assertion", assertion=spec.model_copy(update={"timeout_ms": 3000}),
                             expected=None if spec.expected is None else str(spec.expected), description=raw), \
            confidence
    if m:
        var_name = re.sub(r"\s+", "_", m.group("var").strip().lower())
        if not re.search(r"\b(visible|displayed|shown|present|appears)\b", raw, re.IGNORECASE):
            return ConditionSpec(kind="variable_equals", variable=var_name, expected=strip_quotes(m.group("val")),
                                 description=raw), 0.85
    m = re.match(r"^(?:a\s+|an\s+|the\s+)?(?P<subj>.+?)\s+(?:is|are)\s+(?P<neg>not\s+)?(?:visible|displayed|shown|present)"
                 r"$|^(?:a\s+|an\s+|the\s+)?(?P<subj2>.+?)\s+(?P<neg2>does\s+not\s+)?(?:appears?|shows?\s+up|exists?)$", raw,
                 re.IGNORECASE)
    if m:
        subject = strip_quotes(m.group("subj") or m.group("subj2"))
        negate = bool(m.group("neg") or m.group("neg2"))
        if KIND_NOUN_RE.search(subject):
            return ConditionSpec(kind="element_visible", target=build_target(subject, "hover"), negate=negate,
                                 description=raw), 0.9
        return ConditionSpec(kind="text_visible", expected=subject, negate=negate, description=raw), 0.9
    m = re.match(r"^(?:the\s+)?url\s+(?:contains|includes)\s+(?P<v>.+)$", raw, re.IGNORECASE)
    if m:
        return ConditionSpec(kind="url_contains", expected=strip_quotes(m.group("v")), description=raw), 0.95
    return ConditionSpec(kind="text_visible", expected=raw, description=raw), 0.5


def _looks_like_data(phrase: str | None) -> bool:
    """An e-mail, a value with digits/symbols or a quoted literal - not the name of a field."""

    raw = (phrase or "").strip()
    if QUOTE_RE.fullmatch(raw) or VAR_RE.fullmatch(raw):
        return True  # "{{care_email}}" is data even though its name mentions a field
    text = strip_quotes(raw).strip()
    if not text or looks_like_field(text):
        return False
    return bool(re.fullmatch(r"\S+@\S+\.\S+", text) or re.search(r"\d", text) or re.search(r"[!#$%^&*@]", text))


def _entity_of(claim: str) -> str | None:
    """The named thing a claim is about: "new Interaction for patient Abc" -> "Abc"."""

    m = re.search(r"(?:\bfor|\bof|\bnamed|\bcalled|\bwith\s+name)\s+(?:the\s+)?(?:[a-z]+\s+)?"
                  r"(?P<e>[\"'][^\"']+[\"']|[A-Z0-9][\w.@-]*(?:\s+[A-Z0-9][\w.@-]*)*)\s*$", claim)
    if m:
        return strip_quotes(m.group("e"))
    quoted = QUOTE_RE.search(claim)
    return strip_quotes(quoted.group(0)) if quoted else None
