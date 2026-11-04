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
        if re.match(r"^wait\s+(?:for|until)\s+(?:the\s+)?page\s+(?:to\s+)?(?:fully\s+)?loads?(?:ed)?$", line,
                    re.IGNORECASE):
            return ParsedLine([self._step(ctx, line, ActionType.WAIT, 0.95, wait=WaitSpec(seconds=1.0),
                                          risk_hint="read", notes=["page_load"])], 0.95, "wait load")
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
        m = re.match(r"^(?:select|choose|pick)\s+(?:the\s+)?(?:option\s+)?(?P<opt>\"[^\"]+\"|'[^']+'|.+?)\s+"
                     r"(?:from|in|on|under|within|for)\s+(?:the\s+)?(?P<target>.+)$", line, re.IGNORECASE)
        if m and not re.search(r"\b(page|tab|window|menu bar)\b", m.group("target"), re.IGNORECASE):
            option = strip_quotes(m.group("opt"))
            target = build_target(m.group("target"), "select")
            if not target.preferred_roles:
                target.preferred_roles = ["combobox", "listbox", "radiogroup"]
            target.option_text = option
            value = parse_value(m.group("opt"), ctx)
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
    quoted = [next(g for g in m.groups() if g is not None) for m in QUOTE_RE.finditer(raw)]
    if quoted:
        exact_name = quoted[0]
    work = QUOTE_RE.sub(lambda m: " __Q__ ", raw) if quoted else raw

    context_hint = None
    ctx_m = re.search(r"\s+(?:in|inside|within|on|under|from|of)\s+(?:the\s+)?(?P<ctx>[\w\s'-]+?\s+(?:form|section|"
                      r"dialog|modal|popup|pop-up|panel|header|footer|menu|navigation|nav|navbar|sidebar|card|row|"
                      r"table|frame|iframe|list|column|toolbar|banner|page|area|region|drawer|tab|canvas|map|chart|board|editor|"
                      r"diagram|graph|grid|carousel))$", work, re.IGNORECASE)
    if ctx_m:
        context_hint = DETERMINERS_RE.sub("", ctx_m.group("ctx").strip())
        work = work[: ctx_m.start()]
    else:
        ctx_m = re.search(r"\s+(?:for|next\s+to|beside|near|below|above|under)\s+(?:the\s+)?(?P<ctx>.+)$", work,
                          re.IGNORECASE)
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
    if re.match(r"^(?:the\s+)?page\s+(?:is\s+)?(?:fully\s+)?(?:loaded|loads|displayed|rendered)$", low):
        return a(AssertionType.PAGE_LOADED, 0.95)
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
    m = re.match(r"^(?:the\s+)?(?P<target>.+?)\s+(?:checkbox|check box|toggle|switch)?\s*(?:is|should be)\s+"
                 r"(?P<neg>not\s+|un)?(?:checked|ticked|selected|enabled|on)$", low)
    if m and re.search(r"(checkbox|check box|toggle|switch|remember|agree|terms|subscribe|newsletter|option)", low):
        negate = bool(m.group("neg"))
        target = build_target(text[: m.end("target")], "check")
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
    kinds = list(KIND_NOUN_RE.finditer(subject))
    clean = DETERMINERS_RE.sub("", subject.strip())
    if kinds and kinds[-1].end() >= len(subject.rstrip()) - 1 and kinds[-1].group(1).lower() not in ("text", "message"):
        target = build_target(subject, "hover")
        t = AssertionType.ELEMENT_NOT_VISIBLE if negate else AssertionType.ELEMENT_VISIBLE
        return AssertionSpec(type=t, description=description, target=target), 0.88
    expected = strip_quotes(clean)
    expected = re.sub(r"^(?:text|message|heading|label)\s+", "", expected, flags=re.IGNORECASE)
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
