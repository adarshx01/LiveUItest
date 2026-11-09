"""Interaction executor: performs one grounded step through Playwright MCP.

The logical action (what the test wants) is mapped onto the concrete interaction
the *actual element* needs: a FILL on a native ``<select>`` becomes a select, a
SELECT on a custom combobox becomes open -> (filter) -> pick option, a CHECK on an
already-checked box is a verified no-op, and so on (spec sections 10-12).
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.agents.forms.option_matcher import match_option as _match_option
from backend.agents.forms.value_normalizer import PICKER_TYPES, equivalent_values, normalize_for_input
from backend.agents.governance.redaction import redact_text
from backend.agents.grounding.element_grounder import ElementGrounder, GroundingContext
from backend.agents.grounding.text_match import similarity
from backend.agents.observation.page_observer import PageObserver
from backend.browser.js_probes import (
    ELEMENT_STATE,
    MASK_SENSITIVE_FIELD,
    SCROLL_INFO,
    VALUE_FINGERPRINT,
    find_text_probe,
    find_thing_probe,
    fnv1a,
    scroll_edge_probe,
    scroll_probe,
    set_value_probe,
)
from backend.browser.response_parser import ToolResponse
from backend.browser.session_manager import BrowserSession
from backend.core.config import REPO_ROOT
from backend.core.logging import get_logger
from backend.dsl.models import ActionType, Step, TargetSpec
from backend.dsl.observation import InventoryElement, Observation
from backend.dsl.results import FailureCategory
from backend.execution.dropdown import DropdownDriver
from backend.execution.pickers import DATE_HINT, TIME_HINT, PickerDriver, describe, looks_like_date, looks_like_time

log = get_logger(__name__)

TEST_FILE_DIRS = [REPO_ROOT / "data" / "test-files", REPO_ROOT / "demo" / "fixtures"]


@dataclass
class ActionOutcome:
    ok: bool
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failure_category: FailureCategory | None = None
    code: str | None = None
    downloads: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    no_op: bool = False
    expected: dict[str, Any] = field(default_factory=dict)  # what verification should find
    element_after: InventoryElement | None = None
    tab_switched: bool = False

    @classmethod
    def from_response(cls, response: ToolResponse, **kwargs: Any) -> ActionOutcome:
        return cls(ok=response.ok, tool=response.tool, error=response.error,
                   failure_category=response.failure_category,
                   code=redact_text(response.code) if response.code else None,
                   downloads=list(response.downloads), **kwargs)


def target_of(el: InventoryElement) -> str | None:
    return el.ref or el.css_selector


def match_option(options: list[str], wanted: str) -> tuple[str | None, str]:
    """Backward-compatible wrapper around ``option_matcher.match_option`` (label, how)."""

    result = _match_option(options, wanted)
    return result.label, result.how


def _is_native_select(el: InventoryElement) -> bool:
    return el.tag == "select" or (el.role in ("combobox", "listbox") and bool(el.option_details))


class InteractionExecutor:
    name = "interaction_executor"

    def __init__(self, session: BrowserSession, observer: PageObserver, grounder: ElementGrounder) -> None:
        self.session = session
        self.adapter = session.adapter
        self.observer = observer
        self.grounder = grounder
        self.dropdowns = DropdownDriver(self.adapter, observer)
        self.pickers = PickerDriver(self.adapter)

    # ------------------------------------------------------------------ element actions
    async def click(self, el: InventoryElement, *, double: bool = False, right: bool = False) -> ActionOutcome:
        if not double and not right and _is_native_select(el) and not el.multiple:
            # "Click the Country dropdown": a native popup is drawn by the OS and cannot be operated by
            # clicking; the following "Click India" / "Select India" is performed with select_option
            return ActionOutcome(True, no_op=True, tool="browser_select_option",
                                 notes=["native select: its popup is OS-rendered; the option is chosen with "
                                        "select_option instead of opening it"],
                                 expected={"native_select_opened": True})
        if el.role == "option" and el.context.owner_ref and not double and not right:
            native = await self.dropdowns.select_native(el.context.owner_ref, f"select owning {el.display()}",
                                                        el.name or el.text)
            if native is not None:
                native.notes.insert(0, f"'{el.name}' is an option of a native select: selected it instead of "
                                       "clicking inside the OS-rendered popup")
                return native
        target = target_of(el)
        if target is None:
            if el.bbox is None:
                return ActionOutcome(False, error="element has neither ref, selector nor box",
                                     failure_category=FailureCategory.ELEMENT_NOT_FOUND)
            x, y = el.bbox.center
            response = await self.adapter.mouse_click(x, y, button="right" if right else "left",
                                                      click_count=2 if double else 1)
            return ActionOutcome.from_response(response, args={"x": round(x), "y": round(y)},
                                               notes=["coordinate click on element box"])
        response = await self.adapter.click(target, el.display(), double=double, button="right" if right else None)
        return ActionOutcome.from_response(response, args={"target": target, "element": el.display()})

    async def fill(self, el: InventoryElement, text: str, *, sensitive: bool, slowly: bool = False,
                   observation: Observation | None = None) -> ActionOutcome:
        if _is_native_select(el) and el.tag != "input":
            outcome = await self.select(el, text, observation)
            outcome.notes.insert(0, "field is a native select; used select_option instead of typing")
            return outcome
        target = target_of(el)
        if target is None:
            return ActionOutcome(False, error="element is not addressable", failure_category=FailureCategory.ELEMENT_NOT_FOUND)
        notes: list[str] = []
        input_type = (el.input_type or "").lower()
        if not sensitive and input_type not in PICKER_TYPES:
            picked = await self._custom_picker(el, text)
            if picked is not None:
                return picked
        if not sensitive and (input_type in PICKER_TYPES or input_type == "number"):
            # browser-rendered pickers (date, time, color, range...) accept one machine format and no keystrokes
            normalized = normalize_for_input(text, input_type, min_value=el.min_value, max_value=el.max_value,
                                             step=el.step)
            if not normalized.ok:
                return ActionOutcome(False, error=f"{normalized.note} ({el.display()})",
                                     failure_category=FailureCategory.FORM_VALIDATION_FAILED,
                                     expected={"value": text, "sensitive": False})
            if normalized.changed and normalized.note:
                notes.append(normalized.note)
            text, slowly = normalized.value, False
        elif el.datalist and not sensitive:
            match = _match_option(el.datalist, text)
            if match.found and match.label != text:
                notes.append(f"datalist suggestion {match.describe()} (browser-rendered list, value typed)")
                text = match.label or text
        autocomplete = el.role == "combobox" and el.tag == "input" and not el.datalist and input_type not in PICKER_TYPES
        response = await self.adapter.type_text(target, el.display(), text, slowly=slowly or autocomplete,
                                                sensitive=sensitive)
        outcome = ActionOutcome.from_response(response, args={"target": target, "element": el.display(),
                                                              "text": "••••••" if sensitive else text},
                                              expected={"value": text, "sensitive": sensitive}, notes=notes)
        if response.ok and not autocomplete and input_type not in PICKER_TYPES and el.tag != "select":
            await self._make_value_stick(el, target, text, sensitive, outcome)
        if response.ok and sensitive and text and (el.input_type or "").lower() != "password":
            # browsers already render password inputs as discs; anything else would show the secret in the
            # after-action screenshot and every later one
            masked = await self.adapter.probe(MASK_SENSITIVE_FIELD, target, el.display())
            outcome.notes.append("sensitive field rendered masked for screenshots" if masked
                                 else "WARNING: could not mask the sensitive field for screenshots")
        if response.ok and autocomplete:
            picked = await self._pick_autocomplete(text, el)
            if picked:
                outcome.notes.append(f"selected autocomplete suggestion '{picked}'")
                outcome.expected = {"value": picked, "sensitive": False}
        return outcome

    async def _shown_value(self, el: InventoryElement, target: str, text: str, sensitive: bool) -> tuple[bool, str]:
        """(field holds the typed value, how / what it shows), read from the field that was typed into."""

        for handle in dict.fromkeys(h for h in (target, el.css_selector) if h):
            if sensitive:
                probe = await self.adapter.probe(VALUE_FINGERPRINT, handle, el.display()) or {}
                if probe:
                    return probe.get("hash") == fnv1a(text), f"{probe.get('length')} characters"
            else:
                state = await self.adapter.probe(ELEMENT_STATE, handle, el.display()) or {}
                if state:
                    shown = state.get("value")
                    how = equivalent_values(text, shown)
                    return how is not None, how or repr(shown)
        return True, "not readable"  # the verifier decides; nothing to repair without a reading

    async def _make_value_stick(self, el: InventoryElement, target: str, text: str, sensitive: bool,
                                outcome: ActionOutcome) -> None:
        """Typing succeeded but the application may not have accepted it.

        ``fill`` sets the value without key events: masked editors, keypress filters and fields that format on
        blur then drop or rewrite it. Read the field back; if the value is not there, type it again key by key
        and leave the field (Tab), and as the last resort put it in through the DOM with the events frameworks
        listen for. Each attempt is recorded; the verifier still checks the final value independently.
        """

        ok, shown = await self._shown_value(el, target, text, sensitive)
        if ok:
            return
        outcome.notes.append(f"the field did not keep the value after fill (shows {shown}); retyping key by key")
        await self.adapter.type_text(target, el.display(), "", sensitive=False)  # clear
        await self.adapter.type_text(target, el.display(), text, slowly=True, sensitive=sensitive)
        await self.adapter.press_key("Tab")  # blur: commit / format, as a person leaving the field would
        await asyncio.sleep(0.3)
        ok, shown = await self._shown_value(el, target, text, sensitive)
        if ok:
            outcome.notes.append(f"value accepted after typing key by key and leaving the field ({shown})")
            outcome.args["method"] = "keystrokes+tab"
            return
        if sensitive:
            outcome.notes.append("the field still does not hold the secret value (not injected: secrets are only typed)")
            return
        injected = await self.adapter.probe(set_value_probe(text), target, el.display()) or {}
        await asyncio.sleep(0.3)
        ok, shown = await self._shown_value(el, target, text, sensitive)
        if ok:
            outcome.notes.append(f"value set through the DOM with input/change/blur events ({shown}); keyboard input "
                                 "was not accepted by this field")
            outcome.args["method"] = "dom_injection"
        else:
            outcome.notes.append(f"the field rejects the value: shows {shown} after typing, key-by-key typing and DOM "
                                 f"injection (injected value read back as {injected.get('value')!r})")

    async def _custom_picker(self, el: InventoryElement, text: str) -> ActionOutcome | None:
        """Dates and times entered into page-drawn pickers (read-only inputs, buttons, popover fields).

        Only fields that look like date/time fields are routed here; a picker that does not open returns
        None and the caller continues with its normal typing / dropdown handling."""

        if _is_native_select(el) or el.tag == "select":
            return None
        about = describe(el)
        plain_input = el.tag in ("input", "textarea") and el.editable and not el.readonly
        time = looks_like_time(text)
        if time and (TIME_HINT.search(about) or not plain_input):
            outcome = await self.pickers.set_time(el, time, text)
            if outcome is not None:
                return outcome
        date = None if time else looks_like_date(text)
        if date and (DATE_HINT.search(about) or not plain_input):
            outcome = await self.pickers.set_date(el, date, text)
            if outcome is not None:
                return outcome
        return None

    async def _pick_autocomplete(self, text: str, el: InventoryElement | None = None) -> str | None:
        """After typing into an autocomplete, pick the suggestion the text means (free text stays if none)."""

        target = target_of(el) if el is not None else None
        for _ in range(7):
            await asyncio.sleep(0.35)
            if target and el is not None:
                options = await self.dropdowns._popup_options(target, el)
            else:
                obs = await self.observer.observe(with_tabs=False, with_signals=False)
                options = [{"text": e.name or e.text, "ref": e.ref, "css": e.css_selector, "box": None}
                           for e in obs.elements if e.role == "option" and e.visible and (e.ref or e.css_selector)]
            if not options:
                continue
            match = _match_option([{"label": o["text"], "value": o.get("value")} for o in options], text)
            chosen = options[match.option.index] if match.found and match.option is not None else None
            if chosen is None:
                starts = [o for o in options if o["text"].lower().startswith(text.lower())]
                chosen = starts[0] if len(starts) == 1 else None
            if chosen is None:
                return None
            return chosen["text"] if await self.dropdowns._click_option(chosen) else None
        return None

    async def select(self, el: InventoryElement, option: str, observation: Observation | None) -> ActionOutcome:
        if el.role == "option" and el.context.owner_ref:
            owner = observation.by_ref(el.context.owner_ref) if observation is not None else None
            native = await self.dropdowns.select_native(el.context.owner_ref, f"select owning {el.display()}", option,
                                                        css=owner.css_selector if owner else None)
            if native is not None:
                return native
        if el.role in ("radio", "tab", "button", "option", "menuitemradio", "checkbox", "switch") and \
                not el.haspopup and not el.controls:
            outcome = await self.click(el)
            outcome.expected = {"selected_option": option, "kind": el.role}
            outcome.notes.append(f"choice '{option}' is a {el.role}; clicked it")
            return outcome
        target = target_of(el)
        if target and (_is_native_select(el) or el.tag == "select"):
            pair = self._range_partner(el, option, observation)
            if pair is not None:
                return await self._select_range(el, pair, option)
            native = await self.dropdowns.select_native(target, el.display(), option, css=el.css_selector)
            if native is not None:
                return native
        if target and el.tag in ("input", "textarea") and not el.datalist and el.role not in ("combobox",) \
                and not el.haspopup and not el.controls:
            # "Set the luggage color to red" on <input type=color>, "set budget to 2500" on a slider
            outcome = await self.fill(el, option, sensitive=False)
            outcome.notes.insert(0, f"'{el.display()}' is an input, not a list: the choice is entered as its value")
            return outcome
        if el.datalist and target:
            match = _match_option(el.datalist, option)
            value = match.label if match.found else option
            outcome = await self.fill(el, value or option, sensitive=False)
            outcome.notes.insert(0, f"input with a browser-rendered datalist: typed {match.describe() if match.found else repr(option)}")
            return outcome
        if (el.input_type or "").lower() not in PICKER_TYPES:
            picked = await self._custom_picker(el, option)  # "Click on Start Date and select 09 June 2026"
            if picked is not None:
                return picked
        return await self.dropdowns.select_custom(el, option)

    @staticmethod
    def _range_partner(el: InventoryElement, option: str, observation: Observation | None) -> InventoryElement | None:
        """'Select "202404" and "202409" from the Data period dropdown' on a from ~ to pair of single selects:
        the second value belongs to the select right of this one, on the same row."""

        from backend.agents.grounding.spatial import same_row

        choices = [c.strip() for c in option.split(",") if c.strip()]
        if len(choices) != 2 or el.multiple or observation is None or el.bbox is None:
            return None
        right = [e for e in observation.elements if e is not el and e.visible and e.bbox is not None
                 and (e.tag == "select" or _is_native_select(e)) and not e.multiple and same_row(e, el)
                 and e.bbox.x > el.bbox.x + el.bbox.width * 0.5]
        right.sort(key=lambda e: e.bbox.x if e.bbox else 0)
        return right[0] if right and right[0].bbox and right[0].bbox.x - (el.bbox.x + el.bbox.width) < 160 else None

    async def _select_range(self, first: InventoryElement, second: InventoryElement, option: str) -> ActionOutcome:
        a, b = [c.strip() for c in option.split(",") if c.strip()]
        start = await self.dropdowns.select_native(target_of(first) or "", first.display(), a, css=first.css_selector)
        if start is None or not start.ok:
            return start or ActionOutcome(False, error=f"{first.display()} is not a selectable list",
                                          failure_category=FailureCategory.ELEMENT_NOT_FOUND)
        end = await self.dropdowns.select_native(target_of(second) or "", second.display(), b, css=second.css_selector)
        if end is None or not end.ok:
            outcome = end or ActionOutcome(False, error=f"{second.display()} is not a selectable list",
                                           failure_category=FailureCategory.ELEMENT_NOT_FOUND)
            outcome.notes.insert(0, f"'{a}' selected in the first list of the range")
            return outcome
        start.notes.append(f"range: '{a}' in the first list, '{b}' in the list to its right ({second.display()})")
        start.expected.update({"pair_target": end.expected.get("owner_target"), "pair_css": second.css_selector,
                               "pair_option": b, "pair_description": second.display()})
        start.args["values"] = [a, b]
        return start

    async def set_checked(self, el: InventoryElement, desired: bool | None) -> ActionOutcome:
        current = el.checked
        if desired is not None and current is not None and current == desired:
            return ActionOutcome(True, no_op=True, notes=[f"already {'checked' if desired else 'unchecked'}"],
                                 expected={"checked": desired})
        outcome = await self.click(el)
        outcome.expected = {"checked": (not current) if desired is None and current is not None else desired}
        return outcome

    async def hover(self, el: InventoryElement) -> ActionOutcome:
        target = target_of(el)
        if target is None and el.bbox:
            response = await self.adapter.mouse_move(*el.bbox.center)
            return ActionOutcome.from_response(response, notes=["coordinate hover"])
        response = await self.adapter.hover(target or "", el.display())
        return ActionOutcome.from_response(response, args={"target": target})

    async def drag(self, src: InventoryElement, dst: InventoryElement) -> ActionOutcome:
        s, d = target_of(src), target_of(dst)
        if s and d:
            response = await self.adapter.drag(s, src.display(), d, dst.display())
            return ActionOutcome.from_response(response, args={"from": s, "to": d},
                                               expected={"moved": src.name or src.text, "into": dst.name})
        if src.bbox and dst.bbox:
            (sx, sy), (ex, ey) = src.bbox.center, dst.bbox.center
            response = await self.adapter.mouse_drag(sx, sy, ex, ey)
            return ActionOutcome.from_response(response, notes=["coordinate drag"],
                                               expected={"moved": src.name or src.text, "into": dst.name})
        return ActionOutcome(False, error="drag endpoints not addressable", failure_category=FailureCategory.ELEMENT_NOT_FOUND)

    async def upload(self, el: InventoryElement, file_ref: str) -> ActionOutcome:
        source = resolve_test_file(file_ref)
        if source is None:
            return ActionOutcome(False, error=f"test file {file_ref!r} not found in {', '.join(map(str, TEST_FILE_DIRS))}",
                                 failure_category=FailureCategory.UNKNOWN_FAILURE)
        # MCP only reads files inside the session workspace: stage a copy there
        staged = self.session.uploads_dir / source.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, staged)
        opened = await self.click(el)
        if not opened.ok:
            return opened
        response = await self.adapter.upload([str(staged)])
        return ActionOutcome.from_response(response, args={"file": source.name}, expected={"files": [source.name]},
                                           notes=["file staged in isolated session workspace"])

    async def press_key(self, key: str, el: InventoryElement | None = None, focus: bool = False) -> ActionOutcome:
        if el is not None and focus and target_of(el):
            await self.adapter.click(target_of(el) or "", el.display())
        response = await self.adapter.press_key(key)
        return ActionOutcome.from_response(response, args={"key": key})

    # ------------------------------------------------------------------ page-level actions
    async def navigate(self, url: str) -> ActionOutcome:
        response = await self.adapter.navigate(url)
        return ActionOutcome.from_response(response, args={"url": url}, expected={"url": url})

    async def scroll(self, step: Step) -> ActionOutcome:
        spec = step.scroll
        if spec is None:
            return ActionOutcome(False, error="missing scroll spec")
        before = await self.adapter.probe(SCROLL_INFO) or {}
        if spec.to in ("top", "bottom"):
            for _ in range(spec.max_scrolls if spec.to == "bottom" else 1):
                info = await self.adapter.probe(scroll_edge_probe(spec.to)) or {}
                await asyncio.sleep(0.5)
                after = await self.adapter.probe(SCROLL_INFO) or {}
                if spec.to == "top" or after.get("docHeight") == info.get("docHeight"):
                    break
            after = await self.adapter.probe(SCROLL_INFO) or {}
            return ActionOutcome(True, tool="browser_evaluate", args={"scroll_to": spec.to},
                                 expected={"scroll_to": spec.to, "before": before, "after": after})
        if spec.to == "until_text" and spec.text:
            for i in range(spec.max_scrolls):
                hits = await self.adapter.probe(find_text_probe(spec.text)) or {}
                for hit in hits.get("hits") or []:
                    box = hit.get("box") or [0, 0, 0, 0]
                    vh = before.get("vh") or 800
                    if 0 <= box[1] < vh:
                        return ActionOutcome(True, tool="browser_evaluate", args={"scroll_until": spec.text},
                                             expected={"text_visible": spec.text}, notes=[f"found after {i} scrolls"])
                    await self.adapter.probe(scroll_probe(0, int(box[1] - vh / 3)))
                    return ActionOutcome(True, tool="browser_evaluate", args={"scroll_until": spec.text},
                                         expected={"text_visible": spec.text}, notes=[f"scrolled to text after {i} scrolls"])
                info = await self.adapter.probe(scroll_probe(0, 700)) or {}
                await asyncio.sleep(0.6)  # allow lazy loading / infinite scroll
                if info and info.get("scrollY", 0) + info.get("vh", 0) >= info.get("docHeight", 0) - 2 and i > 3:
                    more = await self.adapter.probe(SCROLL_INFO) or {}
                    if more.get("docHeight") == info.get("docHeight"):
                        break
            return ActionOutcome(False, error=f"text {spec.text!r} not found after scrolling",
                                 failure_category=FailureCategory.ELEMENT_NOT_FOUND)
        amount = spec.amount_px or 600
        dy = amount if spec.direction == "down" else -amount if spec.direction == "up" else 0
        dx = amount if spec.direction == "right" else -amount if spec.direction == "left" else 0
        info = await self.adapter.probe(scroll_probe(dx, dy)) or {}
        return ActionOutcome(True, tool="browser_evaluate", args={"dx": dx, "dy": dy},
                             expected={"scroll_by": dy, "before": before, "after": info})

    async def wait(self, step: Step) -> ActionOutcome:
        spec = step.wait
        if spec is None:
            return ActionOutcome(False, error="missing wait spec")
        if spec.seconds is not None and not spec.until_text and not spec.until_text_gone and not spec.until_stable:
            response = await self.adapter.wait_for(seconds=spec.seconds)
            if not response.ok and "modal state" in (response.error or ""):
                await asyncio.sleep(spec.seconds)  # browser_wait_for refuses while a native dialog is open
                return ActionOutcome(True, args={"seconds": spec.seconds}, notes=["paused (a native dialog is open)"])
            return ActionOutcome.from_response(response, args={"seconds": spec.seconds})
        if spec.until_stable:
            return await self._wait_until_settled(spec.timeout_seconds)
        if spec.until_text:
            return await self._wait_for_thing(spec.until_text, spec.timeout_seconds)
        text = spec.until_text or spec.until_text_gone or ""
        present = await self.adapter.probe(find_text_probe(text)) or {}
        literal_known = bool(present.get("hits")) or bool(spec.until_text)
        if spec.until_text_gone and not present.get("hits"):
            # semantic wait ("the loading spinner") -> wait until busy indicators disappear
            for _ in range(40):
                obs = await self.observer.observe(with_tabs=False)
                if not obs.page_state.loading:
                    return ActionOutcome(True, notes=["busy indicators cleared"], expected={"loading": False})
                await asyncio.sleep(0.5)
            return ActionOutcome(False, error="page still busy after 20s", failure_category=FailureCategory.ACTION_TIMEOUT)
        if literal_known:
            response = await self.adapter.wait_for(text=spec.until_text, text_gone=spec.until_text_gone, seconds=20)
            return ActionOutcome.from_response(response, args={"text": spec.until_text, "gone": spec.until_text_gone})
        return ActionOutcome(False, error="nothing to wait for")

    async def _wait_until_settled(self, timeout: float) -> ActionOutcome:
        """No busy indicator, nothing loading, and the page stopped changing for two consecutive looks."""

        deadline = asyncio.get_running_loop().time() + timeout
        previous: tuple[int, int, str] | None = None
        stable_rounds, looks = 0, 0
        while asyncio.get_running_loop().time() < deadline:
            obs = await self.observer.observe(with_tabs=False)
            looks += 1
            if obs.modal_state is not None:
                return ActionOutcome(True, notes=[f"a native {obs.modal_state.kind} is open: {obs.modal_state.message[:80]}"],
                                     expected={"loading": False})
            signature = (len(obs.elements), len(" ".join(obs.visible_text)), obs.url)
            if not obs.page_state.loading and signature == previous:
                stable_rounds += 1
                if stable_rounds >= 1:
                    return ActionOutcome(True, notes=[f"page settled after {looks} looks"],
                                         expected={"loading": False, "settled": True})
            else:
                stable_rounds = 0
            previous = signature
            await asyncio.sleep(0.6)
        return ActionOutcome(False, error=f"the page was still loading or changing after {timeout:.0f}s",
                             failure_category=FailureCategory.ACTION_TIMEOUT)

    async def _wait_for_thing(self, what: str, timeout: float) -> ActionOutcome:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            found = await self.adapter.probe(find_thing_probe(what)) or {}
            if found.get("found"):
                where = (found.get("text") or found.get("named"))[0]
                how = "text" if found.get("text") else "element named / identified"
                return ActionOutcome(True, notes=[f"'{what}' appeared ({how}: {str(where)[:100]})"],
                                     expected={"text_visible": True, "waited_for": what})
            if asyncio.get_running_loop().time() >= deadline:
                return ActionOutcome(False, error=f"nothing matching '{what}' appeared within {timeout:.0f}s (looked at "
                                                  f"visible text, labels, alt/title text, ids and classes)",
                                     failure_category=FailureCategory.ACTION_TIMEOUT)
            await asyncio.sleep(0.6)

    async def switch_tab(self, step: Step, observation: Observation) -> ActionOutcome:
        spec = step.tab
        tabs = (await self.adapter.tabs("list")).tabs or observation.tabs
        if spec is None:
            return ActionOutcome(False, error="missing tab spec")
        current = next((t.index for t in tabs if t.current), 0)
        if spec.which == "new":
            url = step.target.url if step.target else None
            response = await self.adapter.tabs("new", url=url)
            return ActionOutcome.from_response(response, tab_switched=True, expected={"tab": "new"})
        if spec.which == "close_current":
            response = await self.adapter.tabs("close", index=current)
            return ActionOutcome.from_response(response, tab_switched=True, expected={"tab": "closed"})
        index: int | None = None
        if spec.which == "latest":
            index = max((t.index for t in tabs), default=0)
        elif spec.which == "first":
            index = 0
        elif spec.which == "next":
            index = min(current + 1, max((t.index for t in tabs), default=0))
        elif spec.which == "previous":
            index = max(0, current - 1)
        elif spec.which == "index":
            index = spec.index
        elif spec.which == "title" and spec.title:
            best = max(tabs, key=lambda t: similarity(spec.title or "", t.title), default=None)
            index = best.index if best and similarity(spec.title, best.title) >= 0.5 else None
        if index is None or not any(t.index == index for t in tabs):
            return ActionOutcome(False, error=f"no matching tab among {[t.title for t in tabs]}",
                                 failure_category=FailureCategory.POPUP_OR_DIALOG_ERROR)
        if spec.which == "latest" and len(tabs) == 1:
            return ActionOutcome(False, error="no new tab was opened", failure_category=FailureCategory.POPUP_OR_DIALOG_ERROR)
        response = await self.adapter.tabs("select", index=index)
        tab = next(t for t in tabs if t.index == index)
        return ActionOutcome.from_response(response, tab_switched=True,
                                           expected={"tab_index": index, "tab_title": tab.title, "tab_url": tab.url})

    async def handle_dialog(self, step: Step, observation: Observation, accept: bool,
                            prompt_text: str | None) -> ActionOutcome:
        if observation.modal_state is not None and observation.modal_state.kind in ("alert", "confirm", "prompt",
                                                                                   "beforeunload"):
            response = await self.adapter.handle_dialog(accept, prompt_text)
            return ActionOutcome.from_response(response, args={"accept": accept},
                                               notes=[f"native {observation.modal_state.kind}: "
                                                      f"{observation.modal_state.message[:80]}"],
                                               expected={"modal_cleared": True})
        dialogs = [d for d in observation.dialogs if d.element_ids]
        if not dialogs:
            return ActionOutcome(False, error="no dialog is open", failure_category=FailureCategory.POPUP_OR_DIALOG_ERROR)
        dialog = dialogs[-1]
        hints = ["ok", "yes", "accept", "confirm", "save", "continue", "accept all", "allow"] if accept else \
            ["close", "cancel", "no", "dismiss", "x", "×", "not now", "reject"]
        target = TargetSpec(intent=hints[0], semantic_hints=hints, preferred_roles=["button"],
                            context_hint=dialog.name or None)
        fake = step.model_copy(update={"action": ActionType.CLICK, "target": target})
        decision = self.grounder.ground(fake, observation, GroundingContext())
        el = observation.element(decision.element_id) if decision.element_id else None
        if el is None or el.element_id not in dialog.element_ids:
            return ActionOutcome(False, error=f"no {'accept' if accept else 'dismiss'} control in dialog "
                                              f"{dialog.name!r}", failure_category=FailureCategory.POPUP_OR_DIALOG_ERROR)
        outcome = await self.click(el)
        outcome.notes.append(f"clicked '{el.name}' in dialog '{dialog.name}'")
        outcome.expected = {"dialog_closed": dialog.name}
        return outcome

    async def read_value(self, el: InventoryElement) -> str | None:
        target = target_of(el)
        if not target:
            return el.value_preview or el.text or el.name
        state = await self.adapter.probe(ELEMENT_STATE, target, el.display()) or {}
        if state.get("value") not in (None, ""):
            return str(state["value"])
        return (state.get("text") or el.text or el.name or "").strip() or None


def resolve_test_file(file_ref: str) -> Path | None:
    candidate = Path(file_ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in TEST_FILE_DIRS:
        path = (base / file_ref).resolve()
        if path.exists() and str(path).startswith(str(base.resolve())):
            return path
    return None
