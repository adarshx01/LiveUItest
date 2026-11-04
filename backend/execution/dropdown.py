"""Dropdown engine: native (OS-rendered) selects, datalists and every kind of custom dropdown.

Native ``<select>`` popups are drawn by the browser/OS and cannot be clicked into, so a native select
is always operated with ``select_option`` against its *live* option list (read at action time because
dependent selects fill asynchronously), never by clicking its options. Multi-selects take several
options at once; disabled options are reported, not skipped.

Custom dropdowns are drawn by the page. The engine opens the trigger (typing into it when it is an
editable combobox), then finds the options that belong to *this* trigger: the listbox it controls
(``aria-controls``/``aria-owns``), otherwise the visible list-like popup nearest to it (portals at the
end of ``<body>`` included), otherwise ARIA-less "div soup" menus, otherwise the accessibility tree.
When the option is not rendered it filters (search box), scrolls virtualized lists, and as a last
resort opens with the keyboard. Menu paths ("Export > CSV") walk submenus.

Option choice goes through ``option_matcher`` (exact -> value -> alias -> containment -> ordinal ->
typo-tolerant), so "Indai" picks "India" and the report says why.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.agents.forms.option_matcher import OptionMatch, match_option, split_choices
from backend.browser.js_probes import POPUP_OPTIONS, SCROLL_LIST, SELECT_STATE
from backend.core.logging import get_logger
from backend.dsl.observation import InventoryElement
from backend.dsl.results import FailureCategory

if TYPE_CHECKING:  # pragma: no cover
    from backend.agents.observation.page_observer import PageObserver
    from backend.browser.tool_adapter import BrowserToolAdapter
    from backend.execution.action_executor import ActionOutcome

log = get_logger(__name__)

OPTION_ROLES = ("option", "menuitem", "menuitemradio", "menuitemcheckbox", "treeitem")
NATIVE_POLL_S = 6.0  # dependent selects (country -> state) load their options asynchronously


def target_of(el: InventoryElement) -> str | None:
    return el.ref or el.css_selector


@dataclass
class PickResult:
    chosen: str | None
    how: str = ""
    notes: list[str] = field(default_factory=list)
    visible: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    disabled: bool = False
    option: dict[str, Any] | None = None


class DropdownDriver:
    def __init__(self, adapter: BrowserToolAdapter, observer: PageObserver) -> None:
        self.adapter = adapter
        self.observer = observer
        self._last_search: str | None = None  # search box inside the open popup (never one elsewhere on the page)

    # ------------------------------------------------------------------ native <select>
    async def native_state(self, target: str, description: str) -> dict[str, Any] | None:
        state = await self.adapter.probe(SELECT_STATE, target, description)
        return state if isinstance(state, dict) and state.get("isSelect") else None

    async def select_native(self, target: str, description: str, wanted: str,
                            css: str | None = None) -> ActionOutcome | None:
        """Choose option(s) of a native select. None when ``target`` is not a native select."""

        from backend.execution.action_executor import ActionOutcome

        state = await self.native_state(target, description)
        if state is None:
            return None
        notes: list[str] = ["native select: options chosen with select_option (the OS-rendered popup is never "
                            "clicked)"]
        deadline = asyncio.get_running_loop().time() + NATIVE_POLL_S
        waited = False
        while state.get("disabled") and asyncio.get_running_loop().time() < deadline:
            # dependent selects are disabled while their options load ("Choose a country first", "Loading...")
            waited = True
            await asyncio.sleep(0.4)
            state = await self.native_state(target, description) or state
        if state.get("disabled"):
            return ActionOutcome(False, error=f"{description} stayed disabled for {NATIVE_POLL_S:.0f}s",
                                 failure_category=FailureCategory.ELEMENT_NOT_ENABLED, notes=notes,
                                 expected={"selected_option": wanted})
        while True:
            options = state.get("options") or []
            items = split_choices(wanted, options) if state.get("multiple") else [wanted]
            matches = [match_option(options, item) for item in items]
            missing = [m for m in matches if not m.found]
            real_options = [o for o in options if (o.get("label") or "").strip()]
            if not missing or asyncio.get_running_loop().time() >= deadline:
                break
            if len(real_options) > 1 and waited and not _still_loading(real_options):
                # options are present and stable: waiting longer will not make the option appear
                break
            waited = True
            await asyncio.sleep(0.5)
            state = await self.native_state(target, description) or state
        if waited:
            notes.append("waited for the option list to load")
        if missing:
            m = missing[0]
            labels = [o.get("label") for o in options if o.get("label")]
            return ActionOutcome(False, error=f"option {m_query(items, matches, m)!r} is not offered by {description}"
                                              f" ({m.describe()}); available: {labels[:15]}",
                                 failure_category=FailureCategory.OPTION_NOT_FOUND, notes=notes,
                                 expected={"selected_option": wanted})
        disabled = [m for m in matches if m.option and m.option.disabled]
        if disabled:
            return ActionOutcome(False, error=f"option '{disabled[0].label}' exists but is disabled",
                                 failure_category=FailureCategory.ELEMENT_NOT_ENABLED, notes=notes,
                                 expected={"selected_option": wanted})
        labels = [m.label for m in matches if m.label]
        for item, m in zip(items, matches, strict=False):
            if m.how not in ("exact",):
                notes.append(f"'{item}' -> {m.describe()}")
        # an option label may repeat inside optgroups; the value attribute is then the unambiguous handle
        values = []
        for m in matches:
            assert m.option is not None
            same_label = [o for o in options if o.get("label") == m.option.label]
            values.append(m.option.value if len(same_label) > 1 and m.option.value else m.option.label)
        response = await self.adapter.select_option(target, description, values)
        out = ActionOutcome.from_response(response, args={"target": target, "values": values},
                                          expected={"selected_option": ", ".join(labels), "selected_list": labels,
                                                    "kind": "native", "owner_target": target, "owner_css": css,
                                                    "multiple": bool(state.get("multiple"))},
                                          notes=notes)
        if len(labels) > 1:
            out.notes.append(f"multi-select: {labels}")
        return out

    # ------------------------------------------------------------------ custom dropdowns
    async def select_custom(self, trigger: InventoryElement, wanted: str) -> ActionOutcome:
        from backend.execution.action_executor import ActionOutcome

        target = target_of(trigger)
        if target is None:
            return ActionOutcome(False, error="dropdown trigger is not addressable",
                                 failure_category=FailureCategory.ELEMENT_NOT_FOUND,
                                 expected={"selected_option": wanted})
        path = [p.strip() for p in wanted.split(">")] if ">" in wanted else [wanted]
        if len(path) > 1:
            return await self._menu_path(trigger, target, path)
        editable = trigger.role in ("combobox", "textbox", "searchbox") and (trigger.tag in ("input", "textarea")
                                                                             or trigger.editable)
        opened = False
        items = [wanted]
        if trigger.multiple or re.search(r",|;|\band\b", wanted):
            # several choices? decide against the options actually offered ("Salt and pepper" may be one option)
            visible = await self._open(trigger, target, "" if editable else None)
            opened = True
            items = split_choices(wanted, [o["text"] for o in visible])
        notes: list[str] = []
        chosen: list[str] = []
        for item in items:
            if editable:
                await self._open(trigger, target, item)  # type the option to filter the suggestions
            elif not opened or not await self._popup_options(target, trigger):
                await self._open(trigger, target, None)  # (re)open only when the popup is not showing
            opened = True
            pick = await self._find_and_pick(trigger, target, item, typed=editable)
            notes.extend(pick.notes)
            if pick.chosen is None:
                category = FailureCategory.ELEMENT_NOT_ENABLED if pick.disabled else (
                    FailureCategory.OPTION_NOT_FOUND if pick.visible else FailureCategory.DROPDOWN_NOT_OPEN)
                hint = f"; did you mean {pick.alternatives[:3]}" if pick.alternatives else ""
                return ActionOutcome(False, error=f"option {item!r} not found in {trigger.display()} "
                                                  f"(visible options: {pick.visible[:10]}){hint}",
                                     failure_category=category, notes=notes,
                                     expected={"selected_option": wanted})
            chosen.append(pick.chosen)
        confirmed = None
        still_open = await self._popup_options(target, trigger)
        if still_open:
            marked = {o["text"] for o in still_open if o.get("selected")}
            if chosen and all(c in marked for c in chosen):
                confirmed = "options marked selected (aria-selected) in the open list"
        if len(items) > 1:
            if still_open:
                await self.adapter.press_key("Escape")  # multi-select popups usually stay open
            notes.append(f"multi-select: picked {chosen}")
        return ActionOutcome(True, tool="browser_click", args={"target": target, "options": chosen}, notes=notes,
                             expected={"selected_option": ", ".join(chosen), "selected_list": chosen, "kind": "custom",
                                       "trigger": trigger.name or trigger.label or trigger.text,
                                       "trigger_target": target, "multiple": len(chosen) > 1,
                                       "confirmed": confirmed})

    async def _open(self, trigger: InventoryElement, target: str, typed: str | None) -> list[dict[str, Any]]:
        """Open the popup (typing ``typed`` into editable comboboxes) and return its options."""

        if typed is not None:
            await self.adapter.click(target, trigger.display())
            if typed:
                await self.adapter.type_text(target, trigger.display(), "")  # replace the previous filter
                await self.adapter.type_text(target, trigger.display(), typed, slowly=True)
        else:
            await self.adapter.click(target, trigger.display())
        for attempt in range(6):
            await asyncio.sleep(0.25 if attempt < 3 else 0.4)
            options = await self._popup_options(target, trigger)
            if options:
                return options
            if attempt == 2:
                await self.adapter.press_key("ArrowDown")  # keyboard-opened listboxes
        return []

    async def _popup_options(self, target: str, trigger: InventoryElement) -> list[dict[str, Any]]:
        popup = await self.adapter.probe(POPUP_OPTIONS, target, trigger.display()) or {}
        self._last_search = popup.get("search")
        options = [o for o in popup.get("options") or [] if o.get("text")]
        if options:
            return options
        # fall back to the accessibility tree (options exposed only there, e.g. inside iframes)
        obs = await self.observer.observe(with_tabs=False, with_signals=False)
        out = []
        for e in obs.elements:
            if e.role in OPTION_ROLES and e.visible and (e.ref or e.css_selector):
                if trigger.controls and e.context.list_ref is None and e.context.owner_ref is None:
                    continue
                out.append({"text": e.name or e.text, "ref": e.ref, "css": e.css_selector,
                            "selected": bool(e.selected), "disabled": not e.enabled,
                            "box": [e.bbox.x, e.bbox.y, e.bbox.width, e.bbox.height] if e.bbox else None})
        return out

    async def _find_and_pick(self, trigger: InventoryElement, target: str, item: str, *, typed: bool) -> PickResult:
        result = PickResult(None)
        filtered = typed
        seen: set[str] = set()
        for _round in range(30):
            options = await self._popup_options(target, trigger)
            for o in options:
                seen.add(o["text"])
            result.visible = list(dict.fromkeys(o["text"] for o in options))[:40]
            match: OptionMatch = match_option([{"label": o["text"], "value": o.get("value"),
                                                "disabled": o.get("disabled")} for o in options], item)
            if match.found and match.option is not None:
                option = options[match.option.index]
                if option.get("disabled"):
                    result.disabled = True
                    result.notes.append(f"option '{option['text']}' is disabled")
                    return result
                if option.get("selected"):
                    # clicking a selected entry of a multi-select would deselect it
                    result.chosen, result.how, result.option = option["text"], "already selected", option
                    result.notes.append(f"'{option['text']}' was already selected")
                    return result
                if await self._click_option(option):
                    result.chosen, result.how, result.option = option["text"], match.how, option
                    result.notes.append(f"picked {match.describe()}" + (" after filtering" if filtered else ""))
                    return result
            if options and not filtered:
                search = self._last_search
                if search is not None:
                    await self.adapter.type_text(search, "dropdown search", item, slowly=True)
                    result.notes.append(f"filtered options with '{item}'")
                    filtered = True
                    await asyncio.sleep(0.4)
                    continue
            if options:
                anchor = options[-1].get("ref") or options[-1].get("css")
                scrolled = await self.adapter.probe(SCROLL_LIST, anchor, "option list") if anchor else None
                if scrolled and scrolled.get("scrolled"):
                    if _round == 0:
                        result.notes.append("scrolled the (virtualized) option list")
                    await asyncio.sleep(0.2)
                    continue
            break
        result.alternatives = match_option([{"label": t} for t in seen], item).alternatives
        result.visible = list(seen)[:40] or result.visible
        return result

    async def _click_option(self, option: dict[str, Any]) -> bool:
        handle = option.get("ref") or option.get("css")
        if handle:
            response = await self.adapter.click(handle, f'option "{option["text"][:60]}"')
            if response.ok:
                return True
        box = option.get("box")
        if box and box[2] > 0 and box[3] > 0:
            response = await self.adapter.mouse_click(box[0] + box[2] / 2, box[1] + box[3] / 2)
            return response.ok
        return False

    async def _menu_path(self, trigger: InventoryElement, target: str, path: list[str]) -> ActionOutcome:
        from backend.execution.action_executor import ActionOutcome

        await self.adapter.click(target, trigger.display())
        walked: list[str] = []
        anchor_target, anchor = target, trigger
        for depth, segment in enumerate(path):
            await asyncio.sleep(0.3)
            pick = await self._find_and_pick(anchor, anchor_target, segment, typed=False)
            if pick.chosen is None:
                return ActionOutcome(False, error=f"menu entry {segment!r} not found after {walked or 'opening'} "
                                                  f"(visible: {pick.visible[:10]})",
                                     failure_category=FailureCategory.OPTION_NOT_FOUND,
                                     expected={"selected_option": " > ".join(path)})
            walked.append(pick.chosen)
            if depth < len(path) - 1 and pick.option is not None:
                # the submenu belongs to the entry just activated: look for it next to that entry
                anchor = InventoryElement(element_id="submenu", role="menuitem", name=pick.chosen)
                anchor_target = pick.option.get("ref") or pick.option.get("css") or anchor_target
        return ActionOutcome(True, tool="browser_click", args={"target": target, "path": walked},
                             notes=[f"menu path {' > '.join(walked)}"],
                             expected={"menu_path": walked, "kind": "menu"})


def _still_loading(options: list[dict[str, Any]]) -> bool:
    labels = " ".join((o.get("label") or "").lower() for o in options)
    return "loading" in labels or "please wait" in labels


def m_query(items: list[str], matches: list[OptionMatch], missing: OptionMatch) -> str:
    for item, m in zip(items, matches, strict=False):
        if m is missing:
            return item
    return items[0]
