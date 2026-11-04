"""Page observer: one call produces a complete, structured ``Observation``.

Combines the accessibility snapshot (with bounding boxes and fresh refs), the
DOM inventory probe, page signals (captcha / injection / validation / busy),
tab state and, when requested, a screenshot. Observations are cached
in-process by id; graph state only carries the id and a summary.
"""

from __future__ import annotations

import time
from collections import OrderedDict

from backend.agents.observation.element_inventory import InventoryBuilder
from backend.agents.observation.page_fingerprint import fingerprint
from backend.agents.observation.page_state_analyzer import analyze_page_state
from backend.agents.observation.snapshot_parser import parse_snapshot
from backend.browser.js_probes import DOM_INVENTORY, PAGE_SIGNALS
from backend.browser.tool_adapter import BrowserToolAdapter
from backend.core.ids import new_id
from backend.core.logging import get_logger
from backend.dsl.observation import Observation, TabInfo

log = get_logger(__name__)


class ObservationStore:
    """Bounded per-process cache of full observations."""

    def __init__(self, capacity: int = 200) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, Observation] = OrderedDict()

    def put(self, observation: Observation) -> None:
        self._items[observation.observation_id] = observation
        self._items.move_to_end(observation.observation_id)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def get(self, observation_id: str | None) -> Observation | None:
        if not observation_id:
            return None
        return self._items.get(observation_id)


observation_store = ObservationStore()


class PageObserver:
    name = "page_observer"

    def __init__(self, adapter: BrowserToolAdapter, viewport: tuple[int, int]) -> None:
        self.adapter = adapter
        self.viewport = viewport

    async def observe(self, *, with_tabs: bool = True, with_signals: bool = True) -> Observation:
        observation_id = new_id("obs")
        obs = Observation(observation_id=observation_id, viewport=self.viewport, timestamp=time.time())
        snap = await self.adapter.snapshot()
        if snap.modal_state is not None:
            obs.modal_state = snap.modal_state
            obs.url = snap.page_url or ""
            obs.title = snap.page_title or ""
            observation_store.put(obs)
            return obs
        if not snap.ok or not snap.snapshot_yaml:
            log.warning("snapshot unavailable", error=snap.error)
            obs.url = snap.page_url or ""
            observation_store.put(obs)
            return obs
        obs.url = snap.page_url or ""
        obs.title = snap.page_title or ""
        obs.snapshot_text = snap.snapshot_yaml
        nodes = parse_snapshot(snap.snapshot_yaml)
        dom = await self.adapter.probe(DOM_INVENTORY)
        obs.dom_probe_ok = dom is not None
        if dom:
            obs.url = dom.get("url") or obs.url
            obs.title = dom.get("title") or obs.title
            obs.viewport = (int(dom.get("vw") or self.viewport[0]), int(dom.get("vh") or self.viewport[1]))
        signals = await self.adapter.probe(PAGE_SIGNALS) if with_signals else None
        builder = InventoryBuilder(observation_id, obs.viewport)
        elements, forms, dialogs, headings, visible_text = builder.build(nodes, dom)
        obs.elements, obs.forms, obs.dialogs = elements, forms, dialogs
        obs.headings, obs.visible_text = headings, visible_text
        obs.fingerprint = fingerprint(obs.url, obs.title, elements, forms)
        obs.page_state = analyze_page_state(url=obs.url, title=obs.title, headings=headings, elements=elements,
                                            forms=forms, dialogs=dialogs, signals=signals, dom=dom,
                                            console_errors=snap.console_errors)
        if with_tabs:
            tabs = await self.adapter.tabs("list")
            obs.tabs = tabs.tabs
            current = next((t for t in obs.tabs if t.current), None)
            obs.active_tab = current.index if current else 0
        else:
            obs.tabs = [TabInfo(index=0, title=obs.title, url=obs.url, current=True)]
        observation_store.put(obs)
        return obs
