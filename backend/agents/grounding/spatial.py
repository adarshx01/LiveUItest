"""Spatial relations between elements, the way a person reads a screen.

Test cases point at controls by where they are: "the Add more link next to Upload Documents", "the input
below 01", "the second Data period dropdown", "the delete icon on the right of the row", "the field to the
right of Delivery Date". Those are decided on the rendered layout (CSS-pixel boxes), never on DOM order:

* ``right_of`` / ``left_of`` - on the same visual row (vertical overlap), on that side, nearest first
* ``below`` / ``above`` - in the same column band (horizontal overlap), on that side, nearest first
* ``next_to`` / ``near`` - smallest edge-to-edge distance, same row preferred
* reading order - rows (y bands) top to bottom, then left to right, for "first / second / last"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.agents.grounding.fuzzy import match_score
from backend.agents.planner.lexicon import normalize_text
from backend.dsl.observation import BoundingBox, InventoryElement, Observation

RELATIONS = ("right_of", "left_of", "below", "above", "next_to", "near", "inside")

RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"(?:to\s+the\s+|on\s+the\s+|at\s+the\s+)?right\s+(?:side\s+)?(?:of|to|from)", "right_of"),
    (r"(?:to\s+the\s+|on\s+the\s+|at\s+the\s+)?left\s+(?:side\s+)?(?:of|to|from)", "left_of"),
    (r"(?:directly\s+|right\s+|just\s+)?(?:below|beneath|underneath|under)", "below"),
    (r"(?:directly\s+|right\s+|just\s+)?(?:above|over)", "above"),
    (r"(?:right\s+|just\s+)?(?:next\s+to|beside|alongside|adjacent\s+to|by)", "next_to"),
    (r"(?:near|close\s+to|around)", "near"),
]


def split_relation(phrase: str) -> tuple[str, str, str] | None:
    """'"Add more" link next to "Upload Documents"' -> ('"Add more" link', 'next_to', 'Upload Documents')."""

    for pattern, relation in RELATION_PATTERNS:
        m = re.search(rf"\s+(?:that\s+is\s+|which\s+is\s+|located\s+|placed\s+|shown\s+|displayed\s+)?{pattern}\s+"
                      rf"(?:the\s+)?(?P<anchor>.+)$", phrase, re.IGNORECASE)
        if m and m.start() > 0:
            anchor = m.group("anchor").strip()
            anchor = re.sub(r"\s+(?:in|on|of|within)\s+(?:the\s+)?(?:listing|list|table|grid|page|screen|form|section|"
                            r"results?|panel|dialog)$", "", anchor, flags=re.IGNORECASE).strip().strip("\"'")
            anchor = re.sub(r"\s+(?:field|label|text|heading|section|dropdown|button|link|box|column|header)$", "",
                            anchor, flags=re.IGNORECASE).strip().strip("\"'")
            if anchor:
                return phrase[: m.start()].strip(), relation, anchor
    return None


@dataclass
class Anchor:
    element: InventoryElement
    score: float

    @property
    def box(self) -> BoundingBox:
        assert self.element.bbox is not None
        return self.element.bbox


def find_anchors(observation: Observation, anchor_text: str, limit: int = 4) -> list[Anchor]:
    """Elements whose visible text or name is the anchor phrase (labels, headings, cells, controls)."""

    want = normalize_text(anchor_text)
    if not want:
        return []
    found: list[Anchor] = []
    for el in observation.elements:
        if not el.visible or el.bbox is None or el.bbox.width <= 0:
            continue
        best = 0.0
        for text in (el.name, el.text, el.label, el.value_preview, el.placeholder):
            if not text or len(text) > 160:
                continue
            nt = normalize_text(text)
            if nt == want:
                best = 1.0
                break
            best = max(best, match_score(anchor_text, text) * (0.95 if want in nt else 0.85))
        if best >= 0.72:
            found.append(Anchor(el, best))
    # prefer exact text, then the smallest element carrying it (a label rather than its whole section)
    found.sort(key=lambda a: (-round(a.score, 2), a.box.width * a.box.height))
    return found[:limit]


def _v_overlap(a: BoundingBox, b: BoundingBox) -> float:
    top, bottom = max(a.y, b.y), min(a.y + a.height, b.y + b.height)
    return max(0.0, bottom - top) / max(1.0, min(a.height, b.height))


def _h_overlap(a: BoundingBox, b: BoundingBox) -> float:
    left, right = max(a.x, b.x), min(a.x + a.width, b.x + b.width)
    return max(0.0, right - left) / max(1.0, min(a.width, b.width))


def edge_distance(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(0.0, max(a.x, b.x) - min(a.x + a.width, b.x + b.width))
    dy = max(0.0, max(a.y, b.y) - min(a.y + a.height, b.y + b.height))
    return (dx * dx + dy * dy) ** 0.5


def relation_score(relation: str, candidate: BoundingBox, anchor: BoundingBox) -> float:
    """How well ``candidate`` stands in ``relation`` to ``anchor``: 1 = clearly, 0 = not at all."""

    if candidate.iou(anchor) > 0.6:
        return 0.0  # the anchor itself
    cx, cy = candidate.center
    ax, ay = anchor.center
    if relation in ("right_of", "left_of"):
        same_row = _v_overlap(candidate, anchor) > 0.25 or abs(cy - ay) <= max(candidate.height, anchor.height) * 0.75
        if relation == "right_of":
            ahead = candidate.x >= anchor.x + anchor.width * 0.5 - 4
            gap = candidate.x - (anchor.x + anchor.width)
        else:
            ahead = candidate.x + candidate.width <= anchor.x + anchor.width * 0.5 + 4
            gap = anchor.x - (candidate.x + candidate.width)
        if not (same_row and ahead):
            return 0.0
        return 1.0 / (1.0 + max(0.0, gap) / 220.0)
    if relation in ("below", "above"):
        same_column = _h_overlap(candidate, anchor) > 0.2 or abs(cx - ax) <= max(candidate.width, anchor.width) * 0.6
        if relation == "below":
            beyond = candidate.y >= anchor.y + anchor.height * 0.5 - 2
            gap = candidate.y - (anchor.y + anchor.height)
        else:
            beyond = candidate.y + candidate.height <= anchor.y + anchor.height * 0.5 + 2
            gap = anchor.y - (candidate.y + candidate.height)
        if not (same_column and beyond):
            return 0.0
        return 1.0 / (1.0 + max(0.0, gap) / 160.0)
    if relation in ("next_to", "near"):
        distance = edge_distance(candidate, anchor)
        same_row = _v_overlap(candidate, anchor) > 0.25
        if relation == "next_to":
            # "next to" / "beside" is on the same line: the "Add more" of the section stacked 19px below
            # ("Notes  Add more") is close, but it is under the heading, not next to it
            return min(1.0, (1.15 if same_row else 0.6) / (1.0 + distance / 180.0))
        return min(1.0, (1.15 if same_row else 1.0) / (1.0 + distance / 320.0))
    if relation == "inside":
        inside = (candidate.x >= anchor.x - 2 and candidate.y >= anchor.y - 2
                  and candidate.x + candidate.width <= anchor.x + anchor.width + 2
                  and candidate.y + candidate.height <= anchor.y + anchor.height + 2)
        return 1.0 if inside else 0.0
    return 0.0


def best_relation(relation: str, candidate: InventoryElement, anchors: list[Anchor]) -> tuple[float, str]:
    """Best relation score over the anchors (weighted by how surely each anchor is the named one)."""

    if candidate.bbox is None or not anchors:
        return 0.0, ""
    best, why = 0.0, ""
    for anchor in anchors:
        if anchor.element.element_id == candidate.element_id:
            continue
        score = relation_score(relation, candidate.bbox, anchor.box) * anchor.score
        if score > best:
            best, why = score, f"{relation.replace('_', ' ')} '{(anchor.element.name or anchor.element.text)[:40]}'"
    return best, why


def reading_order_key(el: InventoryElement) -> tuple[float, float]:
    """Rows top to bottom (elements whose vertical centres are within half a line are one row), then x."""

    if el.bbox is None:
        return (1e9, 1e9)
    cx, cy = el.bbox.center
    return (round(cy / max(12.0, el.bbox.height * 0.6)), cx)


def same_row(a: InventoryElement, b: InventoryElement) -> bool:
    if a.bbox is None or b.bbox is None:
        return False
    return _v_overlap(a.bbox, b.bbox) > 0.5 or abs(a.bbox.center[1] - b.bbox.center[1]) <= min(
        a.bbox.height, b.bbox.height) * 0.5


def region_words(el: InventoryElement, viewport: tuple[int, int]) -> str:
    """'top-left', 'middle-right'... where an element sits on the screen (for the model)."""

    if el.bbox is None:
        return ""
    width, height = viewport
    cx, cy = el.bbox.center
    vertical = "top" if cy < height / 3 else "bottom" if cy > 2 * height / 3 else "middle"
    horizontal = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "center"
    return f"{vertical}-{horizontal}"
