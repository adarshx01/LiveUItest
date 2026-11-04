"""Element inventory builder + relationship mapper (spec sections 8.3 / 8.4).

Correlates the accessibility snapshot (roles, names, fresh refs) with the DOM
probe (attributes, labels, test ids, CSS paths, visibility, obstruction) using
bounding boxes as the join key, then derives structural context: owning form,
dialog, landmark, section heading, row text and nearby label text.
"""

from __future__ import annotations

import re
from typing import Any

from backend.agents.observation.element_classifier import classify_element
from backend.agents.observation.snapshot_parser import ancestors, node_text
from backend.agents.planner.lexicon import INTERACTIVE_ROLES, LANDMARK_ROLES, classify_phrase, normalize_text
from backend.core.ids import stable_hash
from backend.dsl.observation import (
    BoundingBox,
    DialogInfo,
    ElementContext,
    FormInfo,
    InventoryElement,
    SnapshotNode,
)

INVENTORY_ROLES = INTERACTIVE_ROLES | {
    "heading", "img", "alert", "status", "dialog", "alertdialog", "row", "cell", "columnheader", "listitem",
    "article", "iframe", "canvas", "progressbar", "tooltip", "paragraph", "generic", "region", "list", "form",
    "radiogroup", "group", "tabpanel", "table", "grid", "figure", "log", "marquee", "note", "strong", "emphasis",
    "term", "definition", "caption", "label",
}
CONTENT_ROLES = {"paragraph", "generic", "strong", "emphasis", "listitem", "cell", "row", "status", "alert", "note",
                 "term", "definition", "caption", "label", "log", "article", "figure", "region", "list", "group",
                 "radiogroup", "table", "grid", "tabpanel", "form"}

ROLE_FROM_TAG = {"canvas": "canvas", "svg": "img", "img": "img", "a": "link", "button": "button", "select": "combobox",
                 "textarea": "textbox", "iframe": "iframe", "summary": "button", "details": "group", "label": "label",
                 "li": "listitem", "td": "cell"}


def _box_close(a: BoundingBox | None, b: list[float] | None, tol: float = 2.5) -> bool:
    if a is None or not b:
        return False
    return (abs(a.x - b[0]) <= tol and abs(a.y - b[1]) <= tol and abs(a.width - b[2]) <= tol
            and abs(a.height - b[3]) <= tol)


def _dom_role(item: dict[str, Any]) -> str:
    if item.get("role"):
        return str(item["role"]).split()[0]
    tag = item.get("tag") or ""
    if tag == "input":
        t = (item.get("type") or "text").lower()
        return {"checkbox": "checkbox", "radio": "radio", "submit": "button", "button": "button", "reset": "button",
                "image": "button", "file": "button", "range": "slider", "number": "spinbutton",
                "search": "searchbox"}.get(t, "textbox")
    return ROLE_FROM_TAG.get(tag, "generic")


def _role_compatible(snapshot_role: str, dom_item: dict[str, Any]) -> bool:
    dom_role = _dom_role(dom_item)
    if snapshot_role == dom_role:
        return True
    pairs = {("button", "button"), ("textbox", "textbox"), ("combobox", "textbox"), ("searchbox", "textbox"),
             ("generic", "generic"), ("img", "img")}
    return (snapshot_role, dom_role) in pairs or dom_item.get("role") == snapshot_role


class InventoryBuilder:
    def __init__(self, observation_id: str, viewport: tuple[int, int]) -> None:
        self.observation_id = observation_id
        self.viewport = viewport

    def build(self, nodes: list[SnapshotNode], dom: dict[str, Any] | None) -> tuple[
            list[InventoryElement], list[FormInfo], list[DialogInfo], list[str], list[str]]:
        dom_items: list[dict[str, Any]] = list((dom or {}).get("elements") or [])
        used_dom: set[int] = set()
        elements: list[InventoryElement] = []
        headings: list[str] = []
        visible_text: list[str] = []
        node_to_element: dict[int, InventoryElement] = {}

        for node in nodes:
            if node.role == "text" and node.text:
                visible_text.append(node.text)
            if node.role == "heading" and node.name:
                headings.append(node.name)
            if node.role not in INVENTORY_ROLES:
                continue
            if node.role in CONTENT_ROLES and not node.ref:
                continue
            if node.role in ("generic", "group", "region", "list") and not (node.name or node.text) \
                    and "cursor" not in node.attrs:
                continue
            dom_item = None
            if node.frame is None and node.box is not None:
                best = None
                for idx, item in enumerate(dom_items):
                    if idx in used_dom or not _box_close(node.box, item.get("box")):
                        continue
                    if _role_compatible(node.role, item):
                        best = idx
                        break
                    if best is None:
                        best = idx
                if best is not None:
                    dom_item = dom_items[best]
                    used_dom.add(best)
            element = self._element_from_node(node, nodes, dom_item, len(elements))
            elements.append(element)
            node_to_element[node.index] = element
            if node.text and node.role not in ("textbox", "combobox"):
                visible_text.append(node.text)

        # DOM-only elements: clickable/visual things the accessibility tree does not expose
        for idx, item in enumerate(dom_items):
            if idx in used_dom or not item.get("visible"):
                continue
            tag = item.get("tag")
            clickable = item.get("cursorPointer") or item.get("draggable") or tag in ("canvas",)
            if not clickable:
                continue
            elements.append(self._element_from_dom(item, len(elements)))

        self._attach_nearby_text(nodes, node_to_element)
        forms = self._forms(nodes, elements)
        dialogs = self._dialogs(nodes, elements, dom_items)
        for el in elements:
            purpose, conf = classify_element(el)
            el.semantic_purpose, el.purpose_confidence = purpose, conf
        for form in forms:
            form.purpose = _form_purpose(form, elements)
            form.fingerprint = stable_hash(*sorted(f"{e.role}:{normalize_text(e.name or e.label)}"
                                                   for e in elements if e.element_id in form.field_ids))
        return elements, forms, dialogs, headings, visible_text

    # ------------------------------------------------------------------ helpers
    def _element_from_node(self, node: SnapshotNode, nodes: list[SnapshotNode], dom: dict[str, Any] | None,
                           position: int) -> InventoryElement:
        ctx = ElementContext(frame=node.frame)
        ancestor_path: list[str] = []
        for anc in ancestors(nodes, node.index):
            if anc.role in ("form",) and ctx.form_ref is None:
                ctx.form_ref, ctx.form_name = anc.ref, anc.name or None
            elif anc.role in ("dialog", "alertdialog") and ctx.dialog_ref is None:
                ctx.dialog_ref, ctx.dialog_name = anc.ref, anc.name or None
            elif anc.role in LANDMARK_ROLES and anc.role != "form" and ctx.landmark is None:
                ctx.landmark, ctx.landmark_name = anc.role, anc.name or None
            elif anc.role in ("row", "listitem", "article") and ctx.row_text is None:
                ctx.row_text = node_text(nodes, anc.index, 200)
            elif anc.role in ("list", "table", "grid", "listbox", "menu", "tablist") and ctx.list_ref is None:
                ctx.list_ref = anc.ref
            elif anc.role == "iframe" and ctx.frame is None:
                ctx.frame = node.frame
            if anc.role not in ("generic",) or anc.name:
                ancestor_path.append(f"{anc.role}:{anc.name}" if anc.name else anc.role)
        ctx.ancestors = ancestor_path[:8]
        ctx.section_heading = _section_heading(nodes, node.index)
        box = node.box
        visible = box is not None and not box.is_empty() or (node.role == "option")
        if node.role == "option" and box is not None and box.is_empty():
            visible = False
        in_viewport = bool(box and not box.is_empty() and box.x + box.width > 0 and box.y + box.height > 0
                           and box.x < self.viewport[0] and box.y < self.viewport[1])
        options = [nodes[c].name for c in node.children if nodes[c].role == "option"] if node.role in (
            "combobox", "listbox") else []
        selected_options = [nodes[c].name for c in node.children
                            if nodes[c].role == "option" and nodes[c].attrs.get("selected")]
        element = InventoryElement(
            element_id=f"el{position}",
            ref=node.ref,
            role=node.role,
            name=node.name,
            text=node.text or (node_text(nodes, node.index, 160) if node.role in ("row", "listitem", "article",
                                                                                 "cell", "alert", "status",
                                                                                 "dialog", "region") else ""),
            placeholder=node.props.get("placeholder"),
            href=node.props.get("url"),
            visible=visible,
            enabled=not bool(node.attrs.get("disabled")),
            checked=_tri(node.attrs.get("checked")),
            selected=_tri(node.attrs.get("selected")),
            expanded=_tri(node.attrs.get("expanded")),
            options=options,
            selected_options=selected_options,
            bbox=box,
            in_viewport=in_viewport,
            clickable_hint=node.attrs.get("cursor") == "pointer",
            context=ctx,
            source="aria",
            snapshot_index=node.index,
            editable=node.role in ("textbox", "searchbox", "combobox", "spinbutton"),
        )
        if dom is not None:
            self._merge_dom(element, dom)
        return element

    @staticmethod
    def _merge_dom(element: InventoryElement, dom: dict[str, Any]) -> None:
        element.source = "aria+dom"
        element.tag = dom.get("tag")
        element.input_type = (dom.get("type") or None) if dom.get("tag") in ("input", "button") else None
        element.dom_id = dom.get("id")
        element.name_attr = dom.get("name")
        element.test_id = dom.get("testid")
        element.autocomplete = dom.get("autocomplete")
        element.title = dom.get("title")
        element.label = dom.get("label") or None
        element.placeholder = element.placeholder or dom.get("placeholder")
        element.css_selector = dom.get("css")
        element.classes = dom.get("classes")
        element.required = bool(dom.get("required"))
        element.readonly = bool(dom.get("readonly"))
        element.obstructed = bool(dom.get("obstructed"))
        element.visible = bool(dom.get("visible", element.visible))
        element.in_viewport = bool(dom.get("inViewport", element.in_viewport))
        if dom.get("disabled"):
            element.enabled = False
        if dom.get("checked") is not None and element.checked is None:
            element.checked = bool(dom["checked"])
        if dom.get("cursorPointer"):
            element.clickable_hint = True
        if element.input_type == "password":
            element.value_preview = "••••••" if dom.get("valueLength") else ""
        elif dom.get("value") is not None:
            element.value_preview = str(dom["value"])[:80]
        if dom.get("options") and not element.options:
            element.options = list(dom["options"])
        if dom.get("selectedOptions") is not None:
            element.selected_options = list(dom["selectedOptions"])
        if dom.get("legend"):
            element.context.fieldset_legend = dom["legend"]
        if dom.get("heading") and not element.context.section_heading:
            element.context.section_heading = dom["heading"]
        if dom.get("formId") and not element.context.form_ref:
            element.context.form_name = element.context.form_name or str(dom["formId"])
        if dom.get("iconHint"):
            element.nearby_text.append(f"icon:{dom['iconHint']}")

    def _element_from_dom(self, item: dict[str, Any], position: int) -> InventoryElement:
        box = item.get("box") or [0, 0, 0, 0]
        bbox = BoundingBox(x=box[0], y=box[1], width=box[2], height=box[3])
        role = _dom_role(item)
        if item.get("draggable"):
            role = role if role != "generic" else "generic"
        el = InventoryElement(
            element_id=f"el{position}", ref=None, role=role,
            name=item.get("ariaLabel") or item.get("title") or item.get("alt") or "",
            text=item.get("text") or "", tag=item.get("tag"), dom_id=item.get("id"), name_attr=item.get("name"),
            test_id=item.get("testid"), title=item.get("title"), css_selector=item.get("css"),
            classes=item.get("classes"), visible=True, enabled=not item.get("disabled"),
            bbox=bbox, in_viewport=bool(item.get("inViewport")), obstructed=bool(item.get("obstructed")),
            clickable_hint=bool(item.get("cursorPointer")), source="dom",
            context=ElementContext(form_name=item.get("formId"), section_heading=item.get("heading") or None),
        )
        if item.get("iconHint"):
            el.nearby_text.append(f"icon:{item['iconHint']}")
        from backend.agents.observation.element_classifier import glyph_words

        glyphs = glyph_words(item.get("text"))
        if glyphs:
            el.nearby_text.append(f"icon:{glyphs}")
        if item.get("draggable"):
            el.nearby_text.append("draggable")
        return el

    @staticmethod
    def _attach_nearby_text(nodes: list[SnapshotNode], mapping: dict[int, InventoryElement]) -> None:
        for index, element in mapping.items():
            node = nodes[index]
            if node.parent_index is None:
                continue
            siblings = nodes[node.parent_index].children
            pos = siblings.index(index)
            for sib_index in reversed(siblings[max(0, pos - 2):pos]):
                sib = nodes[sib_index]
                if sib.role == "text" and sib.text:
                    element.nearby_text.append(sib.text[:80])
                    if not element.label and element.role in ("textbox", "combobox", "searchbox", "spinbutton"):
                        element.label = sib.text[:80]
                    break
            for sib_index in siblings[pos + 1:pos + 2]:
                sib = nodes[sib_index]
                if sib.role == "text" and sib.text and element.role in ("checkbox", "radio", "switch"):
                    element.nearby_text.append(sib.text[:80])

    @staticmethod
    def _forms(nodes: list[SnapshotNode], elements: list[InventoryElement]) -> list[FormInfo]:
        forms: dict[str, FormInfo] = {}
        name_to_key: dict[str, str] = {}
        for el in elements:
            if el.context.form_ref and el.context.form_name:
                name_to_key.setdefault(el.context.form_name, el.context.form_ref)
        for el in elements:
            key = el.context.form_ref or (name_to_key.get(el.context.form_name or "") if el.context.form_name else None) \
                or (f"dom:{el.context.form_name}" if el.context.form_name else None)
            if not key or el.role == "form":
                continue
            if not el.context.form_ref and key in name_to_key.values():
                el.context.form_ref = key
            form = forms.setdefault(key, FormInfo(form_id=key, ref=el.context.form_ref, name=el.context.form_name))
            if el.role in ("textbox", "searchbox", "combobox", "spinbutton", "checkbox", "radio", "switch", "slider",
                           "listbox"):
                form.field_ids.append(el.element_id)
            elif el.role == "button" or (el.input_type in ("submit", "image")):
                form.submit_ids.append(el.element_id)
            if el.role in ("alert",) and el.text:
                form.validation_messages.append(el.text[:120])
        return [f for f in forms.values() if f.field_ids or f.submit_ids]

    def _dialogs(self, nodes: list[SnapshotNode], elements: list[InventoryElement],
                 dom_items: list[dict[str, Any]]) -> list[DialogInfo]:
        dialogs: list[DialogInfo] = []
        for node in nodes:
            if node.role not in ("dialog", "alertdialog"):
                continue
            if node.box is not None and node.box.is_empty():
                continue
            members = [e.element_id for e in elements if e.context.dialog_ref == node.ref and node.ref]
            modal = node.role == "alertdialog" or any(
                d.get("modal") and _box_close(node.box, d.get("box")) for d in dom_items) or bool(
                node.attrs.get("modal"))
            label = (node.name or node_text(nodes, node.index, 120)).lower()
            kind = "cookie_banner" if re.search(r"cookie|consent|gdpr|privacy", label) else (
                "alert" if node.role == "alertdialog" else "modal")
            dialogs.append(DialogInfo(ref=node.ref, name=node.name or node_text(nodes, node.index, 60), modal=modal,
                                      element_ids=members, bbox=node.box, kind=kind))  # type: ignore[arg-type]
        return dialogs


def _tri(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "mixed")


def _section_heading(nodes: list[SnapshotNode], index: int) -> str | None:
    """Nearest heading that precedes the node within an ancestor container."""

    for anc in [nodes[index], *ancestors(nodes, index)]:
        if anc.parent_index is None:
            continue
        siblings = nodes[anc.parent_index].children
        pos = siblings.index(anc.index)
        for sib_index in reversed(siblings[:pos]):
            sib = nodes[sib_index]
            if sib.role == "heading" and sib.name:
                return sib.name
    return None


def _form_purpose(form: FormInfo, elements: list[InventoryElement]) -> str:
    by_id = {e.element_id: e for e in elements}
    purposes = [by_id[i].semantic_purpose for i in form.field_ids if i in by_id]
    submit_text = " ".join(normalize_text(by_id[i].name) for i in form.submit_ids if i in by_id)
    has_password = "password" in purposes
    if has_password and ("confirm_password" in purposes or len(form.field_ids) >= 5 or
                         re.search(r"register|sign up|create", submit_text)):
        return "registration"
    if has_password:
        return "login"
    if purposes and all(p == "search" for p in purposes if p):
        return "search"
    if "card_number" in purposes or "cvv" in purposes:
        return "checkout"
    if "message" in purposes and ("email" in purposes or "full_name" in purposes):
        return "contact"
    match = classify_phrase(form.name or "")
    if match and match.purpose in ("submit_login",):
        return "login"
    return "data_entry"
