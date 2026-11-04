"""Parser for Playwright MCP aria snapshots (YAML-like accessibility tree)."""

from __future__ import annotations

import json
import re

from backend.dsl.observation import BoundingBox, SnapshotNode

NODE_RE = re.compile(
    r"""^(?P<role>[A-Za-z][\w-]*)(?:\s+"(?P<name>(?:[^"\\]|\\.)*)")?(?P<attrs>(?:\s+\[[^\]]*\])*)\s*(?::\s*(?P<text>.*))?$"""
)
ATTR_RE = re.compile(r"\[(?P<key>[\w-]+)(?:=(?P<value>[^\]]*))?\]")
PROP_RE = re.compile(r"^/(?P<key>[\w-]+):\s*(?P<value>.*)$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _unescape_name(name: str) -> str:
    try:
        return json.loads(f'"{name}"')
    except json.JSONDecodeError:
        return name.replace('\\"', '"')


def parse_snapshot(yaml_text: str) -> list[SnapshotNode]:
    """Return nodes in document order with parent/child indices populated."""

    nodes: list[SnapshotNode] = []
    stack: list[tuple[int, int]] = []  # (indent, node index)
    for raw in yaml_text.splitlines():
        if not raw.strip():
            continue
        stripped = raw.lstrip(" ")
        indent = len(raw) - len(stripped)
        if not stripped.startswith("- "):
            continue
        body = stripped[2:].rstrip()
        if body.endswith(":") and not body.endswith("::"):
            body_core = body[:-1]
        else:
            body_core = body
        # YAML may single-quote a whole key containing special characters
        if body_core.startswith("'") and "'" in body_core[1:]:
            end = body_core.rfind("'")
            key_part = body_core[1:end].replace("''", "'")
            rest = body_core[end + 1:]
            body_core = key_part + rest
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent_index = stack[-1][1] if stack else None

        prop = PROP_RE.match(body_core)
        if prop and parent_index is not None:
            nodes[parent_index].props[prop.group("key")] = _unquote(prop.group("value"))
            continue
        m = NODE_RE.match(body_core)
        if not m:
            continue
        role = m.group("role")
        name = _unescape_name(m.group("name")) if m.group("name") is not None else ""
        attrs: dict[str, object] = {}
        ref = None
        box = None
        for am in ATTR_RE.finditer(m.group("attrs") or ""):
            key, value = am.group("key"), am.group("value")
            if key == "ref":
                ref = value
            elif key == "box" and value:
                try:
                    x, y, w, h = (float(v) for v in value.split(","))
                    box = BoundingBox(x=x, y=y, width=w, height=h)
                except ValueError:
                    pass
            elif value is None:
                attrs[key] = True
            else:
                attrs[key] = value
        text = _unquote(m.group("text")) if m.group("text") else ""
        # Frame membership is structural: Playwright MCP prefixes refs per document ("f1e3"),
        # including the top-level page after navigations, so the prefix alone says nothing.
        frame = None
        if parent_index is not None:
            parent = nodes[parent_index]
            frame = (parent.ref or f"iframe{parent.index}") if parent.role == "iframe" else parent.frame
        node = SnapshotNode(ref=ref, role=role, name=name, text=text, attrs=attrs, box=box,
                            depth=len(stack), parent_index=parent_index, index=len(nodes), frame=frame)
        if parent_index is not None:
            nodes[parent_index].children.append(node.index)
        nodes.append(node)
        if body.endswith(":"):
            stack.append((indent, node.index))
        else:
            # leaf nodes can still own deeper property lines (e.g. "- link ... :" handled above)
            stack.append((indent, node.index))
    _absolutize_frame_boxes(nodes)
    return nodes


def _absolutize_frame_boxes(nodes: list[SnapshotNode]) -> None:
    """Boxes inside iframes are frame-relative; shift them by the iframe's position."""

    for node in nodes:
        if node.role != "iframe" or node.box is None:
            continue
        offset_x, offset_y = node.box.x + 2, node.box.y + 2  # default 2px iframe border
        stack = list(node.children)
        while stack:
            idx = stack.pop()
            child = nodes[idx]
            if child.box is not None and child.attrs.get("_abs") is None:
                child.box = child.box.offset(offset_x, offset_y)
                child.attrs["_abs"] = True
            stack.extend(child.children)


def node_text(nodes: list[SnapshotNode], index: int, limit: int = 300) -> str:
    """Concatenated visible text of a subtree (names + inline text)."""

    parts: list[str] = []
    stack = [index]
    while stack and sum(len(p) for p in parts) < limit:
        node = nodes[stack.pop(0)]
        if node.name:
            parts.append(node.name)
        if node.text:
            parts.append(node.text)
        stack[0:0] = node.children
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return " ".join(seen)[:limit]


def ancestors(nodes: list[SnapshotNode], index: int) -> list[SnapshotNode]:
    out = []
    cur = nodes[index].parent_index
    while cur is not None:
        out.append(nodes[cur])
        cur = nodes[cur].parent_index
    return out
