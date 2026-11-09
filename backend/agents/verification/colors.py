"""Named colours for visual checks ("1, 2, 3 are in sky blue", "the Total cell is red").

A tester names a colour family, not a hex value. Colours are compared by hue (with saturation and
lightness bands for the achromatic ones), so "sky blue" matches #87CEEB, #5BC0EB and #00B0F0, while a
deep navy or a teal does not.
"""

from __future__ import annotations

import colorsys
import re

# name -> (hue degrees or None for achromatic, hue tolerance, lightness range, minimum saturation)
NAMED: dict[str, tuple[float | None, float, tuple[float, float], float]] = {
    "red": (0, 22, (0.20, 0.75), 0.35), "crimson": (348, 15, (0.2, 0.6), 0.4), "maroon": (0, 20, (0.1, 0.35), 0.3),
    "orange": (30, 15, (0.3, 0.75), 0.4), "amber": (42, 10, (0.3, 0.75), 0.4), "gold": (50, 10, (0.3, 0.7), 0.4),
    "yellow": (57, 12, (0.35, 0.9), 0.4), "olive": (60, 20, (0.15, 0.45), 0.25), "lime": (95, 25, (0.3, 0.75), 0.4),
    "green": (120, 40, (0.15, 0.75), 0.25), "teal": (180, 12, (0.15, 0.5), 0.3), "turquoise": (174, 14, (0.35, 0.8), 0.3),
    "cyan": (185, 14, (0.3, 0.9), 0.35), "sky blue": (198, 18, (0.45, 0.92), 0.3),
    "light blue": (200, 22, (0.55, 0.95), 0.2), "blue": (220, 25, (0.2, 0.75), 0.35),
    "navy": (235, 20, (0.08, 0.35), 0.3), "royal blue": (225, 12, (0.35, 0.7), 0.45),
    "indigo": (265, 15, (0.15, 0.5), 0.3), "purple": (280, 25, (0.15, 0.7), 0.25), "violet": (285, 25, (0.4, 0.85), 0.25),
    "magenta": (305, 18, (0.3, 0.8), 0.4), "pink": (340, 25, (0.6, 0.95), 0.25), "brown": (25, 20, (0.1, 0.45), 0.25),
    "beige": (45, 20, (0.75, 0.95), 0.15), "white": (None, 0, (0.93, 1.0), 0.0), "black": (None, 0, (0.0, 0.12), 0.0),
    "grey": (None, 0, (0.3, 0.85), 0.0), "gray": (None, 0, (0.3, 0.85), 0.0), "silver": (None, 0, (0.65, 0.85), 0.0),
}


def parse_css_color(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    m = re.match(r"rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)(?:[,\s/]+([\d.]+))?\s*\)", value.strip())
    if m:
        if m.group(4) is not None and float(m.group(4)) == 0:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r"#([0-9a-fA-F]{6})", value.strip())
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return None


def _family(name: str) -> tuple[float | None, float, tuple[float, float], float] | None:
    key = " ".join(name.lower().split())
    if key in NAMED:
        return NAMED[key]
    base = key.split()[-1]
    if base in NAMED:
        hue, tol, (lo, hi), sat = NAMED[base]
        if key.startswith(("light", "pale", "bright")):
            return hue, tol + 5, (max(lo, 0.45), 0.97), sat * 0.6
        if key.startswith(("dark", "deep")):
            return hue, tol + 5, (0.05, min(hi, 0.5)), sat
        return hue, tol, (lo, hi), sat
    return None


def matches_color(rgb: tuple[int, int, int], name: str) -> bool:
    family = _family(name)
    if family is None:
        return False
    hue, tol, (lo, hi), min_sat = family
    h, lightness, s = colorsys.rgb_to_hls(*(c / 255 for c in rgb))
    if not lo <= lightness <= hi:
        return False
    if hue is None:
        return s < 0.18
    if s < min_sat:
        return False
    diff = abs(h * 360 - hue) % 360
    return min(diff, 360 - diff) <= tol


def known_color(name: str) -> bool:
    return _family(name) is not None
