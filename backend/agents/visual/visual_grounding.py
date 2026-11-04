"""Visual and spatial grounding (spec section 9).

Used only when DOM grounding is insufficient: canvas / spatial surfaces,
icon-only or unlabeled controls, and refinement of ambiguous candidates.

* deterministic: colour/shape segmentation of a fresh screenshot inside the
  target surface's bounding box (no model needed for "click the red circle");
* multimodal (when a vision model is configured): set-of-marks prompting — the
  screenshot is annotated with numbered candidate boxes and the model may only
  answer with one of those numbers or a point inside a known surface.

Coordinates are always computed from a *fresh* screenshot of the *current*
viewport and validated (inside the surface, not obstructed) before use.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from backend.core.exceptions import ModelError, ModelUnavailable
from backend.core.logging import get_logger
from backend.dsl.observation import BoundingBox, InventoryElement
from backend.harness.model_gateway import ImageInput, ModelGateway, ModelTier

log = get_logger(__name__)

COLOR_HUES = {  # hue ranges in degrees (HSV)
    "red": [(0, 14), (345, 360)], "orange": [(15, 40)], "yellow": [(41, 70)], "green": [(71, 165)],
    "cyan": [(166, 195)], "teal": [(160, 190)], "blue": [(196, 255)], "purple": [(256, 290)],
    "violet": [(256, 290)], "magenta": [(291, 330)], "pink": [(300, 344)],
}
SHAPES = ("circle", "square", "rectangle", "triangle", "star", "ellipse", "oval", "diamond")


@dataclass
class Blob:
    color: str
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    cx: float
    cy: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def fill_ratio(self) -> float:
        return self.area / max(1, self.width * self.height)

    def shape(self) -> str:
        aspect = self.width / max(1, self.height)
        fill = self.fill_ratio
        if fill > 0.9:
            return "square" if 0.85 <= aspect <= 1.18 else "rectangle"
        if 0.70 <= fill <= 0.86 and 0.8 <= aspect <= 1.25:
            return "circle"
        if 0.70 <= fill <= 0.86:
            return "ellipse"
        if 0.4 <= fill < 0.62:
            return "triangle" if aspect >= 0.8 else "diamond"
        return "shape"


@dataclass
class VisualMatch:
    x: float
    y: float
    confidence: float
    method: Literal["color_segmentation", "vision_model", "box_center"]
    evidence: list[str] = field(default_factory=list)
    box: BoundingBox | None = None


def _rgb_to_hue_sat_val(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = arr.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    sat = np.where(maxc > 0, delta / np.maximum(maxc, 1e-6), 0)
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    rc = np.where(mask, (maxc - r) / np.maximum(delta, 1e-6), 0)
    gc = np.where(mask, (maxc - g) / np.maximum(delta, 1e-6), 0)
    bc = np.where(mask, (maxc - b) / np.maximum(delta, 1e-6), 0)
    hue = np.where(r == maxc, bc - gc, hue)
    hue = np.where(g == maxc, 2.0 + rc - bc, hue)
    hue = np.where(b == maxc, 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0 * 360.0
    return hue, sat, maxc


def segment_color(image: Image.Image, region: BoundingBox | None, color: str, *, min_area: int = 120) -> list[Blob]:
    """Connected components of pixels matching ``color`` inside ``region`` (viewport CSS px)."""

    img = image.convert("RGB")
    ox, oy = 0, 0
    if region is not None:
        ox, oy = max(0, int(region.x)), max(0, int(region.y))
        img = img.crop((ox, oy, min(img.width, int(region.x + region.width)), min(img.height,
                                                                                 int(region.y + region.height))))
    arr = np.asarray(img)
    hue, sat, val = _rgb_to_hue_sat_val(arr)
    if color in ("black",):
        mask = val < 0.2
    elif color in ("white",):
        mask = (val > 0.92) & (sat < 0.08)
    elif color in ("gray", "grey"):
        mask = (sat < 0.12) & (val > 0.25) & (val < 0.85)
    else:
        ranges = COLOR_HUES.get(color)
        if not ranges:
            return []
        hue_mask = np.zeros(hue.shape, dtype=bool)
        for lo, hi in ranges:
            hue_mask |= (hue >= lo) & (hue <= hi)
        mask = hue_mask & (sat > 0.45) & (val > 0.3)
    return _components(mask, color, ox, oy, min_area)


def _components(mask: np.ndarray, color: str, ox: int, oy: int, min_area: int) -> list[Blob]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    blobs: list[Blob] = []
    current = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if labels[sy, sx]:
            continue
        current += 1
        stack = [(sy, sx)]
        labels[sy, sx] = current
        area = 0
        x0 = x1 = sx
        y0 = y1 = sy
        sumx = sumy = 0
        while stack:
            y, x = stack.pop()
            area += 1
            sumx += x
            sumy += y
            x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        if area >= min_area:
            blobs.append(Blob(color, x0 + ox, y0 + oy, x1 + ox, y1 + oy, area, sumx / area + ox, sumy / area + oy))
    blobs.sort(key=lambda b: -b.area)
    return blobs


def safe_point(blob: Blob) -> tuple[float, float]:
    """A click point well inside the shape (centroid; for triangles nudge toward the base)."""

    if blob.shape() == "triangle":
        return blob.cx, blob.y0 + blob.height * 0.66
    return blob.cx, blob.cy


class SetOfMarksChoice(BaseModel):
    mark: int | None = Field(description="Number of the marked box matching the description, or null")
    confidence: float = Field(ge=0, le=1)
    reason: str


class SurfacePoint(BaseModel):
    found: bool
    x_permille: int = Field(ge=0, le=1000, description="X inside the outlined surface, 0..1000")
    y_permille: int = Field(ge=0, le=1000, description="Y inside the outlined surface, 0..1000")
    confidence: float = Field(ge=0, le=1)
    reason: str


def annotate_marks(png: bytes, boxes: list[tuple[int, BoundingBox]], *, outline: tuple[int, int, int] = (255, 0, 102),
                   label: bool = True) -> bytes:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for number, box in boxes:
        x0, y0 = box.x, box.y
        x1, y1 = box.x + box.width, box.y + box.height
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=3)
        if label:
            draw.rectangle([x0, max(0, y0 - 14), x0 + 22, y0], fill=outline)
            draw.text((x0 + 3, max(0, y0 - 13)), str(number), fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class VisualGroundingAgent:
    name = "visual_grounding_agent"

    def __init__(self, gateway: ModelGateway | None = None) -> None:
        self.gateway = gateway

    def locate_in_surface(self, png: bytes, surface: BoundingBox, description: str) -> VisualMatch | None:
        """Deterministic colour/shape grounding inside a spatial surface (canvas, map, chart)."""

        words = description.lower().split()
        color = next((w for w in words if w in COLOR_HUES or w in ("black", "white", "gray", "grey")), None)
        shape = next((w for w in words if w in SHAPES), None)
        if color is None:
            return None
        image = Image.open(io.BytesIO(png))
        blobs = segment_color(image, surface, color)
        if not blobs:
            return None
        ranked: list[tuple[float, Blob]] = []
        for blob in blobs:
            score = 0.6
            evidence_shape = blob.shape()
            if shape:
                if evidence_shape == shape or {evidence_shape, shape} <= {"square", "rectangle"} or \
                        {evidence_shape, shape} <= {"circle", "ellipse", "oval"}:
                    score += 0.35
                else:
                    score -= 0.3
            score += min(0.05, blob.area / 200000)
            ranked.append((score, blob))
        ranked.sort(key=lambda item: -item[0])
        score, blob = ranked[0]
        if len(ranked) > 1 and ranked[1][0] >= score - 0.05:
            score -= 0.25  # two similar shapes: ambiguous
        x, y = safe_point(blob)
        if not surface.contains_point(x, y):
            return None
        return VisualMatch(x=x, y=y, confidence=round(max(0.0, min(0.97, score)), 3), method="color_segmentation",
                           evidence=[f"{blob.color} {blob.shape()} blob {blob.width}x{blob.height}px "
                                     f"fill={blob.fill_ratio:.2f} at ({x:.0f},{y:.0f})"],
                           box=BoundingBox(x=blob.x0, y=blob.y0, width=blob.width, height=blob.height))

    async def choose_mark(self, png: bytes, candidates: list[InventoryElement], description: str) -> tuple[
            InventoryElement | None, float, str]:
        """Ask a vision model to pick among numbered DOM candidates (never free coordinates)."""

        if self.gateway is None or not self.gateway.supports_vision():
            return None, 0.0, "vision model unavailable"
        boxes = [(i + 1, c.bbox) for i, c in enumerate(candidates) if c.bbox is not None]
        annotated = annotate_marks(png, boxes)  # type: ignore[arg-type]
        listing = "\n".join(f"{i}: {c.role} {c.name!r}" for i, c in zip(range(1, len(candidates) + 1), candidates))
        try:
            result = await self.gateway.structured(
                tier=ModelTier.VISION, agent=self.name, output_model=SetOfMarksChoice, max_tokens=1500,
                system="You ground UI elements in screenshots. Choose only among the numbered boxes.",
                prompt=f"Which numbered box is: {description}?\nCandidates:\n{listing}",
                images=[ImageInput(annotated)])
        except (ModelError, ModelUnavailable) as exc:
            return None, 0.0, f"vision model failed: {exc}"
        choice = result.output
        if choice.mark is None or not (1 <= choice.mark <= len(candidates)):
            return None, choice.confidence, choice.reason
        return candidates[choice.mark - 1], choice.confidence, choice.reason

    async def locate_with_model(self, png: bytes, surface: BoundingBox, description: str) -> VisualMatch | None:
        if self.gateway is None or not self.gateway.supports_vision():
            return None
        annotated = annotate_marks(png, [(1, surface)], label=False)
        try:
            result = await self.gateway.structured(
                tier=ModelTier.VISION, agent=self.name, output_model=SurfacePoint, max_tokens=1500,
                system="You locate visual targets inside the outlined surface of a screenshot.",
                prompt=f"Inside the outlined area, where is: {description}? Answer in per-mille of the area.",
                images=[ImageInput(annotated)])
        except (ModelError, ModelUnavailable):
            return None
        point = result.output
        if not point.found:
            return None
        x = surface.x + surface.width * point.x_permille / 1000
        y = surface.y + surface.height * point.y_permille / 1000
        return VisualMatch(x=x, y=y, confidence=min(point.confidence, 0.9), method="vision_model",
                           evidence=[point.reason[:200]])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
