"""Evidence screenshots: highlight the grounded target / click point on the before-image."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont


def annotate_target(png: bytes, *, box: dict[str, float] | None = None, point: tuple[float, float] | None = None,
                    label: str | None = None, color: tuple[int, int, int] = (255, 64, 0)) -> bytes:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    if box:
        x0, y0 = box["x"], box["y"]
        x1, y1 = x0 + box["width"], y0 + box["height"]
        pad = 3
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(*color, 255), width=3,
                       fill=(*color, 40))
        if label:
            text = label[:60]
            tw = draw.textlength(text, font=font) + 8
            ty = y0 - pad - 16 if y0 - pad - 16 > 0 else y1 + pad + 2
            draw.rectangle([x0 - pad, ty, x0 - pad + tw, ty + 14], fill=(*color, 230))
            draw.text((x0 - pad + 4, ty + 1), text, fill=(255, 255, 255, 255), font=font)
    if point:
        px, py = point
        draw.ellipse([px - 9, py - 9, px + 9, py + 9], outline=(*color, 255), width=3)
        draw.line([px - 14, py, px + 14, py], fill=(*color, 255), width=2)
        draw.line([px, py - 14, px, py + 14], fill=(*color, 255), width=2)
    merged = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    out = io.BytesIO()
    merged.save(out, format="PNG", optimize=True)
    return out.getvalue()


def thumbnail(png: bytes, width: int = 640) -> bytes:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.width > width:
        image = image.resize((width, int(image.height * width / image.width)))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=80, optimize=True)
    return out.getvalue()
