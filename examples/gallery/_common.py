"""Shared by the gallery figures: output paths, export, and a colour mix.

Rasterizing is deliberately not vecview's job; cairosvg is example-only.
"""

from __future__ import annotations

from pathlib import Path

import cairosvg

from vecview import Scene

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out" / "gallery"
DOCS = HERE.parents[1] / "docs" / "gallery"


def export(scene: Scene, name: str, *, docs: bool = False) -> Path:
    """Write ``name.svg`` and a 2x ``name.png`` to ``examples/out/gallery``.

    With ``docs``, the SVG is also copied into ``docs/gallery`` for the
    documentation page, which shows the vector original rather than a bitmap.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = scene.save(OUT / f"{name}.svg")
    cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix(".png")), scale=2.0)
    if docs:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def mix(a: str, b: str, t: float) -> str:
    """The colour a fraction ``t`` of the way from hex ``a`` to hex ``b``."""
    t = min(max(t, 0.0), 1.0)
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))
