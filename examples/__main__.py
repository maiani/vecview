"""Build every example, with timings: ``python examples [--docs]``.

Each example is a self-contained script, also runnable on its own as
``python examples/<name>.py``.  This writes each one's SVG to ``examples/out``,
with a 2x PNG preview of every still figure; ``--docs`` also refreshes the
gallery SVGs in ``docs/gallery``.

Rasterizing is deliberately not vecview's job; cairosvg is example-only.
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import cairosvg

import vecview

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DOCS = HERE.parent / "docs" / "gallery"
EXAMPLES = [
    "perovskite",
    "brillouin_zone",
    "fullerene",
    "bloch_sphere",
    "dirac_cone",
    "skyrmion",
    "solenoid",
    "kitaev_chain",
    "mirror_planes",
    "altermagnetic_dot",  # at its active camera, "main"
    "slab_polarizer",  # at its default camera
    "pendulum",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", action="store_true", help="also copy the SVGs into docs/gallery")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in EXAMPLES:
        began = time.perf_counter()
        example = importlib.import_module(name).build()
        built = time.perf_counter() - began
        began = time.perf_counter()
        document = str(example.render())
        rendered = time.perf_counter() - began
        path = OUT / f"{name}.svg"
        path.write_text(document, encoding="utf-8")
        still = isinstance(example, vecview.Scene)
        if still:
            cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix(".png")), scale=2.0)
            if args.docs:
                DOCS.mkdir(parents=True, exist_ok=True)
                (DOCS / path.name).write_text(document, encoding="utf-8")
        print(
            f"{name:>18}  {'scene' if still else 'animation':>9}  build {built * 1000:4.0f} ms"
            f"  render {rendered * 1000:5.0f} ms  {len(document) / 1024:5.0f} kB"
        )


if __name__ == "__main__":
    main()
