"""Build every gallery figure: ``python examples/gallery [--docs]``.

Each figure is also runnable on its own, ``python examples/gallery/<name>.py``.
SVG and a 2x PNG go to ``examples/out/gallery``; ``--docs`` also refreshes the
SVGs the documentation's gallery page shows.  The two device examples one level
up are included, at their default projections.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Callable

from _common import HERE, export

from vecview import Scene

FIGURES = [
    "perovskite",
    "brillouin_zone",
    "fullerene",
    "bloch_sphere",
    "dirac_cone",
    "skyrmion",
    "solenoid",
    "kitaev_chain",
    "mirror_planes",
]


def devices() -> dict[str, Callable[[], Scene]]:
    """The two worked examples outside the gallery, at their default cameras."""
    sys.path.insert(0, str(HERE.parent))
    altermagnetic_dot = importlib.import_module("altermagnetic_dot")
    slab_polarizer = importlib.import_module("slab_polarizer")
    return {
        "altermagnetic_dot": altermagnetic_dot.build,  # its active camera is "main"
        "slab_polarizer": slab_polarizer.build,  # its default camera
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", action="store_true", help="also copy the SVGs into docs/gallery")
    args = ap.parse_args()
    builders: dict[str, Callable[[], Scene]] = {
        name: importlib.import_module(name).build for name in FIGURES
    }
    builders.update(devices())
    for name, build in builders.items():
        began = time.perf_counter()
        scene = build()
        built = time.perf_counter() - began
        began = time.perf_counter()
        document = scene.render()
        rendered = time.perf_counter() - began
        path = export(scene, name, docs=args.docs)
        size = path.stat().st_size / 1024
        count = len(document.elements or [])
        print(
            f"{name:>18}  {count:>5} elements  build {built * 1000:4.0f} ms"
            f"  render {rendered * 1000:5.0f} ms  {size:5.0f} kB"
        )


if __name__ == "__main__":
    main()
