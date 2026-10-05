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

from vecview import OrthographicCamera, Scene

FIGURES = [
    "perovskite",
    "brillouin_zone",
    "fullerene",
    "bloch_sphere",
    "dirac_cone",
    "skyrmion",
    "solenoid",
]


def devices() -> dict[str, Callable[[], Scene]]:
    """The two worked examples outside the gallery, at their default cameras."""
    sys.path.insert(0, str(HERE.parent))
    altermagnetic_dot = importlib.import_module("altermagnetic_dot")
    slab_polarizer = importlib.import_module("slab_polarizer")
    return {
        "altermagnetic_dot": lambda: altermagnetic_dot.build(
            altermagnetic_dot.PROJECTIONS["paper"](altermagnetic_dot.SCALE)
        ),
        "slab_polarizer": lambda: slab_polarizer.build(
            OrthographicCamera(slab_polarizer.AZIM, slab_polarizer.ELEV, slab_polarizer.SCALE),
            False,
        ),
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
        path = export(scene, name, docs=args.docs)
        size = path.stat().st_size / 1024
        print(f"{name:>18}  {len(scene.items):>5} elements  {built * 1000:6.0f} ms  {size:6.0f} kB")


if __name__ == "__main__":
    main()
