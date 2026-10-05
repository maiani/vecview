"""Build the picture at the top of the README.

It is the device sketch of ``examples/altermagnetic_dot.py`` -- gates, leads,
spin densities, and a bias circuit on a layered slab -- seen by its main
camera.  The example leaves its background transparent, for placing on a
page; here it is rendered on white, so it reads on a dark page too.

Usage:
    python docs/readme_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from altermagnetic_dot import build  # noqa: E402

OUT = ROOT / "docs" / "images" / "readme.svg"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    document = build().render(pad=6.0, background="#ffffff")
    OUT.write_text(str(document), encoding="utf-8")
    print(f"wrote {OUT}")
