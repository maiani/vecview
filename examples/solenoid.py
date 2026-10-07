"""A solenoid: a copper coil wound on a core, and the field it makes.

The coil is one ``tube`` along a ``helix``, in a layer sorted by depth with the
core, so each turn passes behind the core and comes round in front of it again
without a layer assigned per half-turn.  The return field lines are dashed, in
the plane of the axis.

Uses: ``helix``, ``tube``, ``cylinder(highlight=...)``, ``arrow3d``,
``sort_by_depth``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vecview
from vecview import OrthographicCamera, Scene

CORE_R, COIL_R, WIRE_R = 0.78, 0.92, 0.1
LENGTH, TURNS = 4.4, 10
COPPER, COPPER_EDGE = "#c8743c", "#5e2f12"
FIELD = "#1f5fa8"
INK = "#2b2b2b"
X, Z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=-66.0, elev_deg=14.0, scale=62.0)
    scene = Scene(cam, pad=14.0, background="#ffffff")
    start = -LENGTH / 2 * X

    # Return field lines, closing outside the coil in the plane of its axis.
    t = np.linspace(0.04, np.pi - 0.04, 120)
    reach = LENGTH / 2 + 1.25
    for k, height in enumerate((2.1, 2.9, -2.1, -2.9)):  # two above the coil, two below
        loop = np.column_stack([reach * np.cos(t), np.zeros_like(t), height * np.sin(t)])
        scene.polyline(
            2,
            loop,
            stroke=FIELD,
            stroke_width=1.2,
            stroke_opacity=0.55,
            stroke_dasharray="6 4",
            id=f"field-line-{k}",
            class_="field-line",
        )

    scene.sort_by_depth(10)
    scene.cylinder(
        10,
        start - 0.3 * X,
        -start + 0.3 * X,
        CORE_R,
        fill="#9aa5b2",
        highlight="#eef1f4",
        stroke="#525c68",
        stroke_width=1.0,
        end_style={"fill": "#c4ccd5", "stroke": "#525c68"},
        id="core",
    )

    # One wire: down-lead, the coil, and the other down-lead.
    coil = vecview.helix(start, X, COIL_R, LENGTH / TURNS, TURNS, n_per_turn=64)
    drop = 1.9 * Z
    wire = np.vstack(
        [
            np.linspace(coil[0] - drop, coil[0], 12)[:-1],
            coil,
            np.linspace(coil[-1], coil[-1] - drop, 12)[1:],
        ]
    )
    scene.tube(
        10, wire, WIRE_R, chunk=3, fill=COPPER, stroke=COPPER_EDGE, stroke_width=0.9, id="coil"
    )

    # The field leaves the core along the axis.
    scene.arrow3d(
        10,
        -start + 0.35 * X,
        X,
        1.5,
        shaft_r=0.08,
        head_r=0.24,
        head_len=0.5,
        fill=FIELD,
        highlight="#cfe0f7",
        stroke="#0d2c55",
        stroke_width=0.7,
        id="field",
    )
    math = dict(font_family="DejaVu Serif", font_style="italic", size=26)
    scene.text(30, -start + 1.9 * X, "B", dx=8, dy=8, fill=FIELD, id="label-B", **math)

    # Current up the first lead and down the other, drawn just outside each.
    beside = [coil[0] - 1.25 * Z - 0.38 * X, coil[-1] - 1.25 * Z + 0.38 * X]
    for k, (at, flow) in enumerate(zip(beside, (Z, -Z), strict=True)):
        scene.arrow3d(
            20,
            at - 0.35 * flow,
            flow,
            0.7,
            shaft_r=0.025,
            head_r=0.085,
            head_len=0.22,
            fill=INK,
            id=f"current-{k}",
            class_="current",
        )
    scene.text(30, beside[0], "I", dx=-22, dy=8, fill=INK, id="label-I", **math)
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "solenoid.svg"))
