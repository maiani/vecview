"""The three mirror planes of a cubic cell, crossing at its centre.

Each pair of planes cuts through the other, so every plane is partly in front
of and partly behind each of the others: no order of drawing whole planes is
right.  In a layer sorted by depth each plane is clipped to the part
of it that shows, the lines where planes meet are hidden only by the third
plane, the body diagonal fades to a faint dash exactly where a plane hides it,
and the cell's hidden edges are dropped where the planes cover them.

Uses: ``sort_by_depth``, ``polyline(back=...)``, ``edges(trim=...)``,
``rect_shape``, and ``sphere``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import vecview
from vecview import OrthographicCamera, Scene

INK = "#1f2430"
CORNER_R = 0.045
PLANES = {  # normal axis: fill, edge
    "x": ("#f2c14e", "#8a6a12"),
    "y": ("#78c091", "#2b6a40"),
    "z": ("#7aa6d8", "#2c5a8f"),
}


def build() -> Scene:
    camera = OrthographicCamera(azim_deg=-35.0, elev_deg=28.0, scale=190.0)
    scene = Scene(camera, pad=14.0, background="#ffffff")
    scene.sort_by_depth(10)

    cell = vecview.box_faces((0, 0, 0), (1, 1, 1))
    scene.edges(
        10,
        cell,
        back={"stroke_dasharray": "5 4", "stroke_opacity": 0.55},
        trim=CORNER_R,  # stop at the corner atoms rather than dash inside them
        id="cell",
        stroke=INK,
        stroke_width=1.3,
        stroke_linejoin="round",
    )

    # Each plane is the unit square spanned by the other two axes.
    for k, (axis, (fill, edge)) in enumerate(PLANES.items()):
        u, v = np.eye(3)[(k + 1) % 3], np.eye(3)[(k + 2) % 3]
        scene.polygon(
            10,
            vecview.rect_shape((0, 0, 0), u, v, 1.0, 1.0),
            fill=fill,
            stroke=edge,
            stroke_width=1.2,
            stroke_linejoin="round",
            id=f"m{axis}",
            class_="mirror-plane",
        )

    # Where two planes meet, a line: the cue the eye reads crossing surfaces by.
    # Each lies on two planes and is hidden only by the third.
    for axis, e in zip("xyz", np.eye(3), strict=True):
        scene.polyline(
            10,
            [-0.5 * e, 0.5 * e],
            stroke=INK,
            stroke_width=1.0,
            stroke_opacity=0.8,
            id=f"meet-{axis}",
            class_="intersection",
        )

    # The body diagonal pierces all three planes at the centre; where a plane
    # hides it, a faint dash shows the way it runs.
    scene.polyline(
        10,
        [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)],
        back={"stroke_dasharray": "3 4", "stroke_opacity": 0.35, "stroke_width": 1.4},
        stroke="#c0392b",
        stroke_width=2.2,
        stroke_linecap="round",
        id="diagonal",
    )
    for k, corner in enumerate(itertools.product((-0.5, 0.5), repeat=3)):
        scene.sphere(
            10,
            corner,
            CORNER_R,
            fill="#3a3f47",
            highlight="#aab1bc",
            stroke=INK,
            stroke_width=0.6,
            id=f"corner-{k}",
            class_="atom",
        )
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "mirror_planes.svg"))
