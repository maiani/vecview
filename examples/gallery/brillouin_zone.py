"""The first Brillouin zone of the fcc lattice, with its high-symmetry path.

The truncated octahedron comes from its 24 corners alone through
``convex_polyhedron``, which merges coplanar triangles into its eight hexagons
and six squares.  Hidden edges are dashed and sit under the translucent faces,
and the path Gamma-X-W-K-Gamma-L-U-W-L-K runs inside.

Uses: ``convex_polyhedron``, ``edges(back=...)``, ``faces(cull=True)``, and
subscripted labels.
"""

from __future__ import annotations

import itertools

import numpy as np
import svg
from _common import export

import vecview
from vecview import OrthographicCamera, Scene

NAME = "brillouin_zone"

INK = "#1f2430"
PATH_COLOR = "#c0392b"
# In units of 2 pi / a, scaled by 2 so the corners are the permutations of (0, +-1, +-2).
POINTS = {
    "Γ": (0.0, 0.0, 0.0),
    "X": (0.0, 0.0, 2.0),
    "W": (1.0, 0.0, 2.0),
    "K": (1.5, 0.0, 1.5),
    "L": (1.0, 1.0, 1.0),
    "U": (0.5, 0.5, 2.0),
}
PATH = ["Γ", "X", "W", "K", "Γ", "L", "U", "W", "L", "K"]
OFFSETS = {  # each label's screen offset from its point
    "Γ": (-24, 20),
    "X": (-26, -6),
    "W": (8, -10),
    "K": (12, 10),
    "L": (10, 18),
    "U": (-6, -12),
}


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=24.0, elev_deg=20.0, scale=95.0)
    scene = Scene(cam, pad=14.0, background="#ffffff")

    # The 24 corners are the permutations of (0, +-1, +-2).
    corners = {
        tuple(sign * value for sign, value in zip(signs, perm, strict=True))
        for perm in itertools.permutations((0.0, 1.0, 2.0))
        for signs in itertools.product((1.0, -1.0), repeat=3)
    }
    zone = vecview.convex_polyhedron(sorted(corners))

    # Front edges over everything; hidden ones under the faces and the path.
    scene.edges(
        30,
        zone,
        back={"stroke_dasharray": "5 4", "stroke_width": 1.0, "stroke_opacity": 0.55},
        back_layer=5,
        id="zone-edge",
        stroke=INK,
        stroke_width=1.4,
        stroke_linejoin="round",
        stroke_linecap="round",
    )
    scene.faces(20, zone, cull=True, fill="#a9c8ea", fill_opacity=0.2, id="zone")

    # Reciprocal axes from Gamma, out through the square faces, with flat heads
    # turned to the camera so one seen end-on still reads as an arrow.
    for label, e in zip("xyz", np.eye(3), strict=True):
        scene.polyline(8, [np.zeros(3), 2.0 * e], stroke="#6b7280", stroke_width=1.0)
        scene.arrow(
            32,
            2.0 * e,
            e,
            0.9,
            normal="camera",
            shaft_w=0.035,
            head_w=0.2,
            head_len=0.3,
            fill="#6b7280",
            id=f"axis-{label}",
            class_="axis",
        )
        scene.text(
            40,
            2.95 * e,
            [svg.TSpan(text="k"), svg.TSpan(text=label, baseline_shift="sub", font_size=14)],
            dx=-6,
            dy=8,
            size=20,
            font_family="DejaVu Serif",
            font_style="italic",
            fill="#4b5563",
            id=f"label-k{label}",
            class_="label",
        )

    # The high-symmetry path, inside the zone and so under its front faces.
    stops = np.array([POINTS[name] for name in PATH])
    scene.polyline(
        10, stops, stroke=PATH_COLOR, stroke_width=2.4, stroke_linejoin="round", id="path"
    )
    for name, at in POINTS.items():
        scene.sphere(
            12,
            at,
            0.07,
            fill=PATH_COLOR,
            stroke="#5c1a12",
            stroke_width=0.6,
            id=f"point-{name}",
            class_="high-symmetry-point",
        )
        dx, dy = OFFSETS[name]
        scene.text(
            40,
            at,
            name,
            dx=dx,
            dy=dy,
            size=22,
            fill=PATH_COLOR,
            id=f"label-{name}",
            class_="label",
        )
    return scene


if __name__ == "__main__":
    export(build(), NAME)
