"""A cubic perovskite ABO3 unit cell, drawn the way a crystallographer would.

The B cation sits in a translucent BO6 octahedron, the A cations on the cell
corners, and the cell edges are dashed where the cell hides them.  Atoms, cell
edges, and octahedron faces share one depth-sorted layer, so every overlap is
resolved without hand-assigned layers.

Uses: ``sort_by_depth``, ``sphere(highlight=...)``, ``convex_polyhedron``,
and ``edges(trim=...)``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import NamedTuple

import numpy as np

import vecview
from vecview import OrthographicCamera, Scene


class Atom(NamedTuple):
    radius: float  # in units of the cell edge
    fill: str
    highlight: str
    legend: str


ATOMS = {
    "A": Atom(0.13, "#3f9b54", "#c8ebd0", "A site"),
    "B": Atom(0.10, "#3b6fb6", "#cfe0f7", "B site"),
    "O": Atom(0.085, "#d23c2c", "#f9d0c8", "oxygen"),
}
HALF = 0.5  # the cell is centred on the B cation


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=-62.0, elev_deg=18.0, scale=230.0)
    scene = Scene(cam, pad=12.0, background="#ffffff")
    scene.sort_by_depth(10)

    corners = [np.array(c) - HALF for c in itertools.product((0.0, 1.0), repeat=3)]
    oxygens = [s * HALF * e for e in np.eye(3) for s in (1.0, -1.0)]  # the face centres

    cell = vecview.box_faces((0, 0, 0), (1, 1, 1))
    scene.edges(
        10,
        cell,
        back={"stroke_dasharray": "5 4", "stroke_width": 1.0, "stroke_opacity": 0.6},
        trim=ATOMS["A"].radius,  # stop at the atoms rather than dash inside them
        id="cell",
        stroke="#2b2b2b",
        stroke_width=1.3,
        stroke_linecap="round",
    )

    # Translucent, so every face shows, and stroked, so its edges are clipped
    # with it wherever an oxygen in front hides them.
    scene.faces(
        10,
        vecview.convex_polyhedron(oxygens),
        fill="#6f9fd8",
        fill_opacity=0.35,
        stroke="#2d5c94",
        stroke_width=1.0,
        stroke_linejoin="round",
        id="octahedron",
    )

    def atom(kind: str, at: np.ndarray, id: str) -> None:
        radius, fill, highlight, _ = ATOMS[kind]
        scene.sphere(
            10,
            at,
            radius,
            fill=fill,
            highlight=highlight,
            stroke="#1d1d1d",
            stroke_width=0.9,
            id=id,
            class_=["atom", f"site-{kind}"],
        )

    for k, at in enumerate(corners):
        atom("A", at, f"A-{k}")
    for k, at in enumerate(oxygens):
        atom("O", at, f"O-{k}")
    atom("B", np.zeros(3), "B")

    # A legend beside the cell, laid out along the screen's own directions.
    right, down = cam.screen_basis(cam.view)
    for k, (kind, entry) in enumerate(ATOMS.items()):
        at = 1.05 * right + (0.32 * k - 0.3) * down
        atom(kind, at, f"legend-{kind}")
        scene.text(20, at, entry.legend, dx=40, dy=6, size=17, id=f"legend-{kind}-label")
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "perovskite.svg"))
