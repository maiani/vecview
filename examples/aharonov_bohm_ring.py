"""An Aharonov-Bohm ring: a mesoscopic loop with two leads, threaded by a flux.

An electron entering from the left lead can pass either arm of the ring, and
the two paths enclose the flux, so the conductance oscillates with it.  The
ring and its leads are one shape with a hole: the union of an annulus and two
strips, which ``outlines.union`` returns as a ``Region`` and
``prism_faces`` extrudes into one solid -- caps with a hole, and walls inside
the hole as well as outside.  The flux line comes down through the hole, in
one depth-sorted layer with the ring: it covers the far arm and the path along
it, and stands inside the ring rather than on top of it.

Uses: ``annulus``, ``outlines.union`` with a hole,
``prism_walls``, ``sort_by_depth``, ``arrow3d``, and ``arc_shape``.
"""

from __future__ import annotations

from pathlib import Path

import vecview
from vecview import OrthographicCamera, Scene, outlines

R_IN, R_OUT = 1.0, 1.4  # the ring's inner and outer radius
LEAD_W, LEAD_X = 0.5, 3.4  # lead width, and how far out each lead runs
T = 0.14  # thickness of the metal
R_PATH = 0.5 * (R_IN + R_OUT)  # where the two electron paths run along the arms

METAL, METAL_TOP = "#8a96a8", "#c7d0dc"
PAD = "#e2b56a"
INK = "#1f2430"


def device() -> vecview.Region:
    """The ring and its two leads as one shape: an outline and the ring's hole."""
    ring = vecview.annulus((0, 0), R_IN, R_OUT, n=96)
    leads = [outlines.rect((-LEAD_X, -LEAD_W / 2), (-R_IN - 0.1, LEAD_W / 2))]
    leads.append(outlines.rect((R_IN + 0.1, -LEAD_W / 2), (LEAD_X, LEAD_W / 2)))
    (shape,) = outlines.union(ring, *leads)  # the leads join the ring; the hole stays
    return shape


def electron_paths(scene: Scene) -> None:
    """The two paths round the arms, which together enclose the flux."""
    for name, (t0, t1), colour in (
        ("upper", (180, 0), "#c0392b"),
        ("lower", (180, 360), "#2e86c1"),
    ):
        path = vecview.arc_shape((0, 0, T + 0.01), (1, 0, 0), (0, 1, 0), R_PATH, t0, t1, n=64)
        scene.polyline(
            20, path, stroke=colour, stroke_width=2.4, stroke_linecap="round", id=f"path-{name}"
        )
        # A head halfway along, pointing the way the electron goes.
        mid = len(path) // 2
        scene.arrow(
            20,
            path[mid],
            path[mid + 1] - path[mid - 1],
            0.32,
            normal=(0, 0, 1),
            shaft_w=0.0,
            head_w=0.24,
            head_len=0.32,
            pivot="mid",
            fill=colour,
            id=f"head-{name}",
        )


def build() -> Scene:
    camera = OrthographicCamera(azim_deg=-62.0, elev_deg=34.0, scale=96.0)
    scene = Scene(camera, pad=16.0, background="#ffffff")
    scene.sort_by_depth(20)

    scene.faces(
        10,
        vecview.box_faces((0, 0, -0.2), (7.6, 4.4, 0.4)),
        cull=True,
        fill="#e4e8ee",
        stroke="#9aa4b2",
        stroke_width=0.8,
        id="substrate",
    )

    # One solid, hole and all: its walls, inside the hole too, as one seamless
    # shape, and the top over them.  The depth sort does the rest.
    shape = device()
    scene.prism_walls(20, shape, 0.0, T, fill=METAL, id="ring-walls")
    top = [f for f in vecview.prism_faces(shape, 0.0, T) if f.name == "+z"]
    scene.faces(20, top, fill=METAL_TOP, stroke=INK, stroke_width=0.7, id="ring")

    for name, x in (("source", -LEAD_X - 0.25), ("drain", LEAD_X + 0.25)):
        pad = vecview.box_faces((x, 0, T), (1.0, 1.4, 0.3))
        scene.faces(20, pad, cull=True, fill=PAD, stroke=INK, stroke_width=0.6, id=name)

    # The flux threads the hole: the arrow ends on the substrate, inside the ring.
    scene.arrow3d(
        20,
        (0, 0, 2.3),
        (0, 0, -1),
        2.3,
        shaft_r=0.06,
        head_r=0.17,
        head_len=0.4,
        fill="#6c3483",
        id="flux",
    )
    scene.text(30, (0, 0, 2.3), "Φ", dx=14, dy=6, size=30, font_style="italic", id="flux-label")
    electron_paths(scene)
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "aharonov_bohm_ring.svg"))
