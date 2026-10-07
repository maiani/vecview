"""A gapped Dirac cone: E = +-sqrt(k^2 + Delta^2), with a chemical potential.

Both bands are meshes of small quads from ``surface_faces``, coloured by energy.
They share one depth-sorted layer with the momentum and energy axes and the
Fermi circle, all drawn as thin tubes, so the axes pass behind the front walls
of the bands and show again inside the upper band, with no layer assigned by
hand.

Uses: ``surface_faces``, ``sort_by_depth``, ``tube``, ``cone``, and ``slot``
holding labels typeset by TeX through VecTeX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vectex

import vecview
from vecview import OrthographicCamera, Scene

GAP = 0.25  # half-gap Delta, in units of the band's energy at k = 1
MU = 0.62  # chemical potential, in the upper band
K_MAX = 1.0
INK = "#1f2430"
UPPER = ("#fde5cf", "#d4560f")  # low to high |E|
LOWER = ("#d4e4f4", "#1f5fa8")
MU_COLOR = "#7a1fa2"
# The labels, typeset by TeX in one run.
LABELS = dict(
    zip(
        ("kx", "ky", "E", "mu", "gap"),
        vectex.render_many(
            [
                vectex.RenderItem("$k_x$", id_prefix="kx"),
                vectex.RenderItem("$k_y$", id_prefix="ky"),
                vectex.RenderItem("$E$", id_prefix="E"),
                vectex.RenderItem(r"$\mu$", id_prefix="mu", color=MU_COLOR),
                vectex.RenderItem(r"$2\Delta$", id_prefix="gap"),
            ],
            size_pt=18,
            color=INK,
        ),
        strict=True,
    )
)


def mix(a: str, b: str, t: float) -> str:
    """The colour a fraction ``t`` of the way from hex ``a`` to hex ``b``."""
    t = min(max(t, 0.0), 1.0)
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


def band(sign: float) -> list[vecview.Face]:
    """The upper (+1) or lower (-1) band as a polar mesh of quads."""
    k = np.linspace(0.0, K_MAX, 13)
    t = np.linspace(0.0, 2.0 * np.pi, 57)
    kk, tt = np.meshgrid(k, t, indexing="ij")
    energy = sign * np.sqrt(kk**2 + GAP**2)
    return vecview.surface_faces(kk * np.cos(tt), kk * np.sin(tt), energy)


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=-58.0, elev_deg=17.0, scale=210.0)
    scene = Scene(cam, pad=14.0, background="#ffffff")
    scene.sort_by_depth(10)
    top = np.sqrt(K_MAX**2 + GAP**2)

    for sign, (light, dark), name in ((1.0, UPPER, "upper"), (-1.0, LOWER, "lower")):
        for face in band(sign):
            level = (abs(face.points[:, 2].mean()) - GAP) / (top - GAP)
            fill = mix(light, dark, level)
            scene.polygon(
                10,
                face.points,
                fill=fill,
                stroke=mix(fill, "#000000", 0.25),
                stroke_width=0.45,
                stroke_linejoin="round",
                id=f"{name}-{face.name}",
                class_=["band", name],
            )

    # The chemical potential cuts the upper band in a circle: the Fermi surface.
    k_f = np.sqrt(MU**2 - GAP**2)
    # It lies exactly on the mesh, so lift it a hair toward the camera: on the
    # surface itself it would tie with the quads beneath it in the depth sort.
    ring = vecview.circle_shape((0, 0, MU), k_f, (0, 0, 1), n=96) + 0.04 * cam.view
    scene.tube(
        10,
        np.vstack([ring, ring[:1]]),
        0.012,
        fill=MU_COLOR,
        chunk=2,
        id="fermi",
        class_="fermi-surface",
    )

    # Axes as thin tubes, so they sort among the band quads, with cone heads.
    axes = {
        "kx": ((-1.35, 0, 0), (1.45, 0, 0)),
        "ky": ((0, -1.35, 0), (0, 1.45, 0)),
        "E": ((0, 0, -1.3), (0, 0, 1.45)),
    }
    for name, (tail, tip) in axes.items():
        tail, tip = np.asarray(tail, float), np.asarray(tip, float)
        along = vecview.unit(tip - tail)
        shaft = np.linspace(tail, tip - 0.1 * along, 60)  # ends inside the head
        scene.tube(10, shaft, 0.006, fill=INK, chunk=2, id=f"axis-{name}", class_="axis")
        scene.cone(
            10,
            tip - 0.12 * along,
            tip,
            0.035,
            fill=INK,
            id=f"axis-{name}-head",
            class_="axis",
        )

    # Each label sits off its point by a screen offset, kept upright by a slot.
    for name, at, dx, dy in (
        ("kx", (1.5, 0, 0), 6, 10),
        ("ky", (0, 1.5, 0), 4, 12),
        ("E", (0, 0, 1.5), -8, -6),
        ("mu", (k_f, 0, MU), 10, -4),
        ("gap", (0, 0, 0), 14, -34),
    ):
        label = LABELS[name]
        scene.slot(
            30,
            at,
            label.width_px,
            label.height_px,
            id=f"label-{name}",
            align="southwest",
            dx=dx,
            dy=dy,
            content=label.to_svg_py(),
        )
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "dirac_cone.svg"))
