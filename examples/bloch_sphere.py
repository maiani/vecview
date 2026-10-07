"""The Bloch sphere: a qubit state at polar angle theta and azimuth phi.

The translucent sphere is one native ellipse; the equator and the two meridians
through the poles, one great circle in each coordinate plane, are split exactly
where they pass behind it and dashed there; the state is a solid
arrow, with the angles marked by arcs in their own planes.

Uses: ``sphere(highlight=...)``, ``sphere_curve``, ``arrow3d``, ``arc_shape``,
the flat ``arrow(normal="camera")`` for the axes, and ``slot`` holding labels
typeset by TeX through VecTeX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vectex

import vecview
from vecview import OrthographicCamera, Scene

THETA, PHI = 52.0, 58.0  # the state, in degrees
INK = "#1f2430"
STATE = "#c0392b"
OUTLINE = "#2f4a63"  # the sphere's edge, heavier than the circles drawn on it
GRID = "#7d9ab6"
AXIS_LEN = 1.38
TEX = {  # name: TeX source, size in points, colour
    "x": ("$x$", 18, INK),
    "y": ("$y$", 18, INK),
    "z": ("$z$", 18, INK),
    "theta": (r"$\theta$", 16, INK),
    "phi": (r"$\varphi$", 16, INK),
    "north": (r"$\lvert 0 \rangle$", 16.5, INK),
    "south": (r"$\lvert 1 \rangle$", 16.5, INK),
    "psi": (r"$\lvert \psi \rangle$", 16.5, STATE),
}
LABELS = dict(
    zip(
        TEX,
        vectex.render_many(
            [
                vectex.RenderItem(source, size_pt=size, color=color, id_prefix=name)
                for name, (source, size, color) in TEX.items()
            ]
        ),
        strict=True,
    )
)


def bloch_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    """The unit vector at polar angle theta and azimuth phi."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def tex(scene: Scene, at, name: str, dx: float, dy: float, *, id: str) -> None:
    """The TeX label ``name`` upright at the world point ``at``, offset on screen."""
    label = LABELS[name]
    scene.slot(
        30,
        at,
        label.width_px,
        label.height_px,
        id=id,
        align="southwest",
        dx=dx,
        dy=dy,
        content=label.to_svg_py(),
    )


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=28.0, elev_deg=16.0, scale=170.0)
    scene = Scene(cam, pad=16.0, background="#ffffff")
    origin = np.zeros(3)
    psi = bloch_vector(THETA, PHI)
    foot = np.array([psi[0], psi[1], 0.0])  # psi dropped onto the equatorial plane

    # Inside the sphere, under its translucent skin: the axes and the
    # construction lines that drop the state onto the equatorial plane.
    thin = dict(stroke=INK, stroke_width=1.1)
    for label, e in zip("xyz", np.eye(3), strict=True):
        scene.polyline(8, [-e, e], id=f"axis-{label}-inner", class_="axis", **thin)
    dashed = dict(stroke="#6b7280", stroke_width=1.0, stroke_dasharray="4 3")
    scene.polyline(9, [origin, foot], id="foot", **dashed)
    scene.polyline(9, [psi, foot], id="drop", **dashed)

    scene.sphere(
        10,
        origin,
        1.0,
        fill="#9fc3e8",
        highlight="#ffffff",
        fill_opacity=0.32,
        stroke=OUTLINE,
        stroke_width=1.8,
        id="sphere",
    )
    hidden = {"stroke_dasharray": "5 4", "stroke_opacity": 0.55}
    # A great circle in each coordinate plane, by its normal: every axis point
    # is where two of them cross.
    circles = {"equator": (0, 0, 1), "meridian-xz": (0, 1, 0), "meridian-yz": (1, 0, 0)}
    for name, normal in circles.items():
        scene.sphere_curve(
            11,
            origin,
            vecview.circle_shape(origin, 1.0, normal, n=180),
            closed=True,
            back=hidden,
            back_layer=9,
            stroke=GRID,
            stroke_width=0.9,
            id=name,
        )

    # Outside the sphere, each axis continues to a flat head turned toward the
    # camera: a cone seen nearly tip-on, as x is here, reads as a blob.
    for e, label, (dx, dy) in zip(np.eye(3), "xyz", ((-20, 14), (10, 8), (-7, -4)), strict=True):
        scene.arrow(
            12,
            e,
            e,
            AXIS_LEN - 1.0,
            normal="camera",
            shaft_w=0.011,
            head_w=0.085,
            head_len=0.15,
            fill=INK,
            id=f"axis-{label}",
            class_="axis",
        )
        tex(scene, AXIS_LEN * e, label, dx, dy, id=f"label-{label}")

    # The state, its angles, and the poles.
    scene.arrow3d(
        20,
        origin,
        psi,
        1.0,
        shaft_r=0.022,
        head_r=0.06,
        head_len=0.16,
        fill=STATE,
        highlight="#f5b7b1",
        stroke="#5c1a12",
        stroke_width=0.6,
        id="state",
    )
    marks = dict(stroke=STATE, stroke_width=1.3)
    scene.polyline(
        21, vecview.arc_shape(origin, (0, 0, 1), psi, 0.36, 0.0, THETA), id="theta", **marks
    )
    scene.polyline(
        21, vecview.arc_shape(origin, (1, 0, 0), (0, 1, 0), 0.3, 0.0, PHI), id="phi", **marks
    )
    # Each angle's label sits just outside the middle of its arc.
    theta_at = 0.45 * bloch_vector(THETA / 2, PHI)
    tex(scene, theta_at, "theta", -4, 6, id="label-theta")
    phi_at = 0.42 * bloch_vector(90.0, PHI / 2)
    tex(scene, phi_at, "phi", -6, 16, id="label-phi")

    for name, z, dy in (("north", 1.0, -12), ("south", -1.0, 36)):
        scene.sphere(22, (0, 0, z), 0.03, fill=INK, id=f"pole-{name}")
        tex(scene, (0, 0, z), name, 10, dy, id=f"ket-{name}")
    tex(scene, psi, "psi", 10, -8, id="ket-psi")
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "bloch_sphere.svg"))
