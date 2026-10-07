"""The Bloch sphere: a qubit state at polar angle theta and azimuth phi.

The translucent sphere is one native ellipse; the equator and a meridian are
split exactly where they pass behind it and dashed there; the state is a solid
arrow, with the angles marked by arcs in their own planes.

Uses: ``sphere(highlight=...)``, ``sphere_curve``, ``arrow3d``, ``arc_shape``,
and the flat ``arrow(normal="camera")`` for the axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vecview
from vecview import OrthographicCamera, Scene

THETA, PHI = 52.0, 58.0  # the state, in degrees
INK = "#1f2430"
STATE = "#c0392b"
AXIS_LEN = 1.38
MATH = dict(font_family="DejaVu Serif", font_style="italic", fill=INK)


def bloch_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    """The unit vector at polar angle theta and azimuth phi."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


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
        stroke="#4a6b8a",
        stroke_width=1.4,
        id="sphere",
    )
    hidden = {"stroke_dasharray": "5 4", "stroke_opacity": 0.55}
    circles = {"equator": (0, 0, 1), "meridian": (0, 1, 0)}
    for name, normal in circles.items():
        scene.sphere_curve(
            11,
            origin,
            vecview.circle_shape(origin, 1.0, normal, n=180),
            closed=True,
            back=hidden,
            back_layer=9,
            stroke="#4a6b8a",
            stroke_width=1.1,
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
        scene.text(30, AXIS_LEN * e, label, dx=dx, dy=dy, size=24, id=f"label-{label}", **MATH)

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
    scene.text(30, theta_at, "θ", dx=-4, dy=6, size=21, id="label-theta", **MATH)
    phi_at = 0.42 * bloch_vector(90.0, PHI / 2)
    scene.text(30, phi_at, "φ", dx=-6, dy=10, size=21, id="label-phi", **MATH)

    for name, z, text, dy in (("north", 1.0, "|0⟩", -12), ("south", -1.0, "|1⟩", 28)):
        scene.sphere(22, (0, 0, z), 0.03, fill=INK, id=f"pole-{name}")
        scene.text(30, (0, 0, z), text, dx=10, dy=dy, size=22, id=f"ket-{name}", fill=INK)
    scene.text(30, psi, "|ψ⟩", dx=10, dy=-8, size=22, id="ket-psi", fill=STATE)
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "bloch_sphere.svg"))
