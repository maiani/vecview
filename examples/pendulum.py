"""A small-angle pendulum: one seamless period, sampled at 60 fps.

The frame callback draws the whole scene at time ``t`` -- here from the closed
form ``theta = theta_0 cos(omega t)``, but it could as well read a simulation.
Everything that does not move is drawn below the swinging rod and bob, so the
animation writes it once instead of in every frame.  The equations are typeset
by TeX through VecTeX, once, and the same fragment is added to every frame.

Uses: ``Animation``, ``add``, ``rect2d``, ``text2d``.
"""

from __future__ import annotations

from math import cos, degrees, pi, radians, sin, sqrt
from pathlib import Path

import svg
import vectex

import vecview

GRAVITY = 9.81  # m/s²
LENGTH = 2.4  # m
AMPLITUDE = radians(15)
OMEGA = sqrt(GRAVITY / LENGTH)
PERIOD = 2 * pi / OMEGA  # one period, so the loop is seamless
CAMERA = vecview.OrthographicCamera(azim_deg=90, elev_deg=0, scale=105)
INK, MUTED, ACCENT = "#20334a", "#64748b", "#db6544"
LAW = vectex.render(
    r"$\theta(t) = \theta_0 \cos \omega t \qquad \omega = \sqrt{g/L}$", size_pt=13, id_prefix="law"
)
VALUES = vectex.render(
    r"$L = 2.4\,\mathrm{m} \qquad g = 9.81\,\mathrm{m/s^2} \qquad \theta_0 = 15^\circ$",
    size_pt=9,
    id_prefix="values",
)


def position(theta: float) -> tuple[float, float, float]:
    return (LENGTH * sin(theta), 0.0, -LENGTH * cos(theta))


def tex(scene: vecview.Scene, x: float, y: float, fragment: vectex.VectexFragment) -> None:
    """A TeX fragment with its baseline starting at screen point ``(x, y)``."""
    px = 4 / 3  # VecTeX measures in TeX points; the scene in px
    top = y - fragment.baseline * px
    place = f"translate({x} {top:g}) scale({px:g})"
    scene.add(0, svg.G(transform=place, elements=[fragment.to_svg_py()]))


def draw_still(scene: vecview.Scene) -> None:
    """The swing's path, its turning points, and the captions."""
    path = [position(AMPLITUDE * (-1 + 2 * i / 100)) for i in range(101)]
    scene.polyline(0, path, stroke="#b4c3d1", stroke_width=1.5, id="trajectory")
    scene.polyline(
        0,
        [(0, 0, 0), (0, 0, -LENGTH - 0.2)],
        stroke="#c6cfd8",
        stroke_dasharray="4 5",
        id="equilibrium",
    )
    for name, angle in [("left", -AMPLITUDE), ("right", AMPLITUDE)]:
        scene.sphere(0, position(angle), 0.12, fill="#f4ede7", stroke="#d9c8ba", id=name)

    scene.text2d(0, -210, -73, "The simple pendulum", size=24, fill=INK, font_weight=600)
    scene.text2d(0, -210, -48, "SMALL-ANGLE MOTION  /  NO DAMPING", size=10, fill=MUTED)
    scene.text2d(0, 96, 115, "ANGLE", size=9, fill=MUTED)
    scene.text2d(0, 96, 197, f"T = {PERIOD:.2f} s", size=11, fill=MUTED)
    scene.text2d(0, -17, 291, "equilibrium", size=10, fill=MUTED, text_anchor="middle")
    scene.rect2d(0, -210, 319, 420, 1, fill="#e2e8f0")
    tex(scene, -210, 348, LAW)
    tex(scene, -210, 375, VALUES)


def frame(t: float) -> vecview.Scene:
    theta = AMPLITUDE * cos(OMEGA * t)
    bob = position(theta)
    scene = vecview.Scene(CAMERA)
    draw_still(scene)
    scene.polyline(1, [(0, 0, 0), bob], stroke=INK, stroke_width=2.5, id="rod")
    scene.sphere(1, bob, 0.135, fill=ACCENT, stroke="#ab452b", stroke_width=1.2, id="bob")
    scene.text2d(1, 96, 141, f"{degrees(theta):+.1f}°", size=22, fill=ACCENT, id="angle")
    scene.text2d(1, 96, 177, f"t = {t:.2f} s", size=11, fill=MUTED, id="clock")
    # The support and its hinge sit over the top of the rod.
    scene.rect2d(2, -33, -7, 66, 7, rx=2, fill=INK, id="support")
    scene.sphere(2, (0, 0, 0), 0.045, fill=INK, id="pivot")
    return scene


def build() -> vecview.Animation:
    return vecview.Animation(
        frame,
        duration=PERIOD,
        view_box=(-240, -105, 480, 510),
        fps=60,
        repeat=None,
        background="#ffffff",
    )


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "pendulum.svg"))
