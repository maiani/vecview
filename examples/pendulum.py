"""A small-angle pendulum: one seamless period, sampled at 60 fps.

The scene is drawn once.  What moves is given a track -- a function of time,
here the closed form ``theta = theta_0 cos(omega t)``, though it could as well
read a simulation -- wherever a still would take a value: the bob's position,
the rod's end, the readouts' text.  The animation reads every track at each
frame and writes once what does not change.  The equations are typeset by TeX
through VecTeX.

Uses: ``Animation``, ``Track``, ``add``, ``rect2d``, ``text2d``.
"""

from __future__ import annotations

from math import cos, degrees, pi, radians, sin, sqrt
from pathlib import Path

import svg
import vectex

import vecview
from vecview.animation import Track

GRAVITY = 9.81  # m/s²
LENGTH = 2.4  # m
AMPLITUDE = radians(15)
OMEGA = sqrt(GRAVITY / LENGTH)
PERIOD = 2 * pi / OMEGA  # one period, so the loop is seamless
# Seen level from the +y side: world x runs to the left on screen and z up, and
# the pendulum swings in the xz plane.  `scale` is screen px per world metre.
CAMERA = vecview.OrthographicCamera(azim_deg=90, elev_deg=0, scale=105)
INK, MUTED, ACCENT = "#20334a", "#64748b", "#db6544"
# Typeset once, before anything is drawn; the fragments are reused below.
LAW, VALUES = vectex.render_many(
    [
        vectex.RenderItem(
            r"$\theta(t) = \theta_0 \cos \omega t \qquad \omega = \sqrt{g/L}$",
            size_pt=13,
            color=INK,
            id_prefix="law",
        ),
        vectex.RenderItem(
            r"$L = 2.4\,\mathrm{m} \qquad g = 9.81\,\mathrm{m/s^2} \qquad \theta_0 = 15^\circ$",
            size_pt=9,
            color=MUTED,
            id_prefix="values",
        ),
    ]
)


def position(theta: float) -> tuple[float, float, float]:
    """Where the bob is, in world metres, at angle ``theta`` from the vertical.

    The pivot is the world origin, so the bob hangs at ``(0, 0, -LENGTH)``.
    """
    return (LENGTH * sin(theta), 0.0, -LENGTH * cos(theta))


# A Track is a function of time in seconds.  It is not called here: every
# drawing call below that is handed one keeps it, and the animation reads it at
# each frame's time.  `map` builds a track from another, so BOB is the bob's
# position at whatever time THETA is read.
THETA = Track(lambda t: AMPLITUDE * cos(OMEGA * t))
BOB = THETA.map(position)


def tex(scene: vecview.Scene, x: float, y: float, fragment: vectex.VectexFragment) -> None:
    """A TeX fragment with its baseline starting at screen point ``(x, y)``.

    A fragment's own origin is its top-left corner, so it is moved up by the
    height of its baseline to sit on the same line as the text around it.
    """
    top = y - fragment.baseline_px
    scene.add(0, svg.G(transform=f"translate({x} {top:g})", elements=[fragment.to_svg_py()]))


def build() -> vecview.Animation:
    # The first number of every call is its layer: layers are painted in
    # increasing order, so 0 is the background, 1 the swinging pendulum, and 2
    # what must stay on top of it.  Within a layer, calls paint in order.
    scene = vecview.Scene(CAMERA)

    # What never moves, in world coordinates: the arc the bob sweeps, the
    # vertical it swings about, and the two turning points.
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

    # What moves: BOB stands in for a point, inside a list of points too, and is
    # read again at every frame.  The ids are what an editor or a figure tool
    # finds the rod and the bob by, the same in every frame.
    scene.polyline(1, [(0, 0, 0), BOB], stroke=INK, stroke_width=2.5, id="rod")
    scene.sphere(1, BOB, 0.135, fill=ACCENT, stroke="#ab452b", stroke_width=1.2, id="bob")
    # The support and its hinge sit over the top of the rod.
    scene.rect2d(2, -33, -7, 66, 7, rx=2, fill=INK, id="support")
    scene.sphere(2, (0, 0, 0), 0.045, fill=INK, id="pivot")

    # The captions are in screen coordinates (`text2d`, `rect2d`): px from where
    # the world origin, the pivot, lands on screen, with y down.  A track gives text too:
    # `angle` follows THETA, and `clock` reads the time itself.
    angle = THETA.map(lambda theta: f"{degrees(theta):+.1f}°")
    clock = Track(lambda t: f"t = {t:.2f} s")
    scene.text2d(0, -210, -73, "The simple pendulum", size=24, fill=INK, font_weight=600)
    scene.text2d(0, -210, -48, "SMALL-ANGLE MOTION  /  NO DAMPING", size=10, fill=MUTED)
    scene.text2d(0, 96, 115, "ANGLE", size=9, fill=MUTED)
    scene.text2d(0, 96, 141, angle, size=22, fill=ACCENT, id="angle")
    scene.text2d(0, 96, 177, clock, size=11, fill=MUTED, id="clock")
    scene.text2d(0, 96, 197, f"T = {PERIOD:.2f} s", size=11, fill=MUTED)
    scene.text2d(0, -17, 291, "equilibrium", size=10, fill=MUTED, text_anchor="middle")
    scene.rect2d(0, -210, 319, 420, 1, fill="#e2e8f0")  # a 1 px rule above the equations
    tex(scene, -210, 348, LAW)
    tex(scene, -210, 375, VALUES)

    # One cycle is one period, cut into ceil(PERIOD * fps) = 187 frames, and
    # `repeat=None` loops it forever.  The view box is the fixed window every
    # frame is seen through, in the same screen px as `text2d`.
    return vecview.Animation(
        scene,
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
