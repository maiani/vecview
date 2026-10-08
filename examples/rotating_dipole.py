"""A rotating electric dipole and the wave it radiates: one seamless period.

The dipole ``p(t) = p0 (cos wt, sin wt, 0)`` is two perpendicular oscillating
dipoles a quarter period apart, the real part of ``p0 (x + iy) exp(-iwt)``, so
its field is the sum of theirs.  Each is the exact retarded field of an
oscillating dipole -- near, induction, and radiation terms together -- in
Gaussian units with ``c = 1``, so ``k = w`` and lengths are in units of
``1/k``, a wavelength over 2 pi.

In the equatorial plane ``E`` lies in the plane and ``B`` is normal to it, so
one picture holds both: arrows for ``E``, and shading for ``B_z``, warm where it
points up and cool where down.  Both fall off as ``1/r`` far out and faster near
the dipole, so each is shown against its own scale at each radius: an arrow is
``r |E|``, capped where the near field is strong, and the shading is ``B_z`` over
its amplitude at that radius.  Turning the dipole is the same as waiting, so the
whole pattern turns rigidly with it: the crests are spirals a wavelength apart
that straighten, close in, into the quasi-static field of the dipole.

Only the shading, the arrows, and the dipole change from frame to frame; the
disk, its rim, the captions, and the equations are written once.  Each frame is
a few hundred flat polygons, with no depth sort, so 48 frames -- enough for the
wave to move smoothly -- render in about a second, into a file of 1.6 MB.

Uses: ``Animation``, ``Track``, ``Part``, ``place``, ``arrow``, ``arrow3d``,
``polygon``, ``arc_shape``.
"""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import numpy as np
import svg
import vectex

import vecview
from vecview.animation import Track

K = 1.0  # the wavenumber, and the angular frequency since c = 1
RADIUS = 3 * pi / K  # of the disk drawn: a wavelength and a half
PERIOD = 4.0  # seconds of animation per turn of the dipole
# Looking down on the plane from 55 degrees up; `scale` is screen px per world unit.
CAMERA = vecview.OrthographicCamera(azim_deg=-60, elev_deg=55, scale=26)
INK, MUTED, ACCENT = "#20334a", "#64748b", "#c0392b"
LEVELS = [0.2, 0.6, 0.9]  # where each shade starts, as a fraction of the peak
UP = ["#fbe3d3", "#f5c3a3", "#ec9b6e"]  # B_z > 0, palest first
DOWN = ["#dbe6f2", "#b6cde6", "#86abd5"]  # B_z < 0
DISK = vecview.circle_shape((0, 0, 0), RADIUS, normal=(0, 0, 1), n=128)
# Typeset once, before anything is drawn; the fragments are reused below.
LAW, ELECTRIC, MAGNETIC, VALUES = vectex.render_many(
    [
        vectex.RenderItem(
            r"$\mathbf{p}(t) = p_0\,(\cos\omega t,\ \sin\omega t,\ 0)"
            r" = \mathrm{Re}\,\mathbf{p}\,e^{-i\omega t}"
            r" \qquad \mathbf{p} = p_0\,(\hat{\mathbf{x}} + i\hat{\mathbf{y}})$",
            size_pt=11,
            color=INK,
            id_prefix="law",
        ),
        vectex.RenderItem(
            r"$\mathbf{E} = \mathrm{Re}\Bigl\{\Bigl["
            r"\frac{k^2}{r}\,(\hat{\mathbf{n}}\times\mathbf{p})\times\hat{\mathbf{n}}"
            r" + \bigl(3\hat{\mathbf{n}}(\hat{\mathbf{n}}\cdot\mathbf{p}) - \mathbf{p}\bigr)"
            r"\Bigl(\frac{1}{r^3} - \frac{ik}{r^2}\Bigr)\Bigr]"
            r"\,e^{i(kr - \omega t)}\Bigr\}$",
            size_pt=11,
            color=INK,
            id_prefix="electric",
        ),
        vectex.RenderItem(
            r"$\mathbf{B} = \mathrm{Re}\,\frac{k^2}{r}\,(\hat{\mathbf{n}}\times\mathbf{p})"
            r"\Bigl(1 - \frac{1}{ikr}\Bigr)\,e^{i(kr - \omega t)}$",
            size_pt=11,
            color=INK,
            id_prefix="magnetic",
        ),
        vectex.RenderItem(
            r"$\text{Gaussian units, } c = 1 \qquad k = \omega = 2\pi/\lambda \qquad"
            r" \text{disk radius } \tfrac{3}{2}\lambda$",
            size_pt=9,
            color=MUTED,
            id_prefix="values",
        ),
    ]
)


def amplitudes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The complex amplitudes of E and B at world points ``(n, 3)``.

    The fields at time ``t`` are the real parts of these times ``exp(-iwt)``.
    They depend on time only through that factor, so they are computed once
    here, and each frame only multiplies by it.
    """
    p = np.array([1.0, 1.0j, 0.0])  # p0 = 1, along x now and along y a quarter period on
    r = np.linalg.norm(points, axis=1, keepdims=True)
    n = points / r  # unit vectors from the dipole
    n_x_p = np.cross(n, p)
    radiation = K**2 / r * np.cross(n_x_p, n)  # the 1/r far field, transverse to n
    near = (3 * n * (n @ p)[:, None] - p) * (1 / r**3 - 1j * K / r**2)  # static and induction
    e = (radiation + near) * np.exp(1j * K * r)  # exp(ikr): each point sees the dipole r ago
    b = K**2 / r * n_x_p * (1 - 1 / (1j * K * r)) * np.exp(1j * K * r)
    return e, b


def on_plane(r: float | np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    """Points of the equatorial plane from polar coordinates."""
    angle = np.asarray(angle)
    return np.stack([r * np.cos(angle), r * np.sin(angle), 0 * angle], axis=-1)


# The phase of B_z along the x axis.  At an angle phi off the axis the field is
# the one the axis sees phi / w later, so this one ray gives the whole plane:
# with b(r) the amplitude on the axis, B_z = |b| cos(phi - (w t - arg b)), which
# peaks at phi = w t - arg b.  `unwrap` keeps arg b continuous rather than
# jumping by 2 pi, so the peak traces one smooth spiral.  The samples crowd
# outward, where a step in r is a longer step along a spiral, and start just
# off the dipole, where the field is singular.
RAY = RADIUS * np.linspace(0, 1, 64)[1:] ** 0.75
_, B_RAY = amplitudes(on_plane(RAY, 0 * RAY))
PHASE = np.unwrap(np.angle(B_RAY[:, 2]))

# The arrows sit on rings SPACING apart, each ring holding as many as keep them
# about SPACING apart along it, and every other ring turned by half a step so
# neighbouring rings interleave.  Their amplitudes are computed once, here.
SPACING = 1.45
GRID = np.vstack(
    [
        on_plane(r, 2 * pi * (np.arange(count) + 0.5 * i) / count)
        for i, r in enumerate(np.arange(1.6, RADIUS, SPACING))
        for count in [round(2 * pi * r / SPACING)]
    ]
)
E_GRID, _ = amplitudes(GRID)

# The one track everything moving follows: w t, the angle the dipole has turned
# through after t seconds.  It is read at every frame's time; the functions
# below take that angle and say what is drawn at it.
TURN = Track(lambda t: 2 * pi * t / PERIOD)


def spiral_strip(lo: float, hi: float) -> Track[np.ndarray]:
    """Where the phase of B_z lies between ``lo`` and ``hi``: a strip between two spirals.

    The strip turns with the dipole, so this returns a track of its outline:
    from the dipole out along one spiral, round the rim, and back in along the
    other.  A polygon handed this track is redrawn at every frame.
    """

    def at(turn: float) -> np.ndarray:
        peak = turn - PHASE  # the angle at which B_z peaks, radius by radius
        rim = vecview.arc_shape(  # the stretch of the disk's edge between the spirals
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            RADIUS,
            np.degrees(peak[-1] + lo),
            np.degrees(peak[-1] + hi),
            n=24,
        )
        ray = np.vstack([on_plane(RAY, peak + lo), rim, on_plane(RAY, peak + hi)[::-1]])
        return np.vstack([[(0, 0, 0)], ray])

    return TURN.map(at)


def electric(turn: float) -> vecview.Part:
    """E in the plane once the dipole has turned through ``turn``.

    One arrow per grid point, its length ``r |E|`` capped at one.  Arrows too
    short to read are left out, so how many there are changes from frame to
    frame -- which one track per arrow could not say.  So this draws them all
    into a part, and the scene places a track of parts: a new one each frame.
    """
    arrows = vecview.Part()
    field = (E_GRID * np.exp(-1j * turn)).real  # the field now, from the amplitudes
    strength = np.minimum(np.linalg.norm(GRID, axis=1) * np.linalg.norm(field, axis=1), 1.0)
    for point, e, s in zip(GRID, field, strength, strict=True):
        if s < 0.12:  # too short to read as an arrow
            continue
        head = min(1.0, 2 * s)  # a short arrow gets a smaller head
        arrows.arrow(  # a flat arrow lying in the plane, centred on its grid point
            0,
            point,
            e,
            1.2 * s,
            normal=(0, 0, 1),
            pivot="mid",
            shaft_w=0.08,
            head_w=0.32 * head,
            head_len=0.38 * head,
            fill=INK,
            class_="e-field",
        )
    return arrows


def legend(scene: vecview.Scene, y: float, label: list[svg.TSpan], colors: list[str]) -> None:
    """A row of swatches and its label, with the text's baseline at ``y``."""
    for i, color in enumerate(colors):
        scene.rect2d(0, -262 + 12 * i, y - 9, 12, 10, fill=color)
    scene.text2d(0, -218, y, label, size=11, fill=INK)


def build() -> vecview.Animation:
    # The first number of every call is its layer, painted in increasing order:
    # the disk (0), the shading (1), the arrows (2), the dipole (3), and the rim
    # over the shading's edge (4).  Within a layer, calls paint in order.
    scene = vecview.Scene(CAMERA)
    scene.polygon(0, DISK, fill="#f4f5f7", id="plane")

    # B_z, normal to the plane: over its amplitude at each radius it is
    # cos(phi - peak), so it is above `level` within arccos(level) of the peak,
    # and below -level as far either side of the trough, half a turn on.  The
    # palest, widest strips go down first and the darker, narrower ones on top.
    for level, up, down in zip(LEVELS, UP, DOWN, strict=True):
        half = np.arccos(level)
        scene.polygon(1, spiral_strip(-half, half), fill=up, class_="b-up")
        scene.polygon(1, spiral_strip(pi - half, pi + half), fill=down, class_="b-down")

    # A track can stand for a whole part: the arrows drawn at this frame's turn.
    scene.place(2, TURN.map(electric))

    # The dipole itself, a solid arrow whose direction is a track.
    scene.arrow3d(
        3,
        (0, 0, 0),
        TURN.map(lambda turn: (cos(turn), sin(turn), 0)),
        2.0,
        pivot="mid",
        shaft_r=0.12,
        head_r=0.3,
        head_len=0.6,
        fill=ACCENT,
        highlight="#f2b8ae",
        id="dipole",
    )
    scene.polygon(4, DISK, fill="none", stroke=MUTED, stroke_width=0.8, id="rim")

    # The captions and equations, in screen coordinates: px from where the world
    # origin lands on screen, with y down.  None of them moves, so the
    # animation writes them once.
    scene.text2d(0, -262, -262, "The rotating dipole", size=24, fill=INK, font_weight=600)
    caption = "EXACT RETARDED FIELDS IN THE EQUATORIAL PLANE"
    scene.text2d(0, -262, -240, caption, size=10, fill=MUTED)
    b_z = [svg.TSpan(text="B"), svg.TSpan(text="z", baseline_shift="sub", font_size=8)]
    legend(scene, 240, [*b_z, svg.TSpan(text=" up, out of the plane")], UP)
    legend(scene, 258, [*b_z, svg.TSpan(text=" down, into it")], DOWN)
    scene.text2d(0, -262, 276, "Arrows: E in the plane, times r", size=11, fill=INK)
    notes = "Shades start at 0.2, 0.6, and 0.9 of the peak at each radius;"
    notes += " arrows saturate near the dipole."
    scene.text2d(0, -262, 292, notes, size=9, fill=MUTED)
    # VecTeX places each equation with its baseline at (x, y), on the line.
    scene.add(0, LAW.to_svg_py(x=-262, y=326, valign="baseline"))
    scene.add(0, ELECTRIC.to_svg_py(x=-262, y=358, valign="baseline"))
    scene.add(0, MAGNETIC.to_svg_py(x=-262, y=390, valign="baseline"))
    scene.add(0, VALUES.to_svg_py(x=-262, y=418, valign="baseline"))

    # One cycle is one turn, cut into PERIOD * fps = 48 frames, and `repeat=None`
    # loops it forever.  The view box is the fixed window every frame is seen
    # through, in the same screen px as the captions.
    return vecview.Animation(
        scene,
        duration=PERIOD,
        view_box=(-282, -292, 564, 724),
        fps=12,
        repeat=None,
        background="#ffffff",
    )


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "rotating_dipole.svg"))
