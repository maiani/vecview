"""Outreach picture: a spin-textured altermagnetic slab as a frequency-tunable polarizer.

Two beams strike the same slab at normal incidence, each drawn with both of the
polarizations the material distinguishes at that frequency -- its principal axes.  In
each case the component along the principal absorption axis theta_+ is drunk by the slab
and the orthogonal one exits.  Because theta_+ swings by ~90 deg across the optical
crossover, the two beams leave polarized along nearly perpendicular directions: the
headline of the paper in one picture.

Schematic, at the author's direction: the two absorption axes are drawn at the values the
two regimes *mean* rather than at one particular frequency's model output.

    phi_q      = 22.5 deg, as in Fig. 3(b)
    locked     theta_+ = 0 deg      -- exactly on a crystal axis
    tracking   theta_+ = 67.5 deg   -- the tracking law 90 deg - phi_q

Both are faithful to the physics: the model gives 1.3 deg at omega = 0.25, and 67.5 deg is
the asymptote theta_+ approaches from above in the tracking regime (78.6 deg at
omega = 0.80, 75.2 at 0.895).  Drawing the asymptote rather than a finite-frequency value
also gives the widest angle to the crystal axes, 22.5 deg, which is what makes the
locked/tracking contrast legible.  Note the angle cannot be widened beyond that by picking
a real frequency: theta_+ is furthest off axis right at the crossover, where the dichroism
dips to its minimum and "one polarization gets through" stops being true.
The higher-frequency beam is also drawn with a shorter wavelength.  That ratio is
compressed relative to the true 3.2x, purely so both beams stay legible.

Camera notes, learned the hard way:
  * Elevation is a real trade-off.  Too low (~24 deg) and the slab plane is so
    foreshortened that both polarization crosses collapse to a similar shallow X,
    destroying the 90 deg flip -- the whole point.  Too high (~40 deg) and the slab
    reads as a flat plan view with no visible underside.  Set to 24 deg at the author's
    direction, to open up the region below the slab.  That is the low end: the two
    polarization crosses are only ~45 deg apart here and differ mainly in orientation
    rather than shape.  If it needs to go lower, carry the flip in face-on polarization
    dials (as outreach-1 does with its purple axis boxes) rather than in the projected
    crosses.
  * The visible side walls at this azimuth are x = +LX/2 and y = +LY/2.

Usage:
    python examples/slab_polarizer.py [--grey]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import cairosvg
import numpy as np
import svg

import vecview
from vecview import Camera, ObliqueCamera, OrthographicCamera, Scene

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# --- model numbers ---------------------------------------------------------
PHI_Q = 22.5
# Incoming linear polarization.  Two constraints: an in-plane direction at the camera
# azimuth (35 deg) projects to almost zero screen amplitude and its wave draws as a flat
# line, so stay ~90 deg away from it; and |cos(THETA_IN - theta_+)| should match across
# the two beams so neither is drawn as losing more than the other.  130 deg does both
# (0.625 absorbed, 0.781 transmitted, for each beam).
THETA_IN = 130.0
AMP_IN = 0.52
THETA_LOW = 0.0  # locked: on the crystal axis
THETA_HIGH = 67.5  # tracking: 90 deg - phi_q
Q_TEXTURE = 0.7  # schematic real-space pitch, as in Fig. 1

# --- geometry -------------------------------------------------------------
LX, LY, THICK = 11.0, 9.0, 0.9
H_IN, H_OUT = 5.2, 4.4  # beam length above / below the slab
SPOT = 3.0  # beam separation along the screen-horizontal world direction
FRONT = 1.7  # how far the beams are walked toward the near edge
SCALE = 62.0
AZIM, ELEV = 35.0, 24.0

COLOR = dict(
    slab_top="#eef1f5",
    slab_side="#cfd6e0",
    slab_edge="#8b96a6",
    sub_a="#e07a2f",
    sub_b="#2f6fb0",
    twist="#1a8f4c",
    beam_low="#d62828",
    beam_high="#6a2fb5",
    absorbed="#4a4e54",
    text="#222222",
    faint="#7d838c",
    phase="#a3a9b2",
    axes="#3a4046",
)
GREY = dict(
    slab_top="#eeeeee",
    slab_side="#cccccc",
    slab_edge="#888888",
    sub_a="#555555",
    sub_b="#9a9a9a",
    twist="#333333",
    beam_low="#404040",
    beam_high="#404040",
    absorbed="#707070",
    text="#222222",
    faint="#999999",
    phase="#bbbbbb",
    axes="#333333",
)


def draw_slab(sc: Scene, colors) -> None:
    """The slab, drawn as a box with only its camera-facing walls.

    `cull=True` culls by outward normal at draw time, so *which* walls those are
    is the camera's business.  This used to be a comment reading "the camera sits
    in the (+x, +y) octant, so these two walls are the visible ones" over two
    hand-listed corner pairs -- true at this azimuth, silently wrong at another.

    Culling inside `faces` rather than with `cam.visible` here is what lets the
    finished scene be reprojected: a cabinet camera sees `-y` where this one sees
    `+y`, and caller-side culling would have baked in the wrong answer.

    Splitting by *name* is camera-independent, so it stays safe: the top is
    translucent so the texture beneath it reads, the walls solid so the slab has
    body.
    """
    edge = dict(stroke=colors["slab_edge"], stroke_width=1.6, stroke_linejoin="round")
    box = vecview.box_faces(center=(0, 0, -THICK / 2), size=(LX, LY, THICK))
    sc.faces(
        10,
        [f for f in box if f.name == "+z"],
        cull=True,
        fill=colors["slab_top"],
        fill_opacity=0.86,
        **edge,
    )
    sc.faces(
        11,
        [f for f in box if f.name != "+z"],
        cull=True,
        fill=colors["slab_side"],
        **edge,
    )


def draw_phase_lines(sc: Scene, colors, layer: int = 14) -> None:
    """Lines of constant helix phase on the slab face -- the visual language of Fig. 1."""
    qh = vecview.in_plane_dir(PHI_Q)
    perp = np.array([-qh[1], qh[0], 0.0])
    t = np.linspace(-11.0, 11.0, 800)
    spacing = (np.pi / 2.0) / Q_TEXTURE
    for k in range(-4, 5):
        pts = qh * k * spacing + np.outer(t, perp)
        inside = (np.abs(pts[:, 0]) <= LX / 2 - 0.06) & (np.abs(pts[:, 1]) <= LY / 2 - 0.06)
        if inside.sum() < 2:
            continue
        seg = pts[inside][[0, -1]]
        sc.polyline(
            layer,
            np.column_stack([seg[:, 0], seg[:, 1], np.full(2, 0.005)]),
            stroke=colors["phase"],
            stroke_width=1.3,
            stroke_opacity=0.45,
            stroke_dasharray="7 6",
        )


def draw_texture(sc: Scene, colors, layer: int = 15) -> None:
    """Fig. 1's checkerboard Neel helix, sparse enough to survive foreshortening."""
    nx, ny = 11, 9
    x, y = np.meshgrid(
        np.linspace(-LX / 2 + 0.75, LX / 2 - 0.75, nx),
        np.linspace(-LY / 2 + 0.7, LY / 2 - 0.7, ny),
    )
    qh = vecview.in_plane_dir(PHI_Q)
    phase = Q_TEXTURE * (qh[0] * x + qh[1] * y)
    u, v = np.cos(phase), np.sin(phase)
    iy, ix = np.indices(u.shape)
    b = (ix + iy) % 2 == 1
    u[b], v[b] = -u[b], -v[b]
    for j in range(ny):
        for i in range(nx):
            sc.polygon(
                layer,
                vecview.arrow_shape(
                    [x[j, i], y[j, i], 0.01],
                    [u[j, i], v[j, i], 0.0],
                    0.64,
                    [0, 0, 1],
                    shaft_w=0.08,
                    head_w=0.25,
                    head_len=0.25,
                    pivot="mid",
                ),
                fill=colors["sub_b"] if b[j, i] else colors["sub_a"],
                fill_opacity=0.6,
            )


def draw_twist_arrow(sc: Scene, colors, layer: int = 16) -> None:
    qh = vecview.in_plane_dir(PHI_Q)
    start = -qh * 1.9
    sc.polygon(
        layer,
        vecview.arrow_shape(
            start + np.array([0, 0, 0.03]),
            qh,
            3.9,
            [0, 0, 1],
            shaft_w=0.10,
            head_w=0.40,
            head_len=0.48,
        ),
        fill=colors["twist"],
        fill_opacity=0.95,
    )


def draw_beam(
    sc: Scene, colors, center, theta_plus, wavelength, beam_color, label, regime, tag
) -> None:
    """One beam, drawn as the incoming polarization decomposed into the two eigenmodes.

    Both components arrive; the one along theta_+ decays inside the slab and the
    orthogonal one passes through unchanged, so the surviving trace runs the full height
    of the picture while its partner dies at the surface.

    The component amplitudes are the physical ones for a linear polarization at THETA_IN:
    cos(THETA_IN - theta_+) = 0.625 and 0.781.  That matters for legibility as well as
    honesty.  An earlier version used arbitrary amplitudes (0.44, 0.273) which happened to
    project to near-equal screen size with opposite sign in the low beam, so the two
    traces crossed at every node and chained into lens shapes instead of reading as sines.
    Each trace is an exact sine either way (checked to 1e-16); it is the *pair* that has
    to be kept visually separable.
    """
    top = np.array([center[0], center[1], H_IN])
    hit = np.array([center[0], center[1], 0.0])
    e_abs, e_trans = vecview.in_plane_dir(theta_plus), vecview.in_plane_dir(theta_plus + 90.0)
    c_abs = np.cos(np.radians(THETA_IN - theta_plus))
    c_trans = np.cos(np.radians(THETA_IN - theta_plus - 90.0))

    # soft shaft: elliptical radial gradient, no mask (a gradient-filled <mask> does not
    # survive cairosvg rasterization -- the glow vanishes silently)
    sc.add_def(
        svg.RadialGradient(
            id=f"gh{tag}",
            cx=0.5,
            cy=0.5,
            r=0.5,
            elements=[
                svg.Stop(offset=0, stop_color=beam_color, stop_opacity=0.26),
                svg.Stop(offset=0.45, stop_color=beam_color, stop_opacity=0.15),
                svg.Stop(offset=1, stop_color=beam_color, stop_opacity=0),
            ],
        )
    )
    x_beam, y_top = sc.cam.at(top + np.array([0, 0, 1.4]))
    _, y_bot = sc.cam.at(np.array([center[0], center[1], -THICK - H_OUT - 1.4]))
    half = 1.05 * sc.cam.scale
    sc.rect2d(6, x_beam - half, y_top, 2 * half, y_bot - y_top, fill=f"url(#gh{tag})")

    ray = dict(stroke=beam_color, stroke_width=1.5, stroke_opacity=0.55)
    sc.polyline(20, np.array([top, hit]), **ray)
    sc.polyline(
        18,
        np.array([hit, hit + np.array([0.0, 0.0, -THICK])]),
        stroke=beam_color,
        stroke_width=1.5,
        stroke_opacity=0.35,
    )
    sc.polyline(
        7,
        np.array([hit + np.array([0.0, 0.0, -THICK]), hit + np.array([0.0, 0.0, -THICK - H_OUT])]),
        **ray,
    )

    t = np.linspace(0.0, H_IN + THICK + H_OUT, 1600)
    z = H_IN - t
    depth = np.clip(-z, 0.0, THICK)
    wave = np.sin(2.0 * np.pi * t / wavelength)
    for e, c0, absorbed in ((e_abs, c_abs, True), (e_trans, c_trans, False)):
        env = AMP_IN * abs(c0) * (np.exp(-depth / 0.30) if absorbed else 1.0)
        disp = env * wave
        pts = np.column_stack([center[0] + disp * e[0], center[1] + disp * e[1], z])
        alive = env > 0.06 * AMP_IN
        dim = 0.62 if absorbed else 0.95
        for layer, region, op in (
            (21, z > 0.0, dim),
            (18, (z <= 0.0) & (z >= -THICK), 0.55 * dim + 0.1),
            (8, z < -THICK, dim),
        ):
            mask = region & alive
            if mask.sum() > 1:
                sc.polyline(
                    layer,
                    pts[mask],
                    stroke=beam_color,
                    stroke_width=2.8,
                    stroke_opacity=op,
                    stroke_linejoin="round",
                )

    marker = dict(shaft_w=0.09, head_w=0.34, head_len=0.30)
    for e, op in ((e_abs, 0.62), (e_trans, 0.95)):
        sc.polygon(
            22,
            vecview.double_arrow_shape(top + np.array([0, 0, 0.62]), e, 1.5, [0, 0, 1], **marker),
            fill=beam_color,
            fill_opacity=op,
        )
    sc.polygon(
        8,
        vecview.double_arrow_shape(
            np.array([center[0], center[1], -THICK - 0.72 * H_OUT]),
            e_trans,
            1.5,
            [0, 0, 1],
            **marker,
        ),
        fill=beam_color,
        fill_opacity=0.95,
    )

    # Locked vs tracking, without words: grey crystal-axis ticks at each landing spot,
    # with the absorption axis theta_+ drawn over them.  For the low beam theta_+ = 1.3
    # deg sits *on* the x tick; for the high beam it lies well off both.  That contrast is
    # exactly the locked/tracking distinction, and it is geometrically true.
    sc.polygon(
        17,
        vecview.circle_shape(hit + np.array([0, 0, 0.02]), 0.62, [0, 0, 1]),
        fill=beam_color,
        fill_opacity=0.12,
        stroke=beam_color,
        stroke_width=1.1,
        stroke_opacity=0.4,
    )
    for d in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        sc.polygon(
            23,
            vecview.double_arrow_shape(
                hit + np.array([0, 0, 0.04]),
                d,
                3.05,
                [0, 0, 1],
                shaft_w=0.065,
                head_w=0.24,
                head_len=0.26,
            ),
            fill=colors["axes"],
            fill_opacity=1.0,
        )
    sc.polygon(
        24,
        vecview.double_arrow_shape(
            hit + np.array([0, 0, 0.06]),
            e_abs,
            2.15,
            [0, 0, 1],
            shaft_w=0.10,
            head_w=0.36,
            head_len=0.34,
        ),
        fill=beam_color,
        fill_opacity=0.95,
    )

    sc.text(
        31,
        top + np.array([0, 0, 2.3]),
        label,
        size=27,
        fill=beam_color,
        text_anchor="middle",
        font_weight="bold",
    )
    # the Letter's own terms for the two regimes
    sc.text(
        31,
        top + np.array([0, 0, 2.3]),
        regime,
        dy=31,
        size=23,
        fill=beam_color,
        text_anchor="middle",
        font_style="italic",
        fill_opacity=0.85,
    )


def build(cam: Camera | None = None, grey: bool = False) -> Scene:
    """Assemble the scene.

    Returns the Scene rather than a rendered document, so a caller can render it
    at its own padding, reproject it with `Scene.with_camera`, or hand
    `to_svg_document()` to a larger figure.

    The camera is a parameter because two things here are derived from it: the
    beam landing spots, via `screen_basis`, and the slab's visible walls. Both
    have to be recomputed per projection, which is why a scene meant to be
    reprojected takes its camera rather than making one.
    """
    colors = GREY if grey else COLOR
    if cam is None:
        cam = OrthographicCamera(AZIM, ELEV, SCALE)
    sc = Scene(cam, pad=28.0, background="#ffffff")
    horiz, down = cam.screen_basis()
    front = FRONT * down

    spots = [-SPOT * horiz + front, +SPOT * horiz + front]
    for c in spots:  # guard: a beam off the slab is a silent disaster
        assert abs(c[0]) < LX / 2 - 0.8 and abs(c[1]) < LY / 2 - 0.8, (
            f"beam lands off the slab at ({c[0]:.2f}, {c[1]:.2f})"
        )

    draw_slab(sc, colors)
    draw_phase_lines(sc, colors)
    draw_texture(sc, colors)
    draw_twist_arrow(sc, colors)
    draw_beam(
        sc,
        colors,
        spots[0],
        THETA_LOW,
        1.70,
        colors["beam_low"],
        "low frequency",
        "locked regime",
        "Low",
    )
    draw_beam(
        sc,
        colors,
        spots[1],
        THETA_HIGH,
        0.85,
        colors["beam_high"],
        "high frequency",
        "tracking regime",
        "High",
    )
    return sc


PROJECTIONS: dict[str, Callable[[float], Camera]] = {
    # The default: a general azimuth/elevation view, so all three axes
    # foreshorten differently.
    "trimetric": lambda scale: OrthographicCamera(AZIM, ELEV, scale),
    "isometric": OrthographicCamera.isometric,
    "dimetric": OrthographicCamera.dimetric,
    "cavalier": ObliqueCamera.cavalier,
    "cabinet": ObliqueCamera.cabinet,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grey", action="store_true", help="composition pass: no colour")
    ap.add_argument(
        "--projection",
        default="trimetric",
        choices=["all", *PROJECTIONS],
        help="which projection to render; 'all' writes one file each",
    )
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    wanted = list(PROJECTIONS) if args.projection == "all" else [args.projection]

    # This scene is rebuilt per projection rather than reprojected with
    # `Scene.with_camera`. `with_camera` replays world-space calls exactly, but
    # `draw_beam` also emits screen-space glow columns via `rect2d`, positioned
    # from `cam.at(...)` at build time. Those are screen space by definition, so
    # a replay leaves them where the first camera put them -- correct by the rule,
    # visibly wrong in the picture. Reach for `with_camera` when a scene is
    # world-space throughout; rebuild when it is not.
    for name in wanted:
        view = build(PROJECTIONS[name](SCALE), args.grey)
        suffix = "_grey" if args.grey else ""
        stem = OUT / f"slab_polarizer_{name}{suffix}"
        out = view.save(stem.with_suffix(".svg"))
        # Rasterizing is deliberately not vecview's job; cairosvg is example-only.
        cairosvg.svg2png(url=str(out), write_to=str(stem.with_suffix(".png")), scale=1.0)
        doc = view.render()
        ratios = ", ".join(f"{r:.2f}" for r in view.cam.foreshortening())
        print(f"{name:>10}  {doc.width:>5.0f}x{doc.height:<5.0f}  axes {ratios}  -> {out.name}")


if __name__ == "__main__":
    main()
