"""A spin-textured altermagnetic slab acting as a frequency-tunable polarizer.

Two beams strike the slab at normal incidence, each drawn as its incoming linear
polarization split into the slab's two eigenmodes.  The component along the
absorption axis theta_+ decays inside the slab and the orthogonal one passes
through.  Because theta_+ swings by about 90 deg between the low- and
high-frequency regimes, the two beams leave polarized along nearly
perpendicular directions.

The absorption axes are drawn at the values the two regimes stand for, rather
than at one frequency's model output:

    phi_q      = 22.5 deg           orientation of the spin texture
    locked     theta_+ = 0 deg      on a crystal axis
    tracking   theta_+ = 67.5 deg   90 deg - phi_q, the high-frequency asymptote

The asymptote also gives the widest angle to the crystal axes, 22.5 deg, which is
what keeps the contrast between the regimes legible.  The high-frequency beam is
drawn with a shorter wavelength, the ratio compressed so both waves stay legible.

Choosing the camera:
  * Elevation trades the polarization crosses against the slab's underside.  Too
    low and the slab plane foreshortens until both crosses collapse to the same
    shallow X, hiding the 90-degree flip; too high and the slab reads as a plan
    view with no underside.  24 deg is the low end of what works; below it, show
    the flip in face-on dials rather than in projected crosses.
  * An in-plane direction near the camera azimuth projects to almost nothing,
    which constrains the incoming polarization; see THETA_IN.

The soft beam glows are screen-space rectangles placed from the camera, so the
scene is rebuilt for each projection rather than rendered under another camera.

Usage:
    python examples/slab_polarizer.py [--grey] [--projection {trimetric,...,all}]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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
Q_TEXTURE = 0.7  # schematic real-space pitch of the spin texture
DECAY = 0.30  # decay length of the absorbed component inside the slab
FADED, SOLID = 0.62, 0.95  # opacity of the absorbed and the transmitted component

# --- geometry -------------------------------------------------------------
LX, LY, THICK = 11.0, 9.0, 0.9
H_IN, H_OUT = 5.2, 4.4  # beam length above / below the slab
SPOT = 3.0  # beam separation along the screen-horizontal world direction
FRONT = 1.7  # how far the beams are walked toward the near edge
SCALE = 62.0
AZIM, ELEV = 35.0, 24.0
GLOW_HALF_WIDTH = 1.05  # of the soft column behind each beam, in world units
GLOW_OVERSHOOT = 1.4  # how far the column runs past each end of the beam
UP = np.array([0.0, 0.0, 1.0])  # the slab normal, which every in-plane arrow lies flat on

COLOR = dict(
    slab_top="#eef1f5",
    slab_side="#cfd6e0",
    slab_edge="#8b96a6",
    sub_a="#e07a2f",
    sub_b="#2f6fb0",
    twist="#1a8f4c",
    beam_low="#d62828",
    beam_high="#6a2fb5",
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
    phase="#bbbbbb",
    axes="#333333",
)


class Beam(NamedTuple):
    key: str  # "low" or "high": its colour, ids, and class
    theta_plus: float  # the absorption axis in this regime (deg)
    wavelength: float
    label: str
    regime: str


BEAMS = (
    Beam("low", THETA_LOW, 1.70, "low frequency", "locked regime"),
    Beam("high", THETA_HIGH, 0.85, "high frequency", "tracking regime"),
)


def draw_slab(sc: Scene, colors: dict[str, str]) -> None:
    """The slab, drawn as a box with only its camera-facing walls.

    `cull=True` culls by outward normal when the scene is rendered, so *which*
    walls those are is the camera's business: a cabinet camera sees `-y` where
    this one sees `+y`.

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
        id="slab",
        class_="slab",
        **edge,
    )
    sc.faces(
        11,
        [f for f in box if f.name != "+z"],
        cull=True,
        fill=colors["slab_side"],
        id="slab",
        class_="slab",
        **edge,
    )


def draw_phase_lines(sc: Scene, colors: dict[str, str], layer: int = 14) -> None:
    """Lines of constant helix phase on the slab face, clipped to the slab."""
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
            class_="phase-line",
        )


def draw_texture(sc: Scene, colors: dict[str, str], layer: int = 15) -> None:
    """A checkerboard Neel helix, sparse enough to survive foreshortening."""
    nx, ny = 11, 9
    x, y = np.meshgrid(
        np.linspace(-LX / 2 + 0.75, LX / 2 - 0.75, nx),
        np.linspace(-LY / 2 + 0.7, LY / 2 - 0.7, ny),
    )
    qh = vecview.in_plane_dir(PHI_Q)
    phase = Q_TEXTURE * (qh[0] * x + qh[1] * y)
    u, v = np.cos(phase), np.sin(phase)
    # The second sublattice points the other way.
    iy, ix = np.indices(u.shape)
    b = (ix + iy) % 2 == 1
    u[b], v[b] = -u[b], -v[b]
    for j in range(ny):
        for i in range(nx):
            sc.arrow(
                layer,
                [x[j, i], y[j, i], 0.01],
                [u[j, i], v[j, i], 0.0],
                0.64,
                normal=UP,
                shaft_w=0.08,
                head_w=0.25,
                head_len=0.25,
                pivot="mid",
                fill=colors["sub_b"] if b[j, i] else colors["sub_a"],
                fill_opacity=0.6,
                class_=["spin", "sublattice-b" if b[j, i] else "sublattice-a"],
            )


def draw_twist_arrow(sc: Scene, colors: dict[str, str], layer: int = 16) -> None:
    """The direction the spin texture twists along, across the middle of the slab."""
    qh = vecview.in_plane_dir(PHI_Q)
    sc.arrow(
        layer,
        -qh * 1.9 + np.array([0, 0, 0.03]),
        qh,
        3.9,
        normal=UP,
        shaft_w=0.10,
        head_w=0.40,
        head_len=0.48,
        fill=colors["twist"],
        fill_opacity=0.95,
        id="twist",
    )


def draw_glow(sc: Scene, beam: Beam, center: np.ndarray, color: str) -> None:
    """A soft column behind the beam: a screen-space rectangle with an elliptical gradient.

    No mask: a gradient-filled ``<mask>`` does not survive cairosvg
    rasterization -- the glow vanishes silently.
    """
    gradient = f"beam-{beam.key}-glow"
    sc.add_def(
        svg.RadialGradient(
            id=gradient,
            cx=0.5,
            cy=0.5,
            r=0.5,
            elements=[
                svg.Stop(offset=0, stop_color=color, stop_opacity=0.26),
                svg.Stop(offset=0.45, stop_color=color, stop_opacity=0.15),
                svg.Stop(offset=1, stop_color=color, stop_opacity=0),
            ],
        )
    )
    x_beam, y_top = sc.camera.at(np.array([center[0], center[1], H_IN + GLOW_OVERSHOOT]))
    _, y_bot = sc.camera.at(np.array([center[0], center[1], -THICK - H_OUT - GLOW_OVERSHOOT]))
    half = GLOW_HALF_WIDTH * sc.camera.scale
    sc.rect2d(6, x_beam - half, y_top, 2 * half, y_bot - y_top, fill=f"url(#{gradient})")


def draw_eigenmodes(sc: Scene, beam: Beam, center: np.ndarray, color: str) -> None:
    """The two eigenmode components of the incoming polarization, as waves along the beam.

    The amplitudes are the physical ones for a linear polarization at THETA_IN:
    cos(THETA_IN - theta_+) = 0.625 and 0.781.  That matters for legibility as
    well as honesty: two amplitudes that project to near-equal screen size with
    opposite sign make the traces cross at every node and chain into lens
    shapes, instead of reading as two sines.
    """
    t = np.linspace(0.0, H_IN + THICK + H_OUT, 1600)
    z = H_IN - t
    depth = np.clip(-z, 0.0, THICK)
    wave = np.sin(2.0 * np.pi * t / beam.wavelength)
    for offset, absorbed in ((0.0, True), (90.0, False)):
        e = vecview.in_plane_dir(beam.theta_plus + offset)
        c0 = np.cos(np.radians(THETA_IN - beam.theta_plus - offset))
        env = AMP_IN * abs(c0) * (np.exp(-depth / DECAY) if absorbed else 1.0)
        disp = env * wave
        pts = np.column_stack([center[0] + disp * e[0], center[1] + disp * e[1], z])
        alive = env > 0.06 * AMP_IN
        dim = FADED if absorbed else SOLID
        # Above the slab over it, inside it dimmed between its top and walls,
        # and below it under everything.
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
                    stroke=color,
                    stroke_width=2.8,
                    stroke_opacity=op,
                    stroke_linejoin="round",
                    class_=["wave", "absorbed" if absorbed else "transmitted"],
                )


def draw_landing_dial(
    sc: Scene, colors: dict[str, str], beam: Beam, hit: np.ndarray, color: str
) -> None:
    """Locked vs tracking, without words: crystal axes and theta_+ where the beam lands.

    Grey crystal-axis ticks at the landing spot, with the absorption axis
    theta_+ drawn over them.  For the low beam theta_+ lies *on* the x tick; for
    the high beam it lies well off both.
    """
    sc.polygon(
        17,
        vecview.circle_shape(hit + np.array([0, 0, 0.02]), 0.62, UP),
        fill=color,
        fill_opacity=0.12,
        stroke=color,
        stroke_width=1.1,
        stroke_opacity=0.4,
    )
    for d in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        sc.polygon(
            23,
            vecview.double_arrow_shape(
                hit + np.array([0, 0, 0.04]), d, 3.05, UP, shaft_w=0.065, head_w=0.24, head_len=0.26
            ),
            fill=colors["axes"],
            fill_opacity=1.0,
            class_="crystal-axis",
        )
    sc.polygon(
        24,
        vecview.double_arrow_shape(
            hit + np.array([0, 0, 0.06]),
            vecview.in_plane_dir(beam.theta_plus),
            2.15,
            UP,
            shaft_w=0.10,
            head_w=0.36,
            head_len=0.34,
        ),
        fill=color,
        fill_opacity=SOLID,
        id=f"beam-{beam.key}-absorption-axis",
    )


def draw_beam(sc: Scene, colors: dict[str, str], beam: Beam, center: np.ndarray) -> None:
    """One beam, drawn as the incoming polarization decomposed into the two eigenmodes.

    Both components arrive; the one along theta_+ decays inside the slab and the
    orthogonal one passes through unchanged, so the surviving trace runs the full height
    of the picture while its partner dies at the surface.
    """
    color = colors[f"beam_{beam.key}"]
    top = np.array([center[0], center[1], H_IN])
    hit = np.array([center[0], center[1], 0.0])
    bottom = hit - np.array([0.0, 0.0, THICK])
    e_abs = vecview.in_plane_dir(beam.theta_plus)
    e_trans = vecview.in_plane_dir(beam.theta_plus + 90.0)

    draw_glow(sc, beam, center, color)
    # The ray: above the slab over it, faint inside it, and below it under it.
    ray = dict(stroke=color, stroke_width=1.5, stroke_opacity=0.55)
    sc.polyline(20, np.array([top, hit]), **ray)
    sc.polyline(18, np.array([hit, bottom]), stroke=color, stroke_width=1.5, stroke_opacity=0.35)
    sc.polyline(7, np.array([bottom, bottom - np.array([0.0, 0.0, H_OUT])]), **ray)
    draw_eigenmodes(sc, beam, center, color)

    # Polarization markers: both components above the slab, the survivor below.
    marker = dict(shaft_w=0.09, head_w=0.34, head_len=0.30)
    for e, op in ((e_abs, FADED), (e_trans, SOLID)):
        sc.polygon(
            22,
            vecview.double_arrow_shape(top + np.array([0, 0, 0.62]), e, 1.5, UP, **marker),
            fill=color,
            fill_opacity=op,
        )
    sc.polygon(
        8,
        vecview.double_arrow_shape(
            np.array([center[0], center[1], -THICK - 0.72 * H_OUT]), e_trans, 1.5, UP, **marker
        ),
        fill=color,
        fill_opacity=SOLID,
    )
    draw_landing_dial(sc, colors, beam, hit, color)

    # The beam's frequency, and the regime's name under it.
    above = top + np.array([0, 0, 2.3])
    text = dict(fill=color, text_anchor="middle", class_="label")
    sc.text(31, above, beam.label, size=27, font_weight="bold", **text)
    sc.text(31, above, beam.regime, dy=31, size=23, font_style="italic", fill_opacity=0.85, **text)


def build(cam: Camera | None = None, grey: bool = False) -> Scene:
    """Assemble the scene for one camera.

    Returns the Scene rather than a rendered document, so a caller can render it
    at its own padding or hand `to_svg_document()` to a larger figure.

    The camera is a parameter because the beam landing spots come from it, via
    `screen_basis`, and so do the screen-space glows behind the beams, via
    `cam.at`.  Both are worked out here, before the scene sees them, so another
    camera needs another build.
    """
    colors = GREY if grey else COLOR
    if cam is None:
        cam = OrthographicCamera(AZIM, ELEV, SCALE)
    sc = Scene(cam, pad=28.0, background="#ffffff")

    # The beams land side by side on screen, walked toward the near edge.
    horiz, down = cam.screen_basis()
    front = FRONT * down
    spots = [-SPOT * horiz + front, +SPOT * horiz + front]
    for c in spots:  # a beam that misses the slab would still draw, so check
        assert abs(c[0]) < LX / 2 - 0.8 and abs(c[1]) < LY / 2 - 0.8, (
            f"beam lands off the slab at ({c[0]:.2f}, {c[1]:.2f})"
        )

    draw_slab(sc, colors)
    draw_phase_lines(sc, colors)
    draw_texture(sc, colors)
    draw_twist_arrow(sc, colors)
    for beam, spot in zip(BEAMS, spots, strict=True):
        draw_beam(sc, colors, beam, spot)
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

    # This scene is rebuilt per projection rather than rendered under another
    # camera: `draw_beam` places screen-space glow columns with `rect2d`, from
    # `cam.at(...)`, while building.  Screen space stays where it was put, so
    # another camera would leave the glows where the first one put them.  A scene
    # that is world-space throughout, like `altermagnetic_dot`, is built once and
    # rendered under each camera.
    for name in wanted:
        view = build(PROJECTIONS[name](SCALE), args.grey)
        suffix = "_grey" if args.grey else ""
        stem = OUT / f"slab_polarizer_{name}{suffix}"
        out = view.save(stem.with_suffix(".svg"))
        # Rasterizing is deliberately not vecview's job; cairosvg is example-only.
        cairosvg.svg2png(url=str(out), write_to=str(stem.with_suffix(".png")), scale=1.0)
        doc = view.render()
        ratios = ", ".join(f"{r:.2f}" for r in view.camera.foreshortening())
        print(f"{name:>10}  {doc.width:>5.0f}x{doc.height:<5.0f}  axes {ratios}  -> {out.name}")


if __name__ == "__main__":
    main()
