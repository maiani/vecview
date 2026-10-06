"""Device sketch: an altermagnetic quantum dot in a two-terminal setup.

A gate-defined dot in a 2D altermagnetic electron gas. Two metallic leads (gold)
contact it along x, through gaps in a ring of four arc-shaped confinement gates
(dark grey), each wired out to the slab edge. An open bias circuit sits below: a
battery V and ground on L, ground on R, and the current I. The two Neel-spin
densities of the ground doublet are drawn in the plane of the gas: the s = +1
(red) orbital is elongated along x and the s = -1 (blue) orbital along y. The
host shows as a faint d-wave sublattice motif: two sublattices with opposite
spins, related by a 90-degree rotation.

The densities are the exact ground states of the model,
|psi_s|^2 ~ exp(-x^2/w_x^2 - y^2/w_y^2), w_{x,y} = (1 +- 2 s b)^(1/4), at an
altermagnetic strength b exaggerated so the deformation reads at column width.
Geometry is in units of the confinement length; one scene unit is 9.6 px at print
size, about 2.5 mm, so stroke widths and type are the printed values.

What it exercises, in one picture:

* ``Part`` and ``Scene.place``: each electrode is drawn once -- seamless
  ``Scene.prism_walls`` under a ``prism_faces`` cap -- and placed several times,
  one arc gate mirrored into all four quadrants and one lead mirrored into the
  other; the ground symbol is placed twice the same way;
* ``outlines.union`` and ``outlines.intersection`` to join each gate's arc,
  an ``annulus_sector``, to its arm and cut the arm off at the slab edge;
* ``Scene.gaussian`` for the soft densities, and ``ellipse_shape`` for their
  1/e contours and the sublattice motif;
* ``Scene.arrow(normal="camera")`` for spins along z and a legend frame that
  read at full width from any viewpoint;
* ``Scene.text`` anchored at world points, so labels follow the geometry, and
  ``class_`` on every kind of object (``.gate``, ``.lead``, ``.density``, ...);
* objects and cameras kept apart: the scene is world-space throughout, so it is
  built once and holds its four projections as named cameras, "main" active,
  and each render picks one by name. The layout -- the wire runs, the legend
  frame -- was tuned for the main camera; the other projections show the same
  objects, not finished layouts, and the frame meets the ground symbol in some
  of them.

The labels are plain SVG text, so the example needs nothing beyond vecview.  For
typeset labels, reserve ``Scene.slot`` groups instead and let the tool that
composes the page fill them with TeX fragments.

Usage:
    python examples/altermagnetic_dot.py [--projection {main,isometric,dimetric,cabinet,all}]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import cairosvg
import numpy as np

import vecview
from vecview import Camera, ObliqueCamera, OrthographicCamera, Part, Scene, outlines

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# --- physics-informed geometry (units of the confinement length) -------------
B_SKETCH = 0.40  # altermagnetic strength, exaggerated for legibility
R_LEAD = 2.0  # how far the lead tips stop from the centre of the dot
W_LEAD, W_TIP = 3.4, 0.8  # lead width at the slab edge and at its rounded tip
R_GATE = 2.5  # inner radius of the arc-shaped confinement gates
W_ARC = 0.55  # radial width of the gate arcs
W_ARM = 0.5  # width of the arms connecting the gates to the slab edge
GAP_LEAD = 24.0  # half-opening of the gate ring around each lead (deg)
GAP_MID = 16.0  # opening between the two gates on one side (deg)

SLAB_X, SLAB_Y = 6.2, 5.0  # half-extents of the heterostructure
Z_GAS = 0.0  # top of the altermagnetic layer, where the 2D gas lives
T_LAYER, T_SUB = 0.22, 1.0  # layer and substrate thickness
T_LEAD, T_GATE = 0.34, 0.26

# --- print-size styling ---------------------------------------------------
PT = 96.0 / 72.0  # px per pt
SCALE = 9.6  # px per unit at print size
LABEL_PX = 8.0 * PT
LW = 0.5 * PT  # reference edge width
DENSITY_OPACITY = 0.65
SPIN_LEN, TRIAD_LEN, NEEL_LEN = 1.05, 1.1, 1.8
X_WIRE = SLAB_X + 3.6  # where the bias wires turn down, clear of the slab on screen
WIRE_DROP_L, WIRE_DROP_R = 4.6, 2.6
TRIAD_AT = (-7.9, -5.2, Z_GAS - 3.0)
NEEL_AT = (SLAB_X - 0.8, SLAB_Y - 1.6, Z_GAS)

COLORS = dict(
    up="#d62828",
    down="#1f6fd1",
    layer_top="#eeecf3",
    layer_side="#cfcadb",
    sub_side="#bfc5cd",
    edge="#6f7682",
    lead_top="#e2b56a",
    lead_side="#b98a40",
    lead_edge="#7d5a22",
    gate_top="#737a84",
    gate_side="#4a5059",
    gate_edge="#2e3238",
    ink="#1a1a1a",
    neel="#5b2a86",
)

X, Y, Z = np.eye(3)

# Which point of a label sits on its anchor, as SVG text alignment.
ALIGN = {
    "center": dict(text_anchor="middle", dominant_baseline="central"),
    "west": dict(text_anchor="start", dominant_baseline="central"),
    "east": dict(text_anchor="end", dominant_baseline="central"),
    "south": dict(text_anchor="middle"),
}


def widths(s: int, b: float = B_SKETCH) -> tuple[float, float]:
    """Density 1/e half-widths (w_x, w_y) of the spin-s ground state."""
    return (1.0 + 2.0 * s * b) ** 0.25, (1.0 - 2.0 * s * b) ** 0.25


def density(s: int, x: float, y: float) -> float:
    """Ground-state density of spin sector s at (x, y), normalized at the centre."""
    wx, wy = widths(s)
    return float(np.exp(-((x / wx) ** 2) - (y / wy) ** 2))


def check_structure() -> None:
    """The properties the sketch exists to show, at the drawn parameters."""
    assert abs(2.0 * B_SKETCH) < 1.0, "outside the confinement bound |b| < 1/2"
    assert np.allclose(widths(+1), widths(-1)[::-1]), "s = -1 is s = +1 rotated by 90 deg"

    def at_contact(s: int, theta_deg: float) -> float:
        """Spin-s density under a lead tip at polar angle ``theta_deg``."""
        th = np.radians(theta_deg)
        return density(s, R_LEAD * np.cos(th), R_LEAD * np.sin(th))

    assert all(at_contact(+1, th) > at_contact(-1, th) for th in (0.0, 180.0)), (
        "leads along x favour s = +1"
    )
    assert at_contact(-1, 90.0) > at_contact(+1, 90.0), "a contact rotated by 90 deg favours s = -1"
    assert np.isclose(at_contact(+1, 45.0), at_contact(-1, 45.0)), "and the diagonal is a node"


# --- footprints -------------------------------------------------------------


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2D points (Andrew's monotone chain)."""
    pts = sorted(map(tuple, points))

    def chain(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while (
                len(out) >= 2
                and (
                    (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                    - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
                )
                <= 0
            ):
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(chain(pts) + chain(pts[::-1]))


def lead_footprint() -> np.ndarray:
    """The right lead: a taper from the slab edge to a rounded tip ``R_LEAD`` from the dot.

    The hull of the two corners at the slab edge and a half-disc tip blends the
    tip into the taper.
    """
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, 17)
    centre = R_LEAD + 0.5 * W_TIP  # so the apex of the tip sits at R_LEAD
    tip = np.column_stack([centre + 0.5 * W_TIP * np.cos(t), 0.5 * W_TIP * np.sin(t)])
    edge = [[SLAB_X, -0.5 * W_LEAD], [SLAB_X, 0.5 * W_LEAD]]
    return convex_hull(np.vstack([edge, tip]))


def gate_footprint() -> np.ndarray:
    """The gate in the first quadrant: an arc around the dot and its arm out to the slab edge.

    The arc spans the quadrant less half the opening for the lead on one side
    and half the gap to the next gate on the other; the arm runs out along its
    mid-angle.  Simple but not convex, which ``prism_walls`` handles.
    """
    th0, th1 = GAP_LEAD, 90.0 - GAP_MID / 2
    arc = vecview.annulus_sector((0, 0), R_GATE, R_GATE + W_ARC, th0, th1, n=80)
    d = vecview.in_plane_dir(0.5 * (th0 + th1))[:2]
    side = 0.5 * W_ARM * np.array([-d[1], d[0]])
    r0, r1 = R_GATE + 0.5 * W_ARC, 2 * SLAB_X  # from inside the arc to past the slab edge
    arm = np.array([r0 * d - side, r1 * d - side, r1 * d + side, r0 * d + side])
    (joined,) = outlines.union(arc, arm)
    (gate,) = outlines.intersection(joined, outlines.rect((-SLAB_X, -SLAB_Y), (SLAB_X, SLAB_Y)))
    return gate


def electrode(kind: str, footprint: np.ndarray, height: float) -> Part:
    """A metal film on the gas: seamless walls (``walls``), then the cap (``cap-pz``) over them.

    ``kind`` picks the colours, ``"gate"`` or ``"lead"``.  Walls-then-cap is
    exact for one solid seen from above; the walls get only a light stroke,
    since a stroke is the one thing that could show a hidden wall.
    """
    part = Part()
    style = dict(stroke=COLORS[f"{kind}_edge"], stroke_linejoin="round")
    part.prism_walls(
        0,
        footprint,
        0.0,
        height,
        fill=COLORS[f"{kind}_side"],
        id="walls",
        stroke_width=0.3 * LW,
        **style,
    )
    cap = vecview.prism_faces(footprint, 0.0, height)[:1]
    part.faces(0, cap, fill=COLORS[f"{kind}_top"], id="cap", stroke_width=0.6 * LW, **style)
    return part


def upright_arrow(
    scene: Scene,
    layer: int,
    tail: tuple[float, float, float] | np.ndarray,
    direction: np.ndarray,
    length: float,
    color: str,
    id: str,
    width: float = 1.0,
) -> None:
    """An arrow along +z or -z, turned to face whichever camera draws it."""
    scene.arrow(
        layer,
        tail,
        direction,
        length,
        normal="camera",
        shaft_w=0.10 * width,
        head_w=0.34 * width,
        head_len=0.34 * width,
        fill=color,
        id=id,
        class_="spin",
    )


def label(
    scene: Scene,
    point: tuple[float, float, float] | np.ndarray,
    s: str,
    align: str,
    offset_pt: tuple[float, float],
    id: str,
) -> None:
    """Upright text at a world point, offset on screen in points."""
    scene.text(
        45,
        point,
        s,
        offset_pt[0] * PT,
        offset_pt[1] * PT,
        size=LABEL_PX,
        font_style="italic",
        fill=COLORS["ink"],
        id=id,
        class_="label",
        **ALIGN[align],
    )


# --- the picture --------------------------------------------------------------


def heterostructure(scene: Scene) -> None:
    """Substrate and altermagnetic layer, the layer's top drawn over its walls."""
    edge = dict(stroke=COLORS["edge"], stroke_width=0.6 * LW, stroke_linejoin="round")
    sub = vecview.box_faces((0, 0, Z_GAS - T_LAYER - T_SUB / 2), (2 * SLAB_X, 2 * SLAB_Y, T_SUB))
    scene.faces(
        10,
        [f for f in sub if f.name != "+z"],
        cull=True,
        fill=COLORS["sub_side"],
        id="substrate",
        **edge,
    )
    am = vecview.box_faces((0, 0, Z_GAS - T_LAYER / 2), (2 * SLAB_X, 2 * SLAB_Y, T_LAYER))
    scene.faces(11, am, cull=True, fill=COLORS["layer_side"], id="am-layer", **edge)
    scene.faces(
        12, [f for f in am if f.name == "+z"], fill=COLORS["layer_top"], id="am-top", **edge
    )


def sublattice_motif(scene: Scene) -> None:
    """Two sublattices with opposite spins (colour), related by a 90-degree rotation.

    A checkerboard of small ellipses on the gas, fading out over the dot so the
    densities read against a clean background.
    """
    pitch, z = 0.6, Z_GAS + 0.004
    clear, ramp = 2.8, 2.0  # no motif within `clear` of the dot, full strength `ramp` further
    for i, x in enumerate(np.arange(-SLAB_X + pitch / 2, SLAB_X, pitch)):
        for j, y in enumerate(np.arange(-SLAB_Y + pitch / 2, SLAB_Y, pitch)):
            fade = np.clip((np.hypot(x, y) - clear) / ramp, 0.0, 1.0)
            if fade <= 0.0:
                continue
            up = (i + j) % 2 == 0
            scene.polygon(
                13,
                vecview.ellipse_shape((x, y, z), X if up else Y, Y if up else X, 0.2, 0.075, n=24),
                fill=COLORS["up" if up else "down"],
                fill_opacity=round(0.32 * fade, 3),
                class_=["sublattice", "up" if up else "down"],
            )


def densities(scene: Scene) -> None:
    """The two Neel-spin densities as Gaussian fills, with their 1/e contours and spins."""
    z = Z_GAS + 0.01
    scene.polygon(
        19,
        vecview.circle_shape((0, 0, z), R_LEAD, Z, n=128),
        fill="none",
        stroke=COLORS["edge"],
        stroke_width=0.5 * LW,
        stroke_dasharray="1.6 1.4",
        id="dot-boundary",
    )
    # Both fills under both contours, so neither contour is veiled by the other fill.
    for s, key in ((+1, "up"), (-1, "down")):
        wx, wy = widths(s)
        scene.gaussian(
            20,
            (0, 0, z),
            X,
            Y,
            wx,
            wy,
            id=f"density-{key}-fill",
            class_="density",
            color=COLORS[key],
            opacity=DENSITY_OPACITY,
        )
    for s, key in ((+1, "up"), (-1, "down")):
        wx, wy = widths(s)
        scene.polygon(
            21,
            vecview.ellipse_shape((0, 0, z), X, Y, wx, wy, n=96),
            fill="none",
            stroke=COLORS[key],
            stroke_width=1.1 * LW,
            id=f"density-{key}",
            class_="density-contour",
        )
    # Each spin stands on its own contour: up on the long axis of s = +1, down
    # hanging from the long axis of s = -1.
    wx_up, _ = widths(+1)
    _, wy_dn = widths(-1)
    upright_arrow(scene, 34, (-wx_up, 0.0, z), Z, SPIN_LEN, COLORS["up"], "spin-up")
    upright_arrow(scene, 34, (0.0, -wy_dn, z), -Z, SPIN_LEN, COLORS["down"], "spin-down")


def electrodes(scene: Scene) -> None:
    """The four arc gates and the two leads, back to front.

    One gate and one lead are drawn; the rest are their mirror images.  The
    gates behind the leads (+y, for the main camera) go under them, the ones in
    front over them.
    """
    gate = electrode("gate", gate_footprint(), T_GATE)
    lead = electrode("lead", lead_footprint(), T_LEAD)
    on_gas = (0.0, 0.0, Z_GAS)
    scene.place(15, gate, at=on_gas, mirror=X, id="gate-back-left", class_="gate")
    scene.place(15, gate, at=on_gas, id="gate-back-right", class_="gate")
    scene.place(30, lead, at=on_gas, mirror=X, id="lead-L", class_="lead")
    scene.place(31, lead, at=on_gas, id="lead-R", class_="lead")
    scene.place(32, gate, at=on_gas, rotate=(Z, 180.0), id="gate-front-left", class_="gate")
    scene.place(32, gate, at=on_gas, mirror=Y, id="gate-front-right", class_="gate")


def frame_and_neel(scene: Scene) -> None:
    """The x, y, z legend frame off the slab, and the Neel vector standing on the layer."""
    o = np.array(TRIAD_AT)
    for d, key in ((X, "x"), (Y, "y"), (Z, "z")):
        scene.arrow(
            40,
            o,
            d,
            TRIAD_LEN,
            normal="camera",
            shaft_w=0.09,
            head_w=0.32,
            head_len=0.34,
            fill=COLORS["ink"],
            id=f"axis-{key}",
            class_="axis",
        )
    label(scene, o + TRIAD_LEN * X, "x", "west", (0.8, 0.4), "label-x")
    label(scene, o + TRIAD_LEN * Y, "y", "west", (1.0, 0.0), "label-y")
    label(scene, o + TRIAD_LEN * Z, "z", "east", (-1.0, 2.0), "label-z")
    upright_arrow(scene, 41, NEEL_AT, Z, NEEL_LEN, COLORS["neel"], "neel-vector", width=1.4)
    label(scene, np.array(NEEL_AT) + NEEL_LEN * Z, "n", "west", (2.4, -0.5), "label-neel")


def ground_symbol() -> Part:
    """Three bars narrowing downward, the top one centred on the part's origin."""
    part = Part()
    for k, half in enumerate((0.55, 0.36, 0.17)):
        part.polyline(
            0,
            [(-half, 0, -0.22 * k), (half, 0, -0.22 * k)],
            stroke=COLORS["ink"],
            stroke_width=1.2 * LW,
            id=str(k),
        )
    return part


def circuit(scene: Scene) -> None:
    """The open bias circuit below the device: V and ground on L, ground on R, and I.

    It is drawn in the plane y = 0, through both contacts.
    """
    xc, zc = SLAB_X - 0.35, Z_GAS + T_LEAD  # the contacts, on top of the leads
    wire = dict(
        stroke=COLORS["ink"], stroke_width=1.2 * LW, stroke_linejoin="round", class_="circuit"
    )
    z_bat, z_gnd, z_gnd_r = zc - 0.45 * WIRE_DROP_L, zc - WIRE_DROP_L, zc - WIRE_DROP_R
    g = 0.14  # half-gap between the battery plates
    scene.polyline(
        50, [(-xc, 0, zc), (-X_WIRE, 0, zc), (-X_WIRE, 0, z_bat + g)], id="wire-L", **wire
    )
    scene.polyline(50, [(-X_WIRE, 0, z_bat - g), (-X_WIRE, 0, z_gnd)], id="wire-V", **wire)
    scene.polyline(50, [(xc, 0, zc), (X_WIRE, 0, zc), (X_WIRE, 0, z_gnd_r)], id="wire-R", **wire)
    # The battery: a long thin plate (+) over a short thick one (-).
    for z, half, w, key in ((z_bat + g, 0.75, 1.2, "plus"), (z_bat - g, 0.4, 2.6, "minus")):
        scene.polyline(
            50,
            [(-X_WIRE - half, 0, z), (-X_WIRE + half, 0, z)],
            stroke=COLORS["ink"],
            stroke_width=w * LW,
            id=f"battery-{key}",
            class_="circuit",
        )
    ground = ground_symbol()
    scene.place(50, ground, at=(-X_WIRE, 0, z_gnd), id="ground-L", class_="circuit")
    scene.place(50, ground, at=(X_WIRE, 0, z_gnd_r), id="ground-R", class_="circuit")
    for side, key in ((-1.0, "L"), (+1.0, "R")):
        scene.polygon(
            51,
            vecview.circle_shape((side * xc, 0, zc), 0.16, Z, n=24),
            fill=COLORS["ink"],
            id=f"contact-{key}",
            class_="circuit",
        )
    i_mid = -0.5 * (xc + X_WIRE)  # the current arrow, over the middle of the left wire
    scene.arrow(
        50,
        (i_mid - 0.6, 0, zc + 0.45),
        X,
        1.2,
        normal=Y,
        shaft_w=0.09,
        head_w=0.32,
        head_len=0.34,
        fill=COLORS["ink"],
        id="current-arrow",
        class_="circuit",
    )
    label(scene, (-X_WIRE - 0.75, 0, z_bat), "V", "east", (-1.0, 0.0), "label-V")
    label(scene, (i_mid, 0, zc + 0.45), "I", "south", (0.0, -1.0), "label-I")
    for side, name in ((-1.0, "L"), (1.0, "R")):
        label(scene, (side * 4.7, 0, Z_GAS + T_LEAD), name, "center", (0.0, 0.0), f"label-{name}")


PROJECTIONS: dict[str, Callable[[float], Camera]] = {
    "main": lambda s: OrthographicCamera(azim_deg=-65.0, elev_deg=36.0, scale=s),
    "isometric": OrthographicCamera.isometric,
    "dimetric": OrthographicCamera.dimetric,
    "cabinet": ObliqueCamera.cabinet,
}


def build() -> Scene:
    check_structure()
    cameras = {name: projection(SCALE) for name, projection in PROJECTIONS.items()}
    scene = Scene("main", cameras=cameras, pad=2.0)
    heterostructure(scene)
    sublattice_motif(scene)
    densities(scene)
    electrodes(scene)
    frame_and_neel(scene)
    circuit(scene)
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--projection",
        default="main",
        choices=["all", *PROJECTIONS],
        help="which projection to render; 'all' writes one file each",
    )
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # Everything is world-space, so the objects are built once and each of the
    # scene's cameras only renders them.
    scene = build()
    wanted = list(scene.cameras) if args.projection == "all" else [args.projection]
    for name in wanted:
        out = scene.save(OUT / f"altermagnetic_dot_{name}.svg", name)
        # Rasterizing is deliberately not vecview's job; cairosvg is example-only.
        # A scale of 6.25 is 600 dpi for this print-size scene.
        cairosvg.svg2png(url=str(out), write_to=str(out.with_suffix(".png")), scale=6.25)
        doc = scene.render(name)
        mm = 25.4 / 96
        print(f"{name:>10}  {doc.width * mm:5.1f} x {doc.height * mm:<5.1f} mm  -> {out.name}")


if __name__ == "__main__":
    main()
