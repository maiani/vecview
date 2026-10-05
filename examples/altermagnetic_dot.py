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

* ``prism_faces`` + ``Scene.prism_walls`` for non-convex, arc-shaped gates and
  tapered leads, with no seams between wall facets;
* ``Scene.gaussian`` for the soft densities, and ``ellipse_shape`` for their
  1/e contours and the sublattice motif;
* ``Scene.arrow(normal="camera")`` for spins along z and a legend frame that
  read at full width from any viewpoint;
* ``Scene.text`` anchored at world points, so labels follow the geometry;
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
from vecview import Camera, ObliqueCamera, OrthographicCamera, Scene

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# --- physics-informed geometry (units of the confinement length) -------------
B_SKETCH = 0.40  # altermagnetic strength, exaggerated for legibility
R_LEAD = 2.0  # radial distance of the lead tips
R_GATE = 2.5  # inner radius of the arc-shaped confinement gates
W_ARC = 0.55  # radial width of the gate arcs
W_ARM = 0.5  # width of the arms connecting the gates to the slab edge
GAP_LEAD = 24.0  # half-opening of the gate ring around each lead (deg)
GAP_MID = 16.0  # opening between the two gates on one side (deg)
THETA_L, THETA_R = 180.0, 0.0  # polar angles of the contacts (deg)

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
TRIAD_LEN, NEEL_LEN = 1.1, 1.8
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

X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

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
    tip = lambda th: R_LEAD * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])  # noqa: E731
    up, down = (lambda th, s=s: density(s, *tip(th)) for s in (+1, -1))
    assert up(THETA_R) > down(THETA_R) and up(THETA_L) > down(THETA_L), "leads favour s = +1"
    assert down(90.0) > up(90.0), "a contact rotated by 90 deg favours s = -1"
    assert np.isclose(up(45.0), down(45.0)), "and the diagonal is a node"


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


def finger(theta_deg: float, r_tip: float, w_out: float, w_tip: float) -> np.ndarray:
    """Footprint of a tapered lead pointing at the dot from ``theta_deg``.

    A trapezoid from the slab edge to a half-disc tip whose apex sits at
    ``r_tip``; the hull blends the tip into the taper.
    """
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    r_out = min(
        SLAB_X / abs(c) if abs(c) > 1e-9 else np.inf,
        SLAB_Y / abs(s) if abs(s) > 1e-9 else np.inf,
    )
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, 17)
    rc = r_tip + 0.5 * w_tip
    tip = np.column_stack([rc + 0.5 * w_tip * np.cos(t), 0.5 * w_tip * np.sin(t)])
    pts = np.vstack([[r_out, -0.5 * w_out], [r_out, 0.5 * w_out], tip])
    return convex_hull(pts @ np.array([[c, -s], [s, c]]).T)


def arc_gate(quadrant: int, n: int = 40) -> np.ndarray:
    """Footprint of one arc-shaped gate and its arm to the slab edge: simple, non-convex.

    An annular sector between ``R_GATE`` and ``R_GATE + W_ARC`` in the first
    quadrant, joined to a straight radial arm along its mid-angle; the other
    quadrants are mirror images.
    """
    r_in, r_out = R_GATE, R_GATE + W_ARC
    th0, th1 = np.radians(GAP_LEAD), np.radians(90.0 - GAP_MID / 2)
    phi = 0.5 * (th0 + th1)
    d = np.array([np.cos(phi), np.sin(phi)])
    nrm = np.array([-d[1], d[0]])
    h = 0.5 * W_ARM
    delta = np.arcsin(h / r_out)  # half-angle the arm cuts out of the outer arc

    def to_edge(side: float) -> np.ndarray:
        p0 = side * h * nrm
        return p0 + min((SLAB_X - p0[0]) / d[0], (SLAB_Y - p0[1]) / d[1]) * d

    def arc(r: float, a: float, b: float) -> np.ndarray:
        t = np.linspace(a, b, n)
        return np.column_stack([r * np.cos(t), r * np.sin(t)])

    pts = np.vstack(
        [
            arc(r_out, th0, phi - delta),
            [to_edge(-1.0), to_edge(+1.0)],
            arc(r_out, phi + delta, th1),
            arc(r_in, th1, th0),
        ]
    )
    sx, sy = {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}[quadrant]
    return pts * (sx, sy)  # prism_faces normalizes the winding a mirror flips


def prism(
    scene: Scene,
    layer: int,
    foot: np.ndarray,
    z0: float,
    z1: float,
    *,
    top: str,
    side: str,
    edge: str,
    id: str,
) -> None:
    """An extruded footprint: seamless walls, then the cap (``{id}-pz``) over them.

    Walls-then-cap is exact for one solid seen from above; the walls get only a
    light stroke, since a stroke is the one thing that could show a hidden wall.
    """
    style = dict(stroke=edge, stroke_linejoin="round")
    scene.prism_walls(
        layer, foot, z0, z1, fill=side, id=f"{id}-walls", stroke_width=0.3 * LW, **style
    )
    cap = [f for f in vecview.prism_faces(foot, z0, z1) if f.name == "+z"]
    scene.faces(layer, cap, fill=top, id=id, stroke_width=0.6 * LW, **style)


def vertical_arrow(
    scene: Scene,
    layer: int,
    base: tuple[float, float, float],
    up: bool,
    color: str,
    length: float,
    id: str,
    width: float = 1.0,
) -> None:
    """An arrow along +z or -z, turned to face whichever camera draws it."""
    origin = np.array(base, float) + (0.0 if up else length) * np.array(Z)
    scene.arrow(
        layer,
        origin,
        Z if up else (0.0, 0.0, -1.0),
        length,
        normal="camera",
        shaft_w=0.10 * width,
        head_w=0.34 * width,
        head_len=0.34 * width,
        fill=color,
        id=id,
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
        **ALIGN[align],
    )


def heterostructure(scene: Scene) -> None:
    """Substrate and altermagnetic layer, with the d-wave sublattice motif on top."""
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

    # Two sublattices with opposite spins (colour), related by a 90-degree
    # rotation, on a checkerboard that fades out over the dot.
    a, z = 0.6, Z_GAS + 0.004
    for i, x in enumerate(np.arange(-SLAB_X + a / 2, SLAB_X, a)):
        for j, y in enumerate(np.arange(-SLAB_Y + a / 2, SLAB_Y, a)):
            fade = np.clip((np.hypot(x, y) - 2.8) / 2.0, 0.0, 1.0)
            if fade <= 0.0:
                continue
            up = (i + j) % 2 == 0
            scene.polygon(
                13,
                vecview.ellipse_shape((x, y, z), X if up else Y, Y if up else X, 0.2, 0.075, n=24),
                fill=COLORS["up" if up else "down"],
                fill_opacity=round(0.32 * fade, 3),
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
        )
    wx_up, _ = widths(+1)
    _, wy_dn = widths(-1)
    vertical_arrow(scene, 34, (-wx_up, 0.0, z), True, COLORS["up"], 1.05, "spin-up")
    vertical_arrow(scene, 34, (0.0, -wy_dn, z - 1.05), False, COLORS["down"], 1.05, "spin-down")


def electrodes(scene: Scene) -> None:
    """Leads at the contact angles and the four arc gates, back to front."""
    gate = dict(top=COLORS["gate_top"], side=COLORS["gate_side"], edge=COLORS["gate_edge"])
    lead = dict(top=COLORS["lead_top"], side=COLORS["lead_side"], edge=COLORS["lead_edge"])
    z1 = Z_GAS + T_GATE
    prism(scene, 15, arc_gate(2), Z_GAS, z1, id="gate-back-left", **gate)
    prism(scene, 15, arc_gate(1), Z_GAS, z1, id="gate-back-right", **gate)
    prism(scene, 30, finger(THETA_L, R_LEAD, 3.4, 0.8), Z_GAS, Z_GAS + T_LEAD, id="lead-L", **lead)
    prism(scene, 31, finger(THETA_R, R_LEAD, 3.4, 0.8), Z_GAS, Z_GAS + T_LEAD, id="lead-R", **lead)
    prism(scene, 32, arc_gate(3), Z_GAS, z1, id="gate-front-left", **gate)
    prism(scene, 32, arc_gate(4), Z_GAS, z1, id="gate-front-right", **gate)


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
        )
    label(scene, o + TRIAD_LEN * np.array(X), "x", "west", (0.8, 0.4), "label-x")
    label(scene, o + TRIAD_LEN * np.array(Y), "y", "west", (1.0, 0.0), "label-y")
    label(scene, o + TRIAD_LEN * np.array(Z), "z", "east", (-1.0, 2.0), "label-z")
    vertical_arrow(scene, 41, NEEL_AT, True, COLORS["neel"], NEEL_LEN, "neel-vector", width=1.4)
    label(scene, np.array(NEEL_AT) + NEEL_LEN * np.array(Z), "n", "west", (2.4, -0.5), "label-neel")


def circuit(scene: Scene) -> None:
    """The open bias circuit below the device: V and ground on L, ground on R, and I."""
    xc, zc = SLAB_X - 0.35, Z_GAS + T_LEAD
    wire = dict(stroke=COLORS["ink"], stroke_width=1.2 * LW, stroke_linejoin="round")
    z_bat, z_gnd, z_gnd_r = zc - 0.45 * WIRE_DROP_L, zc - WIRE_DROP_L, zc - WIRE_DROP_R
    g = 0.14  # half-gap between the battery plates
    scene.polyline(
        50, [(-xc, 0, zc), (-X_WIRE, 0, zc), (-X_WIRE, 0, z_bat + g)], id="wire-L", **wire
    )
    scene.polyline(50, [(-X_WIRE, 0, z_bat - g), (-X_WIRE, 0, z_gnd)], id="wire-V", **wire)
    scene.polyline(50, [(xc, 0, zc), (X_WIRE, 0, zc), (X_WIRE, 0, z_gnd_r)], id="wire-R", **wire)
    for z, half, w, key in ((z_bat + g, 0.75, 1.2, "plus"), (z_bat - g, 0.4, 2.6, "minus")):
        scene.polyline(
            50,
            [(-X_WIRE - half, 0, z), (-X_WIRE + half, 0, z)],
            stroke=COLORS["ink"],
            stroke_width=w * LW,
            id=f"battery-{key}",
        )
    for x, z, key in ((-X_WIRE, z_gnd, "L"), (X_WIRE, z_gnd_r, "R")):
        for k, half in enumerate((0.55, 0.36, 0.17)):
            scene.polyline(
                50,
                [(x - half, 0, z - 0.22 * k), (x + half, 0, z - 0.22 * k)],
                stroke=COLORS["ink"],
                stroke_width=1.2 * LW,
                id=f"ground-{key}-{k}",
            )
    for side, key in ((-1.0, "L"), (+1.0, "R")):
        scene.polygon(
            51,
            vecview.circle_shape((side * xc, 0, zc), 0.16, Z, n=24),
            fill=COLORS["ink"],
            id=f"contact-{key}",
        )
    i_mid = -0.5 * (xc + X_WIRE)
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
    )
    label(scene, (-X_WIRE - 0.75, 0, z_bat), "V", "east", (-1.0, 0.0), "label-V")
    label(scene, (i_mid, 0, zc + 0.45), "I", "south", (0.0, -1.0), "label-I")
    for th, name in ((THETA_L, "L"), (THETA_R, "R")):
        c = np.array([np.cos(np.radians(th)), np.sin(np.radians(th)), 0.0])
        label(scene, 4.7 * c + (0, 0, Z_GAS + T_LEAD), name, "center", (0.0, 0.0), f"label-{name}")


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
