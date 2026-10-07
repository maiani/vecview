"""A minimal Kitaev chain: two quantum dots coupled through a superconductor.

Schematic after T. Dvir, G. Wang, N. van Loo, C.-X. Liu, G. P. Mazur, A. Bordin,
S. L. D. ten Haaf, J.-Y. Wang, D. van Driel, F. Zatelli, X. Li, F. K. Malinowski,
S. Gazibegovic, G. Badawy, E. P. A. M. Bakkers, M. Wimmer and L. P. Kouwenhoven,
"Realization of a minimal Kitaev chain in coupled quantum dots", Nature 614,
445-450 (2023), doi:10.1038/s41586-022-05585-1, arXiv:2206.08045.

An InSb nanowire lies across seven bottom finger gates under a gate dielectric.
Its middle carries a thin Al shell that continues onto the dielectric as the
grounded superconducting lead S, and the wire under it is gated by V_PG.  On
either side a quantum dot forms between two tunnel barriers, its level set by
V_LD or V_RD.  Two Cr/Au films are draped across the wire as the normal leads
N, and the wire runs out past both.  The field B points along the wire.  At
the sweet spot the dots host a pair of "poor man's" Majorana modes, one on each
dot, drawn as soft spots on the wire.

Every film is a solid with its real thickness, exaggerated: the Al is a shell
on three facets plus a slab on the dielectric, and each contact a shell over
all but the bottom facet plus a pad on either side.  Each is a cross-section
in the yz plane extruded along the wire, in convex pieces that depth-sort with
the wire, which is cut wherever a film begins or ends.

Simplified: the gates, dots and hybrid segment are not to scale, and the paper
gives no gate pitch, so the widths here are schematic.  The seven gates follow
the paper's description -- a plunger per dot and V_PG under the hybrid, and the
gates next to the three leads pinched off as two tunnel barriers per dot --
not a measured layout.  The Al is put on the three facets facing the source
of its 15 and 45 degree evaporation, taken to be perpendicular to the wire;
the Pt adatoms on it are not drawn.  The contacts cover the wire evenly,
where an evaporated film would thin out on the lower facets.  Substrate and
dielectric are generic slabs, the AlOx capping layer and the shadow walls are
left out, and B is drawn exactly along the wire rather than a few degrees off.

Uses: ``outlines.regular`` and ``outlines.to_plane`` for the cross-sections,
``extrude`` along x, a ``Part`` placed twice with ``mirror`` for the two
contacts, ``class_`` on every kind of object, ``sort_by_depth``, ``gaussian``,
``arrow3d``, and subscripted labels from ``svg.TSpan`` runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import svg

import vecview
from vecview import OrthographicCamera, Scene, outlines

# Geometry, in units of a finger gate's pitch, roughly.
R = 0.4  # the wire's hexagon circumradius
SUB_X, SUB_T = 4.3, 0.45  # substrate half-length and thickness
SUB_FRONT, SUB_BACK = -2.0, 2.3  # its front and back edge in y
T_GATE, T_DIEL = 0.07, 0.09
Z0 = T_GATE + T_DIEL  # dielectric top, where the wire and the films sit
H_WIRE = 2 * R * np.sin(np.radians(60.0))
ZC = Z0 + H_WIRE / 2  # the wire's axis
WIRE_X = 4.0  # half-length of the wire, which runs out past both contacts
HYBRID = 0.8  # half-length of the Al-covered segment
T_AL, T_N = 0.07, 0.12  # thickness of the Al film and of the Cr/Au contacts
XC, X_END = 2.55, 3.4  # inner and outer edge of each contact
CONTACT_FRONT = -0.8  # the contacts' pads run from here to the back edge
GATE_FRONT, GATE_BACK = -1.55, 1.9  # the finger gates' ends in y
# Finger gates along x, by name: (centre, width, kind).  The dots sit over the
# plungers LD and RD, between a pair of barriers each.
GATES = {
    "L-outer": (-2.22, 0.3, "barrier"),
    "LD": (-1.7, 0.52, "plunger"),
    "L-inner": (-1.18, 0.3, "barrier"),
    "PG": (0.0, 1.6, "plunger"),
    "R-inner": (1.18, 0.3, "barrier"),
    "RD": (1.7, 0.52, "plunger"),
    "R-outer": (2.22, 0.3, "barrier"),
}

INK = "#1f2430"
COLORS = dict(
    substrate="#c9ced6",
    substrate_edge="#7c8592",
    dielectric="#e6eaf0",
    gate="#8a929d",
    gate_edge="#4a5059",
    wire="#7fae8f",
    wire_edge="#2f5a40",
    al="#9dbbe0",
    al_edge="#2d5c94",
    gold="#e2b56a",
    gold_edge="#7d5a22",
    majorana="#c0392b",
    field="#1f5fa8",
)
SUB_MID, SUB_DEPTH = (SUB_FRONT + SUB_BACK) / 2, SUB_BACK - SUB_FRONT
X, Y, Z = np.eye(3)
EDGE = dict(stroke_width=0.8, stroke_linejoin="round")
LABEL_SIZE = 20.0
MATH = dict(font_family="DejaVu Serif", font_style="italic", size=LABEL_SIZE, text_anchor="middle")
COS30 = np.cos(np.radians(30.0))


# --- cross-sections, in the (y, z) plane across the wire ---------------------
# The hexagon lies on its bottom facet.  Corner k is at 60k degrees from +y
# toward +z, and facet k joins corners k and k + 1: facet 1 is the top, 4 the
# bottom, and 5, 0, 1 face a source on the +y side.  Extruding a section along
# the wire builds wall ``side-i`` on its edge i, which is how the walls a film
# hides are picked out below.

WIRE = outlines.regular(6, R, center=(0.0, ZC))


def heel(side: float, t: float, z: float) -> tuple[float, float]:
    """Where the outside of a film ``t`` thick on a lower facet reaches height ``z``.

    ``side`` is +1 for facet 5, on the +y side, and -1 for facet 3.
    """
    return (side * (R / 2 + (t + 0.5 * (z - Z0)) / COS30), z)


def shell(facets: list[int], t: float) -> np.ndarray:
    """A film ``t`` thick over a run of the wire's facets, counter-clockwise.

    The outer side runs over the corners of the wire grown by ``t``, the inner
    side back along the wire.  A film that reaches a bottom corner (5 or 4)
    runs on down to the dielectric.
    """
    grown = outlines.regular(6, R + t / COS30, center=(0.0, ZC))
    corners = [*facets, (facets[-1] + 1) % 6]
    first = heel(+1, t, Z0) if corners[0] == 5 else grown[corners[0]]
    last = heel(-1, t, Z0) if corners[-1] == 4 else grown[corners[-1]]
    outer = [grown[k] for k in corners[1:-1]]
    inner = [WIRE[k] for k in reversed(corners)]
    return np.array([inner[-1], first, *outer, last, *inner[:-1]])


def pad(side: float, t: float, y_far: float) -> np.ndarray:
    """The film on the dielectric beside the wire, from the shell's foot out to ``y_far``."""
    foot, top = heel(side, t, Z0), heel(side, t, Z0 + t)
    pts = [foot, (y_far, Z0), (y_far, Z0 + t), top]
    return np.array(pts if side > 0 else pts[::-1])


def across(section: np.ndarray, x: float) -> np.ndarray:
    """A (y, z) section lifted into the plane across the wire at ``x``."""
    return outlines.to_plane(section, origin=x * X, u=Y, v=Z)


def uncovered(faces: list[vecview.Face], covered: set[str]) -> list[vecview.Face]:
    """The faces of a piece less the ones that lie against something else.

    A covered face is never seen, and keyed inside whatever covers it, it would
    sort in front of that film's end wall.
    """
    return [f for f in faces if f.name not in covered]


AL_SHELL = shell([5, 0, 1], T_AL)  # the three facets facing the evaporation
N_SHELL = shell([5, 0, 1, 2, 3], T_N)  # all but the bottom
# The walls each film hides against the wire, the dielectric, or the shell beside
# it, by section edge: a shell's foot on the dielectric and its inner side along
# the wire, and a pad's foot and its side against the shell.
AL_COVERED = {"side-0", "side-5", "side-6", "side-7"}
N_COVERED = {"side-0", "side-6", "side-7", "side-8", "side-9", "side-10", "side-11"}
PAD_COVERED = {+1.0: {"side-0", "side-3"}, -1.0: {"side-2", "side-3"}}
# The pieces the wire is cut into wherever a film begins or ends, with the faces
# a film or a neighbouring piece covers.  Nothing of the wire under a contact
# shows, so there is no piece there, and only the wire's two ends keep caps.
WIRE_PIECES = {
    "end-L": (-WIRE_X, -X_END, {"end"}),
    "dot-L": (-XC, -HYBRID, {"start", "end"}),
    "hybrid": (-HYBRID, HYBRID, {"start", "end", "side-5", "side-0", "side-1"}),  # under the Al
    "dot-R": (HYBRID, XC, {"start", "end"}),
    "end-R": (X_END, WIRE_X, {"start"}),
}


def sub(base: str, index: str, size: float) -> list[svg.TSpan]:
    """``base`` with an upright subscript, as ``svg.TSpan`` runs.

    The drop is a ``dy`` rather than ``baseline-shift``, which cairosvg ignores.
    """
    drop = round(0.3 * size, 1)
    return [
        svg.TSpan(text=base),
        svg.TSpan(text=index, dy=drop, font_size=round(0.7 * size, 1), font_style="normal"),
    ]


def platform(scene: Scene) -> None:
    """Substrate, the seven finger gates on it, and the dielectric over them."""
    substrate = vecview.box_faces((0, SUB_MID, -SUB_T / 2), (2 * SUB_X, SUB_DEPTH, SUB_T))
    scene.faces(
        10,
        [f for f in substrate if f.name != "+z"],
        cull=True,
        fill=COLORS["substrate"],
        stroke=COLORS["substrate_edge"],
        id="substrate",
        class_="substrate",
        **EDGE,
    )
    for name, (x, w, kind) in GATES.items():
        scene.faces(
            12,
            vecview.box_faces(
                (x, (GATE_FRONT + GATE_BACK) / 2, T_GATE / 2), (w, GATE_BACK - GATE_FRONT, T_GATE)
            ),
            cull=True,
            fill=COLORS["gate"],
            stroke=COLORS["gate_edge"],
            id=f"gate-{name}",
            class_=["gate", kind],
            **EDGE,
        )
    scene.faces(
        14,
        vecview.box_faces((0, SUB_MID, Z0 / 2), (2 * SUB_X, SUB_DEPTH, Z0)),
        cull=True,
        fill=COLORS["dielectric"],
        fill_opacity=0.6,
        stroke=COLORS["substrate_edge"],
        id="dielectric",
        class_="dielectric",
        **EDGE,
    )


def contact() -> vecview.Part:
    """The right Cr/Au contact: a shell draped over the wire and a pad on either side.

    The left contact is its mirror image.
    """
    gold = dict(fill=COLORS["gold"], stroke=COLORS["gold_edge"], class_="lead", **EDGE)
    length = (X_END - XC) * X
    part = vecview.Part()
    drape = vecview.extrude(across(N_SHELL, XC), length)
    part.faces(0, uncovered(drape, N_COVERED), cull=True, id="drape", **gold)
    for side, name, y_far in ((-1.0, "front", CONTACT_FRONT), (1.0, "back", SUB_BACK)):
        beside = vecview.extrude(across(pad(side, T_N, y_far), XC), length)
        part.faces(0, uncovered(beside, PAD_COVERED[side]), cull=True, id=name, **gold)
    return part


def device(scene: Scene) -> None:
    """The wire, the Al film, and the two contacts draped over it, sorted together.

    Each film is convex pieces -- a shell on the wire and a pad on either side
    -- so that each is keyed near what it covers.  The wire is cut wherever a
    film begins or ends, so the piece beside a film sorts in front of the
    film's end wall.
    """
    scene.sort_by_depth(20)
    wire = dict(fill=COLORS["wire"], stroke=COLORS["wire_edge"], class_="semiconductor", **EDGE)
    for name, (x0, x1, covered) in WIRE_PIECES.items():
        piece = vecview.extrude(across(WIRE, x0), (x1 - x0) * X)
        scene.faces(20, uncovered(piece, covered), cull=True, id=f"nanowire-{name}", **wire)

    # The Al shell, and the grounded lead S it continues into on the +y side.
    al = dict(fill=COLORS["al"], stroke=COLORS["al_edge"], class_="superconductor", **EDGE)
    length = 2 * HYBRID * X
    al_shell = vecview.extrude(across(AL_SHELL, -HYBRID), length)
    scene.faces(20, uncovered(al_shell, AL_COVERED), cull=True, id="al-shell", **al)
    al_lead = vecview.extrude(across(pad(+1.0, T_AL, SUB_BACK), -HYBRID), length)
    scene.faces(20, uncovered(al_lead, PAD_COVERED[+1.0]), cull=True, id="al-lead", **al)

    # The two normal leads, mirror images across the middle of the wire.
    normal_contact = contact()
    scene.place(20, normal_contact, mirror=X, id="contact-L")
    scene.place(20, normal_contact, id="contact-R")


def majorana_modes(scene: Scene) -> None:
    """The pair of poor man's Majorana modes, a soft spot on top of each dot."""
    top = Z0 + H_WIRE + 0.004
    for k, gate in enumerate(("LD", "RD"), start=1):
        x = GATES[gate][0]
        scene.gaussian(
            25,
            (x, 0.0, top),
            X,
            Y,
            0.24,
            0.12,
            id=f"majorana-{k}",
            class_="majorana",
            color=COLORS["majorana"],
            opacity=0.9,
        )
        scene.text(
            40,
            (x, 0.0, top),
            sub("\N{GREEK SMALL LETTER GAMMA}", str(k), LABEL_SIZE),
            dy=-21,
            fill=COLORS["majorana"],
            id=f"label-gamma-{k}",
            class_="label",
            **MATH,
        )


def labels(scene: Scene) -> None:
    """The three gate voltages in front, and the roles of the leads, N-S-N, behind."""
    for gate in ("LD", "PG", "RD"):
        x = GATES[gate][0]
        scene.text(
            40,
            (x, GATE_FRONT, Z0),
            sub("V", gate, LABEL_SIZE),
            dy=28,
            fill=INK,
            id=f"label-{gate}",
            class_="label",
            **MATH,
        )
    y = SUB_BACK - 0.6
    roles = {
        "N-L": ("N", (-(XC + X_END) / 2, y, Z0 + T_N)),
        "S": ("S", (0.0, y, Z0 + T_AL)),
        "N-R": ("N", ((XC + X_END) / 2, y, Z0 + T_N)),
    }
    for key, (role, at) in roles.items():
        scene.text(
            40,
            at,
            role,
            dy=7,
            fill=INK,
            id=f"label-{key}",
            class_="label",
            **{**MATH, "font_style": "normal", "font_family": "DejaVu Sans"},
        )


def field(scene: Scene) -> None:
    """The magnetic field B, along the wire."""
    tail, length = np.array([-0.7, 1.5, Z0 + 1.7]), 1.4
    scene.arrow3d(
        30,
        tail,
        X,
        length,
        shaft_r=0.045,
        head_r=0.13,
        head_len=0.32,
        fill=COLORS["field"],
        highlight="#cfe0f7",
        stroke="#0d2c55",
        stroke_width=0.6,
        id="field",
        class_="field",
    )
    scene.text(
        40,
        tail + length * X,
        "B",
        dx=12,
        dy=7,
        fill=COLORS["field"],
        id="label-B",
        class_="label",
        **MATH,
    )


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=-60.0, elev_deg=30.0, scale=96.0)
    scene = Scene(cam, pad=14.0, background="#ffffff")
    platform(scene)
    device(scene)
    majorana_modes(scene)
    labels(scene)
    field(scene)
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "kitaev_chain.svg"))
