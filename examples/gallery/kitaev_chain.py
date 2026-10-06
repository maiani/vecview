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

Uses: ``extrude`` along x, ``class_`` on every kind of
object, ``sort_by_depth``, ``gaussian``, ``arrow3d``, and subscripted labels
from ``svg.TSpan`` runs.
"""

from __future__ import annotations

import itertools

import numpy as np
import svg
from _common import export

import vecview
from vecview import OrthographicCamera, Scene

NAME = "kitaev_chain"

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
COS30 = np.cos(np.radians(30.0))


# --- cross-sections in the (y, z) plane --------------------------------------
# The hexagon lies on its bottom facet.  Corner k is at 60k degrees from +y
# toward +z, and facet k joins corners k and k + 1: facet 1 is the top, 4 the
# bottom, and 5, 0, 1 face a source on the +y side.


def corner(k: int, grow: float = 0.0) -> tuple[float, float]:
    """Corner ``k`` of the wire, or of the wire grown by a film ``grow`` thick."""
    r = R + grow / COS30
    a = np.radians(60.0 * k)
    return (r * np.cos(a), ZC + r * np.sin(a))


def heel(side: float, t: float, z: float) -> tuple[float, float]:
    """Where the outside of a film ``t`` thick on a lower facet reaches height ``z``.

    ``side`` is +1 for facet 5, on the +y side, and -1 for facet 3.
    """
    return (side * (R / 2 + (t + 0.5 * (z - Z0)) / COS30), z)


def shell(corners: range, t: float) -> np.ndarray:
    """A film ``t`` thick over the facets between the given corners, counter-clockwise.

    The outer side runs over the grown corners, the inner side back along the
    wire.  A film that reaches a lower corner (5 or 4) ends on the dielectric.
    """
    ks = list(corners)
    outer = [corner(k, t) for k in ks[1:-1]]
    first = heel(+1, t, Z0) if ks[0] % 6 == 5 else corner(ks[0], t)
    last = heel(-1, t, Z0) if ks[-1] % 6 == 4 else corner(ks[-1], t)
    inner = [corner(k % 6) for k in reversed(ks)]
    return np.array([inner[-1], first, *outer, last, *inner[:-1]])


def pad(side: float, t: float, y_far: float) -> np.ndarray:
    """The film on the dielectric beside the wire, from the shell's foot out to ``y_far``."""
    foot, top = heel(side, t, Z0), heel(side, t, Z0 + t)
    pts = [foot, (y_far, Z0), (y_far, Z0 + t), top]
    return np.array(pts if side > 0 else pts[::-1])


def extrude_x(
    section: np.ndarray, x0: float, x1: float, drop: tuple[int, ...] = ()
) -> list[vecview.Face]:
    """A (y, z) cross-section extruded from x0 to x1; ``drop`` omits walls by edge."""
    ring = np.column_stack([np.full(len(section), x0), section])
    dropped = {f"side-{i}" for i in drop}
    return [f for f in vecview.extrude(ring, (x1 - x0, 0.0, 0.0)) if f.name not in dropped]


HEXAGON = np.array([corner(k) for k in range(6)])
AL_SHELL = shell(range(5, 9), T_AL)  # facets 5, 0, 1: corners 5 to 2
N_SHELL = shell(range(5, 11), T_N)  # all but the bottom: corners 5 to 4
# Walls that lie against the wire, the dielectric, or the shell are never seen.
AL_HIDDEN = (0, 5, 6, 7)
N_HIDDEN = (0, 6, 7, 8, 9, 10, 11)
PAD_HIDDEN = {+1.0: (0, 3), -1.0: (2, 3)}


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


def device(scene: Scene) -> None:
    """The wire, the Al film, and the two contacts draped over it, sorted together.

    Each film is convex pieces -- a shell on the wire and a pad on either side
    -- so that each is keyed near what it covers.  The wire is cut wherever a
    film begins or ends, so the piece beside a film sorts in front of the
    film's end wall.  Facets a film covers are left out rather than drawn and
    painted over: keyed inside the film, they would sort in front of its end
    wall.  That leaves nothing of the wire under a contact, and only the wire's
    two ends keep their caps.
    """
    scene.sort_by_depth(20)
    wire = dict(fill=COLORS["wire"], stroke=COLORS["wire_edge"], class_="semiconductor", **EDGE)
    cuts = [-WIRE_X, -X_END, -XC, -HYBRID, HYBRID, XC, X_END, WIRE_X]
    covered = {"hybrid": (5, 0, 1)}  # by facet; the pieces under the contacts are skipped
    names = ["end-L", None, "dot-L", "hybrid", "dot-R", None, "end-R"]
    for name, (x0, x1) in zip(names, itertools.pairwise(cuts), strict=True):
        if name is None:
            continue
        hidden = {f"side-{k}" for k in covered.get(name, ())}
        hidden |= {"start", "end"} - (
            {"start"} if x0 == -WIRE_X else {"end"} if x1 == WIRE_X else set()
        )
        piece = [f for f in extrude_x(HEXAGON, x0, x1) if f.name not in hidden]
        scene.faces(20, piece, cull=True, id=f"nanowire-{name}", **wire)

    al = dict(fill=COLORS["al"], stroke=COLORS["al_edge"], class_="superconductor", **EDGE)
    scene.faces(20, extrude_x(AL_SHELL, -HYBRID, HYBRID, AL_HIDDEN), cull=True, id="al-shell", **al)
    lead = extrude_x(pad(+1.0, T_AL, SUB_BACK), -HYBRID, HYBRID, PAD_HIDDEN[+1.0])
    scene.faces(20, lead, cull=True, id="al-lead", **al)

    gold = dict(fill=COLORS["gold"], stroke=COLORS["gold_edge"], class_="lead", **EDGE)
    for sign, name in ((-1.0, "L"), (1.0, "R")):
        x0, x1 = sorted((sign * XC, sign * X_END))
        scene.faces(
            20, extrude_x(N_SHELL, x0, x1, N_HIDDEN), cull=True, id=f"contact-{name}-drape", **gold
        )
        for side, part, y_far in ((-1.0, "front", CONTACT_FRONT), (1.0, "back", SUB_BACK)):
            pads = extrude_x(pad(side, T_N, y_far), x0, x1, PAD_HIDDEN[side])
            scene.faces(20, pads, cull=True, id=f"contact-{name}-{part}", **gold)


def annotations(scene: Scene) -> None:
    """Majorana modes on the dots, the field, and the gate labels."""
    top = Z0 + H_WIRE + 0.004
    size = 20.0
    math = dict(font_family="DejaVu Serif", font_style="italic", size=size, text_anchor="middle")
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
            sub("\N{GREEK SMALL LETTER GAMMA}", str(k), size),
            dy=-21,
            fill=COLORS["majorana"],
            id=f"label-gamma-{k}",
            class_="label",
            **math,
        )
    for gate in ("LD", "PG", "RD"):
        x = GATES[gate][0]
        scene.text(
            40,
            (x, GATE_FRONT, Z0),
            sub("V", gate, size),
            dy=28,
            fill=INK,
            id=f"label-{gate}",
            class_="label",
            **math,
        )

    # The roles of the three leads, N-S-N.
    roles = {
        "N-L": (-(XC + X_END) / 2, SUB_BACK - 0.6, Z0 + T_N),
        "S": (0.0, SUB_BACK - 0.6, Z0 + T_AL),
        "N-R": ((XC + X_END) / 2, SUB_BACK - 0.6, Z0 + T_N),
    }
    for key, at in roles.items():
        scene.text(
            40,
            at,
            key[0],
            dy=7,
            fill=INK,
            id=f"label-{key}",
            class_="label",
            **{**math, "font_style": "normal", "font_family": "DejaVu Sans"},
        )

    tail = np.array([-0.7, 1.5, Z0 + 1.7])
    scene.arrow3d(
        30,
        tail,
        X,
        1.4,
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
        tail + 1.4 * X,
        "B",
        dx=12,
        dy=7,
        fill=COLORS["field"],
        id="label-B",
        class_="label",
        **math,
    )


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=-60.0, elev_deg=30.0, scale=96.0)
    scene = Scene(cam, pad=14.0, background="#ffffff")
    platform(scene)
    device(scene)
    annotations(scene)
    return scene


if __name__ == "__main__":
    export(build(), NAME)
