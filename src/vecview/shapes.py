"""World-space glyph geometry.

Functions return world-space point arrays, Regions with holes, or lists of
:class:`Face` for solids.  Nothing here knows about a camera, a style,
or SVG.  That split is deliberate: geometry stays testable as numbers, and the
same polygon can be drawn twice with different fills, or reused as a clip path.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations, pairwise
from typing import Literal

import numpy as np

from vecview._cut import cut
from vecview._face import Face
from vecview._planar import _check_holes, _self_intersects
from vecview._types import Array, Point3, Points3, Region, Shape2, Shape3
from vecview._vec import as_points, as_rings, basis_for, unit

FaceName = Literal["-x", "+x", "-y", "+y", "-z", "+z"]

Pivot = Literal["tail", "mid"]


def in_plane_dir(
    angle_deg: float, u: Point3 = (1.0, 0.0, 0.0), v: Point3 = (0.0, 1.0, 0.0)
) -> Array:
    """Unit direction at ``angle_deg`` from ``u``, rotating toward ``v``.

    The common case -- an angle measured in the ``xy`` plane from ``+x``, as
    crystal and polarization angles usually are -- needs no axes passed.
    """
    a = np.radians(float(angle_deg))
    return unit(np.cos(a) * unit(u) + np.sin(a) * unit(v))


def rect_shape(center: Point3, u: Point3, v: Point3, du: float, dv: float) -> Array:
    """Rectangle centred on ``center``, in the plane spanned by ``u`` and ``v``."""
    c, uh, vh = np.asarray(center, dtype=np.float64), unit(u), unit(v)
    return np.array(
        [
            c - uh * du / 2 - vh * dv / 2,
            c + uh * du / 2 - vh * dv / 2,
            c + uh * du / 2 + vh * dv / 2,
            c - uh * du / 2 + vh * dv / 2,
        ]
    )


def box_faces(center: Point3, size: Point3) -> list[Face]:
    """The six faces of an axis-aligned box, each wound CCW about its outward normal.

    Draw it with ``cull=True`` to show only the walls the rendering camera
    can see::

        slab = box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
        scene.faces(10, slab, cull=True, fill="#cfd6e0")

    Args:
        center: Centre of the box.
        size: Full extents ``(sx, sy, sz)``, not half-extents.
    """
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(size, dtype=np.float64) / 2.0
    if c.shape != (3,) or h.shape != (3,):
        raise ValueError("center and size must each be three numbers")

    axes = np.eye(3)
    faces: list[Face] = []
    for axis, sign in ((a, s) for a in range(3) for s in (+1, -1)):
        normal = axes[axis] * sign
        # In-plane axes chosen so (u, v, normal) is right-handed, which winds
        # the points counter-clockwise as seen from outside the box.
        u = axes[(axis + 1) % 3] * sign
        v = axes[(axis + 2) % 3]
        du, dv = 2 * h[(axis + 1) % 3], 2 * h[(axis + 2) % 3]
        name = f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"
        faces.append(Face(name, rect_shape(c + normal * h[axis], u, v, du, dv), normal))
    return faces


def prism_faces(footprint: Shape2, z0: float, z1: float) -> list[Face]:
    """The faces of a footprint extruded along ``z``, each wound CCW about its normal.

    The generalization of :func:`box_faces` to any simple cross-section -- a
    tapered electrode, an arc-shaped gate, a ring.  The cap is named ``"+z"``
    and the base ``"-z"``, as for a box; wall ``i`` spans footprint vertices
    ``i`` and ``i + 1`` and is named ``"side-{i}"``.

    A footprint with holes -- ``Region(outline, holes)`` -- has caps with
    holes, and the walls of hole ``k`` are named ``"hole{k}_side-{i}"``.
    Hole walls come first, then cap, base and outer walls, so ``faces(...,
    cull=True)`` paints the far side of a hole under the cap.  That is exact for
    one convex hole -- a ring, a washer -- and for holes too shallow to show
    through each other; a non-convex hole can hide its own walls, as a
    non-convex outline can.  Draw those in a sorted layer.  Pick
    the cap of a holed solid by name.

    Args:
        footprint: Simple polygon in the ``xy`` plane, shape ``(n, 2)``, in
            either winding; it may be non-convex. Collinear vertices are
            allowed; repeated ones are not. Or a Region whose holes lie inside
            its exterior, apart from it and each other.
        z0: Height of the base.
        z1: Height of the cap; must exceed ``z0``.

    Raises:
        ValueError: If the footprint is not a simple polygon of at least three
            distinct vertices -- self-intersecting or self-touching outlines
            have no well-defined outside -- if a hole is not inside it, apart
            from it and from the other holes, or if ``z1 <= z0``.

    Back-face culling selects the walls that face the camera, which for a
    convex footprint is exactly the visible set.  For a non-convex one, a
    facing wall can still be hidden behind another wall of the same solid;
    :meth:`~vecview.scene.Scene.prism_walls` draws such a solid correctly.
    """
    foot, *holes = as_rings(footprint, 2)
    if foot.ndim != 2 or foot.shape[1] != 2 or len(foot) < 3:
        raise ValueError(f"footprint must have shape (n, 2) with n >= 3, got {foot.shape}")
    if not z1 > z0:
        raise ValueError(f"z1 must exceed z0, got z0={z0}, z1={z1}")
    x, y = foot[:, 0], foot[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    edges = np.roll(foot, -1, axis=0) - foot
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-12 * float(lengths.max())):
        raise ValueError("footprint has repeated consecutive vertices")
    if abs(area) <= 1e-12 * float(lengths.max()) ** 2:
        raise ValueError("footprint encloses no area")
    if _self_intersects(foot):
        raise ValueError("footprint must be a simple polygon: its edges cross or touch")
    if holes:
        _check_holes([foot, *holes])
    if area < 0:
        foot = foot[::-1]  # clockwise: reverse so the cap faces +z
    up = np.array([0.0, 0.0, float(z1) - float(z0)])
    base, *bores = [np.column_stack([r, np.full(len(r), float(z0))]) for r in (foot, *holes)]
    faces = _swept(base, bores, up)  # checked above, as extrude would check it again
    k = len(faces) - len(foot) - 2  # the hole walls, which come first
    start, end = faces[k], faces[k + 1]
    return [*faces[:k], end._replace(name="+z"), start._replace(name="-z"), *faces[k + 2 :]]


def _walls(ring: Array, sweep: Array, swap: bool, prefix: str) -> list[Face]:
    """The walls a ring sweeps, wall ``i`` on the edge from vertex ``i`` to ``i + 1``."""
    walls: list[Face] = []
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        if swap:
            a, b = b, a
        # The wall's plane holds the edge and the sweep, and (edge, along,
        # outward) is right-handed for an edge with the solid on its left.
        outward = unit(np.cross(b - a, sweep))
        walls.append(Face(f"{prefix}{i}", np.array([a, b, b + sweep, a + sweep]), outward))
    return walls


def extrude(section: Shape3, along: Point3) -> list[Face]:
    """A planar cross-section swept along a vector: a prism in any direction.

    The generalization of :func:`prism_faces` to a section in any plane and a
    sweep in any direction -- a nanowire along ``x``, a waveguide, a fin.  The
    section is the ``"start"`` face, the section moved by ``along`` is the
    ``"end"`` face, and wall ``i`` spans section vertices ``i`` and ``i + 1``
    and is named ``"side-{i}"`` -- in the order given, whichever way the
    section winds, so a wall can be picked out by the edge it was built on.
    Every face is wound counter-clockwise about its outward normal, so it
    culls like a box.  ``along`` need not be perpendicular to the section; an
    oblique sweep gives a slanted prism.

    A regular cross-section comes from :func:`circle_shape` with a small
    ``n``: ``circle_shape((0, 0, 0), r, (1, 0, 0), n=6)`` swept along ``x``
    is a hexagonal wire.

    A section can have holes: ``extrude(Region(outline, [hole]), along)`` is a tube, a
    ring, a plate with vias.  The ends are then faces with holes, and each
    hole sweeps walls of its own, wall ``i`` of hole ``k`` named
    ``"hole{k}_side-{i}"`` and facing into the hole.  Hole walls come first,
    then ``"start"``, ``"end"`` and the outer walls, so a culled tube paints
    its bore under its ends, as :func:`prism_faces` explains.

    Args:
        section: Simple polygon lying in one plane, shape ``(n, 3)``, in
            either winding; it may be non-convex. Repeated consecutive
            vertices are not allowed. Or a Region with holes in the same plane,
            inside the exterior and apart, in any winding.
        along: The sweep, as a world vector; its length is the prism's length.

    Raises:
        ValueError: If the section has fewer than three vertices, is not
            planar, encloses no area, crosses or touches itself, has a hole
            that is not inside it and apart, or if ``along`` lies in its plane.
    """
    pts, *holes = (as_points(ring) for ring in as_rings(section, 3))
    if len(pts) < 3:
        raise ValueError(f"a section needs at least three vertices, got {len(pts)}")
    sweep = np.asarray(along, dtype=np.float64)
    lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    size = float(lengths.max())
    if np.any(lengths <= 1e-12 * size):
        raise ValueError("section has repeated consecutive vertices")
    # Newell's normal: twice the vector area, robust for non-convex polygons.
    vector_area = 0.5 * np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0)
    area = float(np.linalg.norm(vector_area))
    if area <= 1e-12 * size * size:
        raise ValueError("section encloses no area")
    normal = vector_area / area
    if any(np.abs((ring - pts[0]) @ normal).max() > 1e-9 * size for ring in (pts, *holes)):
        raise ValueError("section is not planar")
    u, v = basis_for(normal)
    flat = [np.column_stack([(ring - pts[0]) @ u, (ring - pts[0]) @ v]) for ring in (pts, *holes)]
    if _self_intersects(flat[0]):
        raise ValueError("section must be a simple polygon: its edges cross or touch")
    if holes:
        _check_holes(flat)
    reach = float(np.dot(sweep, normal))
    if abs(reach) <= 1e-12 * max(size, float(np.linalg.norm(sweep))):
        raise ValueError("along lies in the section's plane, so the sweep has no volume")
    return _swept(pts, holes, sweep)


def _swept(pts: Array, holes: list[Array], sweep: Array) -> list[Face]:
    """The faces of a checked section swept along ``sweep``: what :func:`extrude` returns."""
    vector_area = 0.5 * np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0)
    normal = vector_area / float(np.linalg.norm(vector_area))
    reach = float(np.dot(sweep, normal))
    # Wind every face counter-clockwise about the sweep's side of the plane,
    # while wall i keeps the edge from vertex i to i + 1 as the caller gave it.
    flip = reach < 0
    if flip:
        normal = -normal
    ccw = pts[::-1] if flip else pts
    # Holes wind the other way, so the solid lies to the left of every edge.
    winding = [float(np.cross(h, np.roll(h, -1, axis=0)).sum(axis=0) @ normal) for h in holes]
    cw = [h[::-1] if w > 0 else h for h, w in zip(holes, winding, strict=True)]
    walls = [
        wall
        for k, (hole, w) in enumerate(zip(holes, winding, strict=True))
        for wall in _walls(hole, sweep, w > 0, f"hole{k}_side-")
    ]
    start = Face("start", ccw[::-1].copy(), -normal, tuple(h[::-1].copy() for h in cw))
    end = Face("end", ccw + sweep, normal.copy(), tuple(h + sweep for h in cw))
    return [*walls, start, end, *_walls(pts, sweep, flip, "side-")]


def revolve(
    profile: Shape2,
    *,
    sweep_deg: tuple[float, float] = (0.0, 360.0),
    axis: Point3 = (0.0, 0.0, 1.0),
    origin: Point3 = (0.0, 0.0, 0.0),
    n: int = 96,
) -> list[Face]:
    """A profile turned about an axis: the solid a lathe makes.

    ``profile`` is drawn in a half-plane through the axis, as ``(r, z)``
    pairs: ``r >= 0`` is the distance from the axis and ``z`` the height along
    it from ``origin``.  A rectangle away from the axis turns into a thick
    tube, one against it into a cylinder, a profile with a hole into a solid
    with a hollow ring inside.  Vertices on the axis close the solid there.

    The turn runs over ``sweep_deg``, counter-clockwise about ``axis`` by the
    right-hand rule, from the direction of ``+x`` -- of ``+y`` for an axis
    along ``x``.  For the default axis that is from ``+x`` toward ``+y``, as
    :func:`annulus_sector` measures.  A full turn closes on itself; a shorter
    one leaves the solid open along two half-planes through the axis, capped
    by the profile as faces ``"start"`` and ``"end"`` -- the section of a
    cutaway, to draw in a style of its own.

    The walls are flat facets.  Facet ``j`` of profile edge ``i``, the edge
    from vertex ``i`` to ``i + 1`` as given, is named ``"side-{i}-{j}"``; of
    an edge of hole ``k``, ``"hole{k}_side-{i}-{j}"``.  A full turn has ``n``
    per edge and a part of one proportionally fewer, at least one.  A facet
    beside the axis is a triangle, and an edge along the axis has none.  Hole
    walls come first, then ``"start"``, ``"end"`` and the outer walls, as
    :func:`extrude` orders them.  Every face is wound counter-clockwise about
    its outward normal.  :meth:`~vecview.scene.Scene.line_art` draws the
    outline of the result without the seams between facets.

    Args:
        profile: Simple polygon of ``(r, z)`` pairs, shape ``(m, 2)``, in
            either winding, with every ``r >= 0``; or a Region with holes.
        sweep_deg: Start and end of the turn in degrees; the end lies in
            ``(start, start + 360]``.
        axis: Direction of the axis of revolution.
        origin: The point on the axis where ``z = 0``.
        n: Facets per full turn, at least three.

    Raises:
        ValueError: If the profile is not a simple polygon enclosing area, a
            hole is not inside it and apart, a vertex has ``r < 0``, the sweep
            is empty or longer than a turn, the axis is zero, or ``n < 3``.
    """
    rings = as_rings(profile, 2)
    if len(rings[0]) < 3:
        raise ValueError(f"a profile needs at least three vertices, got {len(rings[0])}")
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    a = unit(axis)
    if not a.any():
        raise ValueError("revolve needs a non-zero axis")
    start_deg, end_deg = (float(v) for v in sweep_deg)
    span = end_deg - start_deg
    if not 0.0 < span <= 360.0 + 1e-9:
        raise ValueError(f"the sweep must turn through (0, 360] degrees, got {span:g}")
    size = float(np.ptp(np.vstack(rings), axis=0).max())
    if any(np.any(ring[:, 0] < -1e-12 * size) for ring in rings):
        raise ValueError("the profile must lie at r >= 0, on one side of the axis")
    # Vertices on the axis are put exactly on it, so every facet there meets in one point.
    rings = [
        np.column_stack([np.where(r[:, 0] <= 1e-12 * size, 0.0, r[:, 0]), r[:, 1]]) for r in rings
    ]
    for ring in rings:
        lengths = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        if np.any(lengths <= 1e-12 * size):
            raise ValueError("profile has repeated consecutive vertices")
    if abs(_signed_area(rings[0])) <= 1e-12 * size * size:
        raise ValueError("profile encloses no area")
    if _self_intersects(rings[0]):
        raise ValueError("profile must be a simple polygon: its edges cross or touch")
    if len(rings) > 1:
        _check_holes(rings)

    full = span >= 360.0 - 1e-9
    steps = n if full else max(1, int(np.ceil(n * span / 360.0 - 1e-9)))
    theta = np.radians(start_deg + (360.0 if full else span) * np.arange(steps + 1) / steps)
    ref = np.array([0.0, 1.0, 0.0] if abs(float(a[0])) >= 0.9 else [1.0, 0.0, 0.0])
    e1 = unit(ref - float(ref @ a) * a)
    e2 = np.cross(a, e1)
    radial = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    o = np.asarray(origin, dtype=np.float64)

    def turned(ring: Array) -> Array:
        """The ring at every angle: shape ``(steps + 1, m, 3)``."""
        return o + ring[None, :, :1] * radial[:, None, :] + ring[None, :, 1:] * a

    hole_walls: list[Face] = []
    outer_walls: list[Face] = []
    oriented: list[Array] = []
    for k, ring in enumerate(rings):
        # The solid lies left of every edge, walking the exterior
        # counter-clockwise and each hole clockwise in the (r, z) plane.
        flip = (_signed_area(ring) > 0) != (k == 0)
        oriented.append(ring[::-1] if flip else ring)
        at = turned(ring)
        prefix = "side-" if k == 0 else f"hole{k - 1}_side-"
        for i in range(len(ring)):
            lo, hi = i, (i + 1) % len(ring)
            if flip:
                lo, hi = hi, lo
            on_lo, on_hi = ring[lo, 0] == 0.0, ring[hi, 0] == 0.0
            if on_lo and on_hi:
                continue  # along the axis: no wall
            for j in range(steps):
                ahead = (j + 1) % steps if full else j + 1
                facet = [at[j, lo]]
                facet += [] if on_lo else [at[ahead, lo]]
                facet += [] if on_hi else [at[ahead, hi]]
                facet.append(at[j, hi])
                pts = np.array(facet)
                normal = unit(np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0))
                (outer_walls if k == 0 else hole_walls).append(
                    Face(f"{prefix}{i}-{j}", pts, normal)
                )
    if full:
        return [*hole_walls, *outer_walls]

    # The profile, counter-clockwise, faces back along the turn at its start.
    def cap(name: str, j: int, sign: float) -> Face:
        rims = [turned(ring)[j] for ring in oriented]
        if sign > 0:
            rims = [rim[::-1] for rim in rims]
        tangent = sign * (np.cos(theta[j]) * e2 - np.sin(theta[j]) * e1)
        return Face(name, rims[0], tangent, tuple(rims[1:]))

    return [*hole_walls, cap("start", 0, -1.0), cap("end", steps, 1.0), *outer_walls]


def _signed_area(ring: Array) -> float:
    """Signed area of a 2D ring, positive when it winds counter-clockwise."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def annulus_sector(
    center: tuple[float, float] | Array,
    r_in: float,
    r_out: float,
    theta0_deg: float,
    theta1_deg: float,
    n: int = 32,
) -> Array:
    """Footprint of an annular sector, shape ``(m, 2)``, counter-clockwise.

    The outer arc runs from ``theta0_deg`` to ``theta1_deg`` in ``n`` segments,
    then the inner arc runs back.  With ``r_in = 0`` the inner arc collapses to
    the centre and the result is a pie wedge.  Angles are measured from ``+x``
    toward ``+y``.  Pair with :func:`prism_faces` for an arc-shaped electrode.
    A whole ring has a hole, so it is :func:`annulus`, not a sector.

    Raises:
        ValueError: Unless ``0 <= r_in < r_out``, ``n >= 1``, and the span
            ``theta1_deg - theta0_deg`` lies in ``(0, 360)``.
    """
    if not 0.0 <= r_in < r_out:
        raise ValueError(f"need 0 <= r_in < r_out, got r_in={r_in}, r_out={r_out}")
    span = theta1_deg - theta0_deg
    if span >= 360.0:
        raise ValueError(f"a span of {span} degrees closes the ring: use annulus() for a ring")
    if not span > 0.0:
        raise ValueError(f"theta1_deg - theta0_deg must lie in (0, 360), got {span}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    c = np.asarray(center, dtype=np.float64)
    t = np.radians(np.linspace(theta0_deg, theta1_deg, n + 1))
    arc = np.column_stack([np.cos(t), np.sin(t)])
    inner = c[None, :] if r_in == 0.0 else c + r_in * arc[::-1]
    footprint: Array = np.vstack([c + r_out * arc, inner])
    return footprint


def annulus(
    center: tuple[float, float] | Array,
    r_in: float,
    r_out: float,
    n: int = 32,
    *,
    theta0_deg: float = 0.0,
) -> Region:
    """Footprint of a ring: ``Region(outer, [inner])``, each circle of ``n`` vertices.

    The outer circle runs counter-clockwise and the inner one, the hole,
    clockwise, both from ``theta0_deg`` (measured from ``+x`` toward ``+y``).
    Pair with :func:`prism_faces` for a washer, a microring, or a loop; a full
    disk is :func:`~vecview.outlines.regular`.

    Raises:
        ValueError: Unless ``0 < r_in < r_out`` and ``n >= 3``.
    """
    if not 0.0 < r_in < r_out:
        raise ValueError(f"need 0 < r_in < r_out, got r_in={r_in}, r_out={r_out}")
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    c = np.asarray(center, dtype=np.float64)
    t = np.radians(np.arange(n) * (360.0 / n) + theta0_deg)  # as linspace spaces them
    circle = np.column_stack([np.cos(t), np.sin(t)])
    return Region(c + r_out * circle, (c + r_in * np.roll(circle[::-1], 1, axis=0),))


def arrow_shape(
    origin: Point3,
    direction: Point3,
    length: float,
    normal: Point3,
    shaft_w: float,
    head_w: float,
    head_len: float,
    pivot: Pivot = "tail",
) -> Array:
    """Flat arrow polygon in the plane through ``origin`` with the given normal.

    Args:
        origin: Tail of the arrow, or its midpoint when ``pivot="mid"``.
        direction: Direction the head points.
        length: Total length, tail to tip.
        normal: Normal of the plane the arrow is drawn flat in.
        shaft_w: Width of the shaft.
        head_w: Width of the head at its widest.
        head_len: Length of the head, measured back from the tip.
        pivot: Whether ``origin`` is the ``"tail"`` or the ``"mid"`` point.
    """
    d, n = unit(direction), unit(normal)
    s = unit(np.cross(n, d))
    base = np.asarray(origin, dtype=np.float64) - (d * length / 2.0 if pivot == "mid" else 0.0)
    tip = base + d * length
    neck = base + d * max(length - head_len, 0.0)
    return np.array(
        [
            base + s * shaft_w / 2,
            neck + s * shaft_w / 2,
            neck + s * head_w / 2,
            tip,
            neck - s * head_w / 2,
            neck - s * shaft_w / 2,
            base - s * shaft_w / 2,
        ]
    )


def path_arrow_shape(
    path: Points3,
    normal: Point3,
    shaft_w: float,
    head_w: float,
    head_len: float,
) -> Array:
    """Flat arrow following ``path`` in the plane with the given normal.

    The shaft is the path widened to ``shaft_w`` in that plane, mitred at each
    vertex; the head takes the last ``head_len`` of the path's length and points
    from there to its end.  A two-point path gives exactly :func:`arrow_shape`.
    Pass a finely sampled curve -- an arc, a Bézier, a route around an
    obstacle -- for a smooth arrow.

    Args:
        path: Points from tail to tip, all in one plane perpendicular to ``normal``.
        normal: Normal of that plane.
        shaft_w: Width of the shaft.
        head_w: Width of the head at its widest.
        head_len: Length of the head, measured back along the path from the tip.

    Raises:
        ValueError: If the path has fewer than two distinct points, leaves the
            plane, or bends tighter than half the shaft's width, where the
            widened shaft would fold over itself.
    """
    pts = as_points(path)
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    if len(pts) < 2:
        raise ValueError("path_arrow_shape needs at least two distinct points")
    n = unit(normal)
    size = float(np.ptp(pts, axis=0).max())
    if np.abs((pts - pts[0]) @ n).max() > 1e-9 * max(size, 1.0):
        raise ValueError("path_arrow_shape: the path must lie in a plane perpendicular to normal")

    # Cut the path where the head begins, head_len back from the tip.
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    run = np.r_[0.0, np.cumsum(steps)]
    at = run[-1] - head_len
    tip = pts[-1]
    if at <= 0.0:  # all head, as arrow_shape clamps it
        neck, shaft = pts[0], pts[[0, 0]]
    else:
        k = min(int(np.searchsorted(run, at, side="right")), len(pts) - 1)
        neck = pts[k - 1] + (pts[k] - pts[k - 1]) * (at - run[k - 1]) / steps[k - 1]
        # A neck in the first half of an edge takes the place of the vertex
        # before it, so the shaft never ends on a sliver of an edge, whose
        # mitred outline would fold over.
        drop = k > 1 and at - run[k - 1] < steps[k - 1] / 2
        shaft = np.vstack([pts[: k - 1 if drop else k], neck])
    s = unit(np.cross(n, tip - neck))  # across the head
    # The shaft ends square to the path, not to the head's chord, which on a
    # curve turns away from it.
    onward = unit(np.cross(n, pts[k] - neck)) if at > 0.0 and np.any(pts[k] != neck) else s

    # Each vertex is offset to where the offsets of its two edges meet: a miter.
    across = [unit(np.cross(n, b - a)) if np.any(b != a) else s for a, b in pairwise(shaft)]
    before, after = [across[0], *across], [*across, onward]
    offsets = []
    for u, v in zip(before, after, strict=True):
        cos = float(u @ v)
        if cos < -0.9:
            raise ValueError("path_arrow_shape: the path turns back on itself")
        offsets.append((u + v) / (1.0 + cos) * shaft_w / 2)
    left, right = shaft + offsets, shaft - offsets
    edges = np.diff(shaft, axis=0)
    for side in (left, right):
        if np.any(np.einsum("ij,ij->i", np.diff(side, axis=0), edges) < 0):
            raise ValueError("path_arrow_shape: the path bends tighter than half the shaft width")
    return np.vstack([left, neck + s * head_w / 2, tip, neck - s * head_w / 2, right[::-1]])


def double_arrow_shape(
    center: Point3,
    direction: Point3,
    length: float,
    normal: Point3,
    shaft_w: float,
    head_w: float,
    head_len: float,
) -> Array:
    """Double-headed arrow centred on ``center`` -- an axis or polarization marker."""
    d, n = unit(direction), unit(normal)
    s = unit(np.cross(n, d))
    c = np.asarray(center, dtype=np.float64)
    # A head longer than half the arrow is clamped, as in arrow_shape, rather
    # than letting the two necks cross and the polygon self-intersect.
    h, neck = length / 2.0, max(length / 2.0 - head_len, 0.0)
    return np.array(
        [
            c + d * h,
            c + d * neck + s * head_w / 2,
            c + d * neck + s * shaft_w / 2,
            c - d * neck + s * shaft_w / 2,
            c - d * neck + s * head_w / 2,
            c - d * h,
            c - d * neck - s * head_w / 2,
            c - d * neck - s * shaft_w / 2,
            c + d * neck - s * shaft_w / 2,
            c + d * neck - s * head_w / 2,
        ]
    )


def circle_shape(center: Point3, radius: float, normal: Point3, n: int = 64) -> Array:
    """Circle as an ``n``-gon, in the plane through ``center`` with the given normal."""
    a, b = basis_for(normal)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring: Array = np.asarray(center, dtype=np.float64) + radius * (
        np.outer(np.cos(t), a) + np.outer(np.sin(t), b)
    )
    return ring


def ellipse_shape(center: Point3, u: Point3, v: Point3, a: float, b: float, n: int = 64) -> Array:
    """Ellipse as an ``n``-gon, with semi-axis ``a`` along ``u`` and ``b`` along ``v``.

    The generalization of :func:`circle_shape`. The first vertex lies at
    ``center + a * u`` and the points run counter-clockwise about ``u x v``,
    with no duplicated closing point.

    Raises:
        ValueError: If ``u`` and ``v`` are not perpendicular, since the
            semi-axes would then not be ``a`` and ``b``.
    """
    uh, vh = unit(u), unit(v)
    if abs(float(np.dot(uh, vh))) > 1e-9:
        raise ValueError("u and v must be perpendicular: they are the ellipse's axes")
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring: Array = np.asarray(center, dtype=np.float64) + (
        np.outer(a * np.cos(t), uh) + np.outer(b * np.sin(t), vh)
    )
    return ring


def arc_shape(
    center: Point3,
    u: Point3,
    v: Point3,
    radius: float,
    theta0_deg: float,
    theta1_deg: float,
    n: int = 32,
) -> Array:
    """Circular arc of ``n`` segments in the plane of ``u`` and ``v``.

    Angles are measured from ``u`` toward ``v``; ``v`` need not be perpendicular
    to ``u``, only not parallel, so the arc marking the angle between two
    vectors is ``arc_shape(origin, a, b, r, 0, angle_between)``.

    Raises:
        ValueError: If ``u`` and ``v`` are parallel, which leaves no plane.
    """
    uh = unit(u)
    w = np.asarray(v, dtype=np.float64) - float(np.dot(v, uh)) * uh
    if float(np.linalg.norm(w)) < 1e-12 * max(float(np.linalg.norm(v)), 1.0):
        raise ValueError("u and v are parallel, so they span no plane for the arc")
    vh = unit(w)
    t = np.radians(np.linspace(theta0_deg, theta1_deg, n + 1))
    arc: Array = np.asarray(center, dtype=np.float64) + radius * (
        np.outer(np.cos(t), uh) + np.outer(np.sin(t), vh)
    )
    return arc


def helix(
    start: Point3,
    axis: Point3,
    radius: float,
    pitch: float,
    turns: float,
    n_per_turn: int = 48,
    phase_deg: float = 0.0,
) -> Array:
    """Helix winding about ``axis`` from ``start``, for a coil or a spin spiral.

    ``start`` is on the axis.  The curve begins ``radius`` from it at angle
    ``phase_deg`` (measured in the frame of :func:`circle_shape`), turns
    counter-clockwise about ``axis`` and advances ``pitch`` along it per turn,
    so a positive ``pitch`` is right-handed and a negative one left-handed.

    Raises:
        ValueError: Unless ``turns`` and ``n_per_turn`` are positive.
    """
    if turns <= 0 or n_per_turn < 1:
        raise ValueError(f"need turns > 0 and n_per_turn >= 1, got {turns}, {n_per_turn}")
    a = unit(axis)
    e1, e2 = basis_for(a)
    count = max(round(turns * n_per_turn), 1)
    s = np.linspace(0.0, turns, count + 1)
    t = 2.0 * np.pi * s + np.radians(phase_deg)
    curve: Array = (
        np.asarray(start, dtype=np.float64)
        + radius * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
        + np.outer(s * pitch, a)
    )
    return curve


def surface_faces(x: Array, y: Array, z: Array) -> list[Face]:
    """Quads of a parametric surface sampled on a grid, for band surfaces and cones.

    ``x``, ``y`` and ``z`` are 2D arrays of one shape ``(m, n)`` -- the output
    of :func:`numpy.meshgrid` for a height field ``z = f(x, y)``, or any
    parametrization ``(x(s, t), y(s, t), z(s, t))``.  Quad ``(i, j)`` spans
    samples ``i..i+1`` and ``j..j+1``, is named ``"q-{i}-{j}"``, and is wound
    counter-clockwise about its normal, which points along ``d/di x d/dj``.  For
    a height field from ``np.meshgrid(xs, ys, indexing="ij")`` that is upward;
    the default ``indexing="xy"`` swaps the axes and points it downward.

    A surface is two-sided, so do not cull it; draw the quads in a layer passed
    to :meth:`~vecview.scene.Scene.sort_by_depth`, and use the normal to pick a
    front or back colour if the surface folds toward the camera.

    Raises:
        ValueError: If the arrays differ in shape or have fewer than two
            samples along either axis.
    """
    gx, gy, gz = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    if not gx.shape == gy.shape == gz.shape or gx.ndim != 2:
        raise ValueError(
            f"x, y and z must be 2D arrays of one shape, got {gx.shape}, {gy.shape}, {gz.shape}"
        )
    m, n = gx.shape
    if m < 2 or n < 2:
        raise ValueError(f"a surface needs at least 2 x 2 samples, got {m} x {n}")
    p = np.stack([gx, gy, gz], axis=-1)
    faces: list[Face] = []
    for i in range(m - 1):
        for j in range(n - 1):
            quad = np.array([p[i, j], p[i + 1, j], p[i + 1, j + 1], p[i, j + 1]])
            # Twice the cross product of the two partial derivatives, from the
            # diagonals, which stays defined when one edge collapses to a point.
            normal = unit(np.cross(quad[2] - quad[0], quad[3] - quad[1]))
            faces.append(Face(f"q-{i}-{j}", quad, normal))
    return faces


def convex_polyhedron(vertices: Points3, *, tol: float = 1e-9) -> list[Face]:
    """Faces of the convex hull of ``vertices``, each wound CCW about its outward normal.

    For a Brillouin zone, a coordination octahedron, or any small convex solid
    known by its corners.  Faces are named ``"face-{k}"`` in a deterministic
    order, and coplanar vertices merge into one polygonal face, so a truncated
    octahedron gives eight hexagons and six squares rather than triangles.
    Vertices strictly inside the hull are ignored.

    The search tests every vertex triple against every vertex, which is
    ``O(n^4)``: instant for the tens of vertices this is meant for, and the
    wrong tool for a mesh of thousands.

    Args:
        vertices: World points, shape ``(n, 3)``.
        tol: Coplanarity tolerance, relative to the size of the solid.

    Raises:
        ValueError: For fewer than four vertices, or vertices that are all
            coplanar, since neither encloses a volume.
    """
    pts = as_points(vertices)
    n = len(pts)
    if n < 4:
        raise ValueError(f"a polyhedron needs at least 4 vertices, got {n}")
    size = float(np.ptp(pts, axis=0).max()) or 1.0
    eps = tol * size

    i, j, k = (np.asarray(c) for c in zip(*combinations(range(n), 3), strict=True))
    found: dict[frozenset[int], Array] = {}
    chunk = 4096
    for lo in range(0, len(i), chunk):
        a, b, c = pts[i[lo : lo + chunk]], pts[j[lo : lo + chunk]], pts[k[lo : lo + chunk]]
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        ok = lengths > eps * size
        normals = normals[ok] / lengths[ok, None]
        side = pts @ normals.T - np.einsum("ij,ij->i", a[ok], normals)
        below, above = np.all(side <= eps, axis=0), np.all(side >= -eps, axis=0)
        for col in np.flatnonzero(below | above):
            normal = normals[col] if below[col] else -normals[col]
            on = frozenset(int(v) for v in np.flatnonzero(np.abs(side[:, col]) <= eps))
            found.setdefault(on, normal)
    if not found or all(len(on) == n for on in found):
        raise ValueError("vertices are coplanar, so they enclose no volume")

    faces: list[Face] = []
    for on, normal in found.items():
        idx = sorted(on)
        ring = pts[idx]
        centre = ring.mean(axis=0)
        e1, e2 = basis_for(normal)
        rel = ring - centre
        order = np.argsort(np.arctan2(rel @ e2, rel @ e1), kind="stable")
        faces.append(Face(f"face-{len(faces)}", ring[order], np.asarray(normal)))
    return faces


def trim_corners(faces: Iterable[Face], radius: float, n: int = 8) -> list[Face]:
    """Faces with a disk of ``radius`` cut out at every corner.

    For a polyhedron with an atom or marker on each vertex, when its faces
    must themselves stop at the atoms.  A layer sorted by depth does not need
    this -- it clips each face exactly where an atom hides it.  The ball meets
    each face plane in a disk about the vertex, so the corner is
    replaced by an inward arc of ``n`` segments, centred on the vertex, from
    one edge to the other.  Name and normal are kept.

    Pair with :meth:`~vecview.scene.Scene.edges` ``trim=radius`` for the edges.

    Raises:
        ValueError: If ``radius`` is not smaller than half of every edge, which
            would make neighbouring cuts overlap, or if a face has holes.
    """
    trimmed: list[Face] = []
    for face in faces:
        if face.holes:
            raise ValueError(f"face {face.name!r} has holes, whose corners trim_corners cannot cut")
        ring = np.asarray(face.points, dtype=np.float64)
        edges = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        if radius <= 0:
            trimmed.append(face)
            continue
        if radius >= 0.5 * float(edges.min()):
            raise ValueError(
                f"radius {radius:g} must be under half the shortest edge, "
                f"{0.5 * float(edges.min()):g}, of face {face.name!r}"
            )
        out: list[Array] = []
        for i, v in enumerate(ring):
            a = unit(ring[i - 1] - v)
            b = unit(ring[(i + 1) % len(ring)] - v)
            angle = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
            w = unit(b - np.dot(b, a) * a)
            t = np.linspace(0.0, angle, n + 1)
            out.extend(v + radius * (np.outer(np.cos(t), a) + np.outer(np.sin(t), w)))
        trimmed.append(Face(face.name, np.array(out), face.normal))
    return trimmed


def sine_ribbon(
    start: Point3,
    axis: Point3,
    length: float,
    transverse: Point3,
    amplitude: float | Array,
    wavelength: float,
    n: int = 400,
    phase: float = 0.0,
) -> Array:
    """Transverse wave along ``axis``, displaced along ``transverse``.

    ``amplitude`` may be a scalar or an array of length ``n``, so a component
    that decays inside a medium can be drawn by passing its envelope.
    """
    a, e = unit(axis), unit(transverse)
    t = np.linspace(0.0, length, n)
    disp = np.asarray(amplitude, dtype=np.float64) * np.sin(2.0 * np.pi * t / wavelength + phase)
    ribbon: Array = np.asarray(start, dtype=np.float64) + np.outer(t, a) + np.outer(disp, e)
    return ribbon


__all__ = [
    "Face",
    "FaceName",
    "annulus",
    "annulus_sector",
    "arc_shape",
    "arrow_shape",
    "box_faces",
    "circle_shape",
    "convex_polyhedron",
    "cut",
    "double_arrow_shape",
    "ellipse_shape",
    "extrude",
    "helix",
    "in_plane_dir",
    "path_arrow_shape",
    "prism_faces",
    "rect_shape",
    "revolve",
    "sine_ribbon",
    "surface_faces",
    "trim_corners",
]
