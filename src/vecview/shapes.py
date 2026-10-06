"""World-space glyph geometry.

Every function here returns plain world-space points -- an ``(n, 3)`` array, or
a list of :class:`Face` for solids.  Nothing here knows about a camera, a style,
or SVG.  That split is deliberate: geometry stays testable as numbers, and the
same polygon can be drawn twice with different fills, or reused as a clip path.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from vecview._types import Array, Point3, Points2, Points3
from vecview._vec import as_points, basis_for, unit

FaceName = Literal["-x", "+x", "-y", "+y", "-z", "+z"]

Pivot = Literal["tail", "mid"]


class Face(NamedTuple):
    """One planar polygon of a solid, with the outward normal it was built from.

    Carrying the normal alongside the points is what lets
    :meth:`~vecview.camera.Camera.visible` decide which walls of a box to draw
    without the caller reasoning about which octant the camera sits in.
    """

    name: str
    points: Array
    normal: Array


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

    Pair with :meth:`~vecview.camera.Camera.visible` to draw only the walls a
    given camera can see::

        slab = box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
        scene.faces(10, camera.visible(slab), fill="#cfd6e0")

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


def _self_intersects(poly: Array) -> bool:
    """Whether any two non-adjacent edges of a closed polygon cross or touch."""
    n = len(poly)
    a, b = poly, np.roll(poly, -1, axis=0)
    scale = float(np.ptp(poly, axis=0).max()) or 1.0
    eps = 1e-12 * scale * scale

    def orient(p: Array, q: Array, r: Array) -> Array:
        cross: Array = (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (
            q[..., 1] - p[..., 1]
        ) * (r[..., 0] - p[..., 0])
        return cross

    def within(p: Array, q: Array, r: Array) -> npt.NDArray[np.bool_]:
        """Whether r lies in the bounding box of segment pq (for collinear r)."""
        lo, hi = np.minimum(p, q), np.maximum(p, q)
        inside: npt.NDArray[np.bool_] = np.all(
            (r >= lo - 1e-12 * scale) & (r <= hi + 1e-12 * scale), axis=-1
        )
        return inside

    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))  # the closing edge is adjacent to the first
    i, j = i[keep], j[keep]
    ai, bi, aj, bj = a[i], b[i], a[j], b[j]
    d1, d2 = orient(ai, bi, aj), orient(ai, bi, bj)
    d3, d4 = orient(aj, bj, ai), orient(aj, bj, bi)
    proper = (d1 * d2 < -eps * eps) & (d3 * d4 < -eps * eps)
    touching = (
        ((np.abs(d1) <= eps) & within(ai, bi, aj))
        | ((np.abs(d2) <= eps) & within(ai, bi, bj))
        | ((np.abs(d3) <= eps) & within(aj, bj, ai))
        | ((np.abs(d4) <= eps) & within(aj, bj, bi))
    )
    return bool(np.any(proper | touching))


def prism_faces(footprint: Points2, z0: float, z1: float) -> list[Face]:
    """The faces of a footprint extruded along ``z``, each wound CCW about its normal.

    The generalization of :func:`box_faces` to any simple cross-section -- a
    tapered electrode, an arc-shaped gate.  The cap is named ``"+z"`` and the
    base ``"-z"``, as for a box; wall ``i`` spans footprint vertices ``i`` and
    ``i + 1`` and is named ``"side-{i}"``.

    Args:
        footprint: Simple polygon in the ``xy`` plane, shape ``(n, 2)``, in
            either winding; it may be non-convex. Collinear vertices are
            allowed; repeated ones are not.
        z0: Height of the base.
        z1: Height of the cap; must exceed ``z0``.

    Raises:
        ValueError: If the footprint is not a simple polygon of at least three
            distinct vertices -- self-intersecting or self-touching outlines
            have no well-defined outside -- or ``z1 <= z0``.

    Back-face culling selects the walls that face the camera, which for a
    convex footprint is exactly the visible set.  For a non-convex one, a
    facing wall can still be hidden behind another wall of the same solid;
    :meth:`~vecview.scene.Scene.prism_walls` draws such a solid correctly.
    """
    foot = np.asarray(footprint, dtype=np.float64)
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
    if area < 0:
        foot = foot[::-1]  # clockwise: reverse so the cap faces +z
    base = np.column_stack([foot, np.full(len(foot), float(z0))])
    start, end, *walls = extrude(base, (0.0, 0.0, float(z1) - float(z0)))
    return [end._replace(name="+z"), start._replace(name="-z"), *walls]


def extrude(section: Points3, along: Point3) -> list[Face]:
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

    Args:
        section: Simple polygon lying in one plane, shape ``(n, 3)``, in
            either winding; it may be non-convex. Repeated consecutive
            vertices are not allowed.
        along: The sweep, as a world vector; its length is the prism's length.

    Raises:
        ValueError: If the section has fewer than three vertices, is not
            planar, encloses no area, crosses or touches itself, or if
            ``along`` lies in its plane.
    """
    pts = as_points(section)
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
    if np.abs((pts - pts[0]) @ normal).max() > 1e-9 * size:
        raise ValueError("section is not planar")
    u, v = basis_for(normal)
    if _self_intersects(np.column_stack([(pts - pts[0]) @ u, (pts - pts[0]) @ v])):
        raise ValueError("section must be a simple polygon: its edges cross or touch")
    reach = float(np.dot(sweep, normal))
    if abs(reach) <= 1e-12 * max(size, float(np.linalg.norm(sweep))):
        raise ValueError("along lies in the section's plane, so the sweep has no volume")
    # Wind every face counter-clockwise about the sweep's side of the plane,
    # while wall i keeps the edge from vertex i to i + 1 as the caller gave it.
    flip = reach < 0
    if flip:
        normal = -normal
    ccw = pts[::-1] if flip else pts
    faces = [Face("start", ccw[::-1].copy(), -normal), Face("end", ccw + sweep, normal.copy())]
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        if flip:
            a, b = b, a
        # The wall's plane holds the edge and the sweep, and (edge, along,
        # outward) is right-handed for an edge running counter-clockwise.
        outward = unit(np.cross(b - a, sweep))
        faces.append(Face(f"side-{i}", np.array([a, b, b + sweep, a + sweep]), outward))
    return faces


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

    Raises:
        ValueError: Unless ``0 <= r_in < r_out``, ``n >= 1``, and the span
            ``theta1_deg - theta0_deg`` lies in ``(0, 360)``. A full ring is not a
            simple polygon.
    """
    if not 0.0 <= r_in < r_out:
        raise ValueError(f"need 0 <= r_in < r_out, got r_in={r_in}, r_out={r_out}")
    span = theta1_deg - theta0_deg
    if not 0.0 < span < 360.0:
        raise ValueError(f"theta1_deg - theta0_deg must lie in (0, 360), got {span}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    c = np.asarray(center, dtype=np.float64)
    t = np.radians(np.linspace(theta0_deg, theta1_deg, n + 1))
    ring = np.column_stack([np.cos(t), np.sin(t)])
    outer = c + r_out * ring
    inner = c[None, :] if r_in == 0.0 else c + r_in * ring[::-1]
    footprint: Array = np.vstack([outer, inner])
    return footprint


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
    h, neck = length / 2.0, length / 2.0 - head_len
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

    For a polyhedron with an atom or marker on each vertex.  A face that runs
    into the centre of a sphere cannot be depth-sorted against it, since part
    of the face is inside the ball; cut back to the sphere's surface, it can.
    The ball meets each face plane in a disk about the vertex, so the corner is
    replaced by an inward arc of ``n`` segments, centred on the vertex, from
    one edge to the other.  Name and normal are kept.

    Pair with :meth:`~vecview.scene.Scene.edges` ``trim=radius`` for the edges.

    Raises:
        ValueError: If ``radius`` is not smaller than half of every edge, which
            would make neighbouring cuts overlap.
    """
    trimmed: list[Face] = []
    for face in faces:
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
    "annulus_sector",
    "arc_shape",
    "arrow_shape",
    "box_faces",
    "circle_shape",
    "convex_polyhedron",
    "double_arrow_shape",
    "ellipse_shape",
    "helix",
    "in_plane_dir",
    "prism_faces",
    "rect_shape",
    "sine_ribbon",
    "surface_faces",
    "trim_corners",
]
