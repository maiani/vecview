"""Exact visibility for a layer passed to ``Scene.sort_by_depth(layer, exact=True)``.

The painter's algorithm orders whole elements, so it fails wherever two
elements each hide part of the other.  Here every element of the layer is
described by where it covers the screen and how deep its nearest surface is at
each screen point, and visibility is decided point by point:

* a **surface** keeps its native SVG element, clipped to the region where no
  opaque surface is nearer.  Between two planar surfaces that region is bounded
  by a straight line, found exactly; anywhere else it is the zero contour of
  the depth difference, traced at sub-pixel resolution;
* a **line** is cut where it passes behind an opaque surface, into visible
  runs and hidden runs, which the caller can draw in another style.

Translucent surfaces hide nothing, but are themselves clipped by opaque ones in
front of them.  ``shapely`` and ``contourpy`` are imported only when such a
layer is rendered, so plain scenes do not pay for loading them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vecview._types import Array

if TYPE_CHECKING:
    import svg

type DepthFn = Callable[[Array], Array]
"""Screen points ``(n, 2)`` to the depth of the nearest surface; ``-inf`` where it misses."""

type Plane = tuple[float, float, float]
"""Depth as ``a * x + b * y + c`` over the screen, for a planar piece."""


@dataclass
class Piece:
    """One planar or convex part of a surface.

    Args:
        outline: Screen polygon the piece covers, shape ``(n, 2)``.
        depth: Depth of the piece's front surface at screen points.  Past its
            edge it should go on smoothly -- the rim's depth, say -- rather than
            stop: an element's stroke reaches past the surface, and is hidden
            or not as the surface beside it would be.
        plane: The depth as an affine function of screen position, for a
            planar piece, which makes comparisons with other planes exact.
        covers: Where the piece truly covers, for telling the parts of one
            surface apart; ``None`` to go by ``outline``.
    """

    outline: Array
    depth: DepthFn
    plane: Plane | None = None
    covers: Callable[[Array], Any] | None = None
    width: float = 0.0
    """When positive, ``outline`` is a centre line and the piece covers the band
    this far either side of it -- a tube, drawn as a wide stroke."""


@dataclass
class Surface:
    """Something that fills area and may hide what is behind it."""

    pieces: list[Piece]
    opaque: bool
    pad: float = 0.0
    """Half the stroke width: the element covers its outline grown by this much."""
    chain: tuple[int, int] | None = None
    """``(chain id, position)`` for consecutive pieces of one tube, which overlap
    by construction and are one surface: neighbours in a chain never hide each
    other."""


@dataclass
class Line:
    """Something drawn as strokes, which hides nothing but can be hidden.

    ``runs`` are screen polylines with the depth at each vertex; depth varies
    linearly along each straight segment, which holds for any parallel
    projection.
    """

    runs: list[tuple[Array, Array]]
    rebuild: Callable[[list[Array], list[Array]], list[svg.Element]]
    extra: dict[str, Any] = field(default_factory=dict)


type Shape = Surface | Line


@dataclass
class Visibility:
    """What :func:`resolve` decided for one shape.

    For a surface, ``rings`` is ``None`` when nothing hides it, empty when it is
    hidden entirely, and otherwise the rings of its visible region -- outer
    boundaries counter-clockwise and holes clockwise, in screen coordinates.
    For a line, ``visible`` and ``hidden`` are its runs on either side.
    """

    rings: list[Array] | None = None
    visible: list[Array] = field(default_factory=list)
    hidden: list[Array] = field(default_factory=list)
    hidden_by: list[int] = field(default_factory=list)
    """The shapes, by index, that hide some of this one, and so must be
    painted after it: where two edges meet, the one in front belongs on top."""


def plane_through(screen: Array, depth: Array) -> Plane | None:
    """The plane ``depth = a x + b y + c`` through projected vertices, or ``None`` edge-on.

    Exact for the vertices of any planar polygon under a parallel projection,
    where both screen position and depth are affine in the polygon's plane.
    """
    if len(screen) < 3:
        return None
    centred = screen - screen.mean(axis=0)
    if np.linalg.matrix_rank(centred, tol=1e-9 * max(float(np.abs(centred).max()), 1.0)) < 2:
        return None
    a = np.column_stack([screen, np.ones(len(screen))])
    coef, *_ = np.linalg.lstsq(a, depth, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def planar_depth(plane: Plane) -> DepthFn:
    a, b, c = plane

    def depth(xy: Array) -> Array:
        out: Array = a * xy[:, 0] + b * xy[:, 1] + c
        return out

    return depth


def _import() -> tuple[Any, Any]:
    import contourpy
    import shapely

    return shapely, contourpy


def resolve(
    shapes: Sequence[Shape], *, resolution: float = 0.5, tuck: float = 0.4
) -> list[Visibility]:
    """Decide what of each shape is visible.

    Args:
        shapes: Every shape of one layer.
        resolution: Grid step, in screen units, of the depth contours traced
            where two surfaces are not both planar.
        tuck: How far, in screen units, a hidden surface is kept on under the
            edge of what hides it.
    """
    shapely, contourpy = _import()
    _COVER.clear()
    surfaces: list[tuple[int, Surface]] = [
        (k, s) for k, s in enumerate(shapes) if isinstance(s, Surface)
    ]
    regions = {k: _region(shapely, s) for k, s in surfaces}
    occluders = [k for k, s in surfaces if s.opaque and not regions[k].is_empty]
    solid = dict(surfaces)
    # What a surface hides stops a little inside its own outline, so whatever is
    # behind runs on under its anti-aliased edge instead of both edges fading
    # out over the same pixels and letting the background through.
    tucked = {k: regions[k].buffer(-tuck, quad_segs=2) for k in occluders}
    tree = shapely.STRtree([regions[k] for k in occluders])
    span = _depth_span(shapes)
    eps_surface, eps_line = 1e-9 * span, 1e-4 * span

    result = [Visibility() for _ in shapes]
    for k, surface in surfaces:
        region = regions[k]
        if region.is_empty:
            continue
        hidden = []
        for j in tree.query(region):
            other = occluders[int(j)]
            if other == k or _neighbours(surface, solid[other]):
                continue
            overlap = region.intersection(tucked[other])
            if overlap.is_empty or overlap.area <= 1e-9:
                continue
            front = _in_front(
                shapely, contourpy, surface, solid[other], overlap, eps_surface, resolution, tuck
            )
            if front is not None and not front.is_empty:
                hidden.append(front)
                result[k].hidden_by.append(other)
        if not hidden:
            continue
        visible = region.difference(shapely.union_all(hidden))
        if visible.area <= 1e-3:
            result[k].rings = []  # hidden entirely
            continue
        # A clip path a twentieth of a screen unit off is invisible, and far smaller.
        result[k].rings = _rings(shapely, visible.simplify(0.05)) or None

    for k, shape in enumerate(shapes):
        if isinstance(shape, Line):
            result[k].visible, result[k].hidden = _split_line(
                shapely,
                shape,
                [solid[o] for o in occluders],
                [regions[o] for o in occluders],
                tree,
                eps_line,
            )
    return result


def _cover(shapely: Any, piece: Piece) -> Any:
    """The screen region a piece covers, as a shapely geometry."""
    if piece.width > 0:
        return shapely.LineString(piece.outline).buffer(piece.width, quad_segs=8, cap_style="flat")
    return shapely.make_valid(shapely.Polygon(piece.outline))


def _region(shapely: Any, surface: Surface) -> Any:
    polygons = [_cover(shapely, piece) for piece in surface.pieces]
    region = shapely.union_all(polygons)
    if surface.pad > 0:
        region = region.buffer(surface.pad, quad_segs=4)
    return region


def _depth_span(shapes: Sequence[Shape]) -> float:
    """A depth scale for the tolerances: the spread of the line depths, or 1."""
    values = [d for s in shapes if isinstance(s, Line) for _, d in s.runs]
    if not values:
        return 1.0
    joined = np.concatenate(values)
    return max(float(np.ptp(joined)), 1.0)


def _neighbours(a: Surface, b: Surface) -> bool:
    if a.chain is None or b.chain is None or a.chain[0] != b.chain[0]:
        return False
    return abs(a.chain[1] - b.chain[1]) <= 1


def planes_through(screen: Array, depth: Array) -> list[Plane | None]:
    """:func:`plane_through` for many triangles at once: ``(k, 3, 2)`` and ``(k, 3)``."""
    a = np.concatenate([screen, np.ones((*screen.shape[:2], 1))], axis=2)
    det = np.linalg.det(a)
    scale = np.abs(screen).max() if screen.size else 1.0
    ok = np.abs(det) > 1e-9 * max(float(scale), 1.0) ** 2
    out: list[Plane | None] = [None] * len(screen)
    if ok.any():
        coef = np.linalg.solve(a[ok], depth[ok][..., None])[..., 0]
        for i, row in zip(np.flatnonzero(ok), coef, strict=True):
            out[int(i)] = (float(row[0]), float(row[1]), float(row[2]))
    return out


def capsule_depth(
    segments: Array, radius: float, back: Array, origin: Array, view: Array
) -> tuple[DepthFn, Callable[[Array], Any]]:
    """Exact depth of a chain of capsules -- a tube of ``radius`` along ``segments``.

    ``segments`` is ``(m, 2, 3)`` in world space; ``back`` maps screen points to
    a world point on each ray (``origin + xy @ back.T``) and ``view`` is the
    unit direction toward the camera, so depth is ``x @ view``.  Each ray is
    cast from beyond the nearest point of the chain toward the scene, and the
    first capsule it meets gives the depth.  A ray that meets none takes the
    depth of its closest approach to the nearest one, so the depth goes on
    smoothly past the tube's edge.  Returns the depth and where the tube covers.
    """
    pa, pb = segments[:, 0], segments[:, 1]
    top = float(np.max(segments.reshape(-1, 3) @ view)) + radius + 1.0
    ba = pb - pa
    baba = np.einsum("ij,ij->i", ba, ba)
    rd = -view
    bard = ba @ rd

    def cast(xy: Array) -> tuple[Array, Array]:
        """Distance along each ray to each capsule (``inf`` on a miss), and the
        distance to each capsule's closest approach, for the extension."""
        base = origin + xy @ back.T
        ro = base + (top - base @ view)[:, None] * view  # depth `top`, past everything
        oa = ro[:, None, :] - pa[None, :, :]
        baoa = np.einsum("nmj,mj->nm", oa, ba)
        rdoa = oa @ rd
        oaoa = np.einsum("nmj,nmj->nm", oa, oa)
        a = baba - bard**2
        b = baba * rdoa - baoa * bard
        c = baba * oaoa - baoa**2 - radius**2 * baba
        h = b * b - a * c
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (-b - np.sqrt(np.maximum(h, 0.0))) / a
            y = baoa + t * bard
            body = (h >= 0) & (a > 1e-12) & (y > 0) & (y < baba)
            hit = np.where(body, t, np.inf)
            # The rounded ends: the sphere at whichever end the axis point fell past.
            oc = np.where((y <= 0)[..., None], oa, ro[:, None, :] - pb[None, :, :])
            bc = oc @ rd
            hc = bc * bc - (np.einsum("nmj,nmj->nm", oc, oc) - radius**2)
            hit = np.where(~body & (hc >= 0), -bc - np.sqrt(np.maximum(hc, 0.0)), hit)
        closest = np.take_along_axis(-bc, hc.argmax(axis=1)[:, None], axis=1)[:, 0]
        return hit.min(axis=1), closest

    def depth(xy: Array) -> Array:
        first, closest = cast(xy)
        out: Array = top - np.where(np.isfinite(first), first, closest)
        return out

    def covers(xy: Array) -> Any:
        return np.isfinite(cast(xy)[0])

    return depth, covers


_COVER: dict[int, Any] = {}


def _depth_of(shapely: Any, surface: Surface, xy: Array) -> Array:
    """Nearest depth of a surface at screen points: the front-most of its pieces."""
    if len(surface.pieces) == 1:
        return surface.pieces[0].depth(xy)
    best = np.full(len(xy), -np.inf)
    for piece in surface.pieces:
        if piece.covers is not None:
            inside = piece.covers(xy)
            best[inside] = np.maximum(best[inside], piece.depth(xy[inside]))
            continue
        cover = _COVER.get(id(piece))
        if cover is None:
            cover = _cover(shapely, piece).buffer(surface.pad + 1e-6, quad_segs=2)
            shapely.prepare(cover)
            _COVER[id(piece)] = cover
        inside = shapely.contains_xy(cover, xy[:, 0], xy[:, 1])
        if inside.any():
            best[inside] = np.maximum(best[inside], piece.depth(xy[inside]))
    return best


def _in_front(
    shapely: Any,
    contourpy: Any,
    back: Surface,
    front: Surface,
    overlap: Any,
    eps: float,
    resolution: float,
    tuck: float,
) -> Any:
    """The part of ``overlap`` where ``front`` is nearer than ``back``."""
    pa = back.pieces[0].plane if len(back.pieces) == 1 else None
    pb = front.pieces[0].plane if len(front.pieces) == 1 else None
    if pa is not None and pb is not None:
        a, b, c = (pb[i] - pa[i] for i in range(3))
        # Where two planes cross, the tuck moves the boundary into the hidden side.
        shift = eps + tuck * float(np.hypot(a, b))
        return overlap.intersection(_half_plane(shapely, overlap.bounds, a, b, c - shift))

    probe = _probe_points(shapely, overlap)
    diff = _difference(shapely, front, back, probe)
    if np.all(diff > eps):
        return overlap
    if np.all(diff <= eps):
        return None

    x0, y0, x1, y1 = overlap.bounds
    nx = int(np.clip(np.ceil((x1 - x0) / resolution), 8, 160)) + 1
    ny = int(np.clip(np.ceil((y1 - y0) / resolution), 8, 160)) + 1
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    z = _difference(shapely, front, back, grid)
    top = max(float(z.max()), eps) + 1.0
    generator = contourpy.contour_generator(
        xs, ys, z.reshape(ny, nx), fill_type=contourpy.FillType.OuterOffset
    )
    points, offsets = generator.filled(eps, top)
    polygons = []
    for pts, offs in zip(points, offsets, strict=True):
        rings = [pts[offs[i] : offs[i + 1]] for i in range(len(offs) - 1)]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            polygons.append(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
    if not polygons:
        return None
    return overlap.intersection(shapely.union_all(polygons))


def _difference(shapely: Any, front: Surface, back: Surface, xy: Array) -> Array:
    """How much nearer ``front`` is than ``back``, or ``-1`` where either misses.

    A surface is hidden only where both are present and the other is nearer.
    Where only one is -- the edge of a stroke, say -- the order of painting,
    which puts whatever hides something after it, already gives the answer.
    """
    a, b = _depth_of(shapely, front, xy), _depth_of(shapely, back, xy)
    with np.errstate(invalid="ignore"):
        z = a - b
    z[~(np.isfinite(a) & np.isfinite(b))] = -1.0
    return z


def _half_plane(shapely: Any, bounds: tuple[float, ...], a: float, b: float, c: float) -> Any:
    """The part of a box, grown a little, where ``a x + b y + c > 0`` (Sutherland-Hodgman)."""
    x0, y0, x1, y1 = bounds
    m = 1.0 + 0.01 * max(x1 - x0, y1 - y0)
    box = [(x0 - m, y0 - m), (x1 + m, y0 - m), (x1 + m, y1 + m), (x0 - m, y1 + m)]
    out: list[tuple[float, float]] = []
    for i, p in enumerate(box):
        q = box[(i + 1) % 4]
        fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp > 0:
            out.append(p)
        if (fp > 0) != (fq > 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return shapely.Polygon(out) if len(out) >= 3 else shapely.Polygon()


def _probe_points(shapely: Any, region: Any) -> Array:
    """A few points spread over a region, for a quick test of which surface is in front."""
    x0, y0, x1, y1 = region.bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, 9)[1:-1], np.linspace(y0, y1, 9)[1:-1])
    inside = shapely.contains_xy(region, gx.ravel(), gy.ravel())
    pts = np.column_stack([gx.ravel()[inside], gy.ravel()[inside]])
    rep = region.representative_point()
    return np.vstack([pts, [[rep.x, rep.y]]])


def _rings(shapely: Any, geometry: Any) -> list[Array]:
    """The rings of a region, valid at the precision they are written in.

    Snapped to the 0.01 grid that coordinates are rounded to on output, so the
    written polygon is the valid one shapely computed rather than a rounding of
    it, which can fold a thin spike over itself.  Slivers and pinholes too
    small to see are dropped.
    """
    snapped = shapely.set_precision(geometry, 0.01)
    rings: list[Array] = []
    for poly in getattr(snapped, "geoms", [snapped]):
        if poly.is_empty or poly.geom_type != "Polygon" or poly.area <= 0.01:
            continue
        poly = shapely.geometry.polygon.orient(poly, 1.0)
        rings.append(np.asarray(poly.exterior.coords)[:-1])
        rings += [
            np.asarray(hole.coords)[:-1]
            for hole in poly.interiors
            if shapely.Polygon(hole).area > 0.01
        ]
    return rings


def _split_line(
    shapely: Any,
    line: Line,
    occluders: list[Surface],
    regions: list[Any],
    tree: Any,
    eps: float,
) -> tuple[list[Array], list[Array]]:
    """Cut a line's runs where they pass behind an opaque surface."""
    visible: list[Array] = []
    hidden: list[Array] = []
    for pts, depths in line.runs:
        if len(pts) < 2:
            continue
        candidates = [int(j) for j in tree.query(shapely.LineString(pts))] if occluders else []
        if not candidates:
            visible.append(pts)
            continue

        def behind(xy: Array, d: Array, candidates: list[int] = candidates) -> Array:
            flag = np.zeros(len(xy), dtype=bool)
            for j in candidates:
                inside = shapely.contains_xy(regions[j], xy[:, 0], xy[:, 1])
                if inside.any():
                    front = _depth_of(shapely, occluders[j], xy[inside])
                    flag[inside] |= front > d[inside] + eps
            return flag

        # Sample every segment densely, then refine each change by bisection.
        xy_parts, d_parts = [pts[:1]], [depths[:1]]
        for i in range(len(pts) - 1):
            n = max(int(np.ceil(np.linalg.norm(pts[i + 1] - pts[i]) / 0.5)), 1)
            t = np.linspace(0.0, 1.0, n + 1)[1:, None]
            xy_parts.append(pts[i] + t * (pts[i + 1] - pts[i]))
            d_parts.append(depths[i] + t[:, 0] * (depths[i + 1] - depths[i]))
        xy, d = np.vstack(xy_parts), np.concatenate(d_parts)
        flags = behind(xy, d)

        current = [xy[0]]
        state = bool(flags[0])
        for i in range(1, len(xy)):
            if bool(flags[i]) != state:
                lo, hi = 0.0, 1.0
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    p = xy[i - 1] + mid * (xy[i] - xy[i - 1])
                    dm = d[i - 1] + mid * (d[i] - d[i - 1])
                    if bool(behind(p[None, :], np.array([dm]))[0]) == state:
                        lo = mid
                    else:
                        hi = mid
                cut = xy[i - 1] + 0.5 * (lo + hi) * (xy[i] - xy[i - 1])
                current.append(cut)
                (hidden if state else visible).append(np.array(current))
                current, state = [cut], bool(flags[i])
            current.append(xy[i])
        (hidden if state else visible).append(np.array(current))
    return [_simplify(r) for r in visible if len(r) >= 2], [
        _simplify(r) for r in hidden if len(r) >= 2
    ]


def _simplify(run: Array) -> Array:
    """Drop sample points that lie on a straight stretch, keeping the corners."""
    if len(run) <= 2:
        return run
    keep = [0]
    for i in range(1, len(run) - 1):
        a, b, c = run[keep[-1]], run[i], run[i + 1]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-6 * max(float(np.linalg.norm(c - a)), 1.0) ** 2:
            keep.append(i)
    keep.append(len(run) - 1)
    return run[keep]
