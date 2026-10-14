"""Shared planar topology for outlines, solid validation, and face clipping.

Shapely owns intersection and containment. Public geometry stays NumPy arrays
and Region records, independent of cameras, SVG, and the geometry backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from vecview._types import Array, Region, Shape2
from vecview._vec import as_rings, basis_for


def _polygon(shape: Shape2) -> Polygon:
    """An outline, or a shape with holes, as a shapely polygon, checked to be simple."""
    pts, *holes = as_rings(shape, 2)
    if len(pts) < 3 or any(len(hole) < 3 for hole in holes):
        raise ValueError(f"an outline must have shape (n, 2) with n >= 3, got {pts.shape}")
    for ring in (pts, *holes):
        lengths = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        if np.any(lengths <= 1e-12 * float(lengths.max())):
            raise ValueError("region rings have repeated consecutive vertices")
    polygon = Polygon(pts, holes or None)
    if Polygon(pts).area == 0.0:
        raise ValueError("outline encloses no area")
    if any(Polygon(hole).area == 0.0 for hole in holes):
        raise ValueError("a hole encloses no area")
    if not polygon.is_valid:
        if holes:
            raise ValueError(
                "a shape's holes must lie inside its outline, apart, and none may cross"
            )
        raise ValueError("outline must be a simple polygon: its edges cross or touch")
    _refuse_touching(polygon, "a shape")
    return polygon


def _refuse_touching(polygon: Polygon, what: str) -> None:
    """Refuse a hole that touches the outline or another hole, which shapely allows at a point.

    A shape's rings lie apart, so that its walls, its caps, and a cut through
    it are well defined; ``prism_faces`` and ``extrude`` hold every shape to
    that, and so does everything here.
    """
    if not polygon.interiors:
        return
    rings = [polygon.exterior, *polygon.interiors]
    i, j = shapely.STRtree(rings).query(rings, predicate="intersects")
    if np.any(i != j):
        raise ValueError(
            f"{what} has a hole touching its outline or another hole at a point, "
            "which a shape cannot hold: move them apart, or make them overlap"
        )


def _ring(coords: Any) -> Array:
    return np.asarray(coords, dtype=np.float64)[:-1].copy()  # shapely repeats the first vertex


def _shape(polygon: Polygon) -> Region:
    """A polygon as a Region with counter-clockwise exterior and clockwise holes."""
    _refuse_touching(polygon, "the result")
    polygon = orient(polygon, sign=1.0)  # the outline counter-clockwise, holes clockwise
    outline = _ring(polygon.exterior.coords)
    # Holes in a fixed order, by position, rather than in the order GEOS left them.
    holes = sorted(
        (_ring(hole.coords) for hole in polygon.interiors),
        key=lambda h: tuple(round(float(c), 12) for c in (*h.min(axis=0), *h.max(axis=0))),
    )
    return Region(outline, holes)


def _shapes(geometry: Any) -> list[Region]:
    """Every polygon of a shapely result, largest first, as shapes."""
    parts = [g for g in getattr(geometry, "geoms", [geometry]) if isinstance(g, Polygon)]
    parts = [g for g in parts if not g.is_empty and g.area > 0.0]
    # Largest first, then by position, so the order never depends on GEOS internals.
    parts.sort(key=lambda g: (-round(g.area, 12), *(round(c, 12) for c in g.bounds)))
    return [_shape(g) for g in parts]


def _self_intersects(ring: Array) -> bool:
    return not shapely.LinearRing(ring).is_simple


def _check_holes(rings: list[Array]) -> None:
    """Validate the same planar topology for outlines and solids."""
    _polygon(Region(rings[0], rings[1:]))


def check_holed_polygon(rings: list[Array]) -> None:
    """Refuse a 3D polygon whose holes are off its plane, outside it, or crossing."""
    pts = rings[0]
    vector_area = 0.5 * np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0)
    size = float(np.ptp(np.vstack(rings), axis=0).max())
    if np.linalg.norm(vector_area) <= 1e-12 * size * size:
        raise ValueError("a polygon with holes needs an outline enclosing some area")
    normal = vector_area / np.linalg.norm(vector_area)
    if any(np.abs((ring - pts[0]) @ normal).max() > 1e-9 * size for ring in rings):
        raise ValueError("a polygon's holes must lie in the plane of its outline")
    u, v = basis_for(normal)
    _check_holes([np.column_stack([(r - pts[0]) @ u, (r - pts[0]) @ v]) for r in rings])


def _regions(loops: list[Array], normal: Array) -> list[tuple[Array, list[Array]]]:
    """Group oriented cap boundaries, retaining their original shared 3D vertices."""
    if not loops:
        return []
    u, v = basis_for(normal)
    origin = loops[0][0]
    flat = [np.column_stack([(loop - origin) @ u, (loop - origin) @ v]) for loop in loops]
    polys = [Polygon(ring) for ring in flat]
    outer = [k for k, ring in enumerate(flat) if shapely.is_ccw(shapely.LinearRing(ring))]
    holes: dict[int, list[Array]] = {k: [] for k in outer}
    for k, polygon in enumerate(polys):
        if k in holes:
            continue
        around = [m for m in outer if polys[m].covers(polygon)]
        if not around:
            raise ValueError("the faces do not close up along the cut")
        holes[min(around, key=lambda m: polys[m].area)].append(loops[k])
    return [(loops[k], holes[k]) for k in outer]


def _clip_region(
    region: Region, normal: Array, origin: Array, cut_normal: Array, eps: float
) -> list[Region]:
    """Intersect a planar 3D region with a half-space, returning every component."""
    anchor = region.exterior[0]
    normal = normal / np.linalg.norm(normal)
    projected = cut_normal - np.dot(cut_normal, normal) * normal
    magnitude = float(np.linalg.norm(projected))
    if magnitude < 1e-12:
        return [region] if float((anchor - origin) @ cut_normal) < -eps else []
    u = projected / magnitude
    v = np.cross(normal, u)
    limit = -float((anchor - origin) @ cut_normal) / magnitude
    flat = [np.column_stack([(ring - anchor) @ u, (ring - anchor) @ v]) for ring in region.rings]
    # Shared vertices on the cutting plane must have exactly the same classification.
    for ring in flat:
        ring[np.abs(ring[:, 0] - limit) <= eps / magnitude, 0] = limit
    polygon = Polygon(flat[0], flat[1:])
    lo_x, lo_y, hi_x, hi_y = polygon.bounds
    if limit < lo_x:
        return []
    margin = max(hi_x - lo_x, hi_y - lo_y, 1.0)
    mask = shapely.box(lo_x - margin, lo_y - margin, min(limit, hi_x + margin), hi_y + margin)
    clipped = polygon.intersection(mask)
    # Point contacts created by a tangent cut are legitimate boundary events.
    # Preserve them; input regions, unlike clipped faces, must have separated rings.
    polygons = [
        g
        for g in getattr(clipped, "geoms", [clipped])
        if isinstance(g, Polygon) and g.area > eps * eps
    ]
    polygons.sort(key=lambda g: (-round(g.area, 12), *g.bounds))
    result = []
    for polygon in polygons:
        polygon = orient(polygon, sign=1.0)
        rings = [_ring(polygon.exterior.coords), *[_ring(h.coords) for h in polygon.interiors]]
        lifted = [anchor + np.outer(r[:, 0], u) + np.outer(r[:, 1], v) for r in rings]
        result.append(Region(lifted[0], lifted[1:]))
    return result
