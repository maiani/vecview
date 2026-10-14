"""Flat outlines: footprints and cross-sections, built and combined in 2D.

An outline is a simple polygon as an ``(n, 2)`` array -- the footprint
:func:`~vecview.shapes.prism_faces` extrudes, or a cross-section that
:func:`to_plane` lifts into 3D for :func:`~vecview.shapes.extrude`.  These
functions build them (:func:`rect`, :func:`regular`), grow them
(:func:`offset`, :func:`film`), and combine them (:func:`union`,
:func:`difference`, :func:`intersection`), so a device's layers are drawn as
the cross-sections they are::

    wire = outlines.regular(6, 0.4)
    al = outlines.film(wire, edges=[0, 1, 2], thickness=0.07)  # the three top facets
    section = outlines.to_plane(al, origin=(-1, 0, 0), u=(0, 1, 0), v=(0, 0, 1))
    shell = extrude(section, (2, 0, 0))

Boolean operations return lists of :class:`~vecview.Region`, one per connected
component. Offsets and films return one Region; each has an exterior and a
possibly empty tuple of holes. Point arrays remain shorthand for a simple
region. ``to_plane`` preserves the distinction: an array lifts to an array,
a Region to a Region. No list is interpreted as a list of holes.

Like :mod:`vecview.shapes`, this knows numbers only: no camera, no style, no
SVG.  Every outline returned is wound counter-clockwise, and every hole
clockwise.  ``shapely`` does the clipping.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import shapely
from shapely.geometry import LineString

from vecview._planar import _polygon, _shapes
from vecview._types import Array, Point3, Region, Shape2
from vecview._vec import as_rings

type Join = Literal["mitre", "round", "bevel"]
"""How an offset meets itself at a corner: sharp, rounded, or cut straight."""


def rect(lo: Sequence[float], hi: Sequence[float]) -> Array:
    """The axis-aligned rectangle with corners ``lo`` and ``hi``, counter-clockwise."""
    (x0, y0), (x1, y1) = lo, hi
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"hi must exceed lo in both coordinates, got lo={lo}, hi={hi}")
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float64)


def regular(
    n: int, radius: float, *, center: Sequence[float] = (0.0, 0.0), rotate_deg: float = 0.0
) -> Array:
    """A regular ``n``-gon with circumradius ``radius``, counter-clockwise.

    Vertex ``k`` sits at angle ``rotate_deg + 360 k / n`` from the ``x`` axis,
    so edge ``k`` runs from vertex ``k`` to vertex ``k + 1``.  With the default
    rotation a hexagon has a vertex on ``+x`` and flat edges at top and bottom,
    which is how a nanowire lies on a substrate.
    """
    if n < 3:
        raise ValueError(f"a polygon needs at least three sides, got {n}")
    if not radius > 0:
        raise ValueError(f"radius must be positive, got {radius}")
    t = np.radians(rotate_deg + 360.0 * np.arange(n) / n)
    out: Array = np.asarray(center, dtype=np.float64) + radius * np.column_stack(
        [np.cos(t), np.sin(t)]
    )
    return out


def offset(outline: Shape2, distance: float, *, join: Join = "mitre") -> Region:
    """The outline grown outward by ``distance``, or shrunk for a negative one.

    Holes shrink as the outline grows, and grow as it shrinks; a hole that
    closes is gone, and one an outline closes around -- a C grown into an O
    -- is kept, so the result can be a shape with holes.

    Raises:
        ValueError: If shrinking makes the outline vanish or split in two.
    """
    shape = _polygon(outline).buffer(distance, join_style=join)
    parts = _shapes(shape)
    if len(parts) != 1:
        raise ValueError(
            f"offsetting by {distance} leaves {len(parts)} outlines, not one; "
            "the outline is too thin for that inset"
        )
    return parts[0]


def film(
    outline: Shape2, edges: Iterable[int], thickness: float, *, join: Join = "mitre"
) -> Region:
    """The cross-section of a layer of ``thickness`` deposited on some edges of a body.

    Edge ``i`` runs from vertex ``i`` to vertex ``i + 1`` of ``outline``, as
    given -- the same numbering as the walls of
    :func:`~vecview.shapes.extrude`.  The edges must form one unbroken run
    around the outline (it may wrap past the last vertex), and the film lies
    on their outside, whichever way the outline winds.  Its two ends are cut
    square to the first and last edge, and it never reaches inside the body,
    so a film on a concave run stays out of the notch's interior.  A film on
    every edge is a shell all the way round: ``Region(outside, holes=(body,))``,
    the body its hole.

    Combine it with :func:`union` -- the film on a wire plus the pad it runs
    onto -- and lift it with :func:`to_plane` to extrude it.

    Raises:
        ValueError: For a non-positive thickness, an edge index out of range,
            edges that do not form one run, or an outline with holes.
    """
    if len(as_rings(outline, 2)) > 1:
        raise ValueError("a film lies on the edges of an outline, which must have no holes")
    body = _polygon(outline)
    pts = as_rings(outline, 2)[0]
    n = len(pts)
    if not thickness > 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    chosen = sorted(set(edges))
    if not chosen or chosen[0] < 0 or chosen[-1] >= n:
        raise ValueError(f"edges must be indices in 0..{n - 1}, got {chosen}")
    if len(chosen) == n:
        (ring,) = _shapes(body.buffer(thickness, join_style=join).difference(body))
        return ring
    # Start the run just after an edge that is not in it, so a run that wraps
    # past the last vertex comes out in one piece.
    start = next(i for i in range(n) if (i - 1) % n not in chosen and i in chosen)
    run = [(start + k) % n for k in range(len(chosen))]
    if set(run) != set(chosen):
        raise ValueError(f"edges {chosen} do not form one unbroken run around the outline")
    chain = LineString([pts[i] for i in run] + [pts[(run[-1] + 1) % n]])
    # shapely offsets a single-sided buffer to the left for a positive distance;
    # the outside of a counter-clockwise outline is on the right.
    x, y = pts[:, 0], pts[:, 1]
    ccw = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) > 0
    side = -thickness if ccw else thickness
    layer = chain.buffer(side, single_sided=True, join_style=join, cap_style="flat")
    parts = _shapes(layer.difference(body))
    if len(parts) != 1:
        raise ValueError(f"the film falls apart into {len(parts)} pieces")
    return parts[0]


def union(*outlines: Shape2) -> list[Region]:
    """Everything covered by any of the outlines, as separate shapes, largest first.

    Outlines that close around an uncovered region leave a shape with a hole.
    """
    if not outlines:
        raise ValueError("union needs at least one outline")
    return _shapes(shapely.union_all([_polygon(o) for o in outlines]))


def difference(outline: Shape2, *cuts: Shape2) -> list[Region]:
    """``outline`` with every one of ``cuts`` taken away, as shapes, largest first.

    A cut through the middle leaves two outlines; one that removes everything
    leaves none; one wholly inside leaves a shape with a hole.
    """
    shape = _polygon(outline)
    if cuts:
        shape = shape.difference(shapely.union_all([_polygon(c) for c in cuts]))
    return _shapes(shape)


def intersection(*outlines: Shape2) -> list[Region]:
    """What every one of the outlines covers, as shapes, largest first."""
    if not outlines:
        raise ValueError("intersection needs at least one outline")
    shapes = [_polygon(o) for o in outlines]
    return _shapes(shapely.intersection_all(shapes))


def to_plane(
    outline: Shape2,
    origin: Point3 = (0.0, 0.0, 0.0),
    u: Point3 = (1.0, 0.0, 0.0),
    v: Point3 = (0.0, 1.0, 0.0),
) -> Array | Region:
    """An outline lifted into 3D: the point ``(x, y)`` lands on ``origin + x u + y v``.

    ``u`` and ``v`` are the world vectors for one unit of ``x`` and of ``y``,
    used as given; for a section across a wire along ``x``, ``u = (0, 1, 0)``
    and ``v = (0, 0, 1)``.  The result is what
    :func:`~vecview.shapes.extrude` and ``Scene.polygon`` take.  A shape
    is lifted boundary by boundary and stays a Region.

    Raises:
        ValueError: If ``u`` and ``v`` are parallel, which flattens the outline.
    """
    a, b = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if np.linalg.norm(np.cross(a, b)) <= 1e-12 * max(np.linalg.norm(a) * np.linalg.norm(b), 1.0):
        raise ValueError("u and v are parallel, so the plane they span is a line")
    o = np.asarray(origin, dtype=np.float64)
    lifted: list[Array] = [
        o + np.outer(pts[:, 0], a) + np.outer(pts[:, 1], b) for pts in as_rings(outline, 2)
    ]
    return Region(lifted[0], lifted[1:]) if isinstance(outline, Region) else lifted[0]


__all__ = [
    "Join",
    "difference",
    "film",
    "intersection",
    "offset",
    "rect",
    "regular",
    "to_plane",
    "union",
]
