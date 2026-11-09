"""Cutting a closed solid open by a plane, and capping the cut.

Numbers only, as :mod:`vecview.shapes`, which re-exports :func:`cut`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from vecview._face import Face
from vecview._planar import _clip_region, _regions
from vecview._types import Array, Point3
from vecview._vec import unit


def cut(faces: Iterable[Face], origin: Point3, normal: Point3) -> list[Face]:
    """A closed solid cut open by a plane: the part behind it, with the cut capped.

    The plane passes through ``origin``, and everything on the side ``normal``
    points to is cut away -- the cutaway that shows a wire's core inside its
    shell, or a layer stack from within.  Every face is clipped to the half
    that stays and keeps its name, so ids stay stable; faces wholly cut away
    are dropped.  The cut itself is capped by new faces lying in the plane and
    facing along ``normal``, named ``"cut"`` -- or ``"cut-0"``, ``"cut-1"``, ...
    when the plane meets the solid in several places -- so a cap can be drawn
    in a style of its own.  A face lying in the plane is dropped, the cap
    taking its place.  A plane across a hollow solid -- a tube cut square --
    leaves a cap with a hole in it. Every face is clipped as a planar
    region; a hole whose side lies along the plane opens into a notch.
    Should clipping split a face, the pieces are
    named ``"{name}-0"``, ``"{name}-1"``, ...

    The faces must close a solid, each wound counter-clockwise about its
    outward normal, as every function here returns them; non-convex solids
    are fine.  A plane that misses the solid returns it unchanged, or nothing.

    Raises:
        ValueError: If the faces do not close up along the cut.
    """
    given = list(faces)
    n, o = unit(normal), np.asarray(origin, dtype=np.float64)
    if not n.any():
        raise ValueError("a cut needs a non-zero plane normal")
    if not given:
        return []
    everything = np.vstack([face.points for face in given])
    eps = 1e-9 * max(float(np.ptp(everything, axis=0).max()), 1.0)
    kept: list[Face] = []
    rims: list[tuple[Array, Array]] = []
    for face in given:
        pts = np.asarray(face.points, dtype=np.float64)
        d = (pts - o) @ n
        d[np.abs(d) <= eps] = 0.0
        if np.all(d > 0):
            continue
        if np.all(d == 0):
            continue  # lying in the plane: the cap, built from its neighbours, replaces it
        if np.all(d < 0):
            kept.append(face)
            continue
        pieces, edges = _clip_rings(face, o, n, eps)
        kept += pieces
        rims += edges
    return kept + _caps(rims, n, eps)


def _clip_rings(
    face: Face, o: Array, n: Array, eps: float
) -> tuple[list[Face], list[tuple[Array, Array]]]:
    """Clip every face through the same planar-region operation, including splits."""
    regions = _clip_region(face.region, np.asarray(face.normal), o, n, eps)
    pieces = [
        Face(
            face.name if len(regions) == 1 else f"{face.name}-{k}",
            region.exterior,
            face.normal,
            region.holes,
        )
        for k, region in enumerate(regions)
    ]
    rims: list[tuple[Array, Array]] = []
    for region in regions:
        for loop in region.rings:
            on = np.abs((loop - o) @ n) <= eps
            for i in range(len(loop)):
                j = (i + 1) % len(loop)
                if on[i] and on[j]:
                    rims.append((loop[j], loop[i]))
    return pieces, rims


def _caps(rims: list[tuple[Array, Array]], normal: Array, eps: float) -> list[Face]:
    """Chain the directed edges a cut leaves in its plane into cap faces."""

    def key(p: Array) -> tuple[float, ...]:
        return tuple(np.round(p / (100 * eps)).astype(int).tolist())

    # Two faces running along the same stretch of the plane both ways cancel.
    edges = {(key(a), key(b)): (a, b) for a, b in rims}
    edges = {k: e for k, e in edges.items() if (k[1], k[0]) not in edges}
    following = {k[0]: k for k in edges}
    if len(following) != len(edges):
        raise ValueError("the faces do not close up along the cut")
    loops: list[Array] = []
    while following:
        start = next(iter(following))
        loop, at = [], start
        while at in following:
            edge = following.pop(at)
            loop.append(edges[edge][0])
            at = edge[1]
        if at != start:
            raise ValueError("the faces do not close up along the cut")
        loops.append(np.array(loop))
    regions = _regions(loops, normal)
    return [
        Face("cut" if len(regions) == 1 else f"cut-{k}", loop, normal.copy(), tuple(holes))
        for k, (loop, holes) in enumerate(regions)
    ]
