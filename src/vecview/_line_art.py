"""The lines a draughtsman inks on a faceted solid: its silhouette, creases, and boundaries.

Numbers only: which faces face the camera is decided by the canvas and passed
in, so everything here is testable without one.  The seams between facets of a
smooth surface -- a turned or swept solid -- are left out, which is what makes
a faceted solid read as a curved one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np

from vecview._face import Face
from vecview._types import Array

type _Key = tuple[float, ...]
type _Edge = tuple[_Key, _Key]


def feature_lines(
    faces: Sequence[Face], facing: Sequence[bool], crease_deg: float
) -> tuple[list[Array], list[Array]]:
    """The runs of edges worth drawing, split into near side and far side.

    An edge is drawn when it bounds one face only (an open surface's rim), when
    one face at it faces the camera and another does not (the silhouette), or
    when its faces meet at more than ``crease_deg`` (a crease: a rim, a corner,
    a cut).  On a closed solid, an edge whose faces all face away lies on the
    far side and is returned second; an open surface has no far side, so all of
    its edges are returned first.  Runs are maximal chains through vertices
    where exactly two drawn edges meet, in a deterministic order.
    """
    point: dict[_Key, Array] = {}
    at: dict[_Edge, list[int]] = {}
    for index, face in enumerate(faces):
        for ring in (face.points, *face.holes):
            ring = np.asarray(ring, dtype=np.float64)
            keys = [tuple(np.round(p, 9).tolist()) for p in ring]
            for i, k in enumerate(keys):
                if k not in point:
                    point[k] = ring[i]
            for a, b in zip(keys, keys[1:] + keys[:1], strict=True):
                if a != b:
                    at.setdefault((a, b) if a < b else (b, a), []).append(index)
    closed = all(len(sides) > 1 for sides in at.values())
    limit = math.cos(math.radians(crease_deg))
    near: list[_Edge] = []
    far: list[_Edge] = []
    for edge, sides in at.items():
        seen = {facing[i] for i in sides}
        if len(sides) > 1 and len(seen) == 1:
            normals = [np.asarray(faces[i].normal, dtype=np.float64) for i in sides]
            if all(_cosine(m, n) >= limit for m, n in combinations(normals, 2)):
                continue  # a seam between facets of one smooth surface
        (near if True in seen or not closed else far).append(edge)
    return _chain(near, point), _chain(far, point)


def _cosine(m: Array, n: Array) -> float:
    return float(m @ n) / float(np.linalg.norm(m) * np.linalg.norm(n))


def _chain(edges: Sequence[_Edge], point: dict[_Key, Array]) -> list[Array]:
    """Join edges into runs: from the loose ends, then from junctions, then round loops.

    A run goes on through a vertex where two drawn edges meet, whatever the
    angle -- the corner of a box is one line turning -- and through a junction
    along the straightest edge it has not drawn, if that turns less than
    ``_ONWARD``: a rim goes on whole past the silhouette that meets it.
    """
    links: dict[_Key, list[tuple[_Edge, _Key]]] = {}
    for edge in edges:
        a, b = edge
        links.setdefault(a, []).append((edge, b))
        links.setdefault(b, []).append((edge, a))
    used: set[_Edge] = set()
    runs: list[Array] = []

    def onward(node: _Key, came_from: _Key) -> tuple[_Edge, _Key] | None:
        ahead = [(e, other) for e, other in links[node] if e not in used]
        if len(links[node]) == 2 or not ahead:
            return ahead[0] if ahead else None
        incoming = _direction(point[came_from], point[node])
        best = max(
            ahead, key=lambda link: float(incoming @ _direction(point[node], point[link[1]]))
        )
        turn = float(incoming @ _direction(point[node], point[best[1]]))
        return best if turn >= _ONWARD else None

    def walk(start: _Key, edge: _Edge, node: _Key) -> None:
        run = [start, node]
        used.add(edge)
        while (step := onward(node, run[-2])) is not None:
            edge, node = step
            used.add(edge)
            run.append(node)
        runs.append(np.array([point[k] for k in run]))

    for degrees in ({1}, None, {2}):
        for vertex, adjacent in links.items():
            if degrees is None and len(adjacent) <= 2:
                continue
            if degrees is not None and len(adjacent) not in degrees:
                continue
            for edge, other in adjacent:
                if edge not in used:
                    walk(vertex, edge, other)
    return runs


_ONWARD = math.cos(math.radians(60.0))


def _direction(a: Array, b: Array) -> Array:
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return d / float(np.linalg.norm(d))
