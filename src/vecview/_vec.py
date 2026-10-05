"""Small vector helpers used across the package."""

from __future__ import annotations

import numpy as np

from vecview._types import Array, Point3, Points3


def unit(v: Point3) -> Array:
    """Normalize ``v``, returning it unchanged if it has zero length.

    A zero vector is passed through rather than raising: degenerate directions
    show up naturally in generated geometry (a wave with zero amplitude, a box
    with zero thickness) and a silent zero draws as nothing, which is the
    intended result.
    """
    a: Array = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    return a / n if n else a


def as_points(pts: Points3) -> Array:
    """Coerce a point or sequence of points to a ``(n, 3)`` array."""
    a: Array = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected world points of shape (n, 3), got {a.shape}")
    return a


def basis_for(normal: Point3) -> tuple[Array, Array]:
    """Two orthonormal in-plane directions for the plane with the given normal.

    The choice of in-plane rotation is arbitrary but deterministic, which is
    what matters for reproducible output.
    """
    n = unit(normal)
    seed = (0.0, 0.0, 1.0) if abs(float(n[2])) < 0.9 else (1.0, 0.0, 0.0)
    a = unit(np.cross(n, seed))
    return a, np.asarray(np.cross(n, a), dtype=np.float64)


def convex_hull(pts2: Array) -> list[int]:
    """Indices of the 2D convex hull of ``pts2`` (Andrew's monotone chain).

    Counter-clockwise in a y-up frame, which is clockwise on an SVG screen.
    Collinear boundary points are dropped. Ties break on index, so the result is
    deterministic for repeated points.
    """
    order = sorted(range(len(pts2)), key=lambda i: (float(pts2[i, 0]), float(pts2[i, 1]), i))

    def turn(o: int, a: int, b: int) -> float:
        return float(
            (pts2[a, 0] - pts2[o, 0]) * (pts2[b, 1] - pts2[o, 1])
            - (pts2[a, 1] - pts2[o, 1]) * (pts2[b, 0] - pts2[o, 0])
        )

    chains: list[list[int]] = []
    for sequence in (order, order[::-1]):
        chain: list[int] = []
        for i in sequence:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], i) <= 0:
                chain.pop()
            chain.append(i)
        chains.append(chain[:-1])
    return chains[0] + chains[1]


__all__ = ["as_points", "basis_for", "convex_hull", "unit"]
