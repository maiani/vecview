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


__all__ = ["as_points", "basis_for", "unit"]
