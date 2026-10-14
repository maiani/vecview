"""Shared array type aliases.

``numpy`` accepts a wide range of array-likes.  These aliases name the shapes
the public API actually cares about -- a single 3D point, and a sequence of them
-- so signatures stay readable when type checking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]
"""A ``float64`` array of any shape."""

type Point3 = Sequence[float] | Array
"""One world-space point or direction, ``(x, y, z)``."""

type Points3 = Sequence[Point3] | Array
"""A sequence of world-space points, shape ``(n, 3)``."""

type Points2 = Sequence[Sequence[float]] | Array
"""A sequence of 2D points, shape ``(n, 2)`` -- a footprint in a plane."""


@dataclass(frozen=True, eq=False)
class Region:
    """One connected planar region, in 2D or world-space 3D.

    ``exterior`` is its outer boundary; ``holes`` contains interior boundaries.
    Rings omit the repeated closing vertex. Either winding is accepted on input.
    A collection of disconnected regions is a list of ``Region`` objects, never
    a list of rings. Geometry operations validate topology and planarity.
    """

    exterior: Array
    holes: tuple[Array, ...] = ()

    def __init__(
        self, exterior: Points2 | Points3, holes: Sequence[Points2 | Points3] = ()
    ) -> None:
        rings = [np.array(r, dtype=np.float64, copy=True) for r in (exterior, *holes)]
        dim = rings[0].shape[1] if rings[0].ndim == 2 else 0
        if dim not in (2, 3) or any(r.ndim != 2 or r.shape[1] != dim for r in rings):
            raise ValueError("a region needs rings of shape (n, 2) or (n, 3), all in one dimension")
        if any(len(r) < 3 or not np.isfinite(r).all() for r in rings):
            raise ValueError("each region ring needs at least three finite vertices")
        for ring in rings:
            ring.setflags(write=False)
        object.__setattr__(self, "exterior", rings[0])
        object.__setattr__(self, "holes", tuple(rings[1:]))

    @property
    def rings(self) -> tuple[Array, ...]:
        """Exterior followed by interior boundaries; these are not components."""
        return (self.exterior, *self.holes)


type Shape2 = Points2 | Region
"""One 2D region, or a plain ring of points as shorthand."""

type Shape3 = Points3 | Region
"""One planar world-space region, or a plain ring of points as shorthand."""

type Style = Any
"""An SVG presentation attribute value, passed through to ``svg.py``."""

__all__ = ["Array", "Point3", "Points2", "Points3", "Region", "Shape2", "Shape3", "Style"]
