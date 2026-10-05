"""Shared array type aliases.

``numpy`` accepts a wide range of array-likes.  These aliases name the shapes
the public API actually cares about -- a single 3D point, and a sequence of them
-- so signatures stay readable under ``mypy --strict``.
"""

from __future__ import annotations

from collections.abc import Sequence
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

type Style = Any
"""An SVG presentation attribute value, passed through to ``svg.py``."""

__all__ = ["Array", "Point3", "Points2", "Points3", "Style"]
