"""Copies of a part turned or scaled about a pivot, for what ``place`` alone cannot do."""

from __future__ import annotations

import numpy as np

from vecview._drawing import Part
from vecview._numeric import finite_vector3 as _vector3
from vecview._place import _placement
from vecview._types import Point3


def _moved(
    part: Part, at: Point3, rotate: tuple[Point3, float] | None = None, scale: float = 1.0
) -> Part:
    moved = Part()
    moved.place(0, part, at=at, rotate=rotate, scale=scale)
    return moved


def rotate(part: Part, axis: Point3, angle_deg: float, *, pivot: Point3 = (0, 0, 0)) -> Part:
    """A copy of ``part`` turned by ``angle_deg`` about ``axis`` through ``pivot``.

    Right-handed: counter-clockwise looking down ``axis``.  A point ``p`` lands
    at ``pivot + R @ (p - pivot)``.
    """
    p = np.asarray(_vector3(pivot, "pivot"))
    turn = (axis, float(angle_deg))
    return _moved(part, tuple(p - _placement((0, 0, 0), turn, None, 1.0).rotation @ p), turn)


def scale(part: Part, factor: float, *, pivot: Point3 = (0, 0, 0)) -> Part:
    """A copy of ``part`` scaled by ``factor`` about ``pivot``: ``p`` lands at
    ``pivot + factor * (p - pivot)``."""
    p = np.asarray(_vector3(pivot, "pivot"))
    return _moved(part, tuple((1.0 - factor) * p), scale=factor)
