"""Copies of a part turned or scaled about a pivot, for what ``place`` alone cannot do."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vecview._drawing import Part, _moving, _read_call
from vecview._numeric import finite_vector3 as _vector3
from vecview._place import _placement
from vecview._tracks import Track
from vecview._types import Point3


def _moved(
    part: Part, at: Point3, rotate: tuple[Point3, float] | None = None, scale: float = 1.0
) -> Part:
    moved = Part()
    moved.place(0, part, at=at, rotate=rotate, scale=scale)
    return moved


def _later(make: Callable[..., Part], *args: object, **kwargs: object) -> Part | None:
    """A part redone at every frame, if a track is among the arguments; else ``None``."""
    if not _moving((args, kwargs)):
        return None

    def at(t: float) -> Part:
        now, named = _read_call(args, kwargs, t)
        return make(*now, **named)

    moved = Part()
    moved.place(0, Track(at))
    return moved


def rotate(part: Part, axis: Point3, angle_deg: float, *, pivot: Point3 = (0, 0, 0)) -> Part:
    """A copy of ``part`` turned by ``angle_deg`` about ``axis`` through ``pivot``.

    Right-handed: counter-clockwise looking down ``axis``.  A point ``p`` lands
    at ``pivot + R @ (p - pivot)``.  Any argument may be a track, or the part
    one that holds tracks; the copy then turns as they change.
    """
    if (later := _later(rotate, part, axis, angle_deg, pivot=pivot)) is not None:
        return later
    p = np.asarray(_vector3(pivot, "pivot"))
    turn = (axis, float(angle_deg))
    return _moved(part, tuple(p - _placement((0, 0, 0), turn, None, 1.0).rotation @ p), turn)


def scale(part: Part, factor: float, *, pivot: Point3 = (0, 0, 0)) -> Part:
    """A copy of ``part`` scaled by ``factor`` about ``pivot``: ``p`` lands at
    ``pivot + factor * (p - pivot)``.  Any argument may be a track, as for
    :func:`rotate`."""
    if (later := _later(scale, part, factor, pivot=pivot)) is not None:
        return later
    p = np.asarray(_vector3(pivot, "pivot"))
    return _moved(part, tuple((1.0 - factor) * p), scale=factor)
