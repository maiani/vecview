"""Placing a part: the frame it moves by, and how each recorded call moves with it."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from vecview._types import Array, Point3, Points3, Style
from vecview._vec import as_points, unit
from vecview.shapes import Face


@dataclasses.dataclass(frozen=True)
class _Frame:
    """Where a part is placed: ``p -> scale * rotation @ p + offset``.

    ``rotation`` is orthogonal -- a rotation, possibly with a mirror -- so
    lengths scale uniformly and a sphere stays a sphere.
    """

    rotation: Array
    scale: float
    offset: Array

    @property
    def linear(self) -> Array:
        out: Array = self.scale * self.rotation
        return out

    def then(self, outer: _Frame) -> _Frame:
        """This frame followed by ``outer``: a part placed in a part."""
        return _Frame(
            outer.rotation @ self.rotation,
            outer.scale * self.scale,
            outer.linear @ self.offset + outer.offset,
        )

    def point(self, p: Point3) -> Array:
        out: Array = self.linear @ np.asarray(p, dtype=np.float64) + self.offset
        return out

    def points(self, pts: Points3) -> Array:
        out: Array = as_points(pts) @ self.linear.T + self.offset
        return out

    def vector(self, v: Point3) -> Array:
        out: Array = self.rotation @ np.asarray(v, dtype=np.float64)
        return out

    def face(self, face: Face, *, keep_order: bool = False) -> Face:
        """A face moved with the part, its winding kept true to its normal under a mirror."""
        pts = self.points(face.points)
        if not keep_order and np.linalg.det(self.rotation) < 0:
            pts = pts[::-1].copy()
        return Face(face.name, pts, unit(self.vector(face.normal)))


def _moved_solid(frame: _Frame, solid: list[Any]) -> list[Any]:
    """:meth:`Scene.silhouette`'s argument, faces or vertices, moved with a part."""
    if solid and isinstance(solid[0], Face):
        return [frame.face(face) for face in solid]
    return list(frame.points(solid))


def _moved_normal(frame: _Frame, normal: Point3 | str) -> Point3 | str:
    return normal if isinstance(normal, str) else frame.vector(normal)


# How each recorded call moves with a part: one entry per positional argument
# after ``layer``, and one per keyword that holds geometry.  ``None`` leaves an
# argument alone -- screen units, text, flags.
type _Move = Callable[[_Frame, Any], Any] | None


def _length(frame: _Frame, x: float | None) -> float | None:
    return None if x is None else frame.scale * float(x)


_POINT: _Move = _Frame.point

_POINTS: _Move = _Frame.points

_VECTOR: _Move = _Frame.vector

_EDGE: _Move = lambda frame, v: frame.linear @ np.asarray(v, dtype=np.float64)  # noqa: E731

_FACES: _Move = lambda frame, faces: [frame.face(face) for face in faces]  # noqa: E731

_MOVES: dict[str, tuple[tuple[_Move, ...], dict[str, _Move]]] = {
    "polygon": ((_POINTS,), {}),
    "polyline": ((_POINTS,), {}),
    "faces": ((_FACES,), {}),
    "plane": ((_POINT, _EDGE, _EDGE), {}),
    "slot": ((_POINT, None, None), {}),
    "silhouette": ((_moved_solid,), {}),
    "prism_walls": ((None, None, None), {}),
    "arrow": (
        (_POINT, _VECTOR, _length),
        {"normal": _moved_normal, "shaft_w": _length, "head_w": _length, "head_len": _length},
    ),
    "gaussian": ((_POINT, _VECTOR, _VECTOR, _length, _length), {}),
    "sphere": ((_POINT, _length), {}),
    "cylinder": ((_POINT, _POINT, _length), {"r1": _length}),
    "cone": ((_POINT, _POINT, _length), {}),
    "arrow3d": (
        (_POINT, _VECTOR, _length),
        {"shaft_r": _length, "head_r": _length, "head_len": _length},
    ),
    "tube": ((_POINTS, _length), {}),
    "edges": ((_FACES,), {"trim": _length}),
    "sphere_curve": ((_POINT, _POINTS), {}),
    "text": ((_POINT, None, None, None, None), {}),
}

type _Call = tuple[str, tuple[object, ...], dict[str, Style], tuple[str, ...]]
"""A recorded drawing call: method, positional arguments, keywords, classes."""


def _placement(
    at: Point3, rotate: tuple[Point3, float] | None, mirror: Point3 | None, scale: float
) -> _Frame:
    """The frame :meth:`Part.place` describes, checked."""
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be positive, got {scale}")
    matrix: Array = np.eye(3)
    if mirror is not None:
        n = unit(mirror)
        if not n.any():
            raise ValueError("mirror needs a non-zero plane normal")
        matrix = np.eye(3) - 2.0 * np.outer(n, n)
    if rotate is not None:
        axis, angle_deg = rotate
        k = unit(axis)
        if not k.any():
            raise ValueError("rotate needs a non-zero axis")
        theta = math.radians(angle_deg)
        cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        turn = np.eye(3) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * cross @ cross
        matrix = turn @ matrix
    return _Frame(matrix, float(scale), np.asarray(at, dtype=np.float64))


def _moved(
    call: _Call, frame: _Frame, layer: int, prefix: str | None, extra: tuple[str, ...]
) -> _Call:
    """One recorded call of a part, as it is drawn by a placement."""
    method, (own_layer, *rest), kwargs, classes = call
    positional, keywords = _MOVES[method]
    args = [
        arg if move is None else move(frame, arg)
        for move, arg in zip(positional, rest, strict=True)
    ]
    moved = dict(kwargs)
    for key, move in keywords.items():
        if key in moved and move is not None:
            moved[key] = move(frame, moved[key])
    if moved.get("back_layer") is not None:
        moved["back_layer"] = int(moved["back_layer"]) + layer
    if prefix is not None and moved.get("id") is not None:
        moved["id"] = f"{prefix}-{moved['id']}"
    if method == "prism_walls":
        inner = moved.get("_frame")
        moved["_frame"] = frame if inner is None else inner.then(frame)
    return (
        method,
        (int(cast(int, own_layer)) + layer, *args),
        moved,
        tuple(dict.fromkeys(classes + extra)),
    )
