"""World-space glyph geometry.

Every function here returns plain world-space points -- an ``(n, 3)`` array, or
a list of :class:`Face` for solids.  Nothing here knows about a camera, a style,
or SVG.  That split is deliberate: geometry stays testable as numbers, and the
same polygon can be drawn twice with different fills, or reused as a clip path.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from vecview._types import Array, Point3
from vecview._vec import basis_for, unit

FaceName = Literal["-x", "+x", "-y", "+y", "-z", "+z"]

Pivot = Literal["tail", "mid"]


class Face(NamedTuple):
    """One planar polygon of a solid, with the outward normal it was built from.

    Carrying the normal alongside the points is what lets
    :meth:`~vecview.camera.Camera.visible` decide which walls of a box to draw
    without the caller reasoning about which octant the camera sits in.
    """

    name: str
    points: Array
    normal: Array


def in_plane_dir(
    angle_deg: float, u: Point3 = (1.0, 0.0, 0.0), v: Point3 = (0.0, 1.0, 0.0)
) -> Array:
    """Unit direction at ``angle_deg`` from ``u``, rotating toward ``v``.

    The common case -- an angle measured in the ``xy`` plane from ``+x``, as
    crystal and polarization angles usually are -- needs no axes passed.
    """
    a = np.radians(float(angle_deg))
    return unit(np.cos(a) * unit(u) + np.sin(a) * unit(v))


def rect_shape(center: Point3, u: Point3, v: Point3, du: float, dv: float) -> Array:
    """Rectangle centred on ``center``, in the plane spanned by ``u`` and ``v``."""
    c, uh, vh = np.asarray(center, dtype=np.float64), unit(u), unit(v)
    return np.array(
        [
            c - uh * du / 2 - vh * dv / 2,
            c + uh * du / 2 - vh * dv / 2,
            c + uh * du / 2 + vh * dv / 2,
            c - uh * du / 2 + vh * dv / 2,
        ]
    )


def box_faces(center: Point3, size: Point3) -> list[Face]:
    """The six faces of an axis-aligned box, each wound CCW about its outward normal.

    Pair with :meth:`~vecview.camera.Camera.visible` to draw only the walls a
    given camera can see::

        slab = box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
        scene.faces(10, camera.visible(slab), fill="#cfd6e0")

    Args:
        center: Centre of the box.
        size: Full extents ``(sx, sy, sz)``, not half-extents.
    """
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(size, dtype=np.float64) / 2.0
    if c.shape != (3,) or h.shape != (3,):
        raise ValueError("center and size must each be three numbers")

    axes = np.eye(3)
    faces: list[Face] = []
    for axis, sign in ((a, s) for a in range(3) for s in (+1, -1)):
        normal = axes[axis] * sign
        # In-plane axes chosen so (u, v, normal) is right-handed, which winds
        # the points counter-clockwise as seen from outside the box.
        u = axes[(axis + 1) % 3] * sign
        v = axes[(axis + 2) % 3]
        du, dv = 2 * h[(axis + 1) % 3], 2 * h[(axis + 2) % 3]
        name = f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"
        faces.append(Face(name, rect_shape(c + normal * h[axis], u, v, du, dv), normal))
    return faces


def arrow_shape(
    origin: Point3,
    direction: Point3,
    length: float,
    normal: Point3,
    shaft_w: float,
    head_w: float,
    head_len: float,
    pivot: Pivot = "tail",
) -> Array:
    """Flat arrow polygon in the plane through ``origin`` with the given normal.

    Args:
        origin: Tail of the arrow, or its midpoint when ``pivot="mid"``.
        direction: Direction the head points.
        length: Total length, tail to tip.
        normal: Normal of the plane the arrow is drawn flat in.
        shaft_w: Width of the shaft.
        head_w: Width of the head at its widest.
        head_len: Length of the head, measured back from the tip.
        pivot: Whether ``origin`` is the ``"tail"`` or the ``"mid"`` point.
    """
    d, n = unit(direction), unit(normal)
    s = unit(np.cross(n, d))
    base = np.asarray(origin, dtype=np.float64) - (d * length / 2.0 if pivot == "mid" else 0.0)
    tip = base + d * length
    neck = base + d * max(length - head_len, 0.0)
    return np.array(
        [
            base + s * shaft_w / 2,
            neck + s * shaft_w / 2,
            neck + s * head_w / 2,
            tip,
            neck - s * head_w / 2,
            neck - s * shaft_w / 2,
            base - s * shaft_w / 2,
        ]
    )


def double_arrow_shape(
    center: Point3,
    direction: Point3,
    length: float,
    normal: Point3,
    shaft_w: float,
    head_w: float,
    head_len: float,
) -> Array:
    """Double-headed arrow centred on ``center`` -- an axis or polarization marker."""
    d, n = unit(direction), unit(normal)
    s = unit(np.cross(n, d))
    c = np.asarray(center, dtype=np.float64)
    h, neck = length / 2.0, length / 2.0 - head_len
    return np.array(
        [
            c + d * h,
            c + d * neck + s * head_w / 2,
            c + d * neck + s * shaft_w / 2,
            c - d * neck + s * shaft_w / 2,
            c - d * neck + s * head_w / 2,
            c - d * h,
            c - d * neck - s * head_w / 2,
            c - d * neck - s * shaft_w / 2,
            c + d * neck - s * shaft_w / 2,
            c + d * neck - s * head_w / 2,
        ]
    )


def circle_shape(center: Point3, radius: float, normal: Point3, n: int = 64) -> Array:
    """Circle as an ``n``-gon, in the plane through ``center`` with the given normal."""
    a, b = basis_for(normal)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring: Array = np.asarray(center, dtype=np.float64) + radius * (
        np.outer(np.cos(t), a) + np.outer(np.sin(t), b)
    )
    return ring


def sine_ribbon(
    start: Point3,
    axis: Point3,
    length: float,
    transverse: Point3,
    amplitude: float | Array,
    wavelength: float,
    n: int = 400,
    phase: float = 0.0,
) -> Array:
    """Transverse wave along ``axis``, displaced along ``transverse``.

    ``amplitude`` may be a scalar or an array of length ``n``, so a component
    that decays inside a medium can be drawn by passing its envelope.
    """
    a, e = unit(axis), unit(transverse)
    t = np.linspace(0.0, length, n)
    disp = np.asarray(amplitude, dtype=np.float64) * np.sin(2.0 * np.pi * t / wavelength + phase)
    ribbon: Array = np.asarray(start, dtype=np.float64) + np.outer(t, a) + np.outer(disp, e)
    return ribbon


__all__ = [
    "Face",
    "FaceName",
    "arrow_shape",
    "box_faces",
    "circle_shape",
    "double_arrow_shape",
    "in_plane_dir",
    "rect_shape",
    "sine_ribbon",
]
