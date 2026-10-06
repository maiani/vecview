"""SVG emission shared by the canvases: coordinates, paths, ellipses, text, and ids."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import svg

from vecview._types import Array, Style


def _points(projected: Array, ndigits: int = 2) -> list[svg.Point]:
    """Projected coordinates as the ``x,y`` pairs ``svg.py`` wants for ``points``."""
    # Adding 0.0 turns the -0.0 that rounding can produce into 0.0.
    return [
        svg.Point(round(float(x), ndigits) + 0.0, round(float(y), ndigits) + 0.0)
        for x, y in projected
    ]


def _face_id(base: str, name: str) -> str:
    """Per-face id derived from a base and a face name.

    ``+`` is not a legal XML name character, so the sign is spelled out:
    ``"slab"`` and ``"+z"`` give ``"slab-pz"``.
    """
    return f"{base}-{name.replace('+', 'p').replace('-', 'm')}"


type Align = Literal[
    "center",
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
]

# Which point of a slot's box sits on its anchor, as fractions of (width, height).
_ALIGN: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.5),
    "north": (0.5, 0.0),
    "south": (0.5, 1.0),
    "east": (1.0, 0.5),
    "west": (0.0, 0.5),
    "northeast": (1.0, 0.0),
    "northwest": (0.0, 0.0),
    "southeast": (1.0, 1.0),
    "southwest": (0.0, 1.0),
}


def _num(x: float, ndigits: int = 2) -> float:
    """``x`` rounded for emission, with the ``-0.0`` rounding can produce made ``0.0``."""
    return round(float(x), ndigits) + 0.0


def _path(runs: Iterable[Array], *, closed: bool = False) -> list[svg.PathData]:
    """Path commands tracing each run of screen points as its own subpath."""
    commands: list[svg.PathData] = []
    for run in runs:
        first, *rest = _points(run)
        commands.append(svg.M(first.x, first.y))
        commands += [svg.L(p.x, p.y) for p in rest]
        if closed:
            commands.append(svg.Z())
    return commands


def _ellipse_axes(axes: Array) -> tuple[float, float, float, Array]:
    """Semi-axes, rotation in degrees, and major direction of ``axes @ (cos t, sin t)``.

    The rotation is folded into ``(-90, 90]``, so the same ellipse always
    gets the same attributes.
    """
    u, s, _ = np.linalg.svd(axes)
    major: Array = u[:, 0]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    if angle > 90.0:
        angle, major = angle - 180.0, -major
    elif angle <= -90.0:
        angle, major = angle + 180.0, -major
    return float(s[0]), float(s[1]), angle, major


def _ellipse_element(center: Array, axes: Array, **style: Style) -> svg.Element:
    """The ellipse ``center + axes @ (cos t, sin t)`` as a native SVG shape.

    A ``<circle>`` when the semi-axes agree -- a sphere, or a face-on disk,
    under an orthographic camera -- and a rotated ``<ellipse>`` otherwise, so
    the result stays a shape Inkscape edits as one rather than a path.
    """
    rx, ry, angle, _ = _ellipse_axes(axes)
    cx, cy = _num(center[0]), _num(center[1])
    if abs(rx - ry) <= 1e-9 * max(rx, 1.0):
        return svg.Circle(cx=cx, cy=cy, r=_num(rx, 3), **style)
    rotation = _num(angle, 3)
    transform: list[svg.Transform] | None = [svg.Rotate(rotation, cx, cy)] if rotation else None
    return svg.Ellipse(cx=cx, cy=cy, rx=_num(rx, 3), ry=_num(ry, 3), transform=transform, **style)


type TextContent = str | Sequence[svg.TSpan]
"""A label: plain text, or ``svg.TSpan`` runs for subscripts and mixed styles."""


def _text_length(s: TextContent) -> int:
    """Characters in a label, for the nominal-width bounding-box estimate."""
    return len(s) if isinstance(s, str) else sum(len(run.text or "") for run in s)


def _named(base: str | None, suffix: str) -> dict[str, Style]:
    """``{"id": "{base}-{suffix}"}``, or nothing when there is no base id."""
    return {} if base is None else {"id": f"{base}-{suffix}"}


def _check_unique_ids(elements: Iterable[svg.Element]) -> None:
    """Reject a document in which two elements share an id.

    Duplicate ids are invalid SVG, and a consumer selecting by id silently gets
    the wrong element -- or a gradient fills the wrong shape.  Ids are checked
    in the assembled document because the suffixes a call derives (one per
    face, slice, or hidden part) are only known once a camera has drawn it.
    """
    counts: dict[str, int] = {}
    stack = list(elements)
    while stack:
        element = stack.pop()
        name = getattr(element, "id", None)
        if name is not None:
            counts[str(name)] = counts.get(str(name), 0) + 1
        stack += [child for child in getattr(element, "elements", None) or [] if child is not None]
    repeated = sorted(name for name, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(
            f"ids must be unique in a document; used more than once: {repeated}. "
            "A part placed more than once needs its own id for each placement."
        )


def _slug(color: str) -> str:
    """A colour reduced to characters that are safe in an XML id."""
    return re.sub(r"[^0-9A-Za-z]+", "", color).lower() or "c"


DEFAULT_FONT = "DejaVu Sans, Verdana, sans-serif"

DEFAULT_TEXT_FILL = "#222222"
