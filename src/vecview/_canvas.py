"""The canvas a scene is replayed onto: one camera, the layer stack, and exact visibility.

It draws everything but the curved solids, which :mod:`vecview._solids` adds.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import svg

from vecview._elements import (
    _ALIGN,
    DEFAULT_FONT,
    DEFAULT_TEXT_FILL,
    Align,
    TextContent,
    _check_unique_ids,
    _face_id,
    _num,
    _path,
    _points,
    _text_length,
)
from vecview._occlusion import (
    Line,
    Piece,
    Plane,
    Shape,
    Surface,
    Visibility,
    planar_depth,
    plane_through,
    planes_through,
)
from vecview._place import _Frame
from vecview._types import Array, Point3, Points2, Points3, Style
from vecview._vec import as_points, basis_for, convex_hull, unit
from vecview.camera import Camera, ParallelCamera
from vecview.shapes import Face, Pivot, arrow_shape, ellipse_shape, prism_faces


def _runs(visible: list[bool]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive ``True`` entries in a cyclic sequence.

    Each run is ``(start, count)``; a run may wrap past the end.  A sequence
    that is ``True`` throughout is one run starting at ``0``.
    """
    n = len(visible)
    if all(visible):
        return [(0, n)] if n else []
    # Start scanning just after a False entry, so no run is split by the wrap.
    first = next(i for i in range(n) if not visible[i]) + 1
    runs: list[tuple[int, int]] = []
    count = 0
    for k in range(n):
        i = (first + k) % n
        if visible[i]:
            count += 1
        elif count:
            runs.append(((i - count) % n, count))
            count = 0
    if count:
        runs.append(((first + n - count) % n, count))
    return runs


def _parallel(cam: Camera, what: str) -> ParallelCamera:
    """``cam`` as a :class:`ParallelCamera`, or a clear error naming ``what`` needs it."""
    if not isinstance(cam, ParallelCamera):
        raise TypeError(
            f"{type(cam).__name__} cannot {what}: it needs the affine map only a "
            "parallel projection has"
        )
    return cam


def _spanning(points: Array) -> Array:
    """Three corners of a planar polygon that span it as widely as its first edge allows."""
    first, second = points[0], points[1]
    if not np.any(second != first):
        second = points[2]
    away = np.linalg.norm(np.cross(second - first, points - first), axis=1)
    return np.array([first, second, points[int(np.argmax(away))]])


def _coverage(style: Mapping[str, Style]) -> tuple[bool, float]:
    """Whether a style hides what is behind it, and half its stroke width."""

    def amount(key: str) -> float:
        value = style.get(key, 1.0)
        return float(value) if isinstance(value, int | float) else 1.0

    opaque = style.get("fill") != "none" and amount("opacity") >= 1 and amount("fill_opacity") >= 1
    stroked = style.get("stroke") not in (None, "none")
    return opaque, 0.5 * amount("stroke_width") if stroked else 0.0


def _ball_axes(cam: ParallelCamera, radius: float) -> Array:
    """Axes of the ellipse a ball of ``radius`` projects to: ``radius * sqrt(S S^T)``.

    ``S`` is the camera's linear map, so this is exact for any parallel
    projection.  The square root is the symmetric one, so the result is a
    valid ``axes`` for :func:`_ellipse_element`.
    """
    u, s, _ = np.linalg.svd(cam.matrix * cam.scale)
    axes: Array = radius * (u * s) @ u.T
    return axes


class _Canvas:
    """One projection of a scene: its recorded calls replayed against a camera.

    Every method with a public name implements the :class:`Scene` method of the
    same name, which documents it.  A canvas is built, filled, and turned into a
    document by :meth:`Scene.render`, and never escapes it.
    """

    def __init__(self, cam: Camera) -> None:
        self.cam = cam
        self.items: list[tuple[int, int, svg.Element]] = []
        self.defs: list[svg.Element] = []
        self._seq = 0
        self._lo: Array = np.array([np.inf, np.inf])
        self._hi: Array = np.array([-np.inf, -np.inf])
        self._depths: dict[int, float] = {}
        self._sorted: set[int] = set()
        self._shapes: dict[int, Callable[[], Shape | None]] = {}
        self._opaque: set[int] = set()
        self._rank: dict[int, int] = {}
        self._shared: set[str] = set()
        # The classes of the call being replayed, which every element it emits carries.
        self.classes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not bool(np.all(np.isfinite(self._lo)))

    def bbox(self) -> tuple[Array, Array]:
        return self._lo.copy(), self._hi.copy()

    def document(self, pad: float, background: str | None) -> svg.SVG:
        """Assemble the document, fitting the viewBox to the content plus ``pad``."""
        if self.is_empty:
            raise ValueError("cannot render an empty scene: no geometry to fit a viewBox to")
        if self._sorted:
            self._occlude()
        lo, hi = self._lo - pad, self._hi + pad
        # One rounding for the viewBox and the background, so the one covers the other.
        x, y, w, h = (round(float(v), 1) for v in (*lo, *(hi - lo)))
        elements: list[svg.Element] = []
        if self.defs:
            elements.append(svg.Defs(elements=list(self.defs)))
        if background:
            elements.append(svg.Rect(x=x, y=y, width=w, height=h, fill=background))
        elements += [el for _, _, el in sorted(self.items, key=self._order)]
        _check_unique_ids(elements)
        return svg.SVG(width=w, height=h, viewBox=svg.ViewBoxSpec(x, y, w, h), elements=elements)

    def _grow(self, pts2: np.ndarray) -> None:
        p = np.atleast_2d(pts2)
        self._lo = np.minimum(self._lo, p.min(axis=0))
        self._hi = np.maximum(self._hi, p.max(axis=0))

    def _emit(self, layer: int, element: svg.Element, depth: float | None = None) -> None:
        """Queue an element at ``layer`` without logging it.

        ``depth`` is what a layer passed to :meth:`sort_by_depth` orders by.  It
        is rounded so that equal depths tie exactly and fall back to insertion
        order, rather than to the last bits of a floating-point sum.
        """
        if depth is not None:
            self._depths[self._seq] = _num(depth, 6)
        if self.classes:
            element.class_ = list(self.classes)  # ty: ignore[unresolved-attribute]
        self.items.append((int(layer), self._seq, element))
        self._seq += 1

    def _depth(self, pts3: Points3) -> float:
        """Mean depth of world points: the key one element sorts by."""
        return float(np.mean(self.cam.depth(pts3)))

    def sort_by_depth(self, layer: int) -> None:
        """Decide visibility in ``layer`` by depth instead of by insertion."""
        self._sorted.add(int(layer))

    # --- occlusion --------------------------------------------------------
    def _shape(self, build: Callable[[], Shape | None]) -> None:
        """Describe the element just emitted, for a sorted layer to resolve.

        Kept as a thunk: only the layers sorted by depth pay for building the
        shapes.
        """
        self._shapes[self._seq - 1] = build

    def _planar_piece(self, pts3: Points3) -> Piece | None:
        pts = as_points(pts3)
        screen = self.cam.project(pts)
        plane = plane_through(screen, self.cam.depth(pts))
        return None if plane is None else Piece(screen, planar_depth(plane), plane)

    def _convex_piece(self, faces: Sequence[Face]) -> Piece | None:
        """A convex solid: depth is the nearest of its front faces' planes."""
        cam = _parallel(self.cam, "resolve occlusion")
        screen = cam.project(np.vstack([face.points for face in faces]))
        # A plane per face, from three corners that span it -- which survives a
        # quad collapsed to a triangle at a cone's apex.
        corners = np.array([_spanning(face.points) for face in faces])
        flat = corners.reshape(-1, 3)
        planes = planes_through(cam.project(flat).reshape(-1, 3, 2), cam.depth(flat).reshape(-1, 3))
        front: list[Plane] = []
        rear: list[Plane] = []
        for face, plane in zip(faces, planes, strict=True):
            if plane is not None:
                (front if cam.faces_camera(face.normal) else rear).append(plane)
        if not front:
            return None
        near, far = np.array(front), np.array(rear) if rear else np.zeros((0, 3))
        size = float(np.ptp(screen, axis=0).max()) or 1.0

        def depth(xy: Array) -> Array:
            # A ray enters a convex solid through the last front plane it
            # crosses; past the solid's edge that plane goes on smoothly.
            out: Array = (xy @ near[:, :2].T + near[:, 2]).min(axis=1)
            return out

        def covers(xy: Array) -> Any:
            # ... and leaves through the first back plane: a miss if that comes first.
            exit_ = (xy @ far[:, :2].T + far[:, 2]).max(axis=1)
            return depth(xy) >= exit_ - 1e-9 * size

        outline = screen[convex_hull(screen)]
        return Piece(outline, depth, None, covers if len(far) else None)

    def _grown(self, pad: float) -> float:
        """A stroke's half-width in world units, by which a solid's depth is grown.

        Depth is then defined, and the solid in front, over the whole of its
        drawn outline; otherwise whatever is behind would claim the stroke.
        """
        return pad / self.cam.scale

    def _sphere_piece(self, center: Point3, radius: float, pad: float = 0.0) -> Piece:
        """A sphere: the exact depth of its front surface along each ray."""
        cam = _parallel(self.cam, "resolve occlusion")
        c = np.asarray(center, dtype=np.float64)
        radius += self._grown(pad)
        n = 96
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ring = np.column_stack([np.cos(t), np.sin(t)]) / math.cos(math.pi / n)
        outline = np.asarray(cam.at(c)) + ring @ _ball_axes(cam, radius).T
        back = np.linalg.pinv(cam.matrix * cam.scale)
        view = cam.view

        def reach(xy: Array) -> tuple[Array, Array]:
            base = cam.origin + xy @ back.T  # on each ray
            w = base - c
            along = w @ view
            return base @ view - along, along**2 - np.einsum("ij,ij->i", w, w) + radius**2

        def depth(xy: Array) -> Array:
            # Past the rim, the depth of the rim: the ray's closest approach.
            mid, disc = reach(xy)
            out: Array = mid + np.sqrt(np.maximum(disc, 0.0))
            return out

        def covers(xy: Array) -> Any:
            return reach(xy)[1] >= 0

        return Piece(outline, depth, covers=covers)

    def _line(
        self,
        runs3: Sequence[Array],
        style: Mapping[str, Style],
        back: Mapping[str, Style] | None,
    ) -> Line:
        """A stroked shape: its runs with depths, and how to redraw it once split."""
        runs = [(self.cam.project(run), self.cam.depth(run)) for run in runs3]
        base = style.get("id")

        def rebuild(visible: list[Array], hidden: list[Array]) -> list[svg.Element]:
            out: list[svg.Element] = []
            if visible:
                out.append(svg.Path(d=_path(visible), **style))
            if hidden and back is not None:
                attrs: dict[str, Style] = {**style, **back}
                if base is not None:
                    attrs["id"] = f"{base}-hidden"
                out.append(svg.Path(d=_path(hidden), **attrs))
            return out

        return Line(runs, rebuild)

    def _paint_order(
        self, members: Sequence[tuple[int, Shape]], seen: Sequence[Visibility]
    ) -> None:
        """Order a sorted layer's opaque surfaces so whatever hides another comes after it.

        Clipped, opaque surfaces overlap only along their edges, and there the
        one in front must be painted last.  A topological sort of "hides part
        of" gives that order; depth breaks ties, and a cycle -- planes that
        cross -- falls back to depth for the surfaces caught in it.
        """
        seqs = [seq for seq, _ in members]
        after: dict[int, set[int]] = {seq: set() for seq in seqs}
        for (seq, _), visibility in zip(members, seen, strict=True):
            for other in visibility.hidden_by:
                after[seq].add(seqs[other])
        waiting = {seq: 0 for seq in seqs}
        for seq in seqs:
            for later in after[seq]:
                waiting[later] += 1

        def key(seq: int) -> tuple[float, int]:
            return (self._depths.get(seq, 0.0), seq)

        ready = sorted((seq for seq in seqs if waiting[seq] == 0), key=key)
        left = set(seqs)
        rank = len(self._rank)
        while left:
            if not ready:
                ready = [min(left, key=key)]  # a cycle: take the farthest
            seq = ready.pop(0)
            if seq not in left:
                continue
            left.discard(seq)
            self._rank[seq] = rank
            rank += 1
            for later in sorted(after[seq], key=key):
                waiting[later] -= 1
                if waiting[later] == 0 and later in left:
                    ready.append(later)
            ready.sort(key=key)

    def _occlude(self) -> None:
        """Resolve every sorted layer: clip surfaces to what shows, split lines."""
        from vecview._occlusion import resolve

        outcome: dict[int, tuple[Shape, Visibility]] = {}
        for layer in sorted(self._sorted):
            seqs = [seq for at, seq, _ in self.items if at == layer and seq in self._shapes]
            built = [(seq, self._shapes[seq]()) for seq in seqs]
            members = [(seq, shape) for seq, shape in built if shape is not None]
            seen_all = resolve([shape for _, shape in members])
            for (seq, shape), seen in zip(members, seen_all, strict=True):
                outcome[seq] = (shape, seen)
            self._paint_order(members, seen_all)
        items: list[tuple[int, int, svg.Element]] = []
        for layer, seq, element in self.items:
            if seq not in outcome:
                items.append((layer, seq, element))
                continue
            shape, seen = outcome[seq]
            if isinstance(shape, Surface):
                if shape.opaque:
                    self._opaque.add(seq)
                rings = seen.rings
                if rings is None:
                    items.append((layer, seq, element))
                elif rings:
                    name = getattr(element, "id", None)
                    clip = f"{name}-visible" if name else f"visible-{seq}"
                    region = svg.Path(d=_path(rings, closed=True), clip_rule="evenodd")
                    self.defs.append(svg.ClipPath(id=clip, elements=[region]))
                    element.clip_path = f"url(#{clip})"  # ty: ignore[unresolved-attribute]
                    items.append((layer, seq, element))
            elif seen.hidden:
                rebuilt = shape.rebuild(seen.visible, seen.hidden)
                for part in rebuilt:
                    part.class_ = getattr(element, "class_", None)  # ty: ignore[unresolved-attribute]
                items += [(layer, seq, part) for part in rebuilt]
            else:
                items.append((layer, seq, element))
        self.items = items

    def add(self, layer: int, element: svg.Element) -> None:
        """Add a ready-made ``svg.py`` element at ``layer``, bypassing projection."""
        self._emit(layer, element)

    def add_def(self, element: svg.Element) -> None:
        """Add an element to ``<defs>`` -- a gradient, marker, or clip path."""
        self.defs.append(element)

    def polygon(self, layer: int, pts3: Points3, **style: Style) -> None:
        """Filled polygon through projected world points."""
        p = self.cam.project(pts3)
        self._grow(p)
        self._emit(layer, svg.Polygon(points=_points(p), **style), self._depth(pts3))
        pts = as_points(pts3)
        if style.get("fill") == "none":
            self._shape(lambda: self._line([np.vstack([pts, pts[:1]])], style, None))
        else:
            opaque, pad = _coverage(style)

            def surface() -> Surface | None:
                piece = self._planar_piece(pts)
                return None if piece is None else Surface([piece], opaque, pad)

            self._shape(surface)

    def polyline(
        self, layer: int, pts3: Points3, *, back: Mapping[str, Style] | None = None, **style: Style
    ) -> None:
        """Open path through projected world points; unfilled unless asked."""
        style.setdefault("fill", "none")
        p = self.cam.project(pts3)
        self._grow(p)
        self._emit(layer, svg.Polyline(points=_points(p), **style), self._depth(pts3))
        pts = as_points(pts3)
        self._shape(lambda: self._line([pts], style, back))

    def faces(
        self, layer: int, faces: Iterable[Face], *, cull: bool = False, **style: Style
    ) -> None:
        """Draw each face as a polygon, all with the same style."""
        base = style.pop("id", None)
        given = list(faces)
        drawn = self.cam.visible(given) if cull else given
        for face in drawn:
            if base is None:
                self.polygon(layer, face.points, **style)
            else:
                self.polygon(layer, face.points, id=_face_id(str(base), face.name), **style)

    def plane(
        self,
        layer: int,
        origin: Point3,
        u_edge: Point3,
        v_edge: Point3,
        *,
        id: str,
        **style: Style,
    ) -> None:
        """Reserve an empty group occupying a rectangle of a world plane."""
        cam = _parallel(self.cam, "embed content in a plane")
        matrix = cam.plane_matrix(origin, u_edge, v_edge)
        o = np.asarray(origin, dtype=np.float64)
        u = np.asarray(u_edge, dtype=np.float64)
        v = np.asarray(v_edge, dtype=np.float64)
        self._grow(self.cam.project([o, o + u, o + u + v, o + v]))
        self._emit(
            layer,
            svg.G(
                id=id,
                transform=[svg.Matrix(*(round(value, 4) for value in matrix))],
                **style,
            ),
            self._depth([o + (u + v) / 2.0]),
        )

    def slot(
        self,
        layer: int,
        pt3: Point3,
        w: float,
        h: float,
        *,
        id: str,
        align: Align = "center",
        dx: float = 0.0,
        dy: float = 0.0,
        **style: Style,
    ) -> None:
        """Reserve an empty, screen-aligned group anchored at a projected world point."""
        if align not in _ALIGN:
            raise ValueError(f"unknown align {align!r}; expected one of {sorted(_ALIGN)}")
        x, y = self.cam.at(pt3)
        x, y = x + dx, y + dy
        fx, fy = _ALIGN[align]
        x0, y0 = x - fx * w, y - fy * h
        self._grow(np.array([[x0, y0], [x0 + w, y0 + h]]))
        self._emit(
            layer,
            svg.G(
                id=id,
                transform=[svg.Translate(round(x, 2), round(y, 2))],
                data={"align": align},
                **style,
            ),
            self._depth([pt3]),
        )

    def silhouette(self, layer: int, solid: Iterable[Face] | Points3, **style: Style) -> None:
        """Fill the projected outline of a convex solid as one polygon."""
        given = list(solid)
        if given and isinstance(given[0], Face):
            pts = np.vstack([face.points for face in given if isinstance(face, Face)])
        else:
            pts = as_points(cast(Points3, given))
        p = self.cam.project(pts)
        outline = p[convex_hull(p)]
        self._grow(outline)
        self._emit(layer, svg.Polygon(points=_points(outline), **style), self._depth(pts))
        solid = [face for face in given if isinstance(face, Face)]
        if solid:
            opaque, pad = _coverage(style)

            def surface() -> Surface | None:
                piece = self._convex_piece(solid)
                return None if piece is None else Surface([piece], opaque, pad)

            self._shape(surface)

    def prism_walls(
        self,
        layer: int,
        footprint: Points2,
        z0: float,
        z1: float,
        *,
        _frame: _Frame | None = None,
        **style: Style,
    ) -> None:
        """Draw the camera-facing walls of an extruded footprint as one seamless shape."""
        walls = prism_faces(footprint, z0, z1)[2:]
        if _frame is not None:
            # A placed part: the walls move with it, kept in order so runs stay runs.
            walls = [_frame.face(wall, keep_order=True) for wall in walls]
        facing = {id(face) for face in self.cam.visible(walls)}
        runs = _runs([id(face) in facing for face in walls])
        if not runs:
            return
        n = len(walls)
        outlines: list[Array] = []
        for start, count in runs:
            indices = [(start + k) % n for k in range(count)]
            bottom = [walls[i].points[0] for i in indices] + [walls[indices[-1]].points[1]]
            top = [walls[indices[-1]].points[2]] + [walls[i].points[3] for i in reversed(indices)]
            outlines.append(self.cam.project(np.array(bottom + top)))
            self._grow(outlines[-1])
        depth = self._depth(np.vstack([wall.points for wall in walls]))
        self._emit(layer, svg.Path(d=_path(outlines, closed=True), **style), depth)
        shown = [wall for wall in walls if id(wall) in facing]
        opaque, pad = _coverage(style)

        def surface() -> Surface | None:
            pieces = [piece for wall in shown if (piece := self._planar_piece(wall.points))]
            return Surface(pieces, opaque, pad) if pieces else None

        self._shape(surface)

    def arrow(
        self,
        layer: int,
        origin: Point3,
        direction: Point3,
        length: float,
        *,
        normal: Point3 | Literal["camera"],
        shaft_w: float,
        head_w: float,
        head_len: float,
        pivot: Pivot = "tail",
        **style: Style,
    ) -> None:
        """Flat arrow, as :func:`~vecview.shapes.arrow_shape`, drawn as a polygon."""
        if isinstance(normal, str):
            if normal != "camera":
                raise ValueError(f"normal must be a vector or 'camera', got {normal!r}")
            normal = self._facing_normal(direction)
        shape = arrow_shape(origin, direction, length, normal, shaft_w, head_w, head_len, pivot)
        self.polygon(layer, shape, **style)

    def _facing_normal(self, direction: Point3) -> Array:
        """Normal of the plane through ``direction`` that shows the widest arrow.

        The arrow's width runs along some ``s`` perpendicular to its axis; this
        picks the ``s`` whose screen projection is longest -- the top singular
        vector of the camera matrix restricted to that perpendicular plane.  For
        an orthographic camera that is the plane facing the viewer; for an
        oblique one, ``view`` would give a narrower arrow, so it is not used.
        """
        cam = _parallel(self.cam, "face an arrow toward the camera")
        d = unit(direction)
        if float(np.linalg.norm(cam.matrix @ d)) < 1e-9:
            raise ValueError(
                "arrow points along the projection ray, so no plane through it faces the camera"
            )
        e1, e2 = basis_for(d)
        _, _, vt = np.linalg.svd(cam.matrix @ np.column_stack([e1, e2]))
        width = vt[0, 0] * e1 + vt[0, 1] * e2
        n = unit(np.cross(d, width))
        return n if float(np.dot(n, cam.view)) >= 0 else -n

    def gaussian(
        self,
        layer: int,
        center: Point3,
        u: Point3,
        v: Point3,
        a: float,
        b: float,
        *,
        id: str,
        color: str,
        opacity: float = 1.0,
        extent: float = 2.0,
        stops: int = 9,
        **style: Style,
    ) -> None:
        """A soft Gaussian spot lying in a world plane, as one gradient-filled polygon."""
        cam = _parallel(self.cam, "map a gradient through a plane")
        if stops < 2:
            raise ValueError("a gradient needs at least two stops")
        # Gradient coordinates (s, t) in units of the half-widths, mapped onto the plane.
        matrix = cam.plane_matrix(center, a * unit(u), b * unit(v))
        floor = np.exp(-(extent**2))
        profile: list[svg.Element] = [
            svg.Stop(
                offset=round(k / (stops - 1), 4),
                stop_color=color,
                stop_opacity=round(
                    opacity * (np.exp(-((extent * k / (stops - 1)) ** 2)) - floor) / (1 - floor), 4
                ),
            )
            for k in range(stops)
        ]
        self.defs.append(
            svg.RadialGradient(
                id=f"{id}-profile",
                gradientUnits="userSpaceOnUse",
                gradientTransform=[svg.Matrix(*(round(value, 4) for value in matrix))],
                cx=0,
                cy=0,
                r=extent,
                elements=profile,
            )
        )
        outline = ellipse_shape(center, u, v, a * extent, b * extent)
        self.polygon(layer, outline, fill=f"url(#{id}-profile)", id=id, **style)
        # A soft spot fades to nothing, so it hides nothing behind it.
        build = self._shapes[self._seq - 1]

        def translucent() -> Shape | None:
            shape = build()
            return dataclasses.replace(shape, opaque=False) if isinstance(shape, Surface) else shape

        self._shape(translucent)

    def edges(
        self,
        layer: int,
        faces: Iterable[Face],
        *,
        back: Mapping[str, Style] | None = None,
        back_layer: int | None = None,
        separate: bool = False,
        trim: float = 0.0,
        **style: Style,
    ) -> None:
        """The edges of a convex solid, split into visible and hidden."""
        given = list(faces)
        base = style.pop("id", None)
        style.setdefault("fill", "none")
        facing = {id(face) for face in self.cam.visible(given)}
        ends: dict[tuple[tuple[float, ...], ...], Array] = {}
        shown: dict[tuple[tuple[float, ...], ...], bool] = {}
        for face in given:
            ring: Array = np.asarray(face.points, dtype=np.float64)
            for start, end in zip(ring, np.roll(ring, -1, axis=0), strict=True):
                a, b = cast(Array, start), cast(Array, end)
                key = tuple(sorted((tuple(np.round(a, 9)), tuple(np.round(b, 9)))))
                if key not in ends:
                    along = unit(b - a) * trim
                    ends[key] = np.array([a + along, b - along])
                shown[key] = shown.get(key, False) or id(face) in facing
        for visible, at, extra, suffix in (
            (True, layer, {}, "front"),
            (False, layer if back_layer is None else back_layer, back, "back"),
        ):
            if extra is None:
                continue
            segments = [ends[key] for key in ends if shown[key] is visible]
            groups = [[seg] for seg in segments] if separate else [segments]
            for k, group in enumerate(groups):
                if not group:
                    continue
                projected = [self.cam.project(seg) for seg in group]
                for seg in projected:
                    self._grow(seg)
                attrs: dict[str, Style] = {**style, **extra}
                if base is not None:
                    attrs["id"] = f"{base}-{suffix}-{k}" if separate else f"{base}-{suffix}"
                self._emit(at, svg.Path(d=_path(projected), **attrs), self._depth(np.vstack(group)))
                # Behind another object, a visible edge takes the hidden style and
                # a hidden one disappears.
                hide = back if visible else None
                self._shape(functools.partial(self._line, group, attrs, hide))

    def sphere_curve(
        self,
        layer: int,
        center: Point3,
        pts3: Points3,
        *,
        closed: bool = False,
        back: Mapping[str, Style] | None = None,
        back_layer: int | None = None,
        **style: Style,
    ) -> None:
        """A curve on a sphere's surface, split where it passes behind the sphere."""
        cam = _parallel(self.cam, "split a curve at a sphere's outline")
        style.setdefault("fill", "none")
        base = style.pop("id", None)
        pts = as_points(pts3)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        side = (pts - np.asarray(center, dtype=np.float64)) @ cam.view
        front = side >= 0
        runs: list[tuple[bool, list[Array]]] = []
        current, state = [pts[0]], bool(front[0])
        for i in range(len(pts) - 1):
            if bool(front[i + 1]) != state:
                t = side[i] / (side[i] - side[i + 1])
                cut = pts[i] + t * (pts[i + 1] - pts[i])
                current.append(cut)
                runs.append((state, current))
                current, state = [cut], bool(front[i + 1])
            current.append(pts[i + 1])
        runs.append((state, current))
        if closed and len(runs) > 1 and runs[0][0] == runs[-1][0]:
            runs = [(runs[0][0], runs[-1][1] + runs[0][1][1:]), *runs[1:-1]]
        for visible, at, extra, suffix in (
            (True, layer, {}, "front"),
            (False, layer if back_layer is None else back_layer, back, "back"),
        ):
            if extra is None:
                continue
            chosen = [np.array(run) for state, run in runs if state is visible]
            if not chosen:
                continue
            projected = [self.cam.project(run) for run in chosen]
            for run in projected:
                self._grow(run)
            attrs: dict[str, Style] = {**style, **extra}
            if base is not None:
                attrs["id"] = f"{base}-{suffix}"
            self._emit(
                at,
                svg.Path(d=_path(projected), **attrs),
                self._depth(np.vstack(chosen)),
            )
            hide = back if visible else None
            self._shape(functools.partial(self._line, chosen, attrs, hide))

    def text(
        self,
        layer: int,
        pt3: Point3,
        s: TextContent,
        dx: float = 0.0,
        dy: float = 0.0,
        size: float = 22.0,
        **style: Style,
    ) -> None:
        """Text anchored at a projected world point, offset by ``(dx, dy)`` on screen."""
        x, y = self.cam.at(pt3)
        self._place_text(layer, x + dx, y + dy, s, size, True, self._depth([pt3]), **style)

    def rect2d(
        self,
        layer: int,
        x: float,
        y: float,
        w: float,
        h: float,
        grow: bool = False,
        **style: Style,
    ) -> None:
        """Rectangle in screen coordinates -- a backdrop, glow, or gradient wash."""
        if grow:
            self._grow(np.array([[x, y], [x + w, y + h]]))
        self._emit(
            layer,
            svg.Rect(x=round(x, 1), y=round(y, 1), width=round(w, 1), height=round(h, 1), **style),
        )

    def text2d(
        self,
        layer: int,
        x: float,
        y: float,
        s: TextContent,
        size: float = 22.0,
        grow: bool = True,
        **style: Style,
    ) -> None:
        """Text in screen coordinates; ``s`` as for :meth:`text`."""
        style.setdefault("font_family", DEFAULT_FONT)
        style.setdefault("fill", DEFAULT_TEXT_FILL)
        self._place_text(layer, x, y, s, size, grow, None, **style)

    def _place_text(
        self,
        layer: int,
        x: float,
        y: float,
        s: TextContent,
        size: float,
        grow: bool,
        depth: float | None,
        **style: Style,
    ) -> None:
        """Emit a text element; shared by :meth:`text` and :meth:`text2d`."""
        style.setdefault("font_family", DEFAULT_FONT)
        style.setdefault("fill", DEFAULT_TEXT_FILL)
        if grow:
            # Crude metrics, but enough to keep the fitted viewBox from clipping
            # labels.  Real advance widths would need the font loaded.
            w = 0.56 * size * _text_length(s)
            anchor = style.get("text_anchor", "start")
            x0 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
            self._grow(np.array([[x0, y - 0.82 * size], [x0 + w, y + 0.25 * size]]))
        content: dict[str, Style] = {"text": s} if isinstance(s, str) else {"elements": list(s)}
        self._emit(
            layer,
            svg.Text(
                x=round(x, 2),
                y=round(y, 2),
                font_size=size,
                **content,
                **style,
            ),
            depth,
        )

    def _order(self, item: tuple[int, int, svg.Element]) -> tuple[int, int, float, int]:
        """Sort key: layer, then visibility within a depth-sorted layer, then insertion.

        In a sorted layer the opaque surfaces come first, in the order
        :meth:`_paint_order` found, everything translucent or stroked is then
        painted back to front over them, and screen-space elements go on top.
        """
        layer, seq, _ = item
        if layer not in self._sorted:
            return (layer, 0, 0.0, seq)
        depth = self._depths.get(seq)
        if depth is None:
            return (layer, 2, 0.0, seq)
        if seq in self._opaque:
            return (layer, 0, float(self._rank.get(seq, 0)), seq)
        return (layer, 1, depth, seq)
