"""The layer stack: world-space geometry in, one SVG document out."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import svg

from vecview._types import Array, Point3, Points2, Points3, Style
from vecview._vec import as_points, basis_for, convex_hull, unit
from vecview.camera import Camera, ParallelCamera
from vecview.shapes import Face, Pivot, arrow_shape, ellipse_shape, prism_faces


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


def _ball_axes(cam: ParallelCamera, radius: float) -> Array:
    """Axes of the ellipse a ball of ``radius`` projects to: ``radius * sqrt(S S^T)``.

    ``S`` is the camera's linear map, so this is exact for any parallel
    projection.  The square root is the symmetric one, so the result is a
    valid ``axes`` for :func:`_ellipse_element`.
    """
    u, s, _ = np.linalg.svd(cam.matrix * cam.scale)
    axes: Array = radius * (u * s) @ u.T
    return axes


def _frustum_outline(
    c0: Array,
    c1: Array,
    axes: Array,
    r0: float,
    r1: float,
    *,
    closed: bool = True,
    arcs: tuple[bool, bool] = (True, True),
) -> list[svg.PathData]:
    """Exact outline of a projected frustum: two tangent lines and two elliptical arcs.

    Under a parallel projection the two end circles become the homothetic
    ellipses ``c_k + r_k * axes @ (cos t, sin t)``, and the solid's outline is
    their convex hull.  Mapping the screen back through ``axes`` turns both
    ellipses into circles, where the outer common tangents have a closed form;
    mapping the tangent points forward again gives the outline, with each
    curved side as one SVG arc rather than a polyline.  ``r1 = 0`` is a cone.

    With ``closed=False`` the result is for stroking only: the two sides, and
    the end arcs that ``arcs`` asks for.  That is how a sliced cylinder strokes
    its silhouette without a line across every joint.  The degenerate views
    fall back to the closed outline.
    """
    rx, ry, angle, major = _ellipse_axes(axes)
    det = float(np.linalg.det(axes))
    if abs(det) <= 1e-9 * rx * rx:
        # The end circles are seen edge-on and project to segments.
        ends = np.array(
            [c0 + r0 * rx * major, c0 - r0 * rx * major, c1 + r1 * rx * major, c1 - r1 * rx * major]
        )
        return _path([ends[convex_hull(ends)]], closed=True)

    def at(c: Array, r: float, phi: float) -> svg.Point:
        x, y = c + r * (axes @ np.array([math.cos(phi), math.sin(phi)]))
        return svg.Point(_num(x), _num(y))

    def arc(r: float, large: bool, sweep: bool, end: svg.Point) -> svg.Arc:
        return svg.Arc(_num(rx * r, 3), _num(ry * r, 3), _num(angle, 3), large, sweep, end.x, end.y)

    d = np.linalg.solve(axes, c1 - c0)
    dist = float(np.linalg.norm(d))
    if dist <= abs(r0 - r1) * (1.0 + 1e-12):
        # Seen down the axis: the larger end covers everything else.
        c, r = (c0, r0) if r0 >= r1 else (c1, r1)
        p, q = at(c, r, 0.0), at(c, r, math.pi)
        return [svg.M(p.x, p.y), arc(r, False, True, q), arc(r, False, True, p), svg.Z()]
    theta = math.atan2(float(d[1]), float(d[0]))
    alpha = math.acos(min(max((r0 - r1) / dist, -1.0), 1.0))
    # Both arcs run toward decreasing parameter angle, which is SVG's positive
    # sweep exactly when `axes` reverses orientation.
    sweep = det < 0
    start, back = at(c0, r0, theta + alpha), at(c0, r0, theta - alpha)
    if r1 > 0:
        near, far = at(c1, r1, theta + alpha), at(c1, r1, theta - alpha)
    else:
        near = far = svg.Point(_num(c1[0]), _num(c1[1]))
    far_arc = arc(r1, 2 * alpha > math.pi, sweep, far)
    near_arc = arc(r0, 2 * alpha < math.pi, sweep, start)
    commands: list[svg.PathData] = [svg.M(start.x, start.y), svg.L(near.x, near.y)]
    if closed:
        if r1 > 0:
            commands.append(far_arc)
        return [*commands, svg.L(back.x, back.y), near_arc, svg.Z()]
    commands.append(far_arc if arcs[1] and r1 > 0 else svg.M(far.x, far.y))
    commands.append(svg.L(back.x, back.y))
    if arcs[0]:
        commands.append(near_arc)
    return commands


type TextContent = str | Sequence[svg.TSpan]
"""A label: plain text, or ``svg.TSpan`` runs for subscripts and mixed styles."""


def _text_length(s: TextContent) -> int:
    """Characters in a label, for the nominal-width bounding-box estimate."""
    return len(s) if isinstance(s, str) else sum(len(run.text or "") for run in s)


def _named(base: str | None, suffix: str) -> dict[str, Style]:
    """``{"id": "{base}-{suffix}"}``, or nothing when there is no base id."""
    return {} if base is None else {"id": f"{base}-{suffix}"}


def _slug(color: str) -> str:
    """A colour reduced to characters that are safe in an XML id."""
    return re.sub(r"[^0-9A-Za-z]+", "", color).lower() or "c"


DEFAULT_FONT = "DejaVu Sans, Verdana, sans-serif"
DEFAULT_TEXT_FILL = "#222222"


class Scene:
    """An ordered collection of SVG elements built from world-space geometry.

    Draw order is an explicit integer ``layer`` per call, resolved stably by
    insertion order within a layer.  There is no z-buffer, and no layer is
    depth-sorted unless it asks to be with :meth:`sort_by_depth`: for a schematic
    with a beam passing through a translucent slab, deciding what occludes what
    by hand is worth more than getting it automatically and almost right, while
    for a lattice of hundreds of atoms the painter's algorithm is the only
    practical answer.  :meth:`Camera.visible` handles the one case where the
    answer is unambiguous -- the back faces of a convex solid.

    Args:
        cam: Camera used to project every world-space call.
        pad: Default margin added around the fitted content by :meth:`render`.
        background: Default background fill, or ``None`` for a transparent document.
    """

    def __init__(
        self,
        cam: Camera,
        *,
        pad: float = 26.0,
        background: str | None = None,
    ) -> None:
        self.cam = cam
        self.pad = float(pad)
        self.background = background
        self.items: list[tuple[int, int, svg.Element]] = []
        self.defs: list[svg.Element] = []
        self._log: list[tuple[str, tuple[object, ...], dict[str, Style]]] = []
        self._recording = True
        self._seq = 0
        self._lo: Array = np.array([np.inf, np.inf])
        self._hi: Array = np.array([-np.inf, -np.inf])
        self._depths: dict[int, float] = {}
        self._sorted: set[int] = set()
        self._shared: set[str] = set()

    def __repr__(self) -> str:
        return f"Scene({self.cam!r}, {len(self.items)} elements, {len(self.defs)} defs)"

    def _record(self, method: str, *args: object, **style: Style) -> None:
        """Log one call for :meth:`with_camera` to replay."""
        if self._recording:
            self._log.append((method, args, style))

    @contextmanager
    def _delegating(self) -> Iterator[None]:
        """Suppress logging while a public method fans out to other public ones.

        `faces` records itself and then calls `polygon`; without this the replay
        would draw both, doubling every face.
        """
        previous, self._recording = self._recording, False
        try:
            yield
        finally:
            self._recording = previous

    def with_camera(self, cam: Camera) -> Scene:
        """Return the same scene projected by a different camera.

        Everything added through this object's own methods is replayed against
        ``cam``, so one scene can be rendered isometric, dimetric, and cabinet
        without building it three times::

            scene = build_scene(OrthographicCamera(35, 24, 62))
            scene.with_camera(OrthographicCamera.isometric(62)).save("iso.svg")

        Two things behave as their names promise rather than as a reprojection
        might suggest.  Screen-space calls -- :meth:`rect2d`, :meth:`text2d`, and
        anything handed to :meth:`add` -- are replayed at the same screen
        coordinates, because that is what "screen space" means; they do not
        follow the geometry.  And geometry computed by the *caller* from a camera
        (a point placed via :meth:`Camera.at`, say) is baked in already, so a
        scene meant to be reprojected should take its camera as an argument and
        derive such positions inside.

        The world-space arrays handed to :meth:`polygon` and friends are held by
        reference, not copied, so do not mutate them after adding.
        """
        clone = Scene(cam, pad=self.pad, background=self.background)
        for method, args, style in self._log:
            getattr(clone, method)(*args, **style)
        return clone

    # --- bounds -----------------------------------------------------------
    def _grow(self, pts2: np.ndarray) -> None:
        p = np.atleast_2d(pts2)
        self._lo = np.minimum(self._lo, p.min(axis=0))
        self._hi = np.maximum(self._hi, p.max(axis=0))

    @property
    def is_empty(self) -> bool:
        """Whether anything contributing to the bounding box has been added."""
        return not bool(np.all(np.isfinite(self._lo)))

    def bbox(self) -> tuple[Array, Array]:
        """Screen-space bounds of everything added so far.

        Text extents are estimated from a nominal glyph width, which is enough
        to keep the fitted viewBox from clipping a label but is not exact.
        """
        return self._lo.copy(), self._hi.copy()

    # --- emission ---------------------------------------------------------
    def _emit(self, layer: int, element: svg.Element, depth: float | None = None) -> None:
        """Queue an element at ``layer`` without logging it.

        ``depth`` is what a layer passed to :meth:`sort_by_depth` orders by.  It
        is rounded so that equal depths tie exactly and fall back to insertion
        order, rather than to the last bits of a floating-point sum.
        """
        if depth is not None:
            self._depths[self._seq] = _num(depth, 6)
        self.items.append((int(layer), self._seq, element))
        self._seq += 1

    def _depth(self, pts3: Points3) -> float:
        """Mean depth of world points: the key one element sorts by."""
        return float(np.mean(self.cam.depth(pts3)))

    def sort_by_depth(self, layer: int) -> None:
        """Order ``layer`` back to front by depth instead of by insertion.

        The painter's algorithm, opted into one layer at a time.  Every
        world-space element drawn at ``layer`` is keyed by the mean depth of
        the points that produced it -- a sphere by its centre, a cylinder by its
        axis midpoint, a polygon by its vertices -- and drawn farthest first.
        Equal depths keep insertion order, and screen-space elements, which have
        no depth, go on top in insertion order.  Other layers are untouched.

        This is for many separate objects that do not interpenetrate: the atoms
        and bonds of a lattice, the arrows of a spin texture, the quads of a
        surface.  It is a heuristic, exact for non-overlapping spheres of one
        radius and good for small, similar pieces, and it cannot order two long
        objects that each cover part of the other.  That case, and a beam inside
        a translucent slab, still belong on separate layers.

        The setting is recorded, so :meth:`with_camera` re-sorts for the new
        camera.
        """
        self._record("sort_by_depth", layer)
        self._sorted.add(int(layer))

    def add(self, layer: int, element: svg.Element) -> None:
        """Add a ready-made ``svg.py`` element at ``layer``, bypassing projection."""
        self._record("add", layer, element)
        self._emit(layer, element)

    def add_def(self, element: svg.Element) -> None:
        """Add an element to ``<defs>`` -- a gradient, marker, or clip path."""
        self._record("add_def", element)
        self.defs.append(element)

    # --- world-space primitives -------------------------------------------
    def polygon(self, layer: int, pts3: Points3, **style: Style) -> None:
        """Filled polygon through projected world points."""
        self._record("polygon", layer, pts3, **style)
        p = self.cam.project(pts3)
        self._grow(p)
        self._emit(layer, svg.Polygon(points=_points(p), **style), self._depth(pts3))

    def polyline(self, layer: int, pts3: Points3, **style: Style) -> None:
        """Open path through projected world points; unfilled unless asked."""
        style.setdefault("fill", "none")
        self._record("polyline", layer, pts3, **style)
        p = self.cam.project(pts3)
        self._grow(p)
        self._emit(layer, svg.Polyline(points=_points(p), **style), self._depth(pts3))

    def faces(
        self, layer: int, faces: Iterable[Face], *, cull: bool = False, **style: Style
    ) -> None:
        """Draw each face as a polygon, all with the same style.

        Args:
            layer: Draw order.
            faces: Faces to draw, typically from :func:`~vecview.shapes.box_faces`.
            cull: Drop back faces at draw time, using this scene's camera.
                Prefer this over filtering with :meth:`Camera.visible` yourself
                whenever the scene might be reprojected: culling done by the
                caller bakes in *that* camera's answer, so
                :meth:`with_camera` would keep the original walls and quietly
                draw the wrong ones. With ``cull=True`` the full set is recorded
                and the new camera decides.
            **style: SVG presentation attributes, shared by every face.

        An ``id`` is suffixed per face rather than repeated, since duplicate ids
        are invalid SVG and break selection downstream: ``id="slab"`` over a
        box's visible walls yields ``slab-pz``, ``slab-px``, ``slab-py``.
        """
        base = style.pop("id", None)
        given = list(faces)
        self._record(
            "faces",
            layer,
            given,
            cull=cull,
            **({} if base is None else {"id": base}),
            **style,
        )
        drawn = self.cam.visible(given) if cull else given
        with self._delegating():
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
        """Reserve an empty group occupying a rectangle of a world plane.

        The group carries the affine transform that maps content coordinates in
        :math:`[0,1]^2` onto the projected rectangle, so flat SVG content sits
        *in* the plane -- correctly foreshortened and sheared -- rather than on
        top of the picture.  Nothing is drawn: this package does not parse or
        embed foreign SVG.  A consumer fills the group by ``id``, normalizing
        its content to the unit square.

        Because the group participates in the layer stack like anything else,
        content placed in it can be drawn over the slab it lies on and under the
        beam that crosses it.  The rectangle also grows the fitted viewBox, so
        the embedded content is never clipped.

        Args:
            layer: Draw order, as for every other primitive.
            origin: World point for content ``(0, 0)`` -- the top-left corner,
                since SVG ``y`` grows downward.
            u_edge: World vector along content ``+x``; its length is the width.
            v_edge: World vector along content ``+y`` (downward); its length is
                the height.
            id: Handle a consumer uses to find and fill the group. Required.

        See :meth:`Camera.plane_matrix` for the geometry, including why this is
        exact for an orthographic camera and would not be for a perspective one.
        """
        cam = _parallel(self.cam, "embed content in a plane")
        self._record("plane", layer, origin, u_edge, v_edge, id=id, **style)
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
        """Reserve an empty, screen-aligned group anchored at a projected world point.

        The screen-space sibling of :meth:`plane`: where a plane makes content
        lie *in* the scene, a slot keeps it upright and unforeshortened -- a
        label, an equation, an inset -- while pinning it to a point of the
        geometry.  Nothing is drawn, and this package does not embed foreign
        SVG; a consumer fills the group by ``id``.

        The group is translated to the anchor, ``cam.at(pt3)`` offset by
        ``(dx, dy)``, and records ``align`` as ``data-align``, so a consumer can
        line its content up against the anchor at whatever size it ends up.  A
        ``w`` by ``h`` box aligned the same way grows the fitted viewBox, so
        content of that size is not clipped.  Under :meth:`with_camera` the
        anchor is reprojected, so the slot follows the geometry.

        Args:
            layer: Draw order, so geometry on a higher layer can cover the content.
            pt3: World point the slot is pinned to.
            w: Width of the box to reserve, in scene units.
            h: Height of the box to reserve, in scene units.
            id: Handle a consumer uses to find and fill the group. Required.
            align: Which point of the box sits on the anchor: ``"west"`` puts
                the anchor at the middle of the box's left edge, so the content
                extends to the right.
            dx: Screen offset of the anchor, in scene units.
            dy: Screen offset of the anchor, in scene units, downward.
        """
        if align not in _ALIGN:
            raise ValueError(f"unknown align {align!r}; expected one of {sorted(_ALIGN)}")
        self._record("slot", layer, pt3, w, h, id=id, align=align, dx=dx, dy=dy, **style)
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
        """Fill the projected outline of a convex solid as one polygon.

        Drawing a convex solid's visible walls one polygon each leaves hairline
        seams where neighbouring walls meet, because each edge is anti-aliased
        against the background separately.  Every visible wall lies inside the
        silhouette, so filling the silhouette in the wall colour and drawing the
        cap over it gives the same picture with no seams::

            fin = prism_faces(footprint, 0.0, 0.3)
            scene.silhouette(30, fin, fill="#b98a40", id="lead-walls")
            scene.faces(30, [f for f in fin if f.name == "+z"], fill="#e2b56a")

        The outline is the convex hull of the projected vertices, so it is only
        the silhouette of a convex solid.  It is recomputed by :meth:`with_camera`.

        Args:
            layer: Draw order.
            solid: Faces of the solid, or its world-space vertices.
            **style: SVG presentation attributes of the polygon.
        """
        given = list(solid)
        self._record("silhouette", layer, given, **style)
        if given and isinstance(given[0], Face):
            pts = np.vstack([face.points for face in given if isinstance(face, Face)])
        else:
            pts = as_points(given)
        p = self.cam.project(pts)
        outline = p[convex_hull(p)]
        self._grow(outline)
        self._emit(layer, svg.Polygon(points=_points(outline), **style), self._depth(pts))

    def prism_walls(
        self, layer: int, footprint: Points2, z0: float, z1: float, **style: Style
    ) -> None:
        """Draw the camera-facing walls of an extruded footprint as one seamless shape.

        Each maximal run of consecutive facing walls becomes one strip -- along
        the base, then back along the top -- so no seam shows between
        neighbouring walls, however finely a curved footprint is faceted.  All
        strips go into a single ``<path>``.  Draw the cap over it::

            gate = annulus_sector((0, 0), 2.6, 3.0, 20, 160)
            scene.prism_walls(30, gate, 0.0, 0.26, fill="#4a5059", id="gate-walls")
            scene.faces(30, prism_faces(gate, 0.0, 0.26)[:1], fill="#737a84", id="gate")

        Works for non-convex footprints, which :meth:`silhouette` does not.  With
        an unstroked wall style and a cap facing the camera, walls-then-cap is
        exact at any height: along any view ray the cap is never behind a wall,
        and walls hiding other walls of the same solid are indistinguishable
        when they share one fill.  Only a stroke can show an edge of a wall that
        another wall of the same solid hides -- negligible for a low extrusion,
        visible for a tall non-convex one.  There is no depth sort to fix that,
        by design.

        Args:
            layer: Draw order.
            footprint: Simple polygon, as for :func:`~vecview.shapes.prism_faces`.
            z0: Height of the base.
            z1: Height of the top.
            **style: SVG presentation attributes of the path.

        Culling uses this scene's camera, so :meth:`with_camera` redraws the
        walls the new camera faces.
        """
        self._record("prism_walls", layer, footprint, z0, z1, **style)
        walls = prism_faces(footprint, z0, z1)[2:]
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
        """Flat arrow, as :func:`~vecview.shapes.arrow_shape`, drawn as a polygon.

        ``normal="camera"`` turns the arrow about its own axis to show the
        widest face it can, so it reads from any viewpoint -- a spin along
        ``z``, say.  The normal is resolved from this scene's camera when drawn,
        so :meth:`with_camera` turns the arrow to the new camera too.  Computing
        that normal yourself from ``cam.view`` would bake in the original one.

        Args:
            normal: Normal of the plane the arrow lies flat in, or ``"camera"``.
                The latter needs a :class:`ParallelCamera`.

        Raises:
            ValueError: If ``normal="camera"`` and the arrow points along the
                projection ray, where it has no face to show.

        The remaining arguments are those of :func:`~vecview.shapes.arrow_shape`.
        """
        self._record(
            "arrow",
            layer,
            origin,
            direction,
            length,
            normal=normal,
            shaft_w=shaft_w,
            head_w=head_w,
            head_len=head_len,
            pivot=pivot,
            **style,
        )
        if isinstance(normal, str):
            if normal != "camera":
                raise ValueError(f"normal must be a vector or 'camera', got {normal!r}")
            normal = self._facing_normal(direction)
        shape = arrow_shape(origin, direction, length, normal, shaft_w, head_w, head_len, pivot)
        with self._delegating():
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
        """A soft Gaussian spot lying in a world plane, as one gradient-filled polygon.

        The opacity follows ``exp(-(s/a)**2 - (t/b)**2)`` along the in-plane
        axes ``u`` and ``v``, so ``a`` and ``b`` are the ``1/e`` half-widths.  A
        radial gradient mapped through the plane's affine transform does this in
        one element, foreshortened with the plane, where nested translucent
        ellipses would take many and show their steps.

        The profile is shifted to reach exactly zero at ``extent`` half-widths,
        where the polygon ends, so there is no visible rim.  The gradient goes
        into ``<defs>`` as ``{id}-profile``.

        Args:
            layer: Draw order.
            center: Centre of the spot.
            u: Direction of the ``a`` axis.
            v: Direction of the ``b`` axis; perpendicular to ``u``.
            a: ``1/e`` half-width along ``u``.
            b: ``1/e`` half-width along ``v``.
            id: Id of the polygon. Required, since the gradient id derives from it.
            color: Fill colour.
            opacity: Opacity at the centre.
            extent: Radius drawn, in half-widths.
            stops: Gradient stops sampling the profile; more is smoother.

        Raises:
            TypeError: For a camera that is not a :class:`ParallelCamera`; only
                an affine projection maps a gradient through a plane exactly.

        Note that ``cairosvg`` renders a gradient *fill* correctly; it is a
        gradient *mask* that it silently drops.
        """
        cam = _parallel(self.cam, "map a gradient through a plane")
        if stops < 2:
            raise ValueError("a gradient needs at least two stops")
        self._record(
            "gaussian",
            layer,
            center,
            u,
            v,
            a,
            b,
            id=id,
            color=color,
            opacity=opacity,
            extent=extent,
            stops=stops,
            **style,
        )
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
        with self._delegating():
            self.polygon(layer, outline, fill=f"url(#{id}-profile)", id=id, **style)

    # --- curved solids ----------------------------------------------------
    def sphere(
        self,
        layer: int,
        center: Point3,
        radius: float,
        *,
        highlight: str | None = None,
        **style: Style,
    ) -> None:
        """A sphere, drawn as its exact outline: one ``<circle>`` or ``<ellipse>``.

        A parallel projection maps a sphere onto an ellipse -- a circle for an
        orthographic camera, an ellipse for an oblique one -- so the outline is
        exact and a single native shape, not a polygon.

        Args:
            layer: Draw order.
            center: Centre of the sphere.
            radius: Radius, in world units.
            highlight: A lighter colour for a glossy ball: the ``fill`` becomes
                a radial gradient from ``highlight`` near the upper left to
                ``fill`` at the rim.  This is a fill style, not a lighting model
                -- the highlight sits in the same place on screen whatever the
                camera.  One gradient per colour pair is shared by every sphere
                that uses it, under the id ``ball-{fill}-{highlight}``.
            **style: SVG presentation attributes; ``fill`` is required with
                ``highlight``.
        """
        cam = _parallel(self.cam, "outline a sphere")
        self._record("sphere", layer, center, radius, highlight=highlight, **style)
        c = np.asarray(self.cam.at(center))
        axes = _ball_axes(cam, radius)
        if highlight is not None:
            style["fill"] = self._ball_gradient(style, highlight)
        reach = np.linalg.norm(axes, axis=1)
        self._grow(np.array([c - reach, c + reach]))
        self._emit(layer, _ellipse_element(c, axes, **style), self._depth([center]))

    def _ball_gradient(self, style: Mapping[str, Style], highlight: str) -> str:
        """Add (once) the radial gradient shared by spheres of one colour pair."""
        fill = style.get("fill")
        if not isinstance(fill, str):
            raise ValueError("highlight needs a fill colour to shade toward")
        gid = f"ball-{_slug(fill)}-{_slug(highlight)}"
        if gid not in self._shared:
            self._shared.add(gid)
            self.defs.append(
                svg.RadialGradient(
                    id=gid,
                    cx=0.5,
                    cy=0.5,
                    r=0.5,
                    fx=0.36,
                    fy=0.32,
                    elements=[
                        svg.Stop(offset=0, stop_color=highlight),
                        svg.Stop(offset=1, stop_color=fill),
                    ],
                )
            )
        return f"url(#{gid})"

    def _frustum(
        self,
        p0: Point3,
        p1: Point3,
        r0: float,
        r1: float,
        *,
        ends: tuple[bool, bool],
        end_style: Mapping[str, Style] | None,
        highlight: str | None,
        name: str | None,
        style: Mapping[str, Style],
    ) -> list[svg.Element]:
        """Body outline plus whichever end disks face the camera, as elements.

        ``ends`` says which of the two end disks may be drawn at all; each is
        then drawn only if it faces the camera, after the body, which is exact
        for a convex solid.
        """
        cam = _parallel(self.cam, "outline a solid of revolution")
        if r0 <= 0 or r1 < 0:
            raise ValueError(f"need r0 > 0 and r1 >= 0, got r0={r0}, r1={r1}")
        a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        axis = b - a
        if float(np.linalg.norm(axis)) == 0.0:
            raise ValueError("the two ends coincide, so the solid has no axis")
        axes = cam.scale * cam.matrix @ np.column_stack(basis_for(axis))
        c0, c1 = cam.project([a, b])
        reach = np.linalg.norm(axes, axis=1)
        self._grow(np.array([c0 - r0 * reach, c0 + r0 * reach, c1 - r1 * reach, c1 + r1 * reach]))

        body: dict[str, Style] = dict(style)
        if highlight is not None:
            fill = body.get("fill")
            if not isinstance(fill, str) or name is None:
                raise ValueError("highlight needs a fill colour and an id to name the gradient")
            body["fill"] = f"url(#{name}-shade)"
            self.defs.append(self._shade_across(f"{name}-shade", c0, c1, axes, r0, fill, highlight))
        label: dict[str, Style] = {} if name is None else {"id": name}
        elements: list[svg.Element] = [
            svg.Path(d=_frustum_outline(c0, c1, axes, r0, r1), **label, **body)
        ]
        disk: dict[str, Style] = {**style, **(end_style or {})}
        toward = float(np.dot(unit(axis), cam.view))
        for k, (c, r, faces) in enumerate(((c0, r0, toward < 0), (c1, r1, toward > 0))):
            if ends[k] and faces and r > 0:
                end: dict[str, Style] = {} if name is None else {"id": f"{name}-end{k}"}
                elements.append(_ellipse_element(c, r * axes, **end, **disk))
        return elements

    def _sliced_cylinder(
        self,
        layer: int,
        p0: Point3,
        p1: Point3,
        r0: float,
        r1: float,
        *,
        slices: int,
        ends: bool,
        end_style: Mapping[str, Style] | None,
        highlight: str | None,
        base: str | None,
        style: Mapping[str, Style],
    ) -> None:
        """:meth:`cylinder` cut into ``slices`` separately sorted lengths."""
        cam = _parallel(self.cam, "outline a solid of revolution")
        if r0 <= 0 or r1 < 0:
            raise ValueError(f"need r0 > 0 and r1 >= 0, got r0={r0}, r1={r1}")
        a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        axis = b - a
        if float(np.linalg.norm(axis)) == 0.0:
            raise ValueError("the two ends coincide, so the solid has no axis")
        axes = cam.scale * cam.matrix @ np.column_stack(basis_for(axis))
        reach = np.linalg.norm(axes, axis=1)
        c0, c1 = cam.project([a, b])
        self._grow(np.array([c0 - r0 * reach, c0 + r0 * reach, c1 - r1 * reach, c1 + r1 * reach]))

        fill: dict[str, Style] = {k: v for k, v in style.items() if not k.startswith("stroke")}
        edge: dict[str, Style] = {k: v for k, v in style.items() if k.startswith("stroke")}
        if highlight is not None:
            colour = fill.get("fill")
            if not isinstance(colour, str) or base is None:
                raise ValueError("highlight needs a fill colour and an id to name the gradient")
            fill["fill"] = f"url(#{base}-shade)"
            self.defs.append(
                self._shade_across(f"{base}-shade", c0, c1, axes, max(r0, r1), colour, highlight)
            )
        disk: dict[str, Style] = {**style, **(end_style or {})}
        toward = float(np.dot(unit(axis), cam.view))
        # Each slice's body overruns into its neighbours, so no seam shows at a
        # joint whichever of the two is drawn on top.
        overrun = 0.15 / slices
        cuts = np.linspace(0.0, 1.0, slices + 1)

        def at(t: float) -> tuple[Array, float]:
            return a + t * axis, r0 + t * (r1 - r0)

        for k in range(slices):
            lo, hi = max(cuts[k] - overrun, 0.0), min(cuts[k + 1] + overrun, 1.0)
            (pa, ra), (pb, rb) = at(lo), at(hi)
            qa, qb = cam.project([pa, pb])
            first, last = k == 0, k == slices - 1
            parts: list[svg.Element] = [
                svg.Path(
                    d=_frustum_outline(qa, qb, axes, ra, rb),
                    **_named(base, f"{k}-body"),
                    **fill,
                )
            ]
            if edge.get("stroke"):
                parts.append(
                    svg.Path(
                        d=_frustum_outline(qa, qb, axes, ra, rb, closed=False, arcs=(first, last)),
                        fill="none",
                        **_named(base, f"{k}-edge"),
                        **edge,
                    )
                )
            for cap, centre, r, faces in ((first, qa, ra, toward < 0), (last, qb, rb, toward > 0)):
                if ends and cap and faces and r > 0:
                    end = "0" if centre is qa else "1"
                    label = _named(base, f"end{end}")
                    parts.append(_ellipse_element(centre, r * axes, **label, **disk))
            group = svg.G(elements=parts, **_named(base, str(k)))
            self._emit(layer, group, self._depth([at(cuts[k])[0], at(cuts[k + 1])[0]]))

    @staticmethod
    def _shade_across(
        gid: str, c0: Array, c1: Array, axes: Array, r: float, fill: str, highlight: str
    ) -> svg.Element:
        """A linear gradient across a projected solid, lightest toward the upper left."""
        along = c1 - c0
        length = float(np.linalg.norm(along))
        side = np.array([-along[1], along[0]]) / length if length > 1e-9 else np.array([-1.0, -1.0])
        side = side / float(np.linalg.norm(side))
        if side[0] + side[1] > 0:
            side = -side
        half = r * float(np.linalg.norm(axes.T @ side))
        mid = (c0 + c1) / 2.0
        (x1, y1), (x2, y2) = mid + half * side, mid - half * side
        return svg.LinearGradient(
            id=gid,
            gradientUnits="userSpaceOnUse",
            x1=_num(x1),
            y1=_num(y1),
            x2=_num(x2),
            y2=_num(y2),
            elements=[
                svg.Stop(offset=0, stop_color=fill),
                svg.Stop(offset=0.3, stop_color=highlight),
                svg.Stop(offset=1, stop_color=fill),
            ],
        )

    def cylinder(
        self,
        layer: int,
        p0: Point3,
        p1: Point3,
        radius: float,
        *,
        r1: float | None = None,
        ends: bool = True,
        end_style: Mapping[str, Style] | None = None,
        highlight: str | None = None,
        slices: int = 1,
        **style: Style,
    ) -> None:
        """A cylinder from ``p0`` to ``p1``, or a frustum when ``r1`` differs.

        The outline is exact and compact: two straight sides and two elliptical
        arcs in one ``<path>``, since a parallel projection maps the end circles
        to ellipses whose common tangents have a closed form.  The end disk that
        faces the camera, if any, is drawn over the body as a native
        ``<ellipse>`` (or ``<circle>``).  Everything goes into one ``<g>``, so
        the solid is one object in Inkscape and one element for
        :meth:`sort_by_depth`, keyed by its axis midpoint.

        Args:
            layer: Draw order.
            p0: Centre of the first end.
            p1: Centre of the second end.
            radius: Radius at ``p0``.
            r1: Radius at ``p1``; defaults to ``radius``. ``0`` is a cone, for
                which :meth:`cone` reads better.
            ends: Whether to draw the end disks.  ``False`` leaves an open tube,
                which is what a bond hidden inside two atoms wants.
            end_style: Presentation attributes for the end disks, over
                ``style`` -- a lighter ``fill`` for a flat-shaded cap, say.
            highlight: A lighter colour shading the body across its width,
                lightest toward the upper left: a fill style, not a lighting
                model.  Needs ``fill`` and ``id``; the gradient is ``{id}-shade``.
            slices: Cut the solid into this many lengths along its axis, each
                its own ``<g>`` keyed by its own midpoint, for a long cylinder
                in a layer passed to :meth:`sort_by_depth`.  Keyed by its
                centre alone, a core with a coil wound round it sorts wholly in
                front of the far turns and wholly behind the near ones; sliced,
                each turn meets the slice it wraps.  The slices overlap a little
                and the outline is stroked along the sides only, so the result
                looks like one solid -- except with a translucent fill, where
                the overlaps show.
            **style: SVG presentation attributes. An ``id`` goes on the group;
                the body takes ``{id}-body`` and the end disks ``{id}-body-end0``
                and ``{id}-body-end1``.  Sliced, the groups are ``{id}-0``,
                ``{id}-1``, ... with ``{id}-{k}-body`` and ``{id}-{k}-edge``.

        Raises:
            ValueError: If ``p0 == p1``, or for a negative radius.
        """
        self._record(
            "cylinder",
            layer,
            p0,
            p1,
            radius,
            r1=r1,
            ends=ends,
            end_style=end_style,
            highlight=highlight,
            slices=slices,
            **style,
        )
        base = style.pop("id", None)
        if slices < 1:
            raise ValueError(f"slices must be at least 1, got {slices}")
        if slices > 1:
            self._sliced_cylinder(
                layer,
                p0,
                p1,
                radius,
                radius if r1 is None else r1,
                slices=slices,
                ends=ends,
                end_style=end_style,
                highlight=highlight,
                base=base,
                style=style,
            )
            return
        elements = self._frustum(
            p0,
            p1,
            radius,
            radius if r1 is None else r1,
            ends=(ends, ends),
            end_style=end_style,
            highlight=highlight,
            name=None if base is None else f"{base}-body",
            style=style,
        )
        group = svg.G(elements=elements, **({} if base is None else {"id": base}))
        self._emit(layer, group, self._depth([p0, p1]))

    def cone(
        self,
        layer: int,
        base: Point3,
        apex: Point3,
        radius: float,
        *,
        end: bool = True,
        end_style: Mapping[str, Style] | None = None,
        highlight: str | None = None,
        **style: Style,
    ) -> None:
        """A cone from a disk of ``radius`` at ``base`` to a point at ``apex``.

        :meth:`cylinder` with ``r1=0``, and the same exact outline: the two
        tangents from the apex and one elliptical arc.  ``end`` draws the base
        disk when it faces the camera.
        """
        self._record(
            "cone",
            layer,
            base,
            apex,
            radius,
            end=end,
            end_style=end_style,
            highlight=highlight,
            **style,
        )
        with self._delegating():
            self.cylinder(
                layer,
                base,
                apex,
                radius,
                r1=0.0,
                ends=end,
                end_style=end_style,
                highlight=highlight,
                **style,
            )

    def arrow3d(
        self,
        layer: int,
        origin: Point3,
        direction: Point3,
        length: float,
        *,
        shaft_r: float,
        head_r: float,
        head_len: float,
        pivot: Pivot = "tail",
        end_style: Mapping[str, Style] | None = None,
        highlight: str | None = None,
        **style: Style,
    ) -> None:
        """A solid arrow: a cylindrical shaft and a conical head, in one ``<g>``.

        Unlike the flat :meth:`arrow` it reads as a solid from every side, so it
        suits a field of spins seen at an angle.  The shaft and head are ordered
        within the group by which end is nearer the camera, and the disks that
        face it -- the back of the head, the tail of the shaft -- are drawn.

        Args:
            origin: Tail of the arrow, or its midpoint when ``pivot="mid"``.
            direction: Direction the head points.
            length: Total length, tail to tip.
            shaft_r: Radius of the shaft.
            head_r: Radius of the head at its base.
            head_len: Length of the head, measured back from the tip.
            end_style: Presentation attributes for the visible disks.
            highlight: Shade both parts as :meth:`cylinder` does. Needs an
                ``id``; the parts are ``{id}-shaft`` and ``{id}-head``.
            **style: SVG presentation attributes.

        Raises:
            ValueError: For a zero ``direction``.
        """
        self._record(
            "arrow3d",
            layer,
            origin,
            direction,
            length,
            shaft_r=shaft_r,
            head_r=head_r,
            head_len=head_len,
            pivot=pivot,
            end_style=end_style,
            highlight=highlight,
            **style,
        )
        d = unit(direction)
        if not d.any():
            raise ValueError("an arrow needs a non-zero direction")
        base = style.pop("id", None)
        tail = np.asarray(origin, dtype=np.float64) - (d * length / 2.0 if pivot == "mid" else 0.0)
        tip = tail + d * length
        neck = tail + d * max(length - head_len, 0.0)

        def piece(a: Array, b: Array, r0: float, r1: float, part: str) -> list[svg.Element]:
            return self._frustum(
                a,
                b,
                r0,
                r1,
                ends=(True, False),
                end_style=end_style,
                highlight=highlight,
                name=None if base is None else f"{base}-{part}",
                style=style,
            )

        head = piece(neck, tip, head_r, 0.0, "head")
        shaft = piece(tail, neck, shaft_r, shaft_r, "shaft") if length > head_len else []
        toward = float(np.dot(d, _parallel(self.cam, "draw a solid arrow").view)) >= 0
        elements = shaft + head if toward else head + shaft
        group = svg.G(elements=elements, **({} if base is None else {"id": base}))
        self._emit(layer, group, self._depth([tail, tip]))

    def tube(
        self,
        layer: int,
        pts3: Points3,
        radius: float,
        *,
        chunk: int = 4,
        **style: Style,
    ) -> None:
        """A tube of ``radius`` along a world-space curve -- a coil, a field line, a bond path.

        Drawn as wide strokes rather than as a mesh: for an orthographic
        camera, a tube projects to the curve thickened by the projected radius,
        which a round-joined stroke is exactly.  ``fill`` is the colour of the
        tube and ``stroke`` its outline, drawn as a wider stroke underneath, so
        the call reads like any filled shape.  For an oblique camera the width
        is the mean of the projected radius over directions, an approximation.

        The curve is cut into pieces of ``chunk`` segments, one ``<g>`` each, so
        in a layer passed to :meth:`sort_by_depth` a tube can pass over and
        under itself and others, as a coil does.  Neighbouring pieces overlap,
        and the outline of each stops a segment short of its body, so no seam
        shows where they meet.  The tube's own two ends are
        square.

        Args:
            layer: Draw order.
            pts3: The centre line, shape ``(n, 3)``, with ``n >= 2``.
            radius: Tube radius, in world units.
            chunk: Segments per piece; shorter sorts more finely.
            **style: ``fill`` (default black) and ``stroke`` (default none)
                colour the tube and its outline, ``stroke_width`` (default
                ``1``) is the outline width, and anything else -- ``opacity``,
                say -- goes on each piece.  An ``id`` is suffixed per piece:
                ``coil-0``, ``coil-1``, ...
        """
        cam = _parallel(self.cam, "draw a tube")
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        self._record("tube", layer, pts3, radius, chunk=chunk, **style)
        pts = as_points(pts3)
        if len(pts) < 2:
            raise ValueError("a tube needs at least two points")
        body = style.pop("fill", "#000000")
        edge = style.pop("stroke", None)
        edge_w = float(style.pop("stroke_width", 1.0))
        base = style.pop("id", None)
        s = np.linalg.svd(cam.matrix, compute_uv=False) * cam.scale
        width = 2.0 * radius * math.sqrt(float(s[0] * s[1]))
        p = cam.project(pts)
        margin = width / 2.0 + (edge_w if edge else 0.0)
        self._grow(np.vstack([p - margin, p + margin]))
        last = len(pts) - 1
        line: dict[str, Style] = dict(fill="none", stroke_linecap="butt", stroke_linejoin="round")
        for k, start in enumerate(range(0, last, chunk)):
            # Outlines of neighbouring pieces overlap by a segment, so their
            # square ends leave no notch on the outside of a bend, and each body
            # overruns its own outline by a segment at either inner end, so an
            # outline's square end lies under body colour rather than leaving an
            # anti-aliased seam across the tube.
            stop = min(start + chunk + 3, last)
            strokes: list[svg.Element] = []
            if edge:
                inner = p[(start + 1 if start else 0) : min(start + chunk + 2, last) + 1]
                strokes.append(
                    svg.Polyline(
                        points=_points(inner),
                        stroke=edge,
                        stroke_width=_num(width + 2 * edge_w, 3),
                        **line,
                    )
                )
            strokes.append(
                svg.Polyline(
                    points=_points(p[start : stop + 1]),
                    stroke=body,
                    stroke_width=_num(width, 3),
                    **line,
                )
            )
            label: dict[str, Style] = {} if base is None else {"id": f"{base}-{k}"}
            self._emit(
                layer,
                svg.G(elements=strokes, **label, **style),
                self._depth(pts[start : stop + 1]),
            )

    # --- hidden lines -----------------------------------------------------
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
        """The edges of a convex solid, split into visible and hidden.

        An edge is visible when either face it bounds faces the camera, which is
        exact for a convex solid: a unit cell, a Brillouin zone, a coordination
        polyhedron.  Visible edges go into one ``<path>`` with ``style``.
        Hidden ones are dropped unless ``back`` is given, in which case they
        form a second path styled ``{**style, **back}`` -- the crystallographer's
        dashed back edges::

            cell = vecview.box_faces((0, 0, 0), (1, 1, 1))
            scene.edges(40, cell, back={"stroke_dasharray": "4 3"}, stroke="#333")

        Args:
            layer: Draw order of the visible edges.
            faces: Faces of the solid, sharing vertices exactly where they meet.
            back: Style overrides for the hidden edges, or ``None`` to omit them.
            back_layer: Draw order of the hidden edges; defaults to ``layer``.
                Put it below a translucent solid's faces so they veil it.
            separate: Emit one ``<path>`` per edge, each keyed by its own
                midpoint, so that in a layer passed to :meth:`sort_by_depth`
                the edges interleave with other objects -- the cell edges of a
                crystal among its atoms.
            trim: Shorten every edge by this much, in world units, at both
                ends, so it stops at the surface of an atom or marker sitting
                on each vertex.  A line through the centre of a sphere cannot
                be depth-sorted against it, since part of it is inside; one
                that starts on the surface can.
            **style: SVG presentation attributes; ``fill`` defaults to none.
                An ``id`` becomes ``{id}-front`` and ``{id}-back``, suffixed
                ``-0``, ``-1``, ... per edge when ``separate``.

        Culling uses this scene's camera, so :meth:`with_camera` re-splits.
        """
        given = list(faces)
        base = style.pop("id", None)
        style.setdefault("fill", "none")
        self._record(
            "edges",
            layer,
            given,
            back=back,
            back_layer=back_layer,
            separate=separate,
            trim=trim,
            **({} if base is None else {"id": base}),
            **style,
        )
        facing = {id(face) for face in self.cam.visible(given)}
        ends: dict[tuple[tuple[float, ...], ...], Array] = {}
        shown: dict[tuple[tuple[float, ...], ...], bool] = {}
        for face in given:
            ring = np.asarray(face.points, dtype=np.float64)
            for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True):
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
        """A curve on a sphere's surface, split where it passes behind the sphere.

        For the equator and meridians of a Bloch sphere or a globe.  A point
        ``p`` on the sphere about ``center`` faces the camera when
        ``(p - center) . view >= 0``; the curve is cut exactly where that
        changes sign.  The visible runs form one ``<path>`` with ``style``;
        the hidden ones are dropped unless ``back`` is given, as for
        :meth:`edges`.

        Args:
            layer: Draw order of the visible part.
            center: Centre of the sphere the curve lies on.
            pts3: The curve, shape ``(n, 3)``; :func:`~vecview.shapes.circle_shape`
                and :func:`~vecview.shapes.arc_shape` give the usual ones.
            closed: Join the last point back to the first, as for a circle.
            back: Style overrides for the hidden part, or ``None`` to omit it.
            back_layer: Draw order of the hidden part; defaults to ``layer``.
            **style: SVG presentation attributes; ``fill`` defaults to none.
                An ``id`` becomes ``{id}-front`` and ``{id}-back``.
        """
        cam = _parallel(self.cam, "split a curve at a sphere's outline")
        style.setdefault("fill", "none")
        self._record(
            "sphere_curve",
            layer,
            center,
            pts3,
            closed=closed,
            back=back,
            back_layer=back_layer,
            **style,
        )
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
        """Text anchored at a projected world point, offset by ``(dx, dy)`` on screen.

        ``s`` is a string, or a sequence of ``svg.TSpan`` runs for mixed styling
        such as a subscript::

            scene.text(30, tip, [svg.TSpan(text="k"), svg.TSpan(text="x", baseline_shift="sub")])
        """
        self._record("text", layer, pt3, s, dx, dy, size, **style)
        x, y = self.cam.at(pt3)
        self._place_text(layer, x + dx, y + dy, s, size, True, self._depth([pt3]), **style)

    # --- screen-space primitives ------------------------------------------
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
        """Rectangle in screen coordinates -- a backdrop, glow, or gradient wash.

        Excluded from the bounding box by default, so a soft glow extending past
        the geometry does not inflate the fitted viewBox.
        """
        self._record("rect2d", layer, x, y, w, h, grow, **style)
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
        self._record("text2d", layer, x, y, s, size, grow, **style)
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
        self._emit(
            layer,
            svg.Text(
                x=round(x, 2),
                y=round(y, 2),
                font_size=size,
                **({"text": s} if isinstance(s, str) else {"elements": list(s)}),
                **style,
            ),
            depth,
        )

    # --- output -----------------------------------------------------------
    def render(self, pad: float | None = None, background: str | None = None) -> svg.SVG:
        """Assemble the document, fitting the viewBox to the content.

        Nothing has to be centred by hand: the viewBox follows the geometry.

        Args:
            pad: Margin around the content; defaults to the scene's ``pad``.
            background: Background fill; defaults to the scene's ``background``.

        Raises:
            ValueError: If the scene has nothing to fit a viewBox to.
        """
        if self.is_empty:
            raise ValueError("cannot render an empty scene: no geometry to fit a viewBox to")
        p = self.pad if pad is None else float(pad)
        bg = self.background if background is None else background
        lo, hi = self._lo - p, self._hi + p
        w, h = hi - lo
        elements: list[svg.Element] = []
        if self.defs:
            elements.append(svg.Defs(elements=list(self.defs)))
        if bg:
            elements.append(svg.Rect(x=lo[0], y=lo[1], width=w, height=h, fill=bg))
        elements += [el for _, _, el in sorted(self.items, key=self._order)]
        return svg.SVG(
            width=round(w, 1),
            height=round(h, 1),
            viewBox=svg.ViewBoxSpec(round(lo[0], 1), round(lo[1], 1), round(w, 1), round(h, 1)),
            elements=elements,
        )

    def _order(self, item: tuple[int, int, svg.Element]) -> tuple[int, bool, float, int]:
        """Sort key: layer, then depth within a depth-sorted layer, then insertion."""
        layer, seq, _ = item
        if layer not in self._sorted:
            return (layer, False, 0.0, seq)
        depth = self._depths.get(seq)
        return (layer, depth is None, 0.0 if depth is None else depth, seq)

    def _repr_svg_(self) -> str | None:
        """Show the scene inline in Jupyter; ``None`` (plain repr) while it is empty."""
        return None if self.is_empty else self.to_svg_document()

    def to_svg_document(self) -> str:
        """Return the scene as a complete SVG document.

        This is the whole surface a consumer needs.  A tool that assembles a
        larger document can accept any object exposing this method and place a
        scene without importing this package, or being imported by it.
        """
        return str(self.render())

    def save(self, path: str | Path) -> Path:
        """Write the rendered document to ``path`` and return it.

        Only SVG is written.  Rasterizing and PDF are left to the consumer,
        which is what keeps them out of this package's dependencies -- run
        ``cairosvg`` over the file, or hand :meth:`to_svg_document` to whatever
        assembles the final page.
        """
        out = Path(path)
        out.write_text(self.to_svg_document(), encoding="utf-8")
        return out


__all__ = ["DEFAULT_FONT", "DEFAULT_TEXT_FILL", "Align", "Scene", "TextContent"]
