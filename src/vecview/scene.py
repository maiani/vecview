"""The layer stack: world-space geometry in, one SVG document out."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
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


DEFAULT_FONT = "DejaVu Sans, Verdana, sans-serif"
DEFAULT_TEXT_FILL = "#222222"


class Scene:
    """An ordered collection of SVG elements built from world-space geometry.

    Draw order is an explicit integer ``layer`` per call, resolved stably by
    insertion order within a layer.  There is no z-buffer and no painter's-algorithm
    depth sort, by choice: for a schematic with a beam passing through a translucent
    slab, deciding what occludes what by hand is worth more than getting it
    automatically and almost right.  :meth:`Camera.visible` handles the one case
    where the answer is unambiguous -- the back faces of a convex solid.

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
    def _emit(self, layer: int, element: svg.Element) -> None:
        """Queue an element at ``layer`` without logging it."""
        self.items.append((int(layer), self._seq, element))
        self._seq += 1

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
        self._emit(layer, svg.Polygon(points=_points(p), **style))

    def polyline(self, layer: int, pts3: Points3, **style: Style) -> None:
        """Open path through projected world points; unfilled unless asked."""
        style.setdefault("fill", "none")
        self._record("polyline", layer, pts3, **style)
        p = self.cam.project(pts3)
        self._grow(p)
        self._emit(layer, svg.Polyline(points=_points(p), **style))

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
        self._emit(layer, svg.Polygon(points=_points(outline), **style))

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
        commands: list[svg.PathData] = []
        for start, count in runs:
            indices = [(start + k) % n for k in range(count)]
            bottom = [walls[i].points[0] for i in indices] + [walls[indices[-1]].points[1]]
            top = [walls[indices[-1]].points[2]] + [walls[i].points[3] for i in reversed(indices)]
            outline = self.cam.project(np.array(bottom + top))
            self._grow(outline)
            first, *rest = _points(outline)
            commands.append(svg.M(first.x, first.y))
            commands += [svg.L(p.x, p.y) for p in rest]
            commands.append(svg.Z())
        self._emit(layer, svg.Path(d=commands, **style))

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

    def text(
        self,
        layer: int,
        pt3: Point3,
        s: str,
        dx: float = 0.0,
        dy: float = 0.0,
        size: float = 22.0,
        **style: Style,
    ) -> None:
        """Text anchored at a projected world point, offset by ``(dx, dy)`` on screen."""
        self._record("text", layer, pt3, s, dx, dy, size, **style)
        x, y = self.cam.at(pt3)
        with self._delegating():
            self.text2d(layer, x + dx, y + dy, s, size=size, **style)

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
        s: str,
        size: float = 22.0,
        grow: bool = True,
        **style: Style,
    ) -> None:
        """Text in screen coordinates."""
        style.setdefault("font_family", DEFAULT_FONT)
        style.setdefault("fill", DEFAULT_TEXT_FILL)
        self._record("text2d", layer, x, y, s, size, grow, **style)
        if grow:
            # Crude metrics, but enough to keep the fitted viewBox from clipping
            # labels.  Real advance widths would need the font loaded.
            w = 0.56 * size * len(s)
            anchor = style.get("text_anchor", "start")
            x0 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
            self._grow(np.array([[x0, y - 0.82 * size], [x0 + w, y + 0.25 * size]]))
        self._emit(layer, svg.Text(x=round(x, 2), y=round(y, 2), text=s, font_size=size, **style))

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
        elements += [el for _, _, el in sorted(self.items, key=lambda it: (it[0], it[1]))]
        return svg.SVG(
            width=round(w, 1),
            height=round(h, 1),
            viewBox=svg.ViewBoxSpec(round(lo[0], 1), round(lo[1], 1), round(w, 1), round(h, 1)),
            elements=elements,
        )

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


__all__ = ["DEFAULT_FONT", "DEFAULT_TEXT_FILL", "Align", "Scene"]
