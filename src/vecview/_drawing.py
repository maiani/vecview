"""The flat world-space drawing calls a scene or a part records, and their replay."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Concatenate, Literal, NamedTuple, Self

import svg

from vecview._elements import _ALIGN, Align, TextContent
from vecview._place import _Call, _moved, _placement
from vecview._planar import check_holed_polygon
from vecview._tracks import Track
from vecview._types import Point3, Points3, Shape2, Shape3, Style
from vecview._vec import as_points, as_rings
from vecview.shapes import Face, Pivot, prism_faces

if TYPE_CHECKING:
    from vecview._solid_drawing import Part


class _Tracked(NamedTuple):
    """A drawing call with a track among its arguments, kept as made for ``at`` to replay."""

    method: str
    args: tuple[object, ...]
    kwargs: dict[str, object]


def _moving(value: object) -> bool:
    """Whether ``value`` is a track, or a list, tuple, dict, or part holding one."""
    if isinstance(value, Track):
        return True
    if isinstance(value, _Drawing):
        return value._animated
    if isinstance(value, list | tuple) and type(value) in (list, tuple):
        return any(_moving(v) for v in value)
    if isinstance(value, dict) and type(value) is dict:
        return any(_moving(v) for v in value.values())
    return False


def _read(value: object, t: float) -> object:
    """``value`` with every track in it read at ``t`` seconds."""
    if isinstance(value, Track):
        return _read(value(t), t)  # a track may give a part that moves too
    if isinstance(value, _Drawing) and value._animated:
        return value.at(t)
    if isinstance(value, list | tuple) and type(value) in (list, tuple):
        return type(value)(_read(v, t) for v in value)
    if isinstance(value, dict) and type(value) is dict:
        return {k: _read(v, t) for k, v in value.items()}
    return value


def _read_call(
    args: tuple[object, ...], kwargs: dict[str, object], t: float
) -> tuple[tuple[object, ...], dict[str, object]]:
    """A call's arguments with every track in them read at ``t`` seconds."""
    return tuple(_read(v, t) for v in args), {k: _read(v, t) for k, v in kwargs.items()}


def _trackable[D: _Drawing, **P](
    method: Callable[Concatenate[D, P], None],
) -> Callable[Concatenate[D, P], None]:
    """Let a drawing call take tracks: such a call is checked at ``t = 0`` and kept as made."""

    name = method.__name__  # ty: ignore[unresolved-attribute]

    @functools.wraps(method)
    def call(self: D, *args: P.args, **kwargs: P.kwargs) -> None:
        if not _moving((args, kwargs)):
            return method(self, *args, **kwargs)
        made = _Tracked(name, tuple(args), dict(kwargs))
        now, named = _read_call(made.args, made.kwargs, 0.0)
        method(self._blank(), *now, **named)  # ty: ignore[invalid-argument-type]
        self._log.append(made)
        return None

    return call


def _class_names(value: object) -> tuple[str, ...]:
    """The distinct names in a ``class_`` argument, in the order given; ``None`` gives none."""
    parts = [value] if isinstance(value, str) else () if value is None else value
    if not isinstance(parts, Iterable) or not all(isinstance(p, str) for p in parts):
        raise TypeError(f"class_ must be a string or a sequence of strings, got {value!r}")
    return tuple(dict.fromkeys(name for part in parts for name in part.split()))


class _Drawing:
    """The world-space drawing calls, recorded for a camera to replay.

    Shared by :class:`Scene` and :class:`Part`.  Calls that only make sense for
    a whole document -- screen-space shapes, raw elements, ``<defs>``, and depth
    sorting, which is a property of the layer stack -- live on :class:`Scene`
    alone, so a part cannot hold anything that would not move with it.
    """

    def __init__(self) -> None:
        self._log: list[_Call | _Tracked] = []

    def _add(self, method: str, *args: object, **kwargs: Style) -> None:
        """Record one drawing call, to be replayed against a camera when rendering."""
        classes = _class_names(kwargs.pop("class_", ()))
        self._log.append((method, args, kwargs, classes))

    @property
    def _animated(self) -> bool:
        return any(isinstance(record, _Tracked) for record in self._log)

    def _calls(self) -> list[_Call]:
        """The recorded calls of a drawing with no tracks in it."""
        if self._animated:
            raise RuntimeError("a drawing holding tracks is read at a time first, with at(t)")
        return [record for record in self._log if not isinstance(record, _Tracked)]

    def _blank(self, t: float = 0.0) -> Self:
        """An empty drawing of the same kind, with its settings as they stand at ``t``."""
        return type(self)()

    def at(self, t: float) -> Self:
        """A copy as it stands ``t`` seconds in, with every track read at ``t``.

        Calls without a track are shared with the copy, not made again.  A
        drawing with no tracks gives an unchanged copy at any time.
        """
        still = self._blank(t)
        for record in self._log:
            if isinstance(record, _Tracked):
                args, kwargs = _read_call(record.args, record.kwargs, t)
                getattr(still, record.method)(*args, **kwargs)
            else:
                still._log.append(record)
        return still

    @_trackable
    def place(
        self,
        layer: int,
        part: Part | Track[Part],
        *,
        at: Point3 = (0.0, 0.0, 0.0),
        rotate: tuple[Point3, float] | None = None,
        mirror: Point3 | None = None,
        scale: float = 1.0,
        id: str | None = None,
        class_: str | Iterable[str] | None = None,
    ) -> None:
        """Draw a copy of ``part``: mirrored, then rotated, scaled, and moved to ``at``.

        A point ``p`` of the part lands at ``at + scale * R @ M @ p``, where
        ``M`` reflects through the plane with normal ``mirror`` and ``R`` turns
        by ``rotate = (axis, angle_deg)`` about ``axis``, counter-clockwise
        looking down it.  Both act about the part's own origin.  Only rigid
        motions and one uniform ``scale`` are offered, so a sphere stays a
        sphere and every solid keeps its exact outline.

        The copy is taken now: drawing into ``part`` afterwards changes later
        placements, not this one.  Placing into a part nests, so a unit cell can
        be placed in a supercell that is placed in a scene.

        Args:
            layer: Added to every layer the part draws on, so a part drawn on
                layers ``0`` and ``1`` and placed at ``10`` lands on ``10`` and
                ``11``.  Depth sorting stays the host scene's choice:
                ``scene.sort_by_depth(10)`` sorts the placed atoms with
                everything else on that layer.
            part: The part to draw, or a track of parts for what changes from
                frame to frame -- how many arrows there are, say.
            at: Where the part's origin goes.
            rotate: ``(axis, angle_deg)``, or ``None``.
            mirror: Normal of the plane through the part's origin to reflect
                through, or ``None``.  A mirrored helix turns the other way.
            scale: Uniform factor on every world length -- positions, radii,
                arrow widths.  Screen units (stroke widths, text size, offsets)
                are not scaled.
            id: Prefixed to every id in the part, as ``{id}-{part id}``.  Ids
                must be unique in a document, so a part with ids placed more
                than once needs a different ``id`` for each placement.
            class_: Added to the classes of everything the part draws, so one
                placement can be selected as a whole.

        Raises:
            TypeError: If ``part`` is not a :class:`Part`.
            ValueError: For a non-positive ``scale``, a zero ``rotate`` axis or
                ``mirror`` normal, or an empty ``id``.
        """
        from vecview._solid_drawing import Part  # which draws on this module

        if not isinstance(part, Part):
            raise TypeError(
                f"place takes a Part, got {type(part).__name__}; a scene's cameras and "
                "depth sorting have no meaning inside another scene"
            )
        if id is not None and not id:
            raise ValueError("a placement id must not be empty")
        frame = _placement(at, rotate, mirror, scale)
        extra = _class_names(class_)
        for call in part._calls():
            self._log.append(_moved(call, frame, int(layer), id, extra))

    @_trackable
    def polygon(self, layer: int, pts3: Shape3, **style: Style) -> None:
        """Filled polygon through projected world points.

        Given a Region with holes, as
        :func:`~vecview.outlines.to_plane` returns one -- it is drawn as one
        ``<path>`` with ``fill-rule="evenodd"``, through which whatever is
        behind shows, in a layer sorted by depth as on screen.  Its holes must lie
        in the outline's plane, inside it and apart, or this raises ``ValueError``.
        """
        if len(rings := as_rings(pts3, 3)) > 1:
            check_holed_polygon([as_points(ring) for ring in rings])
        self._add("polygon", layer, pts3, **style)

    @_trackable
    def polyline(
        self, layer: int, pts3: Points3, *, back: Mapping[str, Style] | None = None, **style: Style
    ) -> None:
        """Open path through projected world points; unfilled unless asked.

        In a layer passed to :meth:`sort_by_depth`, the parts an opaque surface hides
        are dropped, or drawn with ``{**style, **back}`` when ``back`` is given
        -- a ray dashed where it passes behind an atom.  Elsewhere ``back`` has
        no effect.
        """
        style.setdefault("fill", "none")
        self._add("polyline", layer, pts3, back=back, **style)

    @_trackable
    def faces(
        self, layer: int, faces: Iterable[Face], *, cull: bool = False, **style: Style
    ) -> None:
        """Draw each face as a polygon, all with the same style.

        Args:
            layer: Draw order.
            faces: Faces to draw, typically from :func:`~vecview.shapes.box_faces`.
            cull: Drop back faces when rendering, using the camera that renders.
                Prefer this over filtering with :meth:`Camera.visible` yourself:
                culling done by the caller bakes in *that* camera's answer, so
                any other camera would keep the original walls and quietly draw
                the wrong ones. With ``cull=True`` the full set is recorded and
                each camera decides.
            **style: SVG presentation attributes, shared by every face.

        An ``id`` is suffixed per face rather than repeated, since duplicate ids
        are invalid SVG and break selection downstream: ``id="slab"`` over a
        box's visible walls yields ``slab-pz``, ``slab-px``, ``slab-py``.
        """
        base = style.pop("id", None)
        given = list(faces)
        self._add(
            "faces",
            layer,
            given,
            cull=cull,
            **({} if base is None else {"id": base}),
            **style,
        )

    @_trackable
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
        self._add("plane", layer, origin, u_edge, v_edge, id=id, **style)

    @_trackable
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
        content: svg.Element | None = None,
        **style: Style,
    ) -> None:
        """A screen-aligned group anchored at a projected world point.

        The screen-space sibling of :meth:`plane`: where a plane makes content
        lie *in* the scene, a slot keeps it upright and unforeshortened -- a
        label, an equation, an inset -- while pinning it to a point of the
        geometry.  Give it ``content`` to hold, or leave it empty for a
        consumer to fill by ``id``.  Either way this package never parses or
        rewrites the content: it places the element it is given.

        The group is translated to the anchor, ``cam.at(pt3)`` offset by
        ``(dx, dy)``, and records ``align`` as ``data-align``, so a consumer can
        line its content up against the anchor at whatever size it ends up.  A
        ``w`` by ``h`` box aligned the same way grows the fitted viewBox, so
        content of that size is not clipped.  Each camera projects the anchor
        afresh, so the slot follows the geometry.

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
            content: An svg.py element to hold, ``w`` by ``h`` in its own
                coordinates from ``(0, 0)`` at its top-left, so the box aligns
                as it would for a consumer.  The same element may be held by
                any number of slots; it is not copied or changed.

        Raises:
            ValueError: For an unknown ``align``.
            TypeError: If ``content`` is not an svg.py element.
        """
        if align not in _ALIGN:
            raise ValueError(f"unknown align {align!r}; expected one of {sorted(_ALIGN)}")
        if content is not None and not isinstance(content, svg.Element):
            raise TypeError(f"slot content must be an svg.py element, got {type(content).__name__}")
        self._add(
            "slot", layer, pt3, w, h, id=id, align=align, dx=dx, dy=dy, content=content, **style
        )

    @_trackable
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
        the silhouette of a convex solid.  Each camera recomputes it.

        Args:
            layer: Draw order.
            solid: Faces of the solid, or its world-space vertices.
            **style: SVG presentation attributes of the polygon.
        """
        given = list(solid)
        self._add("silhouette", layer, given, **style)

    @_trackable
    def prism_walls(
        self, layer: int, footprint: Shape2, z0: float, z1: float, **style: Style
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

        A footprint with holes -- a ring -- adds each hole's facing walls, the
        far side of the hole, as a path of their own drawn first, with the id
        ``"{id}-holes"``.  Those walls lie behind the cap and the outer walls,
        so walls-then-cap stays exact even stroked, for a convex hole.

        Args:
            layer: Draw order.
            footprint: Simple polygon, or one with holes, as for
                :func:`~vecview.shapes.prism_faces`.
            z0: Height of the base.
            z1: Height of the top.
            **style: SVG presentation attributes of the path.

        Culling uses the camera that renders, so each camera draws the walls it
        faces.
        """
        prism_faces(footprint, z0, z1)  # rejects a footprint that is not a simple polygon
        self._add("prism_walls", layer, footprint, z0, z1, **style)

    @_trackable
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
        ``z``, say.  The normal is resolved from the camera that renders, so
        every camera sees the arrow turned toward it.  Computing that normal
        yourself from ``cam.view`` would bake in one camera's answer.

        Args:
            normal: Normal of the plane the arrow lies flat in, or ``"camera"``.
                The latter needs a :class:`ParallelCamera`.

        Raises:
            ValueError: If ``normal="camera"`` and the arrow points along the
                projection ray, where it has no face to show.

        The remaining arguments are those of :func:`~vecview.shapes.arrow_shape`.
        """
        if isinstance(normal, str) and normal != "camera":
            raise ValueError(f"normal must be a vector or 'camera', got {normal!r}")
        self._add(
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

    @_trackable
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
            separate: Emit one ``<path>`` per edge, each with its own id, so
                the edges can be selected one by one.
            trim: Shorten every edge by this much, in world units, at both
                ends, so it stops at the surface of an atom or marker sitting
                on each vertex.  Otherwise, in a layer sorted by depth, the
                part of an edge inside the atom is hidden and drawn in the
                ``back`` style -- a dash across the atom.
            **style: SVG presentation attributes; ``fill`` defaults to none.
                An ``id`` becomes ``{id}-front`` and ``{id}-back``, suffixed
                ``-0``, ``-1``, ... per edge when ``separate``.

        Culling uses the camera that renders, so each camera splits afresh.
        """
        given = list(faces)
        base = style.pop("id", None)
        style.setdefault("fill", "none")
        self._add(
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

    @_trackable
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
        self._add("text", layer, pt3, s, dx, dy, size, **style)
