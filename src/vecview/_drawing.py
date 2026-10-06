"""The world-space drawing calls a scene or a part records, and parts themselves."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np

from vecview._elements import _ALIGN, Align, TextContent
from vecview._place import _Call, _moved, _placement
from vecview._types import Point3, Points2, Points3, Style
from vecview._vec import as_points, unit
from vecview.shapes import Face, Pivot, prism_faces


def _class_names(value: object) -> tuple[str, ...]:
    """The distinct names in a ``class_`` argument, in the order given; ``None`` gives none."""
    parts = [value] if isinstance(value, str) else () if value is None else value
    if not isinstance(parts, Iterable) or not all(isinstance(p, str) for p in parts):
        raise TypeError(f"class_ must be a string or a sequence of strings, got {value!r}")
    return tuple(dict.fromkeys(name for part in parts for name in part.split()))


def _check_highlight(style: Mapping[str, Style], highlight: str | None, *, needs_id: bool) -> None:
    """Reject a ``highlight`` that has no fill to shade toward, or no id to name it by."""
    if highlight is None:
        return
    if not isinstance(style.get("fill"), str):
        raise ValueError("highlight needs a fill colour to shade toward")
    if needs_id and style.get("id") is None:
        raise ValueError("highlight needs an id to name the gradient")


def _check_axis(p0: Point3, p1: Point3, r0: float, r1: float) -> None:
    """Reject a solid of revolution with no axis or a negative radius."""
    if r0 <= 0 or r1 < 0:
        raise ValueError(f"need r0 > 0 and r1 >= 0, got r0={r0}, r1={r1}")
    if np.array_equal(np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)):
        raise ValueError("the two ends coincide, so the solid has no axis")


class _Drawing:
    """The world-space drawing calls, recorded for a camera to replay.

    Shared by :class:`Scene` and :class:`Part`.  Calls that only make sense for
    a whole document -- screen-space shapes, raw elements, ``<defs>``, and depth
    sorting, which is a property of the layer stack -- live on :class:`Scene`
    alone, so a part cannot hold anything that would not move with it.
    """

    def __init__(self) -> None:
        self._log: list[_Call] = []

    def _add(self, method: str, *args: object, **kwargs: Style) -> None:
        """Record one drawing call, to be replayed against a camera when rendering."""
        classes = _class_names(kwargs.pop("class_", ()))
        self._log.append((method, args, kwargs, classes))

    def place(
        self,
        layer: int,
        part: Part,
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
            part: The part to draw.
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
        if not isinstance(part, Part):
            raise TypeError(
                f"place takes a Part, got {type(part).__name__}; a scene's cameras and "
                "depth sorting have no meaning inside another scene"
            )
        if id is not None and not id:
            raise ValueError("a placement id must not be empty")
        frame = _placement(at, rotate, mirror, scale)
        extra = _class_names(class_)
        for call in list(part._log):
            self._log.append(_moved(call, frame, int(layer), id, extra))

    def polygon(self, layer: int, pts3: Points3, **style: Style) -> None:
        """Filled polygon through projected world points."""
        self._add("polygon", layer, pts3, **style)

    def polyline(
        self, layer: int, pts3: Points3, *, back: Mapping[str, Style] | None = None, **style: Style
    ) -> None:
        """Open path through projected world points; unfilled unless asked.

        In a layer sorted with ``exact=True``, the parts an opaque surface hides
        are dropped, or drawn with ``{**style, **back}`` when ``back`` is given
        -- a ray dashed where it passes behind an atom.  Elsewhere ``back`` has
        no effect.
        """
        style.setdefault("fill", "none")
        self._add("polyline", layer, pts3, back=back, **style)

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
        """
        if align not in _ALIGN:
            raise ValueError(f"unknown align {align!r}; expected one of {sorted(_ALIGN)}")
        self._add("slot", layer, pt3, w, h, id=id, align=align, dx=dx, dy=dy, **style)

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

        Culling uses the camera that renders, so each camera draws the walls it
        faces.
        """
        prism_faces(footprint, z0, z1)  # rejects a footprint that is not a simple polygon
        self._add("prism_walls", layer, footprint, z0, z1, **style)

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
        if stops < 2:
            raise ValueError("a gradient needs at least two stops")
        self._add(
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
        _check_highlight(style, highlight, needs_id=False)
        self._add("sphere", layer, center, radius, highlight=highlight, **style)

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
        _check_axis(p0, p1, radius, radius if r1 is None else r1)
        _check_highlight(style, highlight, needs_id=True)
        if slices < 1:
            raise ValueError(f"slices must be at least 1, got {slices}")
        self._add(
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
        _check_axis(base, apex, radius, 0.0)
        _check_highlight(style, highlight, needs_id=True)
        self._add(
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
        if not unit(direction).any():
            raise ValueError("an arrow needs a non-zero direction")
        _check_highlight(style, highlight, needs_id=True)
        self._add(
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
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        if len(as_points(pts3)) < 2:
            raise ValueError("a tube needs at least two points")
        self._add("tube", layer, pts3, radius, chunk=chunk, **style)

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
        style.setdefault("fill", "none")
        self._add(
            "sphere_curve",
            layer,
            center,
            pts3,
            closed=closed,
            back=back,
            back_layer=back_layer,
            **style,
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
        self._add("text", layer, pt3, s, dx, dy, size, **style)


class Part(_Drawing):
    """Objects drawn once and placed many times: a unit cell, a gate, a lens.

    A part records the same world-space calls as a :class:`Scene`, in its own
    coordinates, and :meth:`~Scene.place` draws a moved copy of them into a
    scene or into another part::

        cell = Part()
        cell.sphere(0, (0, 0, 0), 0.2, fill="#3b6fb6", id="atom", class_="atom")
        cell.edges(1, box_faces((0.5, 0.5, 0.5), (1, 1, 1)), stroke="#222", id="edge")
        for i in range(3):
            scene.place(10, cell, at=(i, 0, 0), id=f"cell-{i}")

    A part has no camera, no ``<defs>``, no screen-space calls, and no depth
    sorting: each of those belongs to the document a part ends up in.  To look
    at a part on its own, place it in a scene.
    """

    def __repr__(self) -> str:
        return f"Part({len(self._log)} calls)"
