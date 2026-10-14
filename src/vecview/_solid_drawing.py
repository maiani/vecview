"""The curved solids a scene or a part draws, and parts themselves.

The calls here record like those of :class:`~vecview._drawing._Drawing`; the
canvas that draws them is :class:`~vecview._solids._SolidCanvas`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from vecview._drawing import _Drawing, _trackable
from vecview._types import Point3, Points3, Style
from vecview._vec import as_points, unit
from vecview.shapes import Pivot


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


class _SolidDrawing(_Drawing):
    """The drawing calls of a scene or a part: the flat ones, and spheres, tubes, and cones."""

    @_trackable
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

    @_trackable
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

    @_trackable
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
        the solid is one object in Inkscape, clipped as one where
        :meth:`sort_by_depth` finds it hidden.

        Args:
            layer: Draw order.
            p0: Centre of the first end.
            p1: Centre of the second end.
            radius: Radius at ``p0``.
            r1: Radius at ``p1``; defaults to ``radius``. ``0`` is a cone, for
                which :meth:`cone` reads better.
            ends: Whether to draw the end disks.  ``False`` leaves an open tube,
                which is what a bond hidden inside two atoms, or a beam of
                light, wants: its rims are not edges, so a ``stroke`` outlines
                only the two sides, as a separate ``{id}-edge`` path over the
                unstroked ``{id}-body``.
            end_style: Presentation attributes for the end disks, over
                ``style`` -- a lighter ``fill`` for a flat-shaded cap, say.
            highlight: A lighter colour shading the body across its width,
                lightest toward the upper left: a fill style, not a lighting
                model.  Needs ``fill`` and ``id``; the gradient is ``{id}-shade``.
            slices: Cut the solid into this many lengths along its axis, each
                its own ``<g>``.  Depth sorting does not need it -- a coil wraps
                an unsliced core -- so it is only for a solid wanted in
                separately selectable lengths.  The slices overlap a little and
                the outline is stroked along the sides only, so the result
                looks like one solid -- except with a translucent fill, where
                the overlaps show.
            **style: SVG presentation attributes. An ``id`` goes on the group;
                the body takes ``{id}-body`` and the end disks ``{id}-end0``
                and ``{id}-end1``.  Sliced, the groups are ``{id}-0``,
                ``{id}-1``, ... with ``{id}-{k}-body`` and ``{id}-{k}-edge``;
                the end disks and the gradient are named as unsliced.

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

    @_trackable
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
        disk when it faces the camera; without it the cone is open, and a
        ``stroke`` outlines only the two tangents.
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

    @_trackable
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

    @_trackable
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

        The curve is cut into pieces of ``chunk`` segments, one ``<g>`` each.  In
        a layer passed to :meth:`sort_by_depth` each piece is a surface of its
        own, so a tube can pass over and under itself, as a coil seen at an
        angle does; one piece never hides another part of itself.  Neighbouring pieces overlap,
        and the outline of each stops a segment short of its body, so no seam
        shows where they meet.  The tube's own two ends are
        square.

        Args:
            layer: Draw order.
            pts3: The centre line, shape ``(n, 3)``, with ``n >= 2``.
            radius: Tube radius, in world units.
            chunk: Segments per piece; shorter lets a tightly bent tube hide
                itself more finely.
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

    @_trackable
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


class Part(_SolidDrawing):
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
