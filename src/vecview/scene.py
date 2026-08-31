"""The layer stack: world-space geometry in, one SVG document out."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import svg

from vecview._types import Array, Point3, Points3, Style
from vecview.camera import Camera, ParallelCamera
from vecview.shapes import Face


def _points(projected: Array, ndigits: int = 2) -> list[svg.Point]:
    """Projected coordinates as the ``x,y`` pairs ``svg.py`` wants for ``points``."""
    return [svg.Point(round(float(x), ndigits), round(float(y), ndigits)) for x, y in projected]


def _face_id(base: str, name: str) -> str:
    """Per-face id derived from a base and a face name.

    ``+`` is not a legal XML name character, so the sign is spelled out:
    ``"slab"`` and ``"+z"`` give ``"slab-pz"``.
    """
    return f"{base}-{name.replace('+', 'p').replace('-', 'm')}"


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
        if not isinstance(self.cam, ParallelCamera):
            raise TypeError(
                f"{type(self.cam).__name__} cannot embed content in a plane: the "
                "affine transform an SVG `matrix` carries exists only for a "
                "parallel projection"
            )
        self._record("plane", layer, origin, u_edge, v_edge, id=id, **style)
        matrix = self.cam.plane_matrix(origin, u_edge, v_edge)
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


__all__ = ["DEFAULT_FONT", "DEFAULT_TEXT_FILL", "Scene"]
