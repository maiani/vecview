"""Scenes: world-space objects in, one SVG document per camera out.

A :class:`Scene` only records what was drawn, the way a 3D application keeps
objects separate from the camera that views them.  Rendering replays the record
against a camera on a private canvas, which is where every projection happens.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import svg

from vecview._drawing import Part, _Drawing, _trackable
from vecview._elements import DEFAULT_FONT, DEFAULT_TEXT_FILL, Align, TextContent
from vecview._place import _Call
from vecview._solids import _SolidCanvas
from vecview._tracks import Track
from vecview._types import Array, Style
from vecview.camera import Camera


def _grows(method: str, args: tuple[object, ...]) -> bool:
    """Whether a recorded call can grow the fitted box when it is rendered."""
    if method in ("rect2d", "text2d"):
        return bool(args[-1])  # their ``grow`` argument
    return method not in ("sort_by_depth", "add_def", "add")


type CameraRef = Camera | str
"""A camera, or the name of one in :attr:`Scene.cameras`."""


class Scene(_Drawing):
    """A 3D scene: objects in world space, and the cameras that view them.

    Drawing calls record objects and nothing more; nothing is projected until
    the scene is rendered.  As in a 3D application, a scene can hold any number
    of named cameras, one of them active, and a render uses the active camera
    unless told which other to use::

        scene = Scene(pad=28, background="#ffffff")
        scene.sphere(10, (0, 0, 0), 1.0, fill="#c33")
        scene.cameras["main"] = OrthographicCamera(35, 24, 62)
        scene.cameras["cabinet"] = ObliqueCamera.cabinet(62)
        scene.camera = "main"  # the active camera
        scene.save("main.svg")
        scene.save("cabinet.svg", "cabinet")
        scene.render(OrthographicCamera.isometric(62))  # any camera, named or not

    :attr:`camera` is the active camera, which :meth:`render`, :meth:`save`,
    and :meth:`bbox` use unless given another, and which
    :meth:`to_svg_document` -- the zero-argument embedding contract -- always
    uses.  Set it to the name of a camera in :attr:`cameras`, which keeps
    following that entry if it is replaced, or to a camera directly.  It may be
    ``None`` while the scene is built.  It may also be a :class:`Track` of
    cameras, for a view that moves: :meth:`at` reads it at each time, and
    :attr:`camera` is the camera at ``t = 0``.

    Draw order is an explicit integer ``layer`` per call, resolved stably by
    insertion order within a layer.  There is no z-buffer, and no layer is
    depth-sorted unless it asks to be with :meth:`sort_by_depth`: for a schematic
    with a beam passing through a translucent slab, deciding what occludes what
    by hand is worth more than getting it automatically and almost right, while
    for a lattice of hundreds of atoms deciding visibility by depth is the only
    practical answer.  :meth:`Camera.visible` handles the one case where the
    answer is unambiguous -- the back faces of a convex solid.

    Every drawing call that takes style keywords also takes ``class_``: one
    string, space-separated as in SVG, or a sequence of names.  An ``id`` names
    one object; a class names a kind of object -- ``class_="gate"`` on all four
    gates -- so a stylesheet, a selector, or Inkscape can reach them together.
    Each top-level element a call emits carries the classes: every face of
    :meth:`faces`, both strokes of :meth:`edges`, every piece of a sliced or
    chunked solid, and both parts of a line a depth-sorted layer splits.  The
    elements inside a solid's ``<g>`` do not repeat them, so a selector
    matches each object once.

    Args:
        camera: The active camera, or the name of one in ``cameras``, a track of
            either, or ``None`` to choose one later.
        cameras: Named cameras the scene holds.
        pad: Default margin added around the fitted content.
        background: Default background fill, or ``None`` for a transparent document.
    """

    def __init__(
        self,
        camera: CameraRef | Track[CameraRef] | None = None,
        *,
        cameras: Mapping[str, Camera] | None = None,
        pad: float = 26.0,
        background: str | None = None,
    ) -> None:
        super().__init__()
        self.cameras: dict[str, Camera] = dict(cameras or {})
        self._active: CameraRef | Track[CameraRef] | None = None
        self.camera = camera
        self.pad = float(pad)
        self.background = background

    def __repr__(self) -> str:
        names = ", ".join(self.cameras)
        return f"Scene({self._active!r}, cameras=[{names}], {len(self._log)} calls)"

    # --- cameras ----------------------------------------------------------
    @property
    def camera(self) -> Camera | None:
        """The active camera, looked up by name if it was set by name, at ``t = 0`` if a track."""
        active = self._active(0.0) if isinstance(self._active, Track) else self._active
        return None if active is None else self._lookup(active)

    @camera.setter
    def camera(self, value: CameraRef | Track[CameraRef] | None) -> None:
        first = value(0.0) if isinstance(value, Track) else value
        if isinstance(first, str):
            self._lookup(first)
        elif first is not None and not isinstance(first, Camera):
            raise TypeError(f"a camera must be a Camera or a camera's name, got {first!r}")
        self._active = value

    def _lookup(self, ref: CameraRef) -> Camera:
        if not isinstance(ref, str):
            return ref
        if ref not in self.cameras:
            raise KeyError(f"no camera named {ref!r}; the scene has {sorted(self.cameras)}")
        return self.cameras[ref]

    def _view(self, camera: CameraRef | None) -> Camera:
        if camera is not None:
            return self._lookup(camera)
        active = self.camera
        if active is None:
            raise ValueError("no camera to render with: pass one, or set scene.camera")
        return active

    # --- rendering --------------------------------------------------------
    @property
    def _animated(self) -> bool:
        return isinstance(self._active, Track) or super()._animated

    def _blank(self, t: float = 0.0) -> Scene:
        active = self._active(t) if isinstance(self._active, Track) else self._active
        return Scene(active, cameras=self.cameras, pad=self.pad, background=self.background)

    def _still(self) -> list[_Call]:
        """The calls to render: a scene with tracks is drawn as it stands at ``t = 0``."""
        return (self.at(0.0) if self._animated else self)._calls()

    def _project(self, camera: CameraRef | None) -> _SolidCanvas:
        """Replay every recorded call against a camera, on a fresh canvas."""
        canvas = _SolidCanvas(self._view(camera))
        for method, args, kwargs, classes in self._still():
            canvas.classes = classes
            getattr(canvas, method)(*args, **kwargs)
        return canvas

    @property
    def is_empty(self) -> bool:
        """Whether nothing has been drawn that a viewBox could be fitted to.

        Settings and definitions do not count, and neither do the calls that
        never grow the box: :meth:`add`, and :meth:`rect2d` or :meth:`text2d`
        with ``grow=False``.  So a scene is empty exactly when rendering it
        would raise for want of content.
        """
        return not any(_grows(method, args) for method, args, *_ in self._still())

    def bbox(self, camera: CameraRef | None = None) -> tuple[Array, Array]:
        """Screen-space bounds of the content under ``camera`` (default: the active one).

        Text extents are estimated from a nominal glyph width, which is enough
        to keep the fitted viewBox from clipping a label but is not exact.
        """
        return self._project(camera).bbox()

    def render(
        self,
        camera: CameraRef | None = None,
        *,
        pad: float | None = None,
        background: str | None = None,
    ) -> svg.SVG:
        """Project the scene and assemble the document, fitting the viewBox to the content.

        Nothing has to be centred by hand: the viewBox follows the geometry.
        Rendering is a pure function of the recorded calls and the camera, so
        it can be called any number of times, with any cameras.

        Args:
            camera: A camera, or the name of one in :attr:`cameras`; defaults
                to the active :attr:`camera`.
            pad: Margin around the content; defaults to the scene's ``pad``.
            background: Background fill; defaults to the scene's ``background``.

        Raises:
            ValueError: If there is no camera, or nothing to fit a viewBox to.
            KeyError: For a camera name the scene does not hold.
        """
        canvas = self._project(camera)
        return canvas.document(
            self.pad if pad is None else float(pad),
            self.background if background is None else background,
        )

    def with_camera(self, camera: CameraRef) -> Scene:
        """A copy of this scene whose active camera is ``camera``.

        The recorded calls and the named cameras are copied, so later changes to
        either scene do not affect the other.  Useful for handing the same
        scene, seen two ways, to a tool that only calls :meth:`to_svg_document`.

        Two things behave as their names promise rather than as a new camera
        might suggest.  Screen-space calls -- :meth:`rect2d`, :meth:`text2d`,
        and anything handed to :meth:`add` -- stay at the same screen
        coordinates, because that is what "screen space" means.  And geometry
        computed by the *caller* from a camera (a point placed via
        :meth:`Camera.at`, say) is baked in already.

        The world-space arrays handed to :meth:`polygon` and friends are held by
        reference, not copied, so do not mutate them after adding.
        """
        clone = Scene(camera, cameras=self.cameras, pad=self.pad, background=self.background)
        clone._log = list(self._log)
        return clone

    def _repr_svg_(self) -> str | None:
        """Show the scene inline in Jupyter; ``None`` (plain repr) without a camera or content."""
        if self.camera is None or self.is_empty:
            return None
        return self.to_svg_document()

    def to_svg_document(self) -> str:
        """Return the scene, seen by its active :attr:`camera`, as a complete SVG document.

        This is the whole surface a consumer needs.  A tool that assembles a
        larger document can accept any object exposing this method and place a
        scene without importing this package, or being imported by it.
        """
        return str(self.render())

    def save(self, path: str | Path, camera: CameraRef | None = None) -> Path:
        """Write the document rendered by ``camera`` (default: the active one); return the path.

        Only SVG is written.  Rasterizing and PDF are left to the consumer,
        which is what keeps them out of this package's dependencies -- run
        ``cairosvg`` over the file, or hand :meth:`to_svg_document` to whatever
        assembles the final page.
        """
        out = Path(path)
        out.write_text(str(self.render(camera)), encoding="utf-8")
        return out

    # --- drawing ----------------------------------------------------------
    def sort_by_depth(self, layer: int) -> None:
        """Decide what hides what in ``layer`` by depth, instead of by draw order.

        Visibility is decided point by point.  Each surface keeps its native
        element, clipped to the part of it that no opaque surface hides --
        found exactly between two planar surfaces, and elsewhere traced on a
        grid of half a screen unit, at most 160 steps across an overlap -- so a
        bond can run into an atom's centre, two planes can cross, and a coil
        can wrap its core.  Each line is cut where an opaque surface hides it;
        the hidden part is dropped, or drawn in the ``back`` style of
        :meth:`polyline`, :meth:`edges`, or :meth:`sphere_curve`.  Translucent
        surfaces hide nothing but are clipped by what is in front of them, and
        are painted back to front over the opaque ones.  Screen-space elements
        go on top in insertion order.  Other layers are untouched.

        It costs more than drawing in order -- up to a few seconds for a layer
        of hundreds of solids -- and adds one ``<clipPath>`` per partly hidden
        element.

        A beam inside a translucent slab still belongs on separate layers: a
        translucent face hides nothing, so no rule of visibility orders it.

        The setting is recorded, so rendering with another camera re-sorts.
        """
        self._add("sort_by_depth", layer)

    @_trackable
    def add(self, layer: int, element: svg.Element) -> None:
        """Add a ready-made ``svg.py`` element at ``layer``, bypassing projection."""
        self._add("add", layer, element)

    @_trackable
    def add_def(self, element: svg.Element) -> None:
        """Add an element to ``<defs>`` -- a gradient, marker, or clip path."""
        self._add("add_def", element)

    @_trackable
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
        self._add("rect2d", layer, x, y, w, h, grow, **style)

    @_trackable
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
        self._add("text2d", layer, x, y, s, size, grow, **style)


__all__ = [
    "DEFAULT_FONT",
    "DEFAULT_TEXT_FILL",
    "Align",
    "CameraRef",
    "Part",
    "Scene",
    "TextContent",
]
