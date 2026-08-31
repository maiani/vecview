"""Cameras: world coordinates in, SVG user units out.

Every camera here is a **parallel projection** -- the projection rays are all
mutually parallel, so parallel world edges stay parallel on screen and no
perspective distortion creeps into a lattice or a repeated texture.  That is the
reason schematics of crystals and optical benches are conventionally drawn this
way, and it is what makes an embedded plot an exact affine transform (see
:meth:`Camera.plane_matrix`).

Parallel projections split into two families, and the distinction is worth
keeping straight because the words are often used loosely:

**Axonometric** projections are *orthographic*: the rays meet the projection
plane at right angles.  They are classified by how many of the three world axes
share a foreshortening ratio -- three (isometric), two (dimetric), or none
(trimetric).  :class:`OrthographicCamera` covers all three.

**Oblique** projections keep one face true-shape and push the third axis away at
an angle, with rays that are *not* perpendicular to the projection plane.
Cavalier and cabinet projections are the common cases.
:class:`ObliqueCamera` covers them.  Oblique is emphatically not axonometric,
though both are parallel.

The class hierarchy follows that split, and leaves room for the one projection
this package does not implement.  The concrete projections live in
:mod:`vecview.projections`; this module defines only what a camera *is*::

    Camera                    the projection contract, and nothing more
    +-- ParallelCamera        affine: constant rays, so screen offsets are
    |   |                     position-independent
    |   +-- OrthographicCamera    axonometric, aimed by azimuth and elevation
    |   +-- ObliqueCamera         cavalier and cabinet
    +-- PerspectiveCamera     not implemented; would subclass Camera directly

A perspective camera belongs on the second branch rather than under
:class:`ParallelCamera`, because it is projective rather than affine.  It could
still project points and report depth -- that is all :class:`Camera` promises --
but :meth:`ParallelCamera.direction`, :meth:`ParallelCamera.screen_basis`,
:meth:`ParallelCamera.foreshortening` and :meth:`ParallelCamera.plane_matrix`
would all be meaningless for it, since each assumes a screen offset that does
not depend on where in the scene you are.  Keeping them one level down is what
stops that from becoming a silent wrong answer.

World coordinates are right-handed ``(x, y, z)`` with ``z`` up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from vecview._types import Array, Point3, Points3
from vecview._vec import as_points, basis_for, unit

if TYPE_CHECKING:
    from vecview.shapes import Face


class Camera(ABC):
    """The projection contract every camera satisfies, and nothing more.

    A camera turns world points into SVG user units, reports depth so a caller
    can order things itself, and says which faces point at it.  Everything
    beyond that depends on *which* projection is in use and lives on a subclass.

    Instantiate :class:`OrthographicCamera` or :class:`ObliqueCamera`; subclass
    this directly only to add a projection that is not parallel.

    Args:
        scale: SVG user units per world unit.
        origin: World point that projects to the SVG origin.
    """

    def __init__(self, scale: float, origin: Point3 = (0.0, 0.0, 0.0)) -> None:
        self.scale = float(scale)
        self.origin: Array = np.asarray(origin, dtype=np.float64)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scale={self.scale:g})"

    @abstractmethod
    def project(self, pts: Points3) -> Array:
        """World points -> SVG user units, shape ``(n, 2)``, ``y`` already flipped."""

    @abstractmethod
    def depth(self, pts: Points3) -> Array:
        """Signed depth per point; larger is nearer the camera."""

    @abstractmethod
    def visible(self, faces: Iterable[Face], *, tol: float = 0.0) -> list[Face]:
        """Keep the faces pointing toward the camera.

        Abstract because the test itself differs: for a parallel projection an
        outward normal decides it on its own, while a perspective camera has to
        ask where the face *is* relative to the eye.
        """

    def at(self, pt: Point3) -> tuple[float, float]:
        """Project one world point to an ``(x, y)`` pair."""
        ((x, y),) = self.project([pt])
        return float(x), float(y)


class ParallelCamera(Camera):
    """A parallel projection: all rays mutually parallel, so the map is affine.

    Being affine is the load-bearing property.  It is what makes a screen offset
    independent of position (:meth:`direction`), lets a plane's screen frame be
    solved once (:meth:`screen_basis`), gives every axis a single foreshortening
    ratio (:meth:`foreshortening`), and allows flat content to be embedded in a
    world plane exactly (:meth:`plane_matrix`).

    Instantiate this directly only to define a parallel projection of your own;
    otherwise use :class:`OrthographicCamera` or :class:`ObliqueCamera`.

    Args:
        matrix: The ``(2, 3)`` linear map from world displacements to screen
            ones, *before* ``scale``. Row 0 gives screen ``x``, row 1 gives
            screen ``y`` (downward, as SVG counts it).
        view: Direction from the scene *toward* the camera. A face whose outward
            normal has a positive dot product with this is visible. It need not
            be perpendicular to the projection plane -- for an oblique camera it
            is not.
        scale: SVG user units per world unit.
        origin: World point that projects to the SVG origin.
    """

    def __init__(
        self,
        matrix: Array | Iterable[Iterable[float]],
        view: Point3,
        scale: float,
        origin: Point3 = (0.0, 0.0, 0.0),
    ) -> None:
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (2, 3):
            raise ValueError(
                f"projection matrix must have shape (2, 3), got {m.shape}. "
                "To build a camera from angles use OrthographicCamera("
                "azim_deg, elev_deg, scale)."
            )
        if np.linalg.matrix_rank(m) < 2:
            raise ValueError("projection matrix is rank-deficient: it collapses the scene")
        self.matrix: Array = m
        self.view: Array = unit(view)
        super().__init__(scale, origin)

    # --- projection -------------------------------------------------------
    def project(self, pts: Points3) -> Array:
        """World points -> SVG user units, shape ``(n, 2)``.

        The ``y`` axis is already flipped to SVG's screen convention, so the
        result can go straight into an element's coordinates.
        """
        p = as_points(pts) - self.origin
        out: Array = (p @ self.matrix.T) * self.scale
        return out

    def direction(self, vec: Point3) -> Array:
        """Project a world *direction* (not a point) to a screen displacement.

        Unlike :meth:`project` this ignores ``origin``: a displacement has no
        position, so moving the camera must not change it.
        """
        d = np.asarray(vec, dtype=np.float64)
        out: Array = (self.matrix @ d) * self.scale
        return out

    def depth(self, pts: Points3) -> Array:
        """Signed distance along the view axis; larger is nearer the camera.

        Provided as a query for callers who want to order something by depth
        themselves.  :class:`~vecview.scene.Scene` never sorts by depth on its
        own -- see its class docstring for why.
        """
        depths: Array = np.asarray(as_points(pts) @ self.view, dtype=np.float64)
        return depths

    # --- measurement ------------------------------------------------------
    def foreshortening(self) -> tuple[float, float, float]:
        """Screen length of each unit world axis, in units of ``scale``.

        The three numbers are what classify an axonometric projection, and they
        are the honest way to check you got the view you meant::

            OrthographicCamera.isometric(1.0).foreshortening()
            # (0.8165, 0.8165, 0.8165)
        """
        lengths = np.linalg.norm(self.matrix, axis=0)
        return (float(lengths[0]), float(lengths[1]), float(lengths[2]))

    # --- frames -----------------------------------------------------------
    def screen_basis(self, normal: Point3 = (0.0, 0.0, 1.0)) -> tuple[Array, Array]:
        """In-plane world directions that move a point right and *down* on screen.

        Placing things by eye inside a projected plane is otherwise guesswork:
        at an azimuth of 35 degrees neither ``+x`` nor ``+y`` moves a point
        horizontally across the picture.  This returns the two unit world
        vectors lying in the plane with the given ``normal`` that do.

        Returns:
            ``(horizontal, down)``, both unit vectors in the plane.

        Raises:
            ValueError: If the plane is viewed edge-on.  It then projects to a
                line, so no in-plane direction has a screen ``y`` component and
                the mapping cannot be inverted.
        """
        u, v = basis_for(normal)
        # Columns are the screen displacements of the in-plane basis vectors.
        m = np.column_stack([self.direction(u), self.direction(v)])
        if abs(float(np.linalg.det(m))) < 1e-12:
            raise ValueError(
                "plane is viewed edge-on; it has no in-plane screen basis "
                f"(normal={np.asarray(normal, dtype=np.float64)}, camera={self!r})"
            )
        inv = np.linalg.inv(m)
        right_coeffs = inv @ np.array([1.0, 0.0])  # screen +x
        down_coeffs = inv @ np.array([0.0, 1.0])  # screen +y, i.e. downward
        return (
            unit(right_coeffs[0] * u + right_coeffs[1] * v),
            unit(down_coeffs[0] * u + down_coeffs[1] * v),
        )

    def plane_matrix(
        self, origin: Point3, u_edge: Point3, v_edge: Point3
    ) -> tuple[float, float, float, float, float, float]:
        """Affine map taking the unit square onto a world-space rectangle.

        Returns the six coefficients ``(a, b, c, d, e, f)`` of an SVG
        ``matrix(a b c d e f)``, which sends content coordinates in
        :math:`[0,1]^2` to the parallelogram spanned by ``u_edge`` and
        ``v_edge`` at ``origin``, as projected by this camera.

        This is what lets flat SVG content -- a plot, an equation, a bitmap --
        sit *in* a plane of the scene rather than on top of the picture.  It is
        exact rather than approximate, and for a reason worth knowing: a
        parallel projection is an affine map of world space, so restricting it
        to a plane and composing with that plane's affine parametrization gives
        an affine map of the content, which is precisely what ``matrix``
        expresses.  A perspective camera would be *projective* instead, and the
        restriction a homography that ``matrix`` cannot represent.

        Args:
            origin: World point that content coordinate ``(0, 0)`` lands on --
                the content's top-left corner, since SVG ``y`` grows downward.
            u_edge: World vector from that corner along content ``+x``.  Its
                length is the rectangle's width; it is **not** normalized.
            v_edge: World vector from that corner along content ``+y``, i.e.
                *down* the content.  Its length is the rectangle's height.

        Returns:
            ``(a, b, c, d, e, f)`` for ``matrix(a b c d e f)``.

        Raises:
            ValueError: If the rectangle is degenerate or seen exactly edge-on.
                It then projects to a line, collapsing the content to nothing.

        Content stays upright and unmirrored exactly when ``a > 0`` and
        ``d > 0`` -- its ``+x`` must project rightward and its ``+y`` downward.
        A positive determinant alone is **not** enough: a 180-degree rotation
        has one too.  See :meth:`vecview.scene.Scene.plane` for how to choose the
        edges.
        """
        a, b = self.direction(u_edge)
        c, d = self.direction(v_edge)
        if abs(a * d - b * c) < 1e-12:
            raise ValueError(
                "degenerate plane rectangle: it projects to a line, so content "
                f"placed in it would collapse (camera={self!r})"
            )
        e, f = self.at(origin)
        return (float(a), float(b), float(c), float(d), e, f)

    # --- visibility -------------------------------------------------------
    def faces_camera(self, normal: Point3, *, tol: float = 0.0) -> bool:
        """Whether an outward-facing ``normal`` points toward the camera."""
        return float(np.dot(unit(normal), self.view)) > tol

    def visible(self, faces: Iterable[Face], *, tol: float = 0.0) -> list[Face]:
        """Keep only the faces whose outward normals point toward the camera.

        This is back-face culling and nothing more.  It is exactly right for a
        convex solid such as a box, where it leaves the three visible walls; it
        does not resolve one object occluding another, which is what the layer
        stack is for.

        Args:
            faces: Faces to filter, typically from :func:`~vecview.shapes.box_faces`.
            tol: Raise above ``0`` to also drop faces seen nearly edge-on,
                which project to slivers.
        """
        return [f for f in faces if self.faces_camera(f.normal, tol=tol)]


__all__ = ["Camera", "ParallelCamera"]
