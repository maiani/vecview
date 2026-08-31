"""The parallel projections this package ships.

Two families, and the distinction is worth keeping straight because the words
are often used loosely:

**Axonometric** projections are orthographic -- the rays meet the projection
plane at right angles -- and are classified by how many of the three world axes
share a foreshortening ratio: three (isometric), two (dimetric), or none
(trimetric).  :class:`OrthographicCamera` covers all three.

**Oblique** projections keep one face true-shape and push the third axis away at
an angle, with rays that are *not* perpendicular to the projection plane.
:class:`ObliqueCamera` covers the cavalier and cabinet cases.  Oblique is
emphatically not axonometric, though both are parallel.
"""

from __future__ import annotations

import math

import numpy as np

from vecview._types import Array, Point3
from vecview.camera import ParallelCamera

ISOMETRIC_ELEV_DEG = math.degrees(math.asin(math.tan(math.radians(30.0))))
"""Elevation at which all three axes foreshorten equally: 35.264...  degrees."""

ISOMETRIC_RATIO = math.sqrt(2.0 / 3.0)
"""The shared foreshortening of a true isometric projection: 0.8165."""


class OrthographicCamera(ParallelCamera):
    """Orthographic axonometric camera, aimed by azimuth and elevation.

    The general constructor gives a **trimetric** projection; the two named
    constructors reach the constrained cases.  Which one you have is a question
    about the foreshortening ratios, and :meth:`axonometry` answers it.

    Args:
        azim_deg: Azimuth of the viewing direction, degrees CCW from ``+x``.
        elev_deg: Elevation of the viewing direction above the ``xy`` plane.
        scale: SVG user units per world unit.
        origin: World point that projects to the SVG origin.

    Attributes:
        right: World direction that projects to screen ``+x``.
        up: World direction that projects to screen *up* (SVG ``-y``).

    Unlike the base class, ``right`` and ``up`` here are genuine orthonormal
    world directions, because the projection is orthogonal.
    """

    def __init__(
        self,
        azim_deg: float,
        elev_deg: float,
        scale: float,
        origin: Point3 = (0.0, 0.0, 0.0),
    ) -> None:
        az, el = np.radians(float(azim_deg)), np.radians(float(elev_deg))
        self.azim_deg = float(azim_deg)
        self.elev_deg = float(elev_deg)
        self.right: Array = np.array([-np.sin(az), np.cos(az), 0.0])
        self.up: Array = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
        view = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        super().__init__(np.array([self.right, -self.up]), view, scale, origin)

    def __repr__(self) -> str:
        return (
            f"OrthographicCamera(azim_deg={self.azim_deg:g}, "
            f"elev_deg={self.elev_deg:g}, scale={self.scale:g})"
        )

    @classmethod
    def isometric(
        cls, scale: float, *, azim_deg: float = 45.0, origin: Point3 = (0.0, 0.0, 0.0)
    ) -> OrthographicCamera:
        r"""All three axes equally foreshortened, to ``0.8165`` of true length.

        The classic drafting view: the two horizontal axes come out at exactly
        30 degrees below the horizon, which is why a 30-60 set square draws it.

        Both angles are forced, not just the elevation.  Requiring the two
        horizontal axes to share a ratio gives
        :math:`(\sin^2 a - \cos^2 a)(1 - \sin^2 e) = 0`, so the azimuth must be
        an odd multiple of 45 degrees.  The four such azimuths are all isometric
        and differ only in which octant you view from, which is the one choice
        ``azim_deg`` leaves open.

        Note that "isometric drawing" as draftsmen use the term usually scales
        the result back up by ``1 / 0.8165`` so the axes measure true length;
        pass a correspondingly larger ``scale`` if you want that.

        Raises:
            ValueError: If ``azim_deg`` is not an odd multiple of 45 degrees,
                where the projection would not in fact be isometric.
        """
        if abs((float(azim_deg) - 45.0) % 90.0) > 1e-9:
            raise ValueError(
                f"azim_deg={azim_deg:g} is not isometric: equal foreshortening of the "
                "two horizontal axes requires an odd multiple of 45 degrees "
                "(45, 135, 225, 315), which selects the viewing octant"
            )
        return cls(azim_deg, ISOMETRIC_ELEV_DEG, scale, origin)

    @classmethod
    def dimetric(
        cls, scale: float, *, ratio: float = 0.5, origin: Point3 = (0.0, 0.0, 0.0)
    ) -> OrthographicCamera:
        r"""Two axes equally foreshortened, the vertical one by ``ratio`` of those.

        The azimuth is fixed at 45 degrees, which is what makes ``x`` and ``y``
        share a ratio at any elevation; ``ratio`` then picks the elevation:

        .. math:: \sin^2 e = \frac{2 - r^2}{2 + r^2}

        ``ratio=1`` is isometric. The drafting-standard ``1:1:0.5`` is the
        default, and puts the receding axes 41.4 degrees below the horizon.

        Args:
            ratio: Vertical foreshortening as a fraction of the horizontal one.
                Must lie in ``(0, sqrt(2)]``; at ``sqrt(2)`` the elevation
                reaches zero and the view is edge-on to the ground plane.
        """
        if not 0.0 < ratio <= math.sqrt(2.0):
            raise ValueError(f"ratio must lie in (0, sqrt(2)], got {ratio:g}")
        sin_sq = (2.0 - ratio**2) / (2.0 + ratio**2)
        elev = math.degrees(math.asin(math.sqrt(max(sin_sq, 0.0))))
        return cls(45.0, elev, scale, origin)

    def axonometry(self, *, tol: float = 1e-6) -> str:
        """Classify this projection: ``"isometric"``, ``"dimetric"``, or ``"trimetric"``.

        Decided from :meth:`foreshortening`, so it reports what the camera
        actually does rather than how it was constructed.
        """
        fx, fy, fz = self.foreshortening()
        equal = sum(abs(a - b) <= tol for a, b in ((fx, fy), (fy, fz), (fx, fz)))
        if equal == 3:
            return "isometric"
        if equal >= 1:
            return "dimetric"
        return "trimetric"


class ObliqueCamera(ParallelCamera):
    """Oblique parallel projection: one plane true-shape, the third axis pushed back.

    The ``xz`` plane is drawn at true shape and true angle -- world ``x`` runs
    along screen ``+x`` and world ``z`` straight up -- while world ``y`` recedes
    at ``angle_deg`` above the horizon, foreshortened by ``depth_ratio``.

    Oblique is **not** axonometric: its rays are not perpendicular to the
    projection plane, so no choice of azimuth and elevation reproduces it.  It
    is still a parallel projection, so every guarantee this package rests on --
    parallel edges, exact affine plane embedding -- continues to hold.

    Args:
        scale: SVG user units per world unit.
        depth_ratio: Foreshortening of the receding ``y`` axis. ``1`` is
            cavalier, ``0.5`` cabinet.
        angle_deg: Screen angle of the receding axis above the horizon,
            conventionally 30, 45, or 60. Must not be a multiple of 180, which
            would flatten ``y`` onto the horizontal axis.
        origin: World point that projects to the SVG origin.
    """

    def __init__(
        self,
        scale: float,
        *,
        depth_ratio: float = 0.5,
        angle_deg: float = 45.0,
        origin: Point3 = (0.0, 0.0, 0.0),
    ) -> None:
        if depth_ratio <= 0.0:
            raise ValueError(f"depth_ratio must be positive, got {depth_ratio:g}")
        a = math.radians(float(angle_deg))
        if abs(math.sin(a)) < 1e-12:
            raise ValueError(
                f"angle_deg={angle_deg:g} puts the receding axis on the horizontal, "
                "collapsing the projection"
            )
        self.depth_ratio = float(depth_ratio)
        self.angle_deg = float(angle_deg)
        dx = self.depth_ratio * math.cos(a)
        dy = self.depth_ratio * math.sin(a)
        # Row 0 is screen x, row 1 screen y (downward): x -> right, z -> up, and
        # y -> up-and-right by (dx, dy).
        matrix = np.array([[1.0, dx, 0.0], [0.0, -dy, -1.0]])
        # The projection direction is the null vector of `matrix`; the camera
        # lies along it on the -y side, so front faces (normal -y) are visible.
        view = np.array([dx, -1.0, dy])
        super().__init__(matrix, view, scale, origin)

    def __repr__(self) -> str:
        return (
            f"ObliqueCamera(scale={self.scale:g}, depth_ratio={self.depth_ratio:g}, "
            f"angle_deg={self.angle_deg:g})"
        )

    @classmethod
    def cavalier(
        cls, scale: float, *, angle_deg: float = 45.0, origin: Point3 = (0.0, 0.0, 0.0)
    ) -> ObliqueCamera:
        """Cavalier oblique: the receding axis at full length.

        Nothing is foreshortened, so every edge measures true -- convenient, and
        the reason the result looks too deep to the eye.
        """
        return cls(scale, depth_ratio=1.0, angle_deg=angle_deg, origin=origin)

    @classmethod
    def cabinet(
        cls, scale: float, *, angle_deg: float = 45.0, origin: Point3 = (0.0, 0.0, 0.0)
    ) -> ObliqueCamera:
        """Cabinet oblique: the receding axis at half length.

        Halving the depth is what makes cabinet projection look right where
        cavalier looks stretched, at the cost of one axis no longer measuring
        true.
        """
        return cls(scale, depth_ratio=0.5, angle_deg=angle_deg, origin=origin)


__all__ = [
    "ISOMETRIC_ELEV_DEG",
    "ISOMETRIC_RATIO",
    "ObliqueCamera",
    "OrthographicCamera",
]
