"""The planar face record every solid in :mod:`vecview.shapes` is built from.

Kept apart so that the modules building solids and the one cutting them can
share it; :mod:`vecview.shapes` re-exports it.
"""

from __future__ import annotations

from typing import NamedTuple

from vecview._types import Array, Region


class Face(NamedTuple):
    """One planar polygon of a solid, with the outward normal it was built from.

    Carrying the normal alongside the points is what lets
    :meth:`~vecview.camera.Camera.visible` decide which walls of a box to draw
    without the caller reasoning about which octant the camera sits in.

    ``points`` winds counter-clockwise about ``normal``.  ``holes`` are rings
    cut out of it, each wound the other way -- the cap of a ring or a plate
    with vias; a face with holes is drawn as one even-odd ``<path>``.
    """

    name: str
    points: Array
    normal: Array
    holes: tuple[Array, ...] = ()

    @property
    def region(self) -> Region:
        """The face's planar region, usable by polygon and geometry operations."""
        return Region(self.points, self.holes)
