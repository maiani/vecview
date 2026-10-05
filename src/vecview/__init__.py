"""Layered 3D scenes that render to SVG.

A small projection layer on top of `svg.py <https://pypi.org/project/svg.py/>`_.
``svg.py`` builds the elements; this package supplies what it has no notion of --
a camera, world-space glyph geometry, and an explicit layer stack.

World coordinates are right-handed ``(x, y, z)`` with ``z`` up.  Every camera is
a parallel projection, so parallel edges stay parallel and no perspective
distortion creeps into a lattice.  :class:`OrthographicCamera` covers the
axonometric views (isometric, dimetric, trimetric) and :class:`ObliqueCamera` the
cavalier and cabinet ones::

    import vecview

    cam = vecview.OrthographicCamera(azim_deg=35, elev_deg=24, scale=62)
    scene = vecview.Scene(cam, pad=28, background="#ffffff")

    slab = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
    scene.faces(10, cam.visible(slab), fill="#cfd6e0", stroke="#8b96a6")
    scene.save("slab.svg")

The rendered document is also available as a string from
:meth:`Scene.to_svg_document`, for handing to a tool that assembles a larger
document.
"""

from __future__ import annotations

from vecview._vec import unit
from vecview.camera import Camera, ParallelCamera
from vecview.projections import (
    ISOMETRIC_ELEV_DEG,
    ISOMETRIC_RATIO,
    ObliqueCamera,
    OrthographicCamera,
)
from vecview.scene import Scene
from vecview.shapes import (
    Face,
    annulus_sector,
    arrow_shape,
    box_faces,
    circle_shape,
    double_arrow_shape,
    ellipse_shape,
    in_plane_dir,
    prism_faces,
    rect_shape,
    sine_ribbon,
)

__version__ = "0.1.1"

__all__ = [
    "ISOMETRIC_ELEV_DEG",
    "ISOMETRIC_RATIO",
    "Camera",
    "Face",
    "ObliqueCamera",
    "OrthographicCamera",
    "ParallelCamera",
    "Scene",
    "__version__",
    "annulus_sector",
    "arrow_shape",
    "box_faces",
    "circle_shape",
    "double_arrow_shape",
    "ellipse_shape",
    "in_plane_dir",
    "prism_faces",
    "rect_shape",
    "sine_ribbon",
    "unit",
]
