"""Exact visibility: ``sort_by_depth(layer, exact=True)``.

Visibility is checked as geometry: a clip region must contain the points where
its element is in front, and exclude the points where something else is.
"""

from __future__ import annotations

import builtins
import re

import numpy as np
import pytest
import svg

import vecview
from vecview import OrthographicCamera, Scene

pytest.importorskip("shapely")
pytest.importorskip("contourpy")
import shapely

CAM = OrthographicCamera(35.0, 24.0, 40.0)


def scene(exact: bool = True) -> Scene:
    out = Scene(CAM)
    out.sort_by_depth(0, exact=exact)
    return out


def clip_region(document: svg.SVG, clip_id: str) -> shapely.Geometry:
    """The region of a ``<clipPath>``, as shapely sees it, with even-odd holes."""
    defs = next(el for el in document.elements or [] if isinstance(el, svg.Defs))
    clip = next(el for el in defs.elements or [] if el.id == clip_id)
    (path,) = clip.elements or []
    rings, ring = [], []
    for cmd in path.d or []:
        if isinstance(cmd, svg.M) and ring:
            rings.append(ring)
            ring = []
        if hasattr(cmd, "x"):
            ring.append((cmd.x, cmd.y))
    rings.append(ring)
    region = shapely.Polygon()
    for r in rings:
        region = region.symmetric_difference(shapely.Polygon(r))
    return region


def by_id(document: svg.SVG) -> dict[str, svg.Element]:
    return {el.id: el for el in document.elements or [] if getattr(el, "id", None)}


class TestSurfaces:
    def test_untouched_wholly_clipped_and_dropped(self) -> None:
        s = scene()
        near = 3.0 * CAM.view
        s.sphere(0, near, 1.0, fill="#c33", id="front")
        s.sphere(0, -3.0 * CAM.view, 0.5, fill="#33c", id="hidden")  # behind, inside
        s.sphere(0, -3.0 * CAM.view + (0, 0, 1.3), 0.6, fill="#3c3", id="partly")
        s.sphere(0, (6, -6, 0), 0.5, fill="#999", id="apart")
        elements = by_id(s.render())
        assert "hidden" not in elements
        assert elements["front"].clip_path is None and elements["apart"].clip_path is None
        assert elements["partly"].clip_path == "url(#partly-visible)"

    def test_crossing_planes_each_show_where_they_are_in_front(self) -> None:
        """No order of whole planes draws a cross; clipping does."""
        s = scene()
        a = [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)]  # y = 0
        b = [(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1)]  # x = 0
        s.polygon(0, a, fill="#f4d03f", id="a")
        s.polygon(0, b, fill="#58d68d", id="b")
        document = s.render()
        regions = {k: clip_region(document, f"{k}-visible") for k in "ab"}
        for k in "ab":
            # Halfway out along each of the plane's two in-plane wings, one wing
            # is in front of the other plane and one behind it.
            pts = [(0.5, 0, 0.5), (-0.5, 0, 0.5)] if k == "a" else [(0, 0.5, 0.5), (0, -0.5, 0.5)]
            depth = CAM.depth(pts)
            nearer, farther = (pts[0], pts[1]) if depth[0] > depth[1] else (pts[1], pts[0])
            x, y = CAM.at(nearer)
            assert regions[k].contains(shapely.Point(x, y))
            x, y = CAM.at(farther)
            other = "b" if k == "a" else "a"
            if regions[other].contains(shapely.Point(x, y)):
                assert not regions[k].contains(shapely.Point(x, y))

    def test_a_bond_into_an_atom_stops_at_its_surface(self) -> None:
        s = scene()
        atom = 0.5 * CAM.view  # turned toward the camera, so it covers the bond's end
        s.sphere(0, atom, 0.6, fill="#c33", id="atom")
        s.cylinder(0, atom, atom + np.array([2.0, 0, 0]), 0.15, ends=False, fill="#999", id="bond")
        region = clip_region(s.render(), "bond-visible")
        x, y = CAM.at(atom)
        assert not region.contains(shapely.Point(x, y))
        x, y = CAM.at(atom + np.array([1.5, 0, 0]))
        assert region.contains(shapely.Point(x, y))

    def test_translucent_surfaces_hide_nothing(self) -> None:
        s = scene()
        s.sphere(0, 3.0 * CAM.view, 1.0, fill="#c33", fill_opacity=0.4, id="glass")
        s.sphere(0, -3.0 * CAM.view, 0.5, fill="#33c", id="behind")
        elements = by_id(s.render())
        assert elements["behind"].clip_path is None

    def test_neighbouring_pieces_of_one_tube_do_not_clip_each_other(self) -> None:
        s = scene()
        line = np.column_stack([np.linspace(0, 4, 30), np.zeros(30), np.zeros(30)])
        s.tube(0, line, 0.1, fill="#e67e22", stroke="#333", id="wire")
        assert "clipPath" not in s.to_svg_document()

    def test_a_coil_passes_behind_an_unsliced_core(self) -> None:
        s = scene()
        s.cylinder(0, (0, 0, -1), (0, 0, 1), 0.4, fill="#999", id="core")
        s.tube(0, vecview.helix((0, 0, -0.9), (0, 0, 1), 0.6, 0.6, 3), 0.05, fill="#c80", id="coil")
        document = s.to_svg_document()
        assert 'id="core" clip-path="url(#core-visible)"' in document
        assert re.search(r'id="coil-\d+" clip-path=', document)


class TestLines:
    def test_a_line_behind_a_sphere_is_cut_at_its_outline(self) -> None:
        s = scene()
        s.sphere(0, (0, 0, 0), 1.0, fill="#c33", id="ball")
        behind = -2.0 * CAM.view
        right, _ = CAM.screen_basis(CAM.view)
        s.polyline(
            0,
            [behind - 3 * right, behind + 3 * right],
            back={"stroke_dasharray": "3 2"},
            stroke="#000",
            id="ray",
        )
        elements = by_id(s.render())
        visible, hidden = elements["ray"], elements["ray-hidden"]
        assert isinstance(visible, svg.Path) and isinstance(hidden, svg.Path)
        assert hidden.stroke_dasharray == "3 2"
        ends = np.array([[c.x, c.y] for c in hidden.d or [] if hasattr(c, "x")])
        centre = np.array(CAM.at((0, 0, 0)))
        assert np.allclose(np.linalg.norm(ends - centre, axis=1), CAM.scale, atol=0.05)

    def test_hidden_parts_are_dropped_without_a_back_style(self) -> None:
        s = scene()
        s.sphere(0, (0, 0, 0), 1.0, fill="#c33", id="ball")
        s.polyline(
            0, [-2.0 * CAM.view + (-3, 3, 0), -2.0 * CAM.view + (3, -3, 0)], stroke="#000", id="ray"
        )
        assert "ray-hidden" not in by_id(s.render())

    def test_an_unobstructed_line_is_left_alone(self) -> None:
        plain, exact = scene(exact=False), scene()
        for s in (plain, exact):
            s.polyline(0, [(0, 0, 0), (1, 1, 1)], stroke="#000", id="ray")
        assert plain.to_svg_document() == exact.to_svg_document()

    def test_cell_edges_hidden_by_a_plane_disappear(self) -> None:
        s = scene()
        s.edges(
            0, vecview.box_faces((0, 0, 0), (2, 2, 2)), back={"stroke_dasharray": "2"}, id="cell"
        )
        s.polygon(0, vecview.rect_shape((0, 0, 0), (1, 0, 0), (0, 1, 0), 2, 2), fill="#58d", id="m")
        before = scene(exact=False)
        before.edges(
            0, vecview.box_faces((0, 0, 0), (2, 2, 2)), back={"stroke_dasharray": "2"}, id="cell"
        )
        before.polygon(
            0, vecview.rect_shape((0, 0, 0), (1, 0, 0), (0, 1, 0), 2, 2), fill="#58d", id="m"
        )

        def length(document: svg.SVG, ident: str) -> float:
            path = by_id(document)[ident]
            pts = [np.array([c.x, c.y]) for c in path.d or [] if hasattr(c, "x")]
            moves = [isinstance(c, svg.M) for c in path.d or [] if hasattr(c, "x")]
            return sum(
                float(np.linalg.norm(b - a))
                for a, b, new in zip(pts, pts[1:], moves[1:], strict=False)
                if not new
            )

        assert length(s.render(), "cell-back") < length(before.render(), "cell-back")


class TestContract:
    def test_output_is_deterministic(self) -> None:
        def build() -> str:
            s = scene()
            s.sphere(0, (0, 0, 0), 1.0, fill="#c33", id="a")
            s.cylinder(0, (0, 0, 0), (2, 1, 0.5), 0.2, fill="#999", id="b")
            return s.to_svg_document()

        assert build() == build()

    def test_without_the_extra_the_error_says_what_to_install(self, monkeypatch) -> None:
        real = builtins.__import__

        def refuse(name: str, *args: object, **kwargs: object) -> object:
            if name in ("shapely", "contourpy"):
                raise ImportError(name)
            return real(name, *args, **kwargs)  # type: ignore[arg-type]

        s = scene()
        s.sphere(0, (0, 0, 0), 1.0)
        monkeypatch.setattr(builtins, "__import__", refuse)
        with pytest.raises(ImportError, match=r"vecview\[occlusion\]"):
            s.render()

    def test_only_exact_layers_need_the_extra(self, monkeypatch) -> None:
        real = builtins.__import__

        def refuse(name: str, *args: object, **kwargs: object) -> object:
            if name in ("shapely", "contourpy"):
                raise ImportError(name)
            return real(name, *args, **kwargs)  # type: ignore[arg-type]

        s = scene(exact=False)
        s.sphere(0, (0, 0, 0), 1.0)
        monkeypatch.setattr(builtins, "__import__", refuse)
        s.render()

    def test_exactness_is_replayed_for_another_camera(self) -> None:
        s = scene()
        s.sphere(0, (0, 0, 0), 1.0, fill="#c33", id="a")
        s.sphere(0, (1.2, 0, 0), 0.8, fill="#33c", id="b")
        other = vecview.ObliqueCamera.cabinet(40.0)
        rebuilt = Scene(other)
        rebuilt.sort_by_depth(0, exact=True)
        rebuilt.sphere(0, (0, 0, 0), 1.0, fill="#c33", id="a")
        rebuilt.sphere(0, (1.2, 0, 0), 0.8, fill="#33c", id="b")
        assert str(s.render(other)) == rebuilt.to_svg_document()


def test_every_clip_is_a_valid_polygon_at_its_written_precision() -> None:
    """Rounding a clip after simplifying it once folded a thin spike over itself."""
    s = scene()
    atom = 0.5 * CAM.view
    s.sphere(0, atom, 0.6, fill="#c33", stroke="#000", id="atom")
    s.cylinder(0, atom, atom + np.array([2.0, 0, 0]), 0.15, ends=False, fill="#999", id="bond")
    s.cylinder(0, (3, 0, -1), (3, 0, 1), 0.4, fill="#999", id="core")
    s.tube(0, vecview.helix((3, 0, -0.9), (0, 0, 1), 0.6, 0.6, 3), 0.05, fill="#c80", id="coil")
    document = s.render()
    defs = next(el for el in document.elements or [] if isinstance(el, svg.Defs))
    clips = [el for el in defs.elements or [] if isinstance(el, svg.ClipPath)]
    assert clips
    for clip in clips:
        (path,) = clip.elements or []
        rings, ring = [], []
        for cmd in path.d or []:
            if isinstance(cmd, svg.M) and ring:
                rings.append(ring)
                ring = []
            if hasattr(cmd, "x"):
                ring.append((cmd.x, cmd.y))
        rings.append(ring)
        for r in rings:
            assert shapely.Polygon(r).is_valid, clip.id
