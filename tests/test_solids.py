"""Curved solids, hidden lines, and depth ordering.

Outlines are checked as geometry: a projected solid's outline must contain every
projected surface point and touch the extreme ones.
"""

from __future__ import annotations

import numpy as np
import pytest
import svg

import vecview
from vecview import Camera, ObliqueCamera, OrthographicCamera, Scene

CAMERAS = [
    OrthographicCamera(35.0, 24.0, 10.0),
    OrthographicCamera.isometric(10.0),
    ObliqueCamera.cabinet(10.0),
]
IDS = ["trimetric", "isometric", "cabinet"]


def ids(scene: Scene) -> list[str | None]:
    """The drawn elements' ids in paint order, leaving out ``<defs>``."""
    return [el.id for el in (scene.render().elements or []) if not isinstance(el, svg.Defs)]


def circle_points(center: np.ndarray, axis: np.ndarray, r: float, n: int = 720) -> np.ndarray:
    return vecview.circle_shape(center, r, axis, n=n) if r > 0 else np.array([center])


class TestSphere:
    def test_orthographic_outline_is_a_circle_of_the_scaled_radius(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.sphere(0, (1, 2, 3), 0.5)
        (circle,) = scene.render().elements or []
        assert isinstance(circle, svg.Circle)
        assert circle.r == pytest.approx(5.0)
        assert (circle.cx, circle.cy) == pytest.approx(scene.camera.at((1, 2, 3)), abs=0.01)

    def test_oblique_outline_is_the_ellipse_that_bounds_the_ball(self) -> None:
        cam = ObliqueCamera.cabinet(10.0)
        scene = Scene(cam)
        scene.sphere(0, (0, 0, 0), 1.0)
        (el,) = scene.render().elements or []
        assert isinstance(el, svg.Ellipse)
        rng = np.random.default_rng(0)
        d = rng.normal(size=(4000, 3))
        surface = d / np.linalg.norm(d, axis=1)[:, None]
        p = cam.project(surface)
        t = np.radians(-float(el.transform[0].a)) if el.transform else 0.0
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        q = (p - (el.cx, el.cy)) @ rot.T
        level = (q[:, 0] / el.rx) ** 2 + (q[:, 1] / el.ry) ** 2
        assert level.max() == pytest.approx(1.0, abs=2e-3)

    def test_highlight_shares_one_gradient_per_colour_pair(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        for x in range(3):
            scene.sphere(0, (x, 0, 0), 0.4, fill="#cc3333", highlight="#ffffff")
        scene.sphere(0, (0, 2, 0), 0.4, fill="#3333cc", highlight="#ffffff")
        defs, first, *_ = scene.render().elements or []
        assert [d.id for d in defs.elements] == ["ball-cc3333-ffffff", "ball-3333cc-ffffff"]
        assert first.fill == "url(#ball-cc3333-ffffff)"

    def test_highlight_needs_a_fill(self) -> None:
        with pytest.raises(ValueError, match="fill"):
            Scene(OrthographicCamera(35.0, 24.0, 10.0)).sphere(0, (0, 0, 0), 1, highlight="#fff")


def outline_lines(path: svg.Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """The straight sides of an outline: each L segment with its start point."""
    lines, here = [], None
    for cmd in path.d or []:
        point = np.array([cmd.x, cmd.y], dtype=float) if hasattr(cmd, "x") else None
        if isinstance(cmd, svg.L) and here is not None:
            lines.append((here, point))
        if point is not None:
            here = point
    return lines


class TestFrustumOutline:
    @pytest.mark.parametrize("cam", CAMERAS, ids=IDS)
    @pytest.mark.parametrize(
        ("p1", "r1"), [((2, 1, 0.5), 0.6), ((0, 0, 3), 0.6), ((1.5, -1, 1), 0.2), ((1, 1, 2), 0.0)]
    )
    def test_sides_are_supporting_lines_of_both_ends(
        self, cam: Camera, p1: tuple[float, ...], r1: float
    ) -> None:
        scene = Scene(cam)
        p0, axis = np.zeros(3), np.asarray(p1, dtype=float)
        scene.cylinder(0, p0, axis, 0.6, r1=r1, ends=False)
        (group,) = scene.render().elements or []
        (body,) = group.elements
        surface = cam.project(
            np.vstack([circle_points(p0, axis, 0.6), circle_points(axis, axis, r1)])
        )
        lines = outline_lines(body)
        assert len(lines) == 2
        for a, b in lines:
            edge = b - a
            side = edge[0] * (surface[:, 1] - a[1]) - edge[1] * (surface[:, 0] - a[0])
            side /= np.linalg.norm(edge)
            # Every surface point on one side, and the line touching the surface.
            assert min(side.max(), -side.min()) < 0.02
            assert np.abs(side).min() < 0.02

    def test_seen_down_the_axis_the_outline_is_the_larger_end(self) -> None:
        cam = OrthographicCamera(0.0, 90.0, 10.0)
        scene = Scene(cam)
        scene.cylinder(0, (0, 0, 0), (0, 0, 2), 0.5, r1=0.8, ends=False)
        (group,) = scene.render().elements or []
        arcs = [c for c in group.elements[0].d if isinstance(c, svg.Arc)]
        assert [a.rx for a in arcs] == pytest.approx([8.0, 8.0])

    def test_the_end_facing_the_camera_is_drawn_after_the_body(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.cylinder(0, (0, 0, 0), (0, 0, 1), 0.5, id="post")
        (group,) = scene.render().elements or []
        assert [el.id for el in group.elements] == ["post-body", "post-end1"]

    def test_open_ends_draw_only_the_body(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.cylinder(0, (0, 0, 0), (0, 0, 1), 0.5, ends=False)
        (group,) = scene.render().elements or []
        assert len(group.elements) == 1

    def test_cone_from_below_shows_its_base(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.cone(0, (0, 0, 1), (0, 0, 0), 0.5, id="funnel")
        (group,) = scene.render().elements or []
        assert [el.id for el in group.elements] == ["funnel-body", "funnel-end0"]

    def test_highlight_needs_an_id(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        with pytest.raises(ValueError, match="id"):
            scene.cylinder(0, (0, 0, 0), (1, 0, 0), 0.2, fill="#888", highlight="#eee")

    def test_rejects_coincident_ends(self) -> None:
        with pytest.raises(ValueError, match="coincide"):
            Scene(OrthographicCamera(35.0, 24.0, 10.0)).cylinder(0, (1, 1, 1), (1, 1, 1), 0.2)


class TestSlicedCylinder:
    def build(self, **style: object) -> Scene:
        scene = Scene(OrthographicCamera(-66.0, 14.0, 10.0))
        scene.cylinder(0, (-2, 0, 0), (2, 0, 0), 0.5, slices=4, id="core", **style)
        return scene

    def test_one_group_per_slice(self) -> None:
        assert ids(self.build(stroke="#000")) == ["core-0", "core-1", "core-2", "core-3"]

    def test_only_the_outer_slices_stroke_an_end_arc(self) -> None:
        groups = self.build(stroke="#000").render().elements or []
        edges = [next(el for el in g.elements if el.id.endswith("edge")) for g in groups]
        arcs = [sum(isinstance(c, svg.Arc) for c in e.d) for e in edges]
        assert arcs == [1, 0, 0, 1]

    def test_slices_sort_independently_around_a_coil(self) -> None:
        """A ring round the far end must not sort wholly in front of the near end."""
        cam = OrthographicCamera(-66.0, 14.0, 10.0)
        scene = Scene(cam)
        scene.sort_by_depth(0)
        scene.cylinder(0, (-2, 0, 0), (2, 0, 0), 0.5, slices=4, id="core")
        far_end = np.argmin([cam.depth([(x, 0, 0)])[0] for x in (-2.0, 2.0)])
        x = (-1.75, 1.75)[far_end]
        scene.sphere(0, (x, 0, 0) + 0.6 * cam.view, 0.05, id="front-of-far-end")
        order = ids(scene)
        assert order.index("front-of-far-end") > order.index(f"core-{3 * far_end}")

    def test_highlight_shares_one_gradient(self) -> None:
        scene = self.build(fill="#888888", highlight="#eeeeee")
        defs = (scene.render().elements or [])[0]
        assert [d.id for d in defs.elements] == ["core-shade"]

    @pytest.mark.parametrize("slices", [1, 4])
    def test_slicing_renames_no_end_disk_or_gradient(self, slices: int) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.cylinder(
            0, (0, 0, 0), (0, 0, 2), 0.5, fill="#888", highlight="#eee", slices=slices, id="core"
        )
        document = str(scene.render())
        assert 'id="core-shade"' in document and 'id="core-end1"' in document

    def test_rejects_zero_slices(self) -> None:
        with pytest.raises(ValueError, match="slices"):
            Scene(OrthographicCamera(35.0, 24.0, 10.0)).cylinder(
                0, (0, 0, 0), (1, 0, 0), 0.2, slices=0
            )


class TestArrow3d:
    @pytest.mark.parametrize(
        ("direction", "first"), [((0, 0, 1), "a-shaft"), ((0, 0, -1), "a-head")]
    )
    def test_the_nearer_part_is_drawn_last(self, direction: tuple[int, ...], first: str) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.arrow3d(0, (0, 0, 0), direction, 1.0, shaft_r=0.05, head_r=0.15, head_len=0.3, id="a")
        (group,) = scene.render().elements or []
        assert group.elements[0].id == first

    def test_a_short_arrow_is_all_head(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.arrow3d(0, (0, 0, 0), (1, 0, 0), 0.2, shaft_r=0.05, head_r=0.15, head_len=0.3, id="a")
        (group,) = scene.render().elements or []
        assert all(el.id.startswith("a-head") for el in group.elements)


class TestTube:
    def test_pieces_overlap_and_carry_suffixed_ids(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        pts = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
        scene.tube(0, pts, 0.2, chunk=4, fill="#e67e22", stroke="#333", id="wire")
        groups = scene.render().elements or []
        assert [g.id for g in groups] == ["wire-0", "wire-1", "wire-2"]
        outline, body = groups[0].elements
        assert outline.stroke_width > body.stroke_width == pytest.approx(4.0)
        assert len(body.points) > len(outline.points), "the body overruns its outline"

    def test_without_a_stroke_only_the_body_is_drawn(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.tube(0, [(0, 0, 0), (1, 0, 0)], 0.2, fill="red")
        (group,) = scene.render().elements or []
        assert len(group.elements) == 1


class TestEdges:
    def test_a_box_shows_nine_edges_and_hides_three(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.edges(
            1, vecview.box_faces((0, 0, 0), (1, 1, 1)), back={"stroke_dasharray": "2"}, id="cell"
        )
        front, back = scene.render().elements or []
        moves = lambda p: sum(isinstance(c, svg.M) for c in p.d)  # noqa: E731
        assert (front.id, moves(front)) == ("cell-front", 9)
        assert (back.id, moves(back), back.stroke_dasharray) == ("cell-back", 3, "2")

    def test_separate_edges_are_one_path_each(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.edges(1, vecview.box_faces((0, 0, 0), (1, 1, 1)), separate=True, id="cell")
        assert ids(scene) == [f"cell-front-{k}" for k in range(9)]

    def test_trim_shortens_both_ends(self) -> None:
        def lengths(trim: float) -> np.ndarray:
            scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
            scene.edges(1, vecview.box_faces((0, 0, 0), (2, 2, 2)), separate=True, trim=trim)
            ends = [[(c.x, c.y) for c in path.d] for path in scene.render().elements or []]
            return np.array([np.linalg.norm(np.subtract(*pair)) for pair in ends])

        assert np.allclose(lengths(0.25), 0.75 * lengths(0.0), atol=0.02)

    def test_hidden_edges_are_omitted_by_default(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.edges(1, vecview.box_faces((0, 0, 0), (1, 1, 1)))
        assert len(scene.render().elements or []) == 1

    def test_hidden_edges_can_sit_on_their_own_layer(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.polygon(5, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], id="veil")
        scene.edges(9, vecview.box_faces((0, 0, 0), (1, 1, 1)), back={}, back_layer=1, id="e")
        assert ids(scene) == ["e-back", "veil", "e-front"]


class TestSphereCurve:
    def test_an_equator_seen_from_above_splits_into_two_halves(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        ring = vecview.circle_shape((0, 0, 0), 1.0, (0, 0, 1), n=64)
        scene.sphere_curve(1, (0, 0, 0), ring, closed=True, back={}, id="eq")
        front, back = scene.render().elements or []
        for half in (front, back):
            assert sum(isinstance(c, svg.M) for c in half.d) == 1, "each half is one run"
        # The cut points lie on the outline circle of the unit sphere.
        ends = np.array([[c.x, c.y] for c in front.d if hasattr(c, "x")])[[0, -1]]
        assert np.allclose(np.linalg.norm(ends, axis=1), 10.0, atol=0.02)

    def test_a_curve_entirely_in_front_draws_once(self) -> None:
        cam = OrthographicCamera(35.0, 24.0, 10.0)
        scene = Scene(cam)
        ring = vecview.circle_shape(0.9 * cam.view, 0.3, cam.view)
        scene.sphere_curve(1, (0, 0, 0), ring, closed=True, back={})
        assert len(scene.render().elements or []) == 1


class TestSortByDepth:
    def build(self, cam: Camera, sort: bool) -> Scene:
        scene = Scene(cam)
        if sort:
            scene.sort_by_depth(1)
        # Offset on screen, so the near sphere hides only part of the far one.
        right, _ = cam.screen_basis(cam.view)
        near, far = 2.0 * cam.view, -2.0 * cam.view + 0.6 * right
        scene.sphere(1, near, 0.5, id="near")
        scene.sphere(1, far, 0.5, id="far")
        scene.text2d(1, 0, 0, "label", id="label")
        scene.sphere(0, near, 0.5, id="other-layer")
        return scene

    def test_layers_keep_insertion_order_unless_asked(self) -> None:
        scene = self.build(OrthographicCamera(35.0, 24.0, 10.0), sort=False)
        assert ids(scene) == ["other-layer", "near", "far", "label"]

    def test_a_sorted_layer_paints_what_hides_last_with_screen_space_on_top(self) -> None:
        scene = self.build(OrthographicCamera(35.0, 24.0, 10.0), sort=True)
        assert ids(scene) == ["other-layer", "far", "near", "label"]

    def test_equal_depths_keep_insertion_order(self) -> None:
        scene = Scene(OrthographicCamera(0.0, 0.0, 10.0))
        scene.sort_by_depth(0)
        for k in range(4):
            scene.sphere(0, (0, k, 0), 0.2, id=f"s{k}")  # all at one depth from +x
        assert ids(scene) == ["s0", "s1", "s2", "s3"]

    def test_a_wholly_hidden_surface_is_dropped(self) -> None:
        cam = OrthographicCamera(35.0, 24.0, 10.0)
        scene = Scene(cam)
        scene.sort_by_depth(0)
        scene.sphere(0, 2.0 * cam.view, 0.5, id="near")
        scene.sphere(0, -2.0 * cam.view, 0.4, id="behind")
        assert ids(scene) == ["near"]

    def test_reprojection_resorts(self) -> None:
        scene = self.build(OrthographicCamera(35.0, 24.0, 10.0), sort=True)
        flipped = scene.with_camera(OrthographicCamera(215.0, -24.0, 10.0))
        assert ids(flipped)[1:3] == ["near", "far"]


class TestJupyter:
    def test_repr_svg_is_the_document(self) -> None:
        scene = Scene(OrthographicCamera(35.0, 24.0, 10.0))
        scene.sphere(0, (0, 0, 0), 1.0)
        assert scene._repr_svg_() == scene.to_svg_document()

    def test_an_empty_scene_falls_back_to_its_repr(self) -> None:
        assert Scene(OrthographicCamera(35.0, 24.0, 10.0))._repr_svg_() is None
