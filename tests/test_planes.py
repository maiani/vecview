"""Embedding flat content in a world plane.

The claim under test is that the embedding is *exact*: a parallel projection is
affine, so restricted to a plane it stays affine, and an SVG ``matrix`` carries
it with no error anywhere -- not merely at the corners.
"""

from __future__ import annotations

import numpy as np
import pytest
import svg

from vecview import ObliqueCamera, OrthographicCamera, Scene

UNIT_SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def apply(matrix: tuple[float, ...], x: float, y: float) -> np.ndarray:
    a, b, c, d, e, f = matrix
    return np.array([a * x + c * y + e, b * x + d * y + f])


@pytest.fixture
def cam() -> OrthographicCamera:
    return OrthographicCamera(35.0, 26.0, 62.0)


class TestPlaneMatrix:
    def test_maps_unit_square_corners_onto_the_world_rectangle(
        self, cam: OrthographicCamera
    ) -> None:
        origin, u, v = (
            np.array([-4.0, 3.0, 0.0]),
            np.array([8.0, 0.0, 0.0]),
            np.array([0.0, -6.0, 0.0]),
        )
        m = cam.plane_matrix(origin, u, v)
        for (x, y), corner in zip(
            UNIT_SQUARE, [origin, origin + u, origin + u + v, origin + v], strict=True
        ):
            assert apply(m, x, y) == pytest.approx(cam.at(corner), abs=1e-12)

    def test_is_exact_in_the_interior_too(self, cam: OrthographicCamera) -> None:
        """Corners agreeing is necessary; affine means everywhere agrees."""
        origin, u, v = (
            np.array([1.0, -2.0, 0.5]),
            np.array([5.0, 1.0, 0.0]),
            np.array([0.0, -2.0, -3.0]),
        )
        m = cam.plane_matrix(origin, u, v)
        uv = np.random.default_rng(0).random((200, 2))
        via_matrix = np.array([apply(m, x, y) for x, y in uv])
        direct = cam.project(origin + uv[:, :1] * u + uv[:, 1:2] * v)
        assert np.allclose(via_matrix, direct, atol=1e-10)

    def test_holds_for_an_oblique_camera(self) -> None:
        """Affine is a property of parallel projection, not of orthographic."""
        cam = ObliqueCamera.cabinet(40.0)
        origin, u, v = (
            np.array([0.0, 0.0, 2.0]),
            np.array([4.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -2.0]),
        )
        m = cam.plane_matrix(origin, u, v)
        uv = np.random.default_rng(1).random((100, 2))
        direct = cam.project(origin + uv[:, :1] * u + uv[:, 1:2] * v)
        assert np.allclose(np.array([apply(m, x, y) for x, y in uv]), direct, atol=1e-10)

    def test_edge_lengths_set_the_extent(self, cam: OrthographicCamera) -> None:
        """Edges are not normalized: their magnitude is the rectangle's size."""
        short = cam.plane_matrix((0, 0, 0), (1, 0, 0), (0, -1, 0))
        long = cam.plane_matrix((0, 0, 0), (4, 0, 0), (0, -1, 0))
        assert long[0] == pytest.approx(4 * short[0])
        assert long[1] == pytest.approx(4 * short[1])

    def test_translation_is_the_projected_origin(self, cam: OrthographicCamera) -> None:
        m = cam.plane_matrix((2.0, -1.0, 0.5), (1, 0, 0), (0, -1, 0))
        assert (m[4], m[5]) == pytest.approx(cam.at((2.0, -1.0, 0.5)))

    def test_rejects_a_degenerate_rectangle(self, cam: OrthographicCamera) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            cam.plane_matrix((0, 0, 0), (1, 0, 0), (2, 0, 0))

    def test_rejects_an_edge_on_rectangle(self, cam: OrthographicCamera) -> None:
        """Content in a plane seen edge-on would collapse to a line."""
        with pytest.raises(ValueError, match="degenerate"):
            cam.plane_matrix((0, 0, 0), cam.view, np.cross(cam.view, cam.up))


class TestOrientation:
    """`a > 0 and d > 0` is the readability rule; a positive determinant is not
    enough, because a 180-degree rotation has one too."""

    def test_upright_choice_keeps_content_readable(self, cam: OrthographicCamera) -> None:
        # world +y projects rightward at this azimuth, world +x downward
        a, b, c, d, _, _ = cam.plane_matrix((0, 0, 0), (0, 1, 0), (1, 0, 0))
        assert a > 0 and d > 0
        assert a * d - b * c > 0

    def test_a_rotated_choice_has_positive_determinant_but_is_upside_down(
        self, cam: OrthographicCamera
    ) -> None:
        """The exact trap that put every label upside down in an early example."""
        a, b, c, d, _, _ = cam.plane_matrix((0, 0, 0), (1, 0, 0), (0, -1, 0))
        assert a * d - b * c > 0, "determinant says 'not mirrored'..."
        assert a < 0 and d < 0, "...yet both axes are reversed, so it reads rotated 180"

    def test_mirrored_choice_has_negative_determinant(self, cam: OrthographicCamera) -> None:
        a, b, c, d, _, _ = cam.plane_matrix((0, 0, 0), (1, 0, 0), (0, 1, 0))
        assert a * d - b * c < 0

    def test_screen_basis_edges_are_readable_but_screen_aligned(
        self, cam: OrthographicCamera
    ) -> None:
        """Readable, but the rectangle loses its foreshortening cue on screen."""
        horizontal, down = cam.screen_basis()
        a, b, c, d, _, _ = cam.plane_matrix((0, 0, 0), horizontal, down)
        assert a > 0 and d > 0
        assert b == pytest.approx(0.0, abs=1e-9), "no vertical component: axis-aligned"
        assert c == pytest.approx(0.0, abs=1e-9)


class TestScenePlane:
    def test_emits_a_group_with_the_matrix_transform(self, cam: OrthographicCamera) -> None:
        scene = Scene(cam)
        scene.plane(10, (-4, 3, 0), (8, 0, 0), (0, -6, 0), id="plot")
        (group,) = scene.render().elements or []
        assert isinstance(group, svg.G)
        assert group.id == "plot"
        assert isinstance((group.transform or [])[0], svg.Matrix)

    def test_group_is_empty_for_a_consumer_to_fill(self, cam: OrthographicCamera) -> None:
        """This package never parses or embeds foreign SVG."""
        scene = Scene(cam)
        scene.plane(10, (0, 0, 0), (1, 0, 0), (0, -1, 0), id="plot")
        (group,) = scene.render().elements or []
        assert not group.elements

    def test_transform_matches_the_camera(self, cam: OrthographicCamera) -> None:
        scene = Scene(cam)
        args = ((-4, 3, 0), (8, 0, 0), (0, -6, 0))
        scene.plane(10, *args, id="plot")
        (group,) = scene.render().elements or []
        emitted = (group.transform or [])[0]
        expected = cam.plane_matrix(*args)
        for got, want in zip(
            (emitted.a, emitted.b, emitted.c, emitted.d, emitted.e, emitted.f),
            expected,
            strict=True,
        ):
            assert got == pytest.approx(want, abs=1e-3)

    def test_grows_the_bounding_box_by_the_projected_quad(self, cam: OrthographicCamera) -> None:
        """Otherwise the fitted viewBox clips the content a consumer inserts."""
        origin, u, v = (
            np.array([-4.0, 3.0, 0.0]),
            np.array([8.0, 0.0, 0.0]),
            np.array([0.0, -6.0, 0.0]),
        )
        scene = Scene(cam, pad=0.0)
        scene.plane(10, origin, u, v, id="plot")
        lo, hi = scene.bbox()
        quad = cam.project([origin, origin + u, origin + u + v, origin + v])
        assert np.allclose(lo, quad.min(axis=0))
        assert np.allclose(hi, quad.max(axis=0))

    def test_participates_in_the_layer_stack(self, cam: OrthographicCamera) -> None:
        """The reason to reserve the group in the scene rather than paste it after."""
        scene = Scene(cam)
        scene.polygon(20, [[0, 0, 0], [1, 0, 0], [1, 1, 0]], id="over")
        scene.plane(15, (0, 0, 0), (1, 0, 0), (0, -1, 0), id="plot")
        scene.polygon(10, [[0, 0, 0], [1, 0, 0], [1, 1, 0]], id="under")
        assert [el.id for el in (scene.render().elements or [])] == [
            "under",
            "plot",
            "over",
        ]

    def test_style_passes_through(self, cam: OrthographicCamera) -> None:
        scene = Scene(cam)
        scene.plane(10, (0, 0, 0), (1, 0, 0), (0, -1, 0), id="plot", opacity=0.5)
        (group,) = scene.render().elements or []
        assert group.opacity == 0.5

    def test_rejects_a_degenerate_plane(self, cam: OrthographicCamera) -> None:
        scene = Scene(cam)
        with pytest.raises(ValueError, match="degenerate"):
            scene.plane(10, (0, 0, 0), (1, 0, 0), (2, 0, 0), id="plot")
