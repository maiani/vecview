"""Camera projection, frames, and culling."""

from __future__ import annotations

import numpy as np
import pytest

from vecview import OrthographicCamera


@pytest.fixture
def cam() -> OrthographicCamera:
    return OrthographicCamera(azim_deg=35.0, elev_deg=24.0, scale=62.0)


def test_basis_is_orthonormal(cam: OrthographicCamera) -> None:
    m = np.array([cam.right, cam.up, cam.view])
    assert np.allclose(m @ m.T, np.eye(3))


def test_projection_is_orthographic(cam: OrthographicCamera) -> None:
    """Parallel world segments must stay parallel on screen, at equal length."""
    a = cam.project([[0, 0, 0], [2, 1, 0]])
    b = cam.project([[0, 0, 5], [2, 1, 5]])
    assert np.allclose(a[1] - a[0], b[1] - b[0])


def test_scale_is_svg_units_per_world_unit() -> None:
    cam = OrthographicCamera(0.0, 0.0, 10.0)
    assert np.allclose(np.linalg.norm(cam.project([[0, 1, 0]])), 10.0)


def test_origin_maps_to_svg_origin() -> None:
    cam = OrthographicCamera(35.0, 24.0, 62.0, origin=(1.0, 2.0, 3.0))
    assert cam.at((1.0, 2.0, 3.0)) == (0.0, 0.0)


def test_up_projects_to_negative_svg_y(cam: OrthographicCamera) -> None:
    """SVG y grows downward, so a point higher in the world has smaller y."""
    low, high = cam.project([[0, 0, 0], [0, 0, 1]])
    assert high[1] < low[1]


def test_at_matches_project(cam: OrthographicCamera) -> None:
    assert cam.at((1.0, 2.0, 3.0)) == pytest.approx(tuple(cam.project([[1, 2, 3]])[0]))


def test_project_rejects_wrong_shape(cam: OrthographicCamera) -> None:
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        cam.project([[1.0, 2.0]])


class TestDirection:
    def test_ignores_origin(self) -> None:
        a = OrthographicCamera(35.0, 24.0, 62.0)
        b = OrthographicCamera(35.0, 24.0, 62.0, origin=(9.0, -4.0, 2.0))
        assert np.allclose(a.direction((1, 2, 3)), b.direction((1, 2, 3)))

    def test_is_linear(self, cam: OrthographicCamera) -> None:
        u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 2.0])
        assert np.allclose(
            cam.direction(3 * u - 2 * v), 3 * cam.direction(u) - 2 * cam.direction(v)
        )


class TestScreenBasis:
    def test_horizontal_moves_only_horizontally(self, cam: OrthographicCamera) -> None:
        horizontal, _ = cam.screen_basis()
        dx, dy = cam.direction(horizontal)
        assert dx > 0
        assert dy == pytest.approx(0.0, abs=1e-9)

    def test_down_moves_only_downward(self, cam: OrthographicCamera) -> None:
        _, down = cam.screen_basis()
        dx, dy = cam.direction(down)
        assert dx == pytest.approx(0.0, abs=1e-9)
        assert dy > 0, "SVG y grows downward"

    def test_returns_unit_vectors_in_the_plane(self, cam: OrthographicCamera) -> None:
        normal = np.array([0.0, 1.0, 1.0])
        for d in cam.screen_basis(normal):
            assert np.linalg.norm(d) == pytest.approx(1.0)
            assert float(d @ normal) == pytest.approx(0.0, abs=1e-9)

    def test_independent_of_origin(self) -> None:
        a = OrthographicCamera(35.0, 24.0, 62.0).screen_basis()
        b = OrthographicCamera(35.0, 24.0, 62.0, origin=(3.0, -2.0, 1.0)).screen_basis()
        assert np.allclose(np.array(a), np.array(b))

    def test_rejects_an_edge_on_plane(self, cam: OrthographicCamera) -> None:
        with pytest.raises(ValueError, match="edge-on"):
            cam.screen_basis(normal=cam.up)


class TestVisibility:
    def test_camera_octant_determines_visible_box_walls(self, cam: OrthographicCamera) -> None:
        """At azimuth 35 and elevation 24 the camera sits in the +x+y+z octant."""
        from vecview import box_faces

        visible = cam.visible(box_faces((0, 0, 0), (1, 1, 1)))
        assert sorted(f.name for f in visible) == ["+x", "+y", "+z"]

    def test_a_box_always_shows_three_walls(self) -> None:
        from vecview import box_faces

        faces = box_faces((0, 0, 0), (2, 3, 4))
        for azim in range(5, 360, 10):
            cam = OrthographicCamera(float(azim), 30.0, 1.0)
            assert len(cam.visible(faces)) == 3, f"at azimuth {azim}"

    def test_faces_camera_agrees_with_the_view_axis(self, cam: OrthographicCamera) -> None:
        assert cam.faces_camera(cam.view)
        assert not cam.faces_camera(-cam.view)

    def test_tolerance_drops_edge_on_faces(self, cam: OrthographicCamera) -> None:
        edge_on = np.cross(cam.view, cam.up)
        assert not cam.faces_camera(edge_on, tol=1e-6)


def test_depth_increases_toward_the_camera(cam: OrthographicCamera) -> None:
    near, far = cam.depth([cam.view * 5.0, cam.view * -5.0])
    assert near > far


def test_repr_reports_the_camera_parameters(cam: OrthographicCamera) -> None:
    assert repr(cam) == "OrthographicCamera(azim_deg=35, elev_deg=24, scale=62)"
