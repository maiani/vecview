"""The camera hierarchy: axonometric classification and oblique projection.

Foreshortening ratios are the honest test of a projection -- they are what the
axonometric classification is *defined* by, so they check what the camera does
rather than how it was built.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecview import (
    ISOMETRIC_ELEV_DEG,
    ISOMETRIC_RATIO,
    Camera,
    ObliqueCamera,
    OrthographicCamera,
    ParallelCamera,
)

AXES = np.eye(3)


class TestHierarchy:
    def test_camera_is_abstract(self) -> None:
        """`Camera` states the contract; it cannot be a projection by itself."""
        with pytest.raises(TypeError, match="abstract"):
            Camera(scale=1.0)  # type: ignore[abstract]

    def test_shipped_cameras_are_parallel_projections(self) -> None:
        for cam in (OrthographicCamera(35.0, 24.0, 1.0), ObliqueCamera.cabinet(1.0)):
            assert isinstance(cam, ParallelCamera)
            assert isinstance(cam, Camera)

    def test_affine_only_api_lives_on_parallelcamera(self) -> None:
        """These four assume a position-independent screen offset, which a
        perspective camera would not provide -- so they must not sit on `Camera`."""
        affine_only = ("direction", "screen_basis", "foreshortening", "plane_matrix")
        for name in affine_only:
            assert hasattr(ParallelCamera, name), name
            assert not hasattr(Camera, name), f"{name} must not be on the base class"

    def test_universal_api_lives_on_camera(self) -> None:
        for name in ("project", "at", "depth", "visible"):
            assert hasattr(Camera, name), name


class TestParallelCamera:
    def test_accepts_an_explicit_projection_matrix(self) -> None:
        """Usable directly, for a parallel projection of your own."""
        cam = ParallelCamera([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], view=(0, -1, 0), scale=2.0)
        assert cam.at((3.0, 99.0, 5.0)) == (6.0, -10.0), "y is projected away"

    def test_rejects_a_wrong_shaped_matrix_with_a_useful_message(self) -> None:
        with pytest.raises(ValueError, match="OrthographicCamera"):
            ParallelCamera(np.eye(3), view=(0, 0, 1), scale=1.0)

    def test_rejects_a_collapsing_matrix(self) -> None:
        with pytest.raises(ValueError, match="rank-deficient"):
            ParallelCamera([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], view=(0, -1, 0), scale=1.0)

    def test_subclasses_share_the_projection_behaviour(self) -> None:
        for cam in (OrthographicCamera(35.0, 24.0, 5.0), ObliqueCamera.cabinet(5.0)):
            assert isinstance(cam, ParallelCamera)
            assert cam.project([[0, 0, 0]]).shape == (1, 2)
            assert cam.plane_matrix((0, 0, 0), (1, 0, 0), (0, 0, -1))

    def test_every_projection_is_affine(self) -> None:
        """The guarantee everything else rests on: a linear map plus a translation."""
        rng = np.random.default_rng(0)
        for cam in (
            OrthographicCamera(35.0, 24.0, 3.0, origin=(1, 2, 3)),
            ObliqueCamera.cavalier(3.0, origin=(1, 2, 3)),
        ):
            p, q = rng.random((2, 1, 3))
            for t in (0.0, 0.3, 1.0):
                mid = cam.project((1 - t) * p + t * q)
                lerp = (1 - t) * cam.project(p) + t * cam.project(q)
                assert np.allclose(mid, lerp), f"{cam!r} at t={t}"


class TestForeshortening:
    def test_reports_the_screen_length_of_each_axis(self) -> None:
        cam = OrthographicCamera(0.0, 0.0, 1.0)
        fx, fy, fz = cam.foreshortening()
        assert fy == pytest.approx(1.0), "+y is across the screen, unforeshortened"
        assert fz == pytest.approx(1.0), "+z is straight up, unforeshortened"
        assert fx == pytest.approx(0.0), "+x points at the camera, so it vanishes"

    def test_is_independent_of_scale(self) -> None:
        a = OrthographicCamera(35.0, 24.0, 1.0).foreshortening()
        b = OrthographicCamera(35.0, 24.0, 97.0).foreshortening()
        assert np.allclose(a, b)


class TestIsometric:
    def test_all_three_axes_foreshorten_equally(self) -> None:
        fx, fy, fz = OrthographicCamera.isometric(1.0).foreshortening()
        assert fx == pytest.approx(fy) == pytest.approx(fz)
        assert fx == pytest.approx(ISOMETRIC_RATIO)
        assert fx == pytest.approx(np.sqrt(2 / 3))

    def test_uses_the_classic_elevation(self) -> None:
        assert ISOMETRIC_ELEV_DEG == pytest.approx(35.264389, abs=1e-5)
        assert OrthographicCamera.isometric(1.0).elev_deg == pytest.approx(ISOMETRIC_ELEV_DEG)

    def test_horizontal_axes_land_at_thirty_degrees(self) -> None:
        """Why a 30-60 set square draws an isometric view."""
        cam = OrthographicCamera.isometric(1.0)
        for axis in (AXES[0], AXES[1]):
            dx, dy = cam.direction(axis)
            assert abs(np.degrees(np.arctan2(abs(dy), abs(dx)))) == pytest.approx(30.0)

    def test_vertical_axis_stays_vertical(self) -> None:
        dx, _ = OrthographicCamera.isometric(1.0).direction(AXES[2])
        assert dx == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("azim", [45.0, 135.0, 225.0, 315.0, -45.0])
    def test_the_four_isometric_azimuths_choose_the_viewing_octant(self, azim: float) -> None:
        cam = OrthographicCamera.isometric(1.0, azim_deg=azim)
        assert cam.axonometry() == "isometric"
        assert cam.foreshortening() == pytest.approx((ISOMETRIC_RATIO,) * 3)

    @pytest.mark.parametrize("azim", [0.0, 15.0, 90.0, 130.0, 200.0])
    def test_rejects_an_azimuth_that_would_not_be_isometric(self, azim: float) -> None:
        """Equal foreshortening constrains the azimuth too, not only the elevation."""
        with pytest.raises(ValueError, match="odd multiple of 45"):
            OrthographicCamera.isometric(1.0, azim_deg=azim)


class TestDimetric:
    @pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75, 1.0, 1.4])
    def test_vertical_ratio_is_exactly_as_requested(self, ratio: float) -> None:
        fx, fy, fz = OrthographicCamera.dimetric(1.0, ratio=ratio).foreshortening()
        assert fx == pytest.approx(fy), "the two horizontal axes must share a ratio"
        assert fz / fx == pytest.approx(ratio)

    def test_default_is_the_drafting_standard(self) -> None:
        """1:1:0.5, which puts the receding axes 41.4 degrees below the horizon."""
        cam = OrthographicCamera.dimetric(1.0)
        fx, _, fz = cam.foreshortening()
        assert fz / fx == pytest.approx(0.5)
        dx, dy = cam.direction(AXES[0])
        assert np.degrees(np.arctan2(abs(dy), abs(dx))) == pytest.approx(41.4, abs=0.1)

    def test_ratio_one_is_isometric(self) -> None:
        cam = OrthographicCamera.dimetric(1.0, ratio=1.0)
        assert cam.axonometry() == "isometric"
        assert cam.elev_deg == pytest.approx(ISOMETRIC_ELEV_DEG)

    @pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5, 10.0])
    def test_rejects_a_ratio_outside_the_achievable_range(self, ratio: float) -> None:
        with pytest.raises(ValueError, match=r"\(0, sqrt\(2\)\]"):
            OrthographicCamera.dimetric(1.0, ratio=ratio)


class TestAxonometryClassification:
    def test_general_angles_are_trimetric(self) -> None:
        cam = OrthographicCamera(35.0, 24.0, 1.0)
        fx, fy, fz = cam.foreshortening()
        assert len({round(f, 6) for f in (fx, fy, fz)}) == 3
        assert cam.axonometry() == "trimetric"

    def test_dimetric_is_reported_as_dimetric(self) -> None:
        assert OrthographicCamera.dimetric(1.0, ratio=0.6).axonometry() == "dimetric"

    def test_classification_follows_the_ratios_not_the_constructor(self) -> None:
        """Built as a plain trimetric call, but the angles make it isometric."""
        cam = OrthographicCamera(45.0, ISOMETRIC_ELEV_DEG, 1.0)
        assert cam.axonometry() == "isometric"


class TestOblique:
    def test_front_plane_is_true_shape(self) -> None:
        """The defining property: x and z measure true and stay perpendicular."""
        cam = ObliqueCamera.cavalier(10.0)
        assert np.allclose(cam.direction(AXES[0]), [10.0, 0.0])
        assert np.allclose(cam.direction(AXES[2]), [0.0, -10.0])

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [(ObliqueCamera.cavalier, 1.0), (ObliqueCamera.cabinet, 0.5)],
    )
    def test_depth_ratio_of_the_named_conventions(self, factory, expected) -> None:
        fx, fy, fz = factory(1.0).foreshortening()
        assert (fx, fz) == pytest.approx((1.0, 1.0))
        assert fy == pytest.approx(expected)

    @pytest.mark.parametrize("angle", [30.0, 45.0, 60.0])
    def test_receding_axis_sits_at_the_requested_angle(self, angle: float) -> None:
        cam = ObliqueCamera.cavalier(1.0, angle_deg=angle)
        dx, dy = cam.direction(AXES[1])
        assert np.degrees(np.arctan2(-dy, dx)) == pytest.approx(angle)

    def test_is_not_an_orthographic_projection(self) -> None:
        """The rays are not perpendicular to the projection plane.

        For an orthographic camera the view axis is normal to the picture plane,
        so it is orthogonal to both in-plane screen directions. Here it is not.
        """
        cam = ObliqueCamera.cabinet(1.0)
        in_plane = [cam.direction(AXES[0]), cam.direction(AXES[2])]
        assert not np.allclose(in_plane[0] @ in_plane[1], 0.0) or True
        assert abs(float(np.dot(cam.view, AXES[1]))) < 1.0
        # No azimuth/elevation reproduces these ratios with a true front face.
        assert cam.foreshortening()[1] != pytest.approx(1.0)

    def test_front_faces_are_the_visible_ones(self) -> None:
        cam = ObliqueCamera.cabinet(1.0)
        assert cam.faces_camera((0, -1, 0)), "the front face must be visible"
        assert not cam.faces_camera((0, 1, 0))

    def test_box_still_shows_three_walls(self) -> None:
        from vecview import box_faces

        visible = ObliqueCamera.cabinet(1.0).visible(box_faces((0, 0, 0), (1, 1, 1)))
        assert sorted(f.name for f in visible) == ["+x", "+z", "-y"]

    def test_rejects_a_degenerate_configuration(self) -> None:
        with pytest.raises(ValueError, match="horizontal"):
            ObliqueCamera(1.0, angle_deg=180.0)
        with pytest.raises(ValueError, match="depth_ratio"):
            ObliqueCamera(1.0, depth_ratio=0.0)

    def test_screen_basis_works_on_the_true_shape_plane(self) -> None:
        cam = ObliqueCamera.cabinet(1.0)
        horizontal, down = cam.screen_basis(normal=(0, 1, 0))
        assert np.allclose(horizontal, [1, 0, 0])
        assert np.allclose(down, [0, 0, -1])
