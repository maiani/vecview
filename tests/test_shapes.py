"""World-space geometry: shapes are checked as numbers, not as rendered output."""

from __future__ import annotations

import numpy as np
import pytest

import vecview
from vecview import Face


def plane_normal(pts: np.ndarray) -> np.ndarray:
    """Best-fit normal of a set of coplanar points, via the smallest singular vector."""
    centred = pts - pts.mean(axis=0)
    return np.linalg.svd(centred)[2][-1]


def assert_in_plane(pts: np.ndarray, normal: np.ndarray) -> None:
    offsets = (pts - pts.mean(axis=0)) @ vecview.unit(normal)
    assert np.allclose(offsets, 0.0, atol=1e-12), "points are not coplanar with the normal"


class TestUnit:
    def test_normalizes(self) -> None:
        assert np.allclose(vecview.unit([0.0, 3.0, 4.0]), [0.0, 0.6, 0.8])

    def test_passes_a_zero_vector_through(self) -> None:
        """A degenerate direction draws as nothing rather than raising."""
        assert np.allclose(vecview.unit([0.0, 0.0, 0.0]), 0.0)


class TestInPlaneDir:
    @pytest.mark.parametrize(
        ("angle", "expected"),
        [(0.0, [1, 0, 0]), (90.0, [0, 1, 0]), (180.0, [-1, 0, 0]), (-90.0, [0, -1, 0])],
    )
    def test_measures_from_x_toward_y(self, angle: float, expected: list[float]) -> None:
        assert np.allclose(vecview.in_plane_dir(angle), expected, atol=1e-15)

    def test_is_always_a_unit_vector(self) -> None:
        for angle in range(0, 360, 7):
            assert np.linalg.norm(vecview.in_plane_dir(float(angle))) == pytest.approx(1.0)

    def test_accepts_a_custom_plane(self) -> None:
        d = vecview.in_plane_dir(90.0, u=(1, 0, 0), v=(0, 0, 1))
        assert np.allclose(d, [0, 0, 1], atol=1e-15)


class TestRectShape:
    def test_is_centred_with_the_requested_extents(self) -> None:
        pts = vecview.rect_shape((1, 2, 3), (1, 0, 0), (0, 1, 0), 4.0, 6.0)
        assert np.allclose(pts.mean(axis=0), [1, 2, 3])
        assert np.ptp(pts[:, 0]) == pytest.approx(4.0)
        assert np.ptp(pts[:, 1]) == pytest.approx(6.0)

    def test_normalizes_its_axes(self) -> None:
        """A caller passing an unnormalized axis must not get a stretched rectangle."""
        a = vecview.rect_shape((0, 0, 0), (5, 0, 0), (0, 9, 0), 2.0, 2.0)
        b = vecview.rect_shape((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 2.0)
        assert np.allclose(a, b)


class TestBoxFaces:
    @pytest.fixture
    def faces(self) -> list[Face]:
        return vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))

    def test_has_six_uniquely_named_faces(self, faces: list[Face]) -> None:
        assert sorted(f.name for f in faces) == ["+x", "+y", "+z", "-x", "-y", "-z"]

    def test_declared_normals_match_the_winding(self, faces: list[Face]) -> None:
        """Culling is only correct if the winding agrees with the stated normal."""
        for f in faces:
            wound = np.cross(f.points[1] - f.points[0], f.points[2] - f.points[1])
            assert np.allclose(vecview.unit(wound), f.normal, atol=1e-12), f.name

    def test_size_is_full_extent_not_half(self) -> None:
        corners = np.concatenate([f.points for f in vecview.box_faces((0, 0, 0), (2, 4, 6))])
        assert np.allclose(corners.max(axis=0) - corners.min(axis=0), [2, 4, 6])

    def test_faces_sit_at_the_box_surface(self, faces: list[Face]) -> None:
        for f in faces:
            offset = f.points.mean(axis=0) - np.array([0, 0, -0.45])
            assert np.allclose(vecview.unit(offset), f.normal, atol=1e-12), f.name

    def test_each_face_is_planar(self, faces: list[Face]) -> None:
        for f in faces:
            assert_in_plane(f.points, f.normal)

    def test_a_flat_box_still_yields_six_faces(self) -> None:
        """Zero thickness is a legitimate request: a bare plane drawn as a box."""
        assert len(vecview.box_faces((0, 0, 0), (5, 5, 0))) == 6

    def test_rejects_malformed_arguments(self) -> None:
        with pytest.raises(ValueError, match="three numbers"):
            vecview.box_faces((0, 0), (1, 1, 1))


class TestArrowShape:
    def test_spans_the_requested_length(self) -> None:
        pts = vecview.arrow_shape(
            (0, 0, 0), (1, 0, 0), 10.0, (0, 0, 1), shaft_w=1, head_w=3, head_len=2
        )
        assert pts[:, 0].max() == pytest.approx(10.0)
        assert pts[:, 0].min() == pytest.approx(0.0)

    def test_head_is_wider_than_the_shaft(self) -> None:
        pts = vecview.arrow_shape(
            (0, 0, 0), (1, 0, 0), 10.0, (0, 0, 1), shaft_w=1, head_w=3, head_len=2
        )
        assert np.ptp(pts[:, 1]) == pytest.approx(3.0)

    def test_mid_pivot_centres_on_the_origin(self) -> None:
        pts = vecview.arrow_shape((0, 0, 0), (1, 0, 0), 10.0, (0, 0, 1), 1, 3, 2, pivot="mid")
        assert pts[:, 0].min() == pytest.approx(-5.0)
        assert pts[:, 0].max() == pytest.approx(5.0)

    def test_lies_in_the_plane_of_its_normal(self) -> None:
        normal = vecview.unit([1.0, 1.0, 1.0])
        pts = vecview.arrow_shape((1, 2, 3), (1, -1, 0), 4.0, normal, 0.2, 0.8, 0.6)
        assert_in_plane(pts, normal)

    def test_a_head_longer_than_the_arrow_is_clamped(self) -> None:
        """Otherwise the neck runs behind the tail and the polygon self-crosses."""
        pts = vecview.arrow_shape(
            (0, 0, 0), (1, 0, 0), 1.0, (0, 0, 1), shaft_w=1, head_w=3, head_len=5
        )
        assert pts[:, 0].min() == pytest.approx(0.0)


class TestDoubleArrowShape:
    def test_is_symmetric_about_its_centre(self) -> None:
        pts = vecview.double_arrow_shape(
            (0, 0, 0), (1, 0, 0), 6.0, (0, 0, 1), shaft_w=0.2, head_w=1.0, head_len=0.5
        )
        assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-12)
        assert np.ptp(pts[:, 0]) == pytest.approx(6.0)

    def test_is_centred_on_the_given_point(self) -> None:
        pts = vecview.double_arrow_shape((3, -1, 2), (0, 1, 0), 4.0, (0, 0, 1), 0.2, 1.0, 0.5)
        assert np.allclose(pts.mean(axis=0), [3, -1, 2], atol=1e-12)


class TestCircleShape:
    def test_all_points_lie_at_the_radius(self) -> None:
        pts = vecview.circle_shape((1, 2, 3), 2.5, (0, 0, 1))
        assert np.allclose(np.linalg.norm(pts - [1, 2, 3], axis=1), 2.5)

    def test_lies_in_the_plane_of_its_normal(self) -> None:
        normal = vecview.unit([1.0, 2.0, -1.0])
        pts = vecview.circle_shape((0, 0, 0), 1.0, normal)
        assert np.allclose(pts @ normal, 0.0, atol=1e-12)
        assert np.allclose(vecview.unit(plane_normal(pts)), normal, atol=1e-9) or np.allclose(
            vecview.unit(plane_normal(pts)), -normal, atol=1e-9
        )

    def test_segment_count_is_honoured_without_a_duplicate_seam(self) -> None:
        pts = vecview.circle_shape((0, 0, 0), 1.0, (0, 0, 1), n=12)
        assert pts.shape == (12, 3)
        assert not np.allclose(pts[0], pts[-1]), "closing point would double the seam"


class TestSineRibbon:
    def test_is_an_exact_sine_along_the_axis(self) -> None:
        pts = vecview.sine_ribbon(
            (0, 0, 0), (0, 0, 1), 10.0, (1, 0, 0), amplitude=0.5, wavelength=2.0, n=64
        )
        t = np.linspace(0.0, 10.0, 64)
        assert np.allclose(pts[:, 0], 0.5 * np.sin(2 * np.pi * t / 2.0), atol=1e-15)
        assert np.allclose(pts[:, 2], t)

    def test_accepts_an_amplitude_envelope(self) -> None:
        """A component absorbed inside a medium is drawn by passing its envelope."""
        env = np.linspace(1.0, 0.0, 32)
        pts = vecview.sine_ribbon(
            (0, 0, 0), (0, 0, 1), 4.0, (1, 0, 0), amplitude=env, wavelength=1.0, n=32
        )
        assert abs(pts[-1, 0]) < 1e-12
        assert np.abs(pts[:, 0]).max() <= 1.0 + 1e-12

    def test_phase_shifts_the_wave(self) -> None:
        kwargs = dict(wavelength=2.0, n=16)
        a = vecview.sine_ribbon((0, 0, 0), (0, 0, 1), 4.0, (1, 0, 0), 1.0, **kwargs)
        b = vecview.sine_ribbon((0, 0, 0), (0, 0, 1), 4.0, (1, 0, 0), 1.0, phase=np.pi, **kwargs)
        assert np.allclose(a[:, 0], -b[:, 0], atol=1e-15)
