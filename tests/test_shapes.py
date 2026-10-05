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


class TestEllipseShape:
    def test_points_satisfy_the_ellipse_equation(self) -> None:
        u, v = np.array([1.0, 1.0, 0.0]), np.array([-1.0, 1.0, 0.0])
        ring = vecview.ellipse_shape((1, 2, 3), u, v, 2.0, 0.5, n=48)
        rel = ring - [1, 2, 3]
        s, t = rel @ vecview.unit(u), rel @ vecview.unit(v)
        assert np.allclose((s / 2.0) ** 2 + (t / 0.5) ** 2, 1.0)
        assert_in_plane(ring, np.cross(u, v))

    def test_starts_on_the_a_axis_and_winds_ccw_about_u_cross_v(self) -> None:
        ring = vecview.ellipse_shape((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0, n=16)
        assert np.allclose(ring[0], [2.0, 0.0, 0.0])
        assert np.cross(ring[1] - ring[0], ring[2] - ring[1])[2] > 0

    def test_has_no_duplicate_seam(self) -> None:
        ring = vecview.ellipse_shape((0, 0, 0), (1, 0, 0), (0, 0, 1), 2.0, 1.0, n=10)
        assert len(ring) == 10
        assert not np.allclose(ring[0], ring[-1])

    def test_equal_axes_give_a_circle(self) -> None:
        ring = vecview.ellipse_shape((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.5, 1.5)
        assert np.allclose(np.linalg.norm(ring, axis=1), 1.5)

    def test_rejects_axes_that_are_not_perpendicular(self) -> None:
        with pytest.raises(ValueError, match="perpendicular"):
            vecview.ellipse_shape((0, 0, 0), (1, 0, 0), (1, 1, 0), 1.0, 1.0)


class TestPrismFaces:
    TAPER = ((0.0, -1.0), (3.0, -0.3), (3.4, 0.0), (3.0, 0.3), (0.0, 1.0))

    @pytest.fixture
    def faces(self) -> list[Face]:
        return vecview.prism_faces(self.TAPER, 0.0, 0.5)

    def test_names_cap_base_and_one_wall_per_edge(self, faces: list[Face]) -> None:
        names = [f.name for f in faces]
        assert names[:2] == ["+z", "-z"]
        assert names[2:] == [f"side-{i}" for i in range(len(self.TAPER))]

    def test_declared_normals_match_the_winding(self, faces: list[Face]) -> None:
        for face in faces:
            p = face.points
            assert np.dot(np.cross(p[1] - p[0], p[2] - p[1]), face.normal) > 0, face.name

    def test_normals_point_outward(self, faces: list[Face]) -> None:
        centre = np.vstack([f.points for f in faces]).mean(axis=0)
        for face in faces:
            assert np.dot(face.points.mean(axis=0) - centre, face.normal) > 0, face.name

    def test_each_face_is_planar_and_walls_are_vertical(self, faces: list[Face]) -> None:
        for face in faces:
            assert_in_plane(face.points, face.normal)
        assert all(f.normal[2] == 0 for f in faces[2:])

    def test_cap_and_base_sit_at_their_heights(self, faces: list[Face]) -> None:
        assert np.allclose(faces[0].points[:, 2], 0.5)
        assert np.allclose(faces[1].points[:, 2], 0.0)

    def test_clockwise_footprint_gives_the_same_solid(self) -> None:
        """Winding is normalized, so the cap still faces +z and walls still face out."""
        ccw = vecview.prism_faces(self.TAPER, 0.0, 0.5)
        cw = vecview.prism_faces(self.TAPER[::-1], 0.0, 0.5)
        cap = cw[0].points
        assert np.cross(cap[1] - cap[0], cap[2] - cap[1])[2] > 0
        normals = sorted(tuple(np.round(f.normal, 12)) for f in ccw)
        assert sorted(tuple(np.round(f.normal, 12)) for f in cw) == normals

    def test_a_box_footprint_culls_like_box_faces(self) -> None:
        cam = vecview.OrthographicCamera(35.0, 24.0, 62.0)
        prism = vecview.prism_faces([(-1, -1), (1, -1), (1, 1), (-1, 1)], -1.0, 1.0)
        assert len(cam.visible(prism)) == len(cam.visible(vecview.box_faces((0, 0, 0), (2, 2, 2))))

    def test_collinear_vertices_are_allowed(self) -> None:
        vecview.prism_faces([(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)], 0.0, 1.0)

    @pytest.mark.parametrize(
        "footprint",
        [
            [(0, 0), (2, 0), (2, 2), (0, 2), (2, 1)],  # bow-tie-like crossing
            [(np.cos(t), np.sin(t)) for t in np.arange(5) * 4 * np.pi / 5],  # star
            [(0, 0), (4, 0), (4, 2), (2, 0), (2, 2), (0, 2)],  # touches itself at (2, 0)
        ],
        ids=["crossing", "star", "touching"],
    )
    def test_rejects_a_self_intersecting_footprint(
        self, footprint: list[tuple[float, float]]
    ) -> None:
        with pytest.raises(ValueError, match="simple polygon"):
            vecview.prism_faces(footprint, 0.0, 1.0)

    def test_rejects_malformed_arguments(self) -> None:
        with pytest.raises(ValueError, match="z1 must exceed z0"):
            vecview.prism_faces(self.TAPER, 1.0, 1.0)
        with pytest.raises(ValueError, match="shape"):
            vecview.prism_faces([(0, 0), (1, 0)], 0.0, 1.0)
        with pytest.raises(ValueError, match="repeated"):
            vecview.prism_faces([(0, 0), (1, 0), (1, 0), (0, 1)], 0.0, 1.0)


def inside(polygon: np.ndarray, point: np.ndarray) -> bool:
    """Even-odd point-in-polygon test, for checking outward normals of a concave shape."""
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


class TestNonConvexPrism:
    L_SHAPE = ((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3))

    @pytest.mark.parametrize(
        "footprint",
        [L_SHAPE, L_SHAPE[::-1], vecview.annulus_sector((0, 0), 2.6, 3.0, 20, 160, n=12)],
        ids=["L", "L-clockwise", "sector"],
    )
    def test_walls_wind_ccw_about_normals_that_point_outward(self, footprint: object) -> None:
        faces = vecview.prism_faces(footprint, 0.0, 0.3)  # type: ignore[arg-type]
        cap = faces[0].points[:, :2]
        for face in faces:
            # Newell's normal: the winding of the whole polygon, valid at concave corners.
            p, q = face.points, np.roll(face.points, -1, axis=0)
            assert np.dot(np.cross(p, q).sum(axis=0), face.normal) > 0, face.name
        for face in faces[2:]:
            mid = face.points.mean(axis=0)[:2]
            assert not inside(cap, mid + 1e-3 * face.normal[:2]), face.name
            assert inside(cap, mid - 1e-3 * face.normal[:2]), face.name

    def test_a_concave_footprint_has_facing_walls_that_overlap_on_screen(self) -> None:
        """Why culling is not a complete answer here, and prism_walls exists."""
        cam = vecview.OrthographicCamera(35.0, 24.0, 62.0)
        faces = vecview.prism_faces(self.L_SHAPE, 0.0, 0.3)
        assert len(cam.visible(faces[2:])) > 2

    def test_rejects_a_footprint_without_area(self) -> None:
        with pytest.raises(ValueError, match="area"):
            vecview.prism_faces([(0, 0), (1, 0), (2, 0)], 0.0, 1.0)


class TestAnnulusSector:
    def test_points_lie_on_the_two_radii(self) -> None:
        foot = vecview.annulus_sector((1, 2), 2.0, 3.0, 10, 100, n=8)
        r = np.linalg.norm(foot - [1, 2], axis=1)
        assert np.allclose(r[:9], 3.0) and np.allclose(r[9:], 2.0)

    def test_is_counter_clockwise_and_spans_the_angles(self) -> None:
        foot = vecview.annulus_sector((0, 0), 1.0, 2.0, -30, 210, n=16)
        x, y = foot[:, 0], foot[:, 1]
        assert np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) > 0
        assert np.allclose(
            foot[0], 2.0 * np.array([np.cos(np.radians(-30)), np.sin(np.radians(-30))])
        )
        assert np.allclose(
            foot[16], 2.0 * np.array([np.cos(np.radians(210)), np.sin(np.radians(210))])
        )

    def test_area_matches_the_exact_sector_as_n_grows(self) -> None:
        foot = vecview.annulus_sector((0, 0), 1.0, 2.0, 0, 90, n=400)
        x, y = foot[:, 0], foot[:, 1]
        area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
        assert area == pytest.approx(np.pi / 4 * (4 - 1), rel=1e-4)

    def test_zero_inner_radius_gives_a_wedge(self) -> None:
        foot = vecview.annulus_sector((5, 5), 0.0, 1.0, 0, 90, n=4)
        assert len(foot) == 6
        assert np.allclose(foot[-1], [5, 5])

    def test_extrudes_into_a_valid_prism(self) -> None:
        vecview.prism_faces(vecview.annulus_sector((0, 0), 2.6, 3.0, 20, 160), 0.0, 0.26)

    @pytest.mark.parametrize(
        ("r_in", "r_out", "t0", "t1", "n"),
        [
            (2, 1, 0, 90, 8),
            (-1, 1, 0, 90, 8),
            (1, 2, 90, 90, 8),
            (1, 2, 0, 360, 8),
            (1, 2, 0, 90, 0),
        ],
    )
    def test_rejects_malformed_arguments(
        self, r_in: float, r_out: float, t0: float, t1: float, n: int
    ) -> None:
        with pytest.raises(ValueError):
            vecview.annulus_sector((0, 0), r_in, r_out, t0, t1, n=n)
