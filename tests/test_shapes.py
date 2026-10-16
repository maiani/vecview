"""World-space geometry: shapes are checked as numbers, not as rendered output."""

from __future__ import annotations

import itertools

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


class TestExtrude:
    HEXAGON = vecview.circle_shape((0, 0, 0), 0.4, (1, 0, 0), n=6)
    # An L in the yz plane: non-convex, so outward cannot be read off a centroid.
    L_SECTION = np.array([(0, 0, 0), (0, 3, 0), (0, 3, 1), (0, 1, 1), (0, 1, 3), (0, 0, 3)], float)

    @staticmethod
    def newell(points: np.ndarray) -> np.ndarray:
        return np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)

    @pytest.mark.parametrize(
        "along", [(5, 0, 0), (-5, 0, 0), (4, 1, 0.5)], ids=["x", "-x", "oblique"]
    )
    @pytest.mark.parametrize("reverse", [False, True], ids=["ccw", "cw"])
    def test_every_face_winds_about_an_outward_normal(self, along: tuple, reverse: bool) -> None:
        section = self.L_SECTION[::-1] if reverse else self.L_SECTION
        faces = vecview.extrude(section, along)
        assert [f.name for f in faces] == ["start", "end"] + [f"side-{i}" for i in range(6)]
        for face in faces:
            assert np.dot(self.newell(face.points), face.normal) > 0, face.name
            assert_in_plane(face.points, face.normal)
        # Each wall is built on edge i, and its normal points out of the section there.
        yz = section[:, 1:]
        for i, face in enumerate(faces[2:]):
            a, b = section[i], section[(i + 1) % 6]
            assert {tuple(a), tuple(b)} <= {tuple(p) for p in face.points}, face.name
            mid = ((a + b) / 2)[1:]
            assert not inside(yz, mid + 1e-3 * face.normal[1:]), face.name
            assert inside(yz, mid - 1e-3 * face.normal[1:]), face.name

    def test_caps_are_the_section_and_its_sweep(self) -> None:
        start, end, *_ = vecview.extrude(self.HEXAGON, (5, 0, 0))
        assert {tuple(p) for p in start.points} == {tuple(p) for p in self.HEXAGON}
        assert np.allclose(
            sorted(map(tuple, end.points)),
            sorted(map(tuple, self.HEXAGON + np.array([5.0, 0.0, 0.0]))),
        )
        assert np.allclose(start.normal, (-1, 0, 0)) and np.allclose(end.normal, (1, 0, 0))

    def test_a_convex_wire_culls_to_the_facets_a_camera_sees(self) -> None:
        cam = vecview.OrthographicCamera(35.0, 24.0, 62.0)
        faces = vecview.extrude(self.HEXAGON, (5, 0, 0))
        for face in cam.visible(faces):
            assert np.dot(face.normal, cam.view) > 0
        # Normals point away from the wire's axis, so a convex wire shows its near half.
        for face in faces[2:]:
            radial = face.points.mean(axis=0) - np.array([face.points.mean(axis=0)[0], 0, 0])
            assert np.dot(radial, face.normal) > 0, face.name

    def test_prism_faces_is_extrusion_along_z(self) -> None:
        foot = [(0, 0), (2, 0), (1, 1)]
        prism = vecview.prism_faces(foot, 0.5, 2.0)
        swept = vecview.extrude([(x, y, 0.5) for x, y in foot], (0, 0, 1.5))
        assert [f.name for f in prism] == ["+z", "-z", "side-0", "side-1", "side-2"]
        for a, b in zip(prism, [swept[1], swept[0], *swept[2:]], strict=True):
            assert np.array_equal(a.points, b.points) and np.array_equal(a.normal, b.normal)

    @pytest.mark.parametrize(
        ("section", "along", "message"),
        [
            ([(0, 0, 0), (0, 1, 0)], (1, 0, 0), "three vertices"),
            ([(0, 0, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1)], (1, 0, 0), "repeated"),
            ([(0, 0, 0), (0, 1, 0), (0, 2, 0)], (1, 0, 0), "area"),
            ([(0, 0, 0), (0, 1, 0), (0.5, 1, 1), (0, 0, 1)], (1, 0, 0), "planar"),
            ([(0, 0, 0), (0, 2, 1), (0, 2, 0), (0, 0, 2)], (1, 0, 0), "simple"),
            ([(0, 0, 0), (0, 1, 0), (0, 0, 1)], (0, 1, 1), "plane"),
        ],
        ids=["two", "repeated", "collinear", "warped", "bowtie", "in-plane"],
    )
    def test_rejects_what_has_no_volume_or_outside(self, section, along, message) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match=message):
            vecview.extrude(section, along)


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


def depth_at(cam: vecview.OrthographicCamera, face: Face, at: np.ndarray) -> float:
    """The depth of ``face``'s plane under screen point ``at``, from three of its corners."""
    s, d = cam.project(face.points[:3]), cam.depth(face.points[:3])
    a, b, c = np.linalg.solve(np.column_stack([s, np.ones(3)]), d)
    return float(a * at[0] + b * at[1] + c)


def top_of(faces: list[vecview.Face]) -> vecview.Face:
    return next(f for f in faces if f.name == "+z")


def area(face: Face) -> float:
    """The area a face fills, even-odd: its outline less its holes."""
    rings = [face.points, *face.holes]
    newell = [np.cross(r, np.roll(r, -1, axis=0)).sum(axis=0) / 2 @ face.normal for r in rings]
    return float(newell[0] + sum(newell[1:]))


def closed_up(faces: list[Face]) -> bool:
    """Whether every edge of every ring is run once each way, as on a closed surface."""
    edges = [
        (tuple(np.round(a, 9)), tuple(np.round(b, 9)))
        for f in faces
        for r in (f.points, *f.holes)
        for a, b in zip(r, np.roll(r, -1, axis=0), strict=True)
    ]
    return sorted(edges) == sorted((b, a) for a, b in edges)


def enclosed(faces: list[Face]) -> float:
    """Divergence theorem over every ring: the volume a closed surface with holed faces encloses."""
    rings = [r for f in faces for r in (f.points, *f.holes)]
    return sum(float(np.cross(r, np.roll(r, -1, axis=0)).sum(axis=0) @ r[0]) for r in rings) / 6


class TestCutsThroughHoles:
    SQUARE = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)

    def square_with(self, lo: float, hi: float) -> list[Face]:
        hole = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=float)
        return vecview.prism_faces(vecview.Region(self.SQUARE, [hole]), 0.0, 1.0)

    @pytest.mark.parametrize("theta0", [0.0, 90.0, 180.0, 270.0])
    def test_a_plane_touching_a_hole_works_from_any_starting_vertex(self, theta0: float) -> None:
        """The hole's first vertex can lie on the cut; containment must not hinge on it."""
        ring = vecview.annulus((0, 0), 1.0, 2.0, n=32, theta0_deg=theta0)
        faces = vecview.cut(vecview.prism_faces(ring, 0.0, 1.0), (-1, 0, 0), (-1, 0, 0))
        assert closed_up(faces)
        top = next(f for f in faces if f.name == "+z")
        assert len(top.holes) == 1
        whole = vecview.prism_faces(ring, 0.0, 1.0)
        cap = next(f for f in whole if f.name == "+z")
        assert enclosed(faces) == pytest.approx(area(cap) - 2.430289446294295)

    def test_a_plane_through_a_diamond_hole_tip(self) -> None:
        diamond = np.array([[2, 1], [3, 2], [2, 3], [1, 2]], dtype=float)
        faces = vecview.cut(
            vecview.prism_faces(vecview.Region(self.SQUARE, [diamond]), 0, 1), (1, 2, 0), (-1, 0, 0)
        )
        assert closed_up(faces) and enclosed(faces) == pytest.approx(12.0 - 2.0)

    @pytest.mark.parametrize(
        ("lo", "hi", "x", "volume"),
        [(1.0, 2.0, 2.0, 7.0), (2.5, 3.5, 3.5, 13.0), (1.0, 2.0, 1.0, 4.0)],
        ids=["far-side", "far-side-near-edge", "near-side"],
    )
    def test_a_plane_along_a_hole_wall_opens_the_hole(
        self, lo: float, hi: float, x: float, volume: float
    ) -> None:
        faces = vecview.cut(self.square_with(lo, hi), (x, 0, 0), (1, 0, 0))
        assert closed_up(faces)
        assert enclosed(faces) == pytest.approx(volume)
        assert all(not f.holes for f in faces)  # the hole is now a notch, or gone

    def test_a_plane_along_an_inner_wall_of_a_u(self) -> None:
        u = [(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)]
        faces = vecview.cut(vecview.prism_faces(u, 0.0, 2.0), (1, 0, 0), (1, 0, 0))
        assert closed_up(faces) and enclosed(faces) == pytest.approx(4.0)

    def test_a_plane_along_a_face_replaces_it_with_one_cap(self) -> None:
        faces = vecview.cut(vecview.prism_faces(self.SQUARE, 0.0, 1.0), (0, 4, 0), (0, 1, 0))
        assert closed_up(faces) and enclosed(faces) == pytest.approx(16.0)
        assert [f.name for f in faces].count("cut") == 1

    def test_random_cuts_through_holes_close_up(self) -> None:
        rng = np.random.default_rng(1)
        shapes = [
            vecview.Region(self.SQUARE, [np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=float)]),
            vecview.annulus((0, 0), 1.0, 2.0, n=24),
        ]
        for shape in shapes:
            solid = vecview.prism_faces(shape, 0.0, 1.0)
            corners = np.vstack(shape.rings)
            for trial in range(60):
                theta = rng.uniform(0, 2 * np.pi)
                normal = (np.cos(theta), np.sin(theta), 0.0)
                if trial % 4 == 0:
                    normal = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)][trial // 4 % 4]
                origin = (*corners[rng.integers(len(corners))], 0.0)  # through a vertex
                assert closed_up(vecview.cut(solid, origin, normal)), (origin, normal)


class TestHoles:
    OUTER = vecview.outlines.rect((-2, -2), (2, 2))
    VIAS = tuple(vecview.outlines.regular(6, 0.4, center=(x, 0)) for x in (-1.0, 1.0))

    @staticmethod
    def newell(points: np.ndarray) -> np.ndarray:
        return np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)

    @pytest.mark.parametrize("flip", [False, True], ids=["as-built", "holes-ccw"])
    def test_a_plate_with_vias_is_a_closed_outward_wound_solid(self, flip: bool) -> None:
        vias = [v[::-1] for v in self.VIAS] if flip else self.VIAS
        faces = vecview.prism_faces(vecview.Region(self.OUTER, vias), 0.0, 0.5)
        names = [f.name for f in faces]
        # Hole walls first, so a culled solid paints them under its cap.
        assert names[:12] == [f"hole{k}_side-{i}" for k in (0, 1) for i in range(6)]
        assert names[12:] == ["+z", "-z", "side-0", "side-1", "side-2", "side-3"]
        for face in faces:
            assert np.dot(self.newell(face.points), face.normal) > 0, face.name
            for hole in face.holes:
                assert np.dot(self.newell(hole), face.normal) < 0, face.name
                assert_in_plane(hole, face.normal)
        assert closed_up(faces)
        via = 1.5 * np.sqrt(3) * 0.4**2
        assert area(faces[12]) == pytest.approx(16.0 - 2 * via)
        assert area(faces[13]) == pytest.approx(16.0 - 2 * via)

    def test_hole_walls_face_into_the_hole(self) -> None:
        faces = vecview.prism_faces(vecview.Region(self.OUTER, self.VIAS), 0.0, 0.5)
        for face in faces[:6]:
            mid = face.points.mean(axis=0)[:2]
            assert np.linalg.norm(mid + 0.01 * face.normal[:2] - (-1, 0)) < np.linalg.norm(
                mid - (-1, 0)
            ), face.name

    @pytest.mark.parametrize("elev", [15.0, 35.0, 70.0])
    @pytest.mark.parametrize("azim", [-120.0, 30.0, 160.0])
    def test_a_culled_holed_solid_paints_nearer_faces_last(self, azim: float, elev: float) -> None:
        """In an unsorted layer, list order is paint order: it must never put far over near."""
        import shapely

        cam = vecview.OrthographicCamera(azim_deg=azim, elev_deg=elev, scale=10.0)
        ring = vecview.annulus((0, 0), 1.0, 2.0, n=24)
        plate = vecview.Region(self.OUTER, self.VIAS)
        # Exact for one hole however deep, and for holes too shallow to show through each
        # other; deeper ones need a sorted layer, as the docs say.
        for solid in (vecview.prism_faces(ring, 0.0, 3.0), vecview.prism_faces(plate, 0.0, 0.2)):
            drawn = cam.visible(solid)
            screen = [
                shapely.Polygon(cam.project(f.points), [cam.project(h) for h in f.holes])
                for f in drawn
            ]
            lo, hi = (
                np.min([s.bounds[:2] for s in screen], 0),
                np.max([s.bounds[2:] for s in screen], 0),
            )
            # Wherever faces overlap, the one painted last must be the nearest.
            for at in itertools.product(
                *(np.linspace(a, b, 41)[1:-1] for a, b in zip(lo, hi, strict=True))
            ):
                over = [k for k, s in enumerate(screen) if s.contains(shapely.Point(at))]
                if len(over) > 1:
                    depths = [depth_at(cam, drawn[k], np.array(at)) for k in over]
                    assert depths[-1] >= max(depths) - 1e-9, [drawn[k].name for k in over]

    def test_extrude_keeps_hole_walls_in_the_given_order(self) -> None:
        ring = vecview.annulus((0, 0), 1.0, 2.0, n=8)
        section = vecview.outlines.to_plane(ring, u=(0, 1, 0), v=(0, 0, 1))
        faces = vecview.extrude(section, (-3, 0, 0))
        inner = section.holes[0]
        for i in range(8):
            wall = next(f for f in faces if f.name == f"hole0_side-{i}")
            assert wall.name == f"hole0_side-{i}"
            ends = {tuple(np.round(p, 9)) for p in wall.points}
            assert {tuple(np.round(inner[i], 9)), tuple(np.round(inner[(i + 1) % 8], 9))} <= ends
        assert closed_up(faces)

    def test_without_holes_nothing_changes(self) -> None:
        plain = vecview.prism_faces(self.OUTER, 0.0, 1.0)
        listed = vecview.prism_faces(vecview.Region(self.OUTER), 0.0, 1.0)
        assert [f.name for f in plain] == [f.name for f in listed]
        assert all(f.holes == () for f in plain)

    @pytest.mark.parametrize(
        ("holes", "message"),
        [
            ([vecview.outlines.rect((3, 3), (4, 4))], "holes"),
            ([vecview.outlines.rect((1, 1), (3, 3))], "cross"),
            (
                [
                    vecview.outlines.rect((-1, -1), (1, 1)),
                    vecview.outlines.rect((-0.5, -0.5), (0.5, 0.5)),
                ],
                "holes",
            ),
            ([[(0, 0), (1, 0), (2, 0)]], "no area"),
            ([[(0, 0), (1, 0)]], "three"),
        ],
        ids=["outside", "crossing", "nested", "flat", "two"],
    )
    def test_refuses_holes_that_are_not_holes(self, holes: list, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            vecview.prism_faces(vecview.Region(self.OUTER, holes), 0.0, 1.0)

    def test_an_annulus_is_a_ring(self) -> None:
        outer, inner = vecview.annulus((1, 2), 1.0, 2.0, n=24, theta0_deg=30).rings
        assert len(outer) == len(inner) == 24
        assert np.allclose(np.linalg.norm(outer - (1, 2), axis=1), 2.0)
        assert np.allclose(np.linalg.norm(inner - (1, 2), axis=1), 1.0)
        assert np.allclose(inner[0] - (1, 2), 0.5 * (outer[0] - (1, 2)))  # both from theta0
        assert np.allclose(
            outer[0] - (1, 2), 2.0 * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
        )

    @pytest.mark.parametrize(
        ("r_in", "r_out", "n"), [(0.0, 1.0, 12), (2.0, 1.0, 12), (1.0, 2.0, 2)]
    )
    def test_an_annulus_rejects_malformed_arguments(
        self, r_in: float, r_out: float, n: int
    ) -> None:
        with pytest.raises(ValueError):
            vecview.annulus((0, 0), r_in, r_out, n=n)

    def test_a_full_turn_sector_points_to_annulus(self) -> None:
        with pytest.raises(ValueError, match="annulus"):
            vecview.annulus_sector((0, 0), 1.0, 2.0, 0, 360)

    def test_many_holes_are_checked_quickly(self) -> None:
        """A perforated plate: rings that lie apart are not tested edge against edge."""
        import time

        plate = vecview.outlines.rect((-1, -1), (20, 20))
        holes = [
            vecview.outlines.regular(32, 0.3, center=(i, j)) for i in range(20) for j in range(20)
        ]
        began = time.perf_counter()
        faces = vecview.prism_faces(vecview.Region(plate, holes), 0.0, 0.2)
        assert time.perf_counter() - began < 5.0  # took minutes and gigabytes when quadratic
        assert len(faces) == 400 * 32 + 2 + 4

    def test_trim_corners_refuses_a_face_with_holes(self) -> None:
        with pytest.raises(ValueError, match="holes"):
            vecview.trim_corners(
                vecview.prism_faces(vecview.Region(self.OUTER, self.VIAS), 0, 1), 0.01
            )

    def test_a_cut_across_a_ring_leaves_two_caps_and_c_shaped_ends(self) -> None:
        ring = vecview.annulus((0, 0), 1.0, 2.0, n=16)
        faces = vecview.cut(vecview.prism_faces(ring, 0.0, 1.0), (0, 0.2, 0), (0, 1, 0))
        caps = {f.name: f for f in faces if f.name.startswith("cut")}
        assert sorted(caps) == ["cut-0", "cut-1"]
        assert all(not f.holes for f in faces)
        assert closed_up(faces)
        for cap in caps.values():
            assert np.allclose(cap.points[:, 1], 0.2) and np.allclose(cap.normal, (0, 1, 0))
        top = next(f for f in faces if f.name == "+z")
        assert area(top) > 0 and area(top) < area(top_of(vecview.prism_faces(ring, 0, 1)))

    def test_a_cut_through_a_ring_wall_keeps_its_hole(self) -> None:
        ring = vecview.annulus((0, 0), 1.0, 2.0, n=16)
        faces = vecview.cut(vecview.prism_faces(ring, 0.0, 1.0), (0, 0, 0.4), (0, 0, 1))
        (cap,) = (f for f in faces if f.name == "cut")
        assert len(cap.holes) == 1
        assert area(cap) == pytest.approx(area(top_of(vecview.prism_faces(ring, 0, 1))))
        assert closed_up(faces)

    def test_a_cut_through_a_via_keeps_the_rest_of_the_plate(self) -> None:
        faces = vecview.cut(
            vecview.prism_faces(vecview.Region(self.OUTER, self.VIAS), 0.0, 0.5),
            (-1, 0, 0),
            (1, 0, 0),
        )
        top = next(f for f in faces if f.name == "+z")
        assert not top.holes  # the via cut in half is now a notch in the outline
        assert closed_up(faces)
        via = 1.5 * np.sqrt(3) * 0.4**2
        assert area(top) == pytest.approx(4.0 - via / 2)


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
            (1, 2, 0, 361, 8),
            (1, 2, 0, 360, 2),
            (1, 2, 0, 90, 0),
        ],
    )
    def test_rejects_malformed_arguments(
        self, r_in: float, r_out: float, t0: float, t1: float, n: int
    ) -> None:
        with pytest.raises(ValueError):
            vecview.annulus_sector((0, 0), r_in, r_out, t0, t1, n=n)


class TestArcShape:
    def test_lies_on_the_circle_between_the_angles(self) -> None:
        arc = vecview.arc_shape((1, 2, 3), (1, 0, 0), (0, 1, 0), 2.0, 0.0, 90.0, n=8)
        assert np.allclose(np.linalg.norm(arc - (1, 2, 3), axis=1), 2.0)
        assert np.allclose(arc[0], (3, 2, 3)) and np.allclose(arc[-1], (1, 4, 3))

    def test_v_need_not_be_perpendicular(self) -> None:
        """The arc between two vectors is measured in their plane, from the first."""
        a, b = np.array([0.0, 0.0, 1.0]), vecview.unit([1.0, 0.0, 1.0])
        arc = vecview.arc_shape((0, 0, 0), a, b, 1.0, 0.0, 45.0)
        assert np.allclose(arc[-1], b)

    def test_rejects_parallel_axes(self) -> None:
        with pytest.raises(ValueError, match="parallel"):
            vecview.arc_shape((0, 0, 0), (1, 0, 0), (2, 0, 0), 1.0, 0.0, 90.0)


class TestHelix:
    def test_radius_and_pitch(self) -> None:
        h = vecview.helix((0, 0, 0), (0, 0, 1), 0.5, 2.0, 3.0, n_per_turn=40)
        assert np.allclose(np.hypot(h[:, 0], h[:, 1]), 0.5)
        assert np.isclose(h[-1, 2] - h[0, 2], 6.0)
        assert np.allclose(h[40, :2], h[0, :2]), "one full turn returns to the same angle"

    @pytest.mark.parametrize(("pitch", "sign"), [(1.0, 1.0), (-1.0, -1.0)])
    def test_handedness_follows_the_pitch(self, pitch: float, sign: float) -> None:
        h = vecview.helix((0, 0, 0), (0, 0, 1), 1.0, pitch, 1.0)
        # Right-handed: counter-clockwise about +z while rising.
        turning = h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]
        assert turning > 0 and np.sign(h[1, 2] - h[0, 2]) == sign

    def test_rejects_no_turns(self) -> None:
        with pytest.raises(ValueError, match="turns"):
            vecview.helix((0, 0, 0), (0, 0, 1), 1.0, 1.0, 0.0)


class TestSurfaceFaces:
    def test_one_quad_per_grid_cell(self) -> None:
        x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3), indexing="ij")
        faces = vecview.surface_faces(x, y, x * y)
        assert len(faces) == 3 * 2
        assert faces[0].name == "q-0-0" and faces[-1].name == "q-2-1"

    def test_ij_height_field_normals_point_up_and_wind_ccw(self) -> None:
        x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), indexing="ij")
        for face in vecview.surface_faces(x, y, 0.1 * x**2):
            assert face.normal[2] > 0
            p = face.points
            assert np.dot(np.cross(p[1] - p[0], p[2] - p[0]), face.normal) > 0

    def test_rejects_mismatched_grids(self) -> None:
        with pytest.raises(ValueError, match="one shape"):
            vecview.surface_faces(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 4)))


class TestConvexPolyhedron:
    def test_cube_has_six_square_faces(self) -> None:
        corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        faces = vecview.convex_polyhedron(corners)
        assert sorted(len(f.points) for f in faces) == [4] * 6

    def test_truncated_octahedron_merges_coplanar_triangles(self) -> None:
        """The fcc Brillouin zone: eight hexagons and six squares."""
        verts = {
            tuple(s * v for s, v in zip(signs, perm, strict=True))
            for perm in __import__("itertools").permutations((0, 1, 2))
            for signs in __import__("itertools").product((1, -1), repeat=3)
        }
        faces = vecview.convex_polyhedron(sorted(verts))
        assert sorted(len(f.points) for f in faces) == [4] * 6 + [6] * 8

    def test_faces_wind_ccw_about_outward_normals(self) -> None:
        rng = np.random.default_rng(1)
        faces = vecview.convex_polyhedron(rng.normal(size=(30, 3)))
        for face in faces:
            p = face.points
            area = sum(np.cross(p[i] - p[0], p[i + 1] - p[0]) for i in range(1, len(p) - 1))
            assert np.dot(area, face.normal) > 0
            assert_in_plane(p, face.normal)

    def test_interior_points_are_ignored(self) -> None:
        corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        assert len(vecview.convex_polyhedron([*corners, (0, 0, 0), (0.5, 0, 0)])) == 6

    def test_rejects_a_flat_point_set(self) -> None:
        with pytest.raises(ValueError, match="coplanar"):
            vecview.convex_polyhedron([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])


class TestTrimCorners:
    def test_every_point_keeps_its_distance_from_the_corners(self) -> None:
        square = vecview.box_faces((0, 0, 0), (2, 2, 2))[:1]
        (cut,) = vecview.trim_corners(square, 0.3)
        corners = square[0].points
        gaps = np.linalg.norm(cut.points[:, None, :] - corners[None, :, :], axis=2).min(axis=1)
        assert gaps.min() == pytest.approx(0.3)
        assert_in_plane(cut.points, cut.normal)

    def test_keeps_name_normal_and_winding(self) -> None:
        face = vecview.box_faces((0, 0, 0), (2, 2, 2))[0]
        (cut,) = vecview.trim_corners([face], 0.3)
        p = cut.points
        area = sum(np.cross(p[i] - p[0], p[i + 1] - p[0]) for i in range(1, len(p) - 1))
        assert cut.name == face.name and np.dot(area, face.normal) > 0

    def test_rejects_cuts_that_would_overlap(self) -> None:
        with pytest.raises(ValueError, match="half the shortest edge"):
            vecview.trim_corners(vecview.box_faces((0, 0, 0), (1, 1, 1)), 0.5)


class TestCut:
    BOX = vecview.box_faces((0, 0, 0), (2, 2, 2))
    L_PRISM = vecview.prism_faces([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)], 0.0, 1.0)

    @staticmethod
    def newell(points: np.ndarray) -> np.ndarray:
        return np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)

    @staticmethod
    def volume(faces: list[Face]) -> float:
        """Divergence theorem: a closed, outward-wound surface encloses this volume."""
        return sum(float(np.dot(f.points[0], TestCut.newell(f.points))) for f in faces) / 6.0

    @pytest.mark.parametrize(
        ("solid", "origin", "normal"),
        [
            ("BOX", (0.3, 0, 0), (1, 0, 0)),
            ("BOX", (0, 0, 0), (1, 1, 1)),
            ("L_PRISM", (0, 0, 0.5), (0, 0, 1)),
            ("L_PRISM", (2, 0, 0), (1, 0.2, 0.3)),
        ],
        ids=["box-axis", "box-diagonal", "L-flat", "L-oblique"],
    )
    def test_the_result_is_closed_and_wound_outward(self, solid: str, origin, normal) -> None:  # type: ignore[no-untyped-def]
        faces = vecview.cut(getattr(self, solid), origin, normal)
        for face in faces:
            assert np.dot(self.newell(face.points), face.normal) > 0, face.name
            assert_in_plane(face.points, face.normal)
        # Closed: every edge is run once each way.
        edges = [
            (tuple(np.round(a, 9)), tuple(np.round(b, 9)))
            for f in faces
            for a, b in zip(f.points, np.roll(f.points, -1, axis=0), strict=True)
        ]
        assert sorted(edges) == sorted((b, a) for a, b in edges)

    def test_keeps_the_volume_behind_the_plane(self) -> None:
        assert self.volume(vecview.cut(self.BOX, (0.5, 0, 0), (1, 0, 0))) == pytest.approx(6.0)
        assert self.volume(vecview.cut(self.L_PRISM, (0, 0, 0.25), (0, 0, -1))) == pytest.approx(
            3.75
        )

    def test_the_cap_lies_in_the_plane_facing_along_the_normal(self) -> None:
        *_, cap = vecview.cut(self.BOX, (0.3, 0, 0), (1, 0, 0))
        assert cap.name == "cut" and np.allclose(cap.normal, (1, 0, 0))
        assert np.allclose(cap.points[:, 0], 0.3)
        assert np.linalg.norm(self.newell(cap.points)) / 2 == pytest.approx(4.0)

    def test_faces_keep_their_names_and_the_cut_side_goes(self) -> None:
        names = [f.name for f in vecview.cut(self.BOX, (0.3, 0, 0), (1, 0, 0))]
        assert names == ["-x", "+y", "-y", "+z", "-z", "cut"]

    def test_a_plane_through_two_solids_caps_each(self) -> None:
        two = vecview.box_faces((0, 0, 0), (1, 1, 1)) + vecview.box_faces((3, 0, 0), (1, 1, 1))
        caps = [f.name for f in vecview.cut(two, (0, 0, 0), (0, 0, 1)) if f.name.startswith("cut")]
        assert caps == ["cut-0", "cut-1"]

    def test_a_plane_that_misses_keeps_all_or_nothing(self) -> None:
        assert len(vecview.cut(self.BOX, (5, 0, 0), (1, 0, 0))) == 6
        assert vecview.cut(self.BOX, (-5, 0, 0), (1, 0, 0)) == []

    def test_a_plane_through_a_cavity_leaves_a_cap_with_a_hole(self) -> None:
        outer = vecview.box_faces((0, 0, 0), (4, 4, 2))
        cavity = [
            f._replace(points=f.points[::-1].copy(), normal=-f.normal)
            for f in vecview.box_faces((0, 0, 0), (2, 2, 1))
        ]
        *_, cap = vecview.cut(outer + cavity, (0, 0, 0), (0, 0, 1))
        assert cap.name == "cut" and len(cap.holes) == 1
        assert np.dot(self.newell(cap.holes[0]), cap.normal) < 0  # wound the other way
        assert area(cap) == pytest.approx(16.0 - 4.0)
