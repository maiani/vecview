"""Flat arrows -- straight, double-headed, and along a path -- checked as numbers."""

from __future__ import annotations

import numpy as np
import pytest

import vecview


def assert_in_plane(pts: np.ndarray, normal: np.ndarray) -> None:
    offsets = (pts - pts.mean(axis=0)) @ vecview.unit(normal)
    assert np.allclose(offsets, 0.0, atol=1e-12), "points are not coplanar with the normal"


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


def quarter_arc(radius: float = 2.0, n: int = 40) -> np.ndarray:
    t = np.linspace(0, np.pi / 2, n)
    return np.c_[radius * np.cos(t), radius * np.sin(t), np.zeros(n)]


def sides(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two sides of a path arrow's shaft, tail to neck, from its polygon."""
    m = (len(pts) - 3) // 2
    return pts[:m], pts[-1 : -m - 1 : -1]


class TestPathArrowShape:
    @pytest.mark.parametrize(
        ("direction", "normal"), [((1, 0, 0), (0, 0, 1)), ((1, -1, 0), (1, 1, 1))]
    )
    def test_a_straight_path_is_arrow_shape(self, direction, normal) -> None:
        d = np.cross(vecview.unit(normal), np.cross(direction, vecview.unit(normal)))
        start = np.array([1.0, 2.0, 3.0])
        path = [start, start + 4.0 * vecview.unit(d)]
        expected = vecview.arrow_shape(start, d, 4.0, normal, 0.2, 0.8, 0.6)
        assert np.allclose(vecview.path_arrow_shape(path, normal, 0.2, 0.8, 0.6), expected)

    def test_a_head_longer_than_the_path_is_clamped_as_arrow_shape_does(self) -> None:
        path = [(0, 0, 0), (1, 0, 0)]
        expected = vecview.arrow_shape((0, 0, 0), (1, 0, 0), 1.0, (0, 0, 1), 1, 3, 5)
        assert np.allclose(vecview.path_arrow_shape(path, (0, 0, 1), 1, 3, 5), expected)

    def test_the_shaft_is_the_path_widened_on_both_sides(self) -> None:
        pts = vecview.path_arrow_shape(quarter_arc(), (0, 0, 1), 0.2, 0.6, 0.5)
        left, right = sides(pts)
        middle = (left + right) / 2
        for side in (left, right):
            for a, b, c in zip(middle[:-1], middle[1:], side[:-1], strict=True):
                along = vecview.unit(b - a)
                offset = (c - a) - ((c - a) @ along) * along
                assert np.linalg.norm(offset) == pytest.approx(0.1)
        assert np.allclose(middle[:-1], quarter_arc()[: len(middle) - 1])

    def test_the_head_takes_the_last_head_len_of_the_path(self) -> None:
        arc = quarter_arc()
        pts = vecview.path_arrow_shape(arc, (0, 0, 1), 0.2, 0.6, 0.5)
        assert np.allclose(pts[len(pts) // 2], arc[-1])  # the tip
        left, right = sides(pts)
        neck = (left[-1] + right[-1]) / 2
        steps = np.linalg.norm(np.diff(arc, axis=0), axis=1)
        k = int(np.searchsorted(np.cumsum(steps), steps.sum() - 0.5))
        before = steps[:k].sum() + np.linalg.norm(neck - arc[k])
        assert before == pytest.approx(steps.sum() - 0.5)
        head = pts[len(pts) // 2 - 1] - pts[len(pts) // 2 + 1]
        assert np.linalg.norm(head) == pytest.approx(0.6)

    def test_lies_in_the_plane_of_its_normal(self) -> None:
        normal = vecview.unit([1.0, 1.0, 1.0])
        u, v = vecview.unit([1.0, -1.0, 0.0]), vecview.unit([1.0, 1.0, -2.0])
        t = np.linspace(0, 2, 30)[:, None]
        path = 3.0 + t * u + np.sin(t) * v
        assert_in_plane(vecview.path_arrow_shape(path, normal, 0.1, 0.4, 0.3), normal)

    def test_a_head_starting_just_past_a_vertex_does_not_fold_the_shaft(self) -> None:
        """A regression: the shaft once ended square to the head's chord, not the path."""
        arc = quarter_arc(n=40)
        run = np.r_[0, np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1))]
        head_len = run[-1] - run[30] - 1e-6
        pts = vecview.path_arrow_shape(arc, (0, 0, 1), 0.3, 0.8, head_len)
        left, right = sides(pts)
        assert np.linalg.norm(left[-1] - right[-1]) == pytest.approx(0.3, rel=1e-3)  # a miter

    def test_repeated_points_are_ignored(self) -> None:
        arc = quarter_arc(n=10)
        doubled = np.repeat(arc, 2, axis=0)
        assert np.allclose(
            vecview.path_arrow_shape(doubled, (0, 0, 1), 0.2, 0.6, 0.5),
            vecview.path_arrow_shape(arc, (0, 0, 1), 0.2, 0.6, 0.5),
        )

    @pytest.mark.parametrize(
        ("path", "match"),
        [
            ([(1, 1, 0), (1, 1, 0)], "two distinct points"),
            ([(0, 0, 0), (1, 0, 0.5)], "lie in a plane"),
            (quarter_arc(radius=0.05), "bends tighter"),
            ([(0, 0, 0), (2, 0, 0), (0, 0.01, 0)], "turns back"),
        ],
    )
    def test_rejects_paths_it_cannot_draw(self, path, match) -> None:
        with pytest.raises(ValueError, match=match):
            vecview.path_arrow_shape(path, (0, 0, 1), 0.2, 0.6, 0.01)


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

    def test_heads_longer_than_half_meet_at_the_centre(self) -> None:
        """Otherwise each neck runs past the centre and the polygon self-crosses."""
        pts = vecview.double_arrow_shape((0, 0, 0), (1, 0, 0), 2.0, (0, 0, 1), 0.2, 1.0, 5.0)
        half = len(pts) // 2
        # Along the arrow, the outline runs monotonically from one tip to the other.
        assert np.all(np.diff(pts[: half + 1, 0]) <= 1e-12)
        assert np.ptp(pts[:, 0]) == pytest.approx(2.0)
