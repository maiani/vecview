"""Flat outlines: checked as regions, through shapely, not as vertex lists."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

import vecview
from vecview import outlines

SQUARE = outlines.rect((0, 0), (1, 1))
WIRE = outlines.regular(6, 0.4)
L_SHAPE = np.array([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)], dtype=float)


def signed_area(outline: np.ndarray) -> float:
    outline = outline.exterior if isinstance(outline, vecview.Region) else outline
    x, y = outline[:, 0], outline[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def region(outline: np.ndarray) -> Polygon:
    shape = (
        Polygon(outline.exterior, outline.holes)
        if isinstance(outline, vecview.Region)
        else Polygon(outline)
    )
    assert shape.is_valid
    return shape


def shaped(rings: list[np.ndarray]) -> Polygon:
    """A shape with holes as shapely fills it: the outline less every hole."""
    shape = Polygon(rings.exterior, rings.holes)
    assert shape.is_valid
    return shape


class TestBuilding:
    def test_rect_is_counter_clockwise(self) -> None:
        assert signed_area(outlines.rect((-1, 2), (3, 5))) == pytest.approx(12.0)

    def test_rect_needs_hi_above_lo(self) -> None:
        with pytest.raises(ValueError, match="exceed"):
            outlines.rect((0, 0), (1, 0))

    def test_regular_polygon_has_its_vertices_on_the_circle(self) -> None:
        hexagon = outlines.regular(6, 0.4, center=(1, 2), rotate_deg=30)
        assert np.allclose(np.linalg.norm(hexagon - (1, 2), axis=1), 0.4)
        assert np.allclose(hexagon[0], (1 + 0.4 * np.cos(np.pi / 6), 2 + 0.4 * np.sin(np.pi / 6)))
        assert signed_area(hexagon) > 0

    def test_a_default_hexagon_lies_on_a_flat_edge(self) -> None:
        assert np.isclose(WIRE[:, 1].min(), WIRE[4, 1]) and np.isclose(WIRE[4, 1], WIRE[5, 1])


class TestOffset:
    def test_a_mitred_offset_keeps_square_corners(self) -> None:
        grown = outlines.offset(SQUARE, 0.1)
        assert region(grown).area == pytest.approx(1.2**2)
        assert signed_area(grown) > 0

    def test_a_round_offset_is_the_minkowski_sum_with_a_disk(self) -> None:
        grown = outlines.offset(SQUARE, 0.1, join="round")
        assert region(grown).area == pytest.approx(1 + 4 * 0.1 + np.pi * 0.01, rel=1e-3)

    def test_a_negative_offset_shrinks(self) -> None:
        assert region(outlines.offset(SQUARE, -0.1)).area == pytest.approx(0.8**2)

    def test_shrinking_past_nothing_says_so(self) -> None:
        with pytest.raises(ValueError, match="too thin"):
            outlines.offset(SQUARE, -0.6)


class TestFilm:
    def test_lies_on_the_chosen_edges_and_outside_the_body(self) -> None:
        film = region(outlines.film(WIRE, [0, 1, 2], 0.07))
        assert film.intersection(region(WIRE)).area == pytest.approx(0.0, abs=1e-12)
        for i in range(6):
            mid = (WIRE[i] + WIRE[(i + 1) % 6]) / 2
            just_outside = Point(mid * (1 + 0.05 / np.linalg.norm(mid)))  # edge normals are radial
            assert film.contains(just_outside) == (i in (0, 1, 2)), f"edge {i}"

    def test_thickness_is_the_distance_from_the_body(self) -> None:
        film = region(outlines.film(SQUARE, [2], 0.25))  # the top edge
        assert film.bounds == pytest.approx((0.0, 1.0, 1.0, 1.25))

    def test_a_run_may_wrap_past_the_last_vertex(self) -> None:
        wrapped = region(outlines.film(WIRE, [5, 0], 0.05))
        direct = region(outlines.film(np.roll(WIRE, -5, axis=0), [0, 1], 0.05))
        assert wrapped.symmetric_difference(direct).area == pytest.approx(0.0, abs=1e-12)

    def test_either_winding_gives_the_same_film(self) -> None:
        ccw = region(outlines.film(SQUARE, [1, 2], 0.1))
        # Reversed, the edge from vertex i to i + 1 is the old edge 2 - i, wrapped.
        cw = region(outlines.film(SQUARE[::-1], [1, 2], 0.1))
        assert ccw.area == pytest.approx(cw.area)
        assert cw.intersection(region(SQUARE)).area == pytest.approx(0.0, abs=1e-12)

    def test_a_concave_run_stays_out_of_the_body(self) -> None:
        film = region(outlines.film(L_SHAPE, [1, 2, 3], 0.2))
        assert film.intersection(region(L_SHAPE)).area == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("edges", "thickness", "message"),
        [
            ([0, 2], 0.1, "unbroken run"),
            ([6], 0.1, "indices"),
            ([], 0.1, "indices"),
            ([0], 0.0, "positive"),
        ],
        ids=["gap", "out-of-range", "none", "zero"],
    )
    def test_refuses_what_is_not_one_film(self, edges, thickness, message) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match=message):
            outlines.film(WIRE, edges, thickness)


class TestBooleans:
    def test_overlapping_outlines_unite_into_one(self) -> None:
        (both,) = outlines.union(SQUARE, outlines.rect((0.5, 0.5), (2, 2)))
        assert region(both).area == pytest.approx(1 + 2.25 - 0.25)
        assert signed_area(both) > 0

    def test_disjoint_outlines_stay_separate_largest_first(self) -> None:
        small, big = SQUARE, outlines.rect((5, 0), (7, 2))
        assert [region(o).area for o in outlines.union(small, big)] == pytest.approx([4.0, 1.0])

    def test_a_cut_through_the_middle_leaves_two(self) -> None:
        parts = outlines.difference(outlines.rect((0, 0), (3, 1)), outlines.rect((1, -1), (2, 2)))
        assert [region(p).area for p in parts] == pytest.approx([1.0, 1.0])
        assert (
            parts[0].exterior[:, 0].min() < parts[1].exterior[:, 0].min()
        )  # ties ordered by position

    def test_a_cut_that_removes_everything_leaves_none(self) -> None:
        assert outlines.difference(SQUARE, outlines.rect((-1, -1), (2, 2))) == []

    @pytest.mark.parametrize(
        "make",
        [
            lambda: outlines.difference(
                outlines.rect((0, 0), (3, 3)), outlines.rect((1, 1), (2, 2))
            ),
            lambda: outlines.union(
                outlines.rect((0, 0), (3, 1)),
                outlines.rect((0, 2), (3, 3)),
                outlines.rect((0, 0), (1, 3)),
                outlines.rect((2, 0), (3, 3)),
            ),
        ],
        ids=["difference", "union"],
    )
    def test_a_hole_is_kept_as_a_ring_of_its_own(self, make) -> None:  # type: ignore[no-untyped-def]
        (ring,) = make()
        outline, hole = ring.rings
        assert signed_area(outline) == pytest.approx(9.0)
        assert signed_area(hole) == pytest.approx(-1.0)  # clockwise
        assert shaped(ring).area == pytest.approx(8.0)
        assert not shaped(ring).contains(Point(1.5, 1.5))

    def test_a_result_without_holes_is_still_a_region(self) -> None:
        (both,) = outlines.union(SQUARE, outlines.rect((0.5, 0.5), (2, 2)))
        assert isinstance(both, vecview.Region) and not both.holes

    def test_holed_shapes_combine_further(self) -> None:
        (ring,) = outlines.difference(outlines.rect((0, 0), (3, 3)), outlines.rect((1, 1), (2, 2)))
        # Filling the window closes the hole; cutting across the frame opens it.
        (closed,) = outlines.union(ring, outlines.rect((0.5, 0.5), (2.5, 2.5)))
        assert isinstance(closed, vecview.Region) and region(closed).area == pytest.approx(9.0)
        (opened,) = outlines.difference(ring, outlines.rect((1.2, -1), (1.8, 1.5)))
        assert isinstance(opened, vecview.Region) and region(opened).area == pytest.approx(7.4)

    def test_two_holes_come_in_order_of_position(self) -> None:
        plate = outlines.rect((0, 0), (5, 2))
        vias = [outlines.regular(8, 0.3, center=(x, 1)) for x in (3.5, 1.5)]
        (result,) = outlines.difference(plate, *vias)
        first, second = result.holes
        assert first[:, 0].mean() == pytest.approx(1.5) and second[:, 0].mean() == pytest.approx(
            3.5
        )

    @pytest.mark.parametrize(
        "make",
        [
            lambda: outlines.difference(SQUARE * 4, [[2, 0], [3, 1], [1, 1]]),
            lambda: outlines.difference(
                SQUARE * 4, [[1, 1], [2, 1], [2, 2]], [[2, 2], [3, 2], [3, 3]]
            ),
            lambda: outlines.union(
                outlines.rect((0, 0), (3, 1)),
                outlines.rect((0, 0), (1, 3)),
                outlines.rect((2, 0), (3, 2)),
                outlines.rect((1, 2), (2, 3)),
            ),
            lambda: outlines.offset(vecview.Region(SQUARE * 4, [[[2, 0], [3, 1], [1, 1]]]), 0.0),
        ],
        ids=["hole-on-outline", "holes-at-a-vertex", "c-closed-at-a-point", "given"],
    )
    def test_a_hole_touching_at_a_point_is_refused(self, make) -> None:  # type: ignore[no-untyped-def]
        """What extrude would refuse is refused here too, so a result passes on as it comes."""
        with pytest.raises(ValueError, match="touching"):
            make()

    def test_a_hole_outside_the_outline_is_refused(self) -> None:
        with pytest.raises(ValueError, match="holes"):
            outlines.union(vecview.Region(SQUARE, [SQUARE + 5]))

    def test_intersection(self) -> None:
        (common,) = outlines.intersection(SQUARE, outlines.rect((0.5, -1), (2, 0.5)))
        assert region(common).bounds == pytest.approx((0.5, 0.0, 1.0, 0.5))
        assert outlines.intersection(SQUARE, outlines.rect((5, 5), (6, 6))) == []

    def test_a_self_crossing_outline_is_refused(self) -> None:
        with pytest.raises(ValueError, match="simple"):
            outlines.union([(0, 0), (1, 1), (1, 0), (0, 2)])


class TestHoledShapes:
    def test_a_film_on_every_edge_is_a_shell_round_the_body(self) -> None:
        outside, body = outlines.film(WIRE, range(6), 0.1).rings
        assert np.allclose(sorted(map(tuple, body)), sorted(map(tuple, WIRE)))
        assert signed_area(body) < 0 < signed_area(outside)
        assert shaped(vecview.Region(outside, [body])).area == pytest.approx(
            region(outlines.offset(WIRE, 0.1)).area - region(WIRE).area
        )

    def test_growing_a_c_closes_it_into_a_ring(self) -> None:
        slit = outlines.rect((1.9, 2.5), (2.1, 4.5))
        (c,) = outlines.difference(
            outlines.rect((0, 0), (4, 4)), outlines.rect((1, 1), (3, 3)), slit
        )
        assert isinstance(c, vecview.Region)
        _, hole = outlines.offset(c, 0.2).rings  # the slit closes, the window stays open
        assert region(hole).area == pytest.approx(1.6**2)

    def test_a_film_refuses_an_outline_with_holes(self) -> None:
        with pytest.raises(ValueError, match="holes"):
            outlines.film(
                vecview.Region(outlines.rect((0, 0), (3, 3)), [outlines.rect((1, 1), (2, 2))]),
                [0],
                0.1,
            )

    def test_lifted_ring_by_ring(self) -> None:
        ring = vecview.Region(outlines.rect((0, 0), (3, 3)), [outlines.rect((1, 1), (2, 2))[::-1]])
        outline, hole = outlines.to_plane(ring, origin=(0, 0, 1)).rings
        assert np.allclose(outline[:, 2], 1.0) and np.allclose(hole[:, :2], ring.holes[0])


class TestToPlane:
    def test_maps_x_and_y_onto_the_given_vectors(self) -> None:
        pts = outlines.to_plane(SQUARE, origin=(5, 0, 0), u=(0, 1, 0), v=(0, 0, 2))
        assert np.allclose(pts, [(5, 0, 0), (5, 1, 0), (5, 1, 2), (5, 0, 2)])

    def test_parallel_axes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="parallel"):
            outlines.to_plane(SQUARE, u=(1, 0, 0), v=(2, 0, 0))

    def test_a_lifted_film_extrudes_with_walls_on_its_edges(self) -> None:
        section = outlines.to_plane(
            outlines.film(WIRE, [0, 1, 2], 0.07), (-1, 0, 0), (0, 1, 0), (0, 0, 1)
        )
        faces = vecview.extrude(section, (2, 0, 0))
        assert [f.name for f in faces[:2]] == ["start", "end"]
        assert len(faces) == 2 + len(section.exterior)
