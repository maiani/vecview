"""Planar regions compose without confusing holes, components, or edge order."""

from collections import Counter

import numpy as np
import pytest
import shapely
import svg

import vecview as vv
from vecview._vec import basis_for


def polygon(region: vv.Region) -> shapely.Polygon:
    return shapely.Polygon(region.exterior, region.holes)


def assert_closed(faces: list[vv.Face]) -> None:
    edges = Counter(
        (tuple(np.round(a, 8)), tuple(np.round(b, 8)))
        for face in faces
        for ring in face.region.rings
        for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True)
    )
    assert all(count == edges[b, a] for (a, b), count in edges.items())
    for face in faces:
        u, v = basis_for(face.normal)
        rings = [np.column_stack([r @ u, r @ v]) for r in face.region.rings]
        assert shapely.Polygon(rings[0], rings[1:]).is_valid


def test_region_copies_input_and_exposes_readonly_boundaries() -> None:
    points = vv.outlines.rect((0, 0), (2, 2))
    region = vv.Region(points)
    points[0] = 100
    assert np.array_equal(region.exterior[0], (0, 0))
    with pytest.raises(ValueError, match="read-only"):
        region.exterior[0] = 100


def test_disconnected_regions_and_island_in_a_hole_survive_booleans() -> None:
    frame = vv.Region(vv.outlines.rect((-3, -3), (3, 3)), [vv.outlines.rect((-2, -2), (2, 2))])
    island = vv.outlines.rect((-1, -1), (1, 1))
    parts = vv.outlines.union(frame, island)
    assert len(parts) == 2
    assert [polygon(r).area for r in parts] == pytest.approx([20, 4])
    for part in parts:
        solid = vv.extrude(vv.outlines.to_plane(part), (0.5, 0.2, 2))
        assert_closed(solid)
    # A list of components cannot silently become an exterior followed by holes.
    with pytest.raises(ValueError, match="component"):
        vv.outlines.to_plane(parts)
    with pytest.raises(ValueError, match="Region"):
        vv.prism_faces([frame.exterior, *frame.holes], 0, 1)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("azimuth", [-130, -35, 35, 130])
def test_seamless_hole_walls_cover_exactly_the_facing_quads(reverse: bool, azimuth: float) -> None:
    outer = vv.outlines.rect((-3, -3), (3, 3))
    hole = vv.outlines.regular(8, 1)
    region = vv.Region(outer, [hole[::-1] if reverse else hole])
    camera = vv.OrthographicCamera(azimuth, 24, 40)
    scene = vv.Scene(camera)
    scene.prism_walls(0, region, 0, 1, fill="#888", id="walls")
    # The hole walls are a path of their own, painted before the outer walls.
    paths = [
        el for el in scene.render().elements if getattr(el, "id", None) in ("walls-holes", "walls")
    ]
    assert [p.id for p in paths] == ["walls-holes", "walls"]
    rings = []
    for command in (c for path in paths for c in path.d):
        if isinstance(command, svg.M):
            rings.append([])
        if hasattr(command, "x"):
            rings[-1].append((command.x, command.y))
    strips = [shapely.Polygon(r) for r in rings]
    assert all(p.is_valid for p in strips)
    expected = shapely.union_all(
        [
            shapely.Polygon(camera.project(face.points))
            for face in camera.visible(vv.prism_faces(region, 0, 1))
            if "side-" in face.name
        ]
    )
    actual = shapely.union_all(strips)
    # Emission rounds coordinates to two decimals; allow only that perimeter error.
    assert actual.symmetric_difference(expected).area < expected.length * 0.01


def test_two_cuts_through_a_washer_leave_two_closed_bars() -> None:
    region = vv.Region(vv.outlines.rect((-3, -3), (3, 3)), [vv.outlines.rect((-1, -1), (1, 1))])
    solid = vv.prism_faces(region, 0, 1)
    solid = vv.cut(solid, (0, 0.5, 0), (0, 1, 0))
    solid = vv.cut(solid, (0, -0.5, 0), (0, -1, 0))
    assert_closed(solid)
    caps = [face for face in solid if face.name.startswith("+z")]
    assert len(caps) == 2
    assert sum(shapely.Polygon(f.points[:, :2]).area for f in caps) == pytest.approx(4)
    # A third, oblique cut must consume those split faces too.
    assert_closed(vv.cut(solid, (0, 0, 0.5), (0.1, 0.2, 1)))


def test_a_concave_face_without_holes_uses_the_same_clipping_path() -> None:
    footprint = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    solid = vv.cut(vv.prism_faces(footprint, 0, 1), (0, 2, 0), (0, -1, 0))
    assert_closed(solid)
    caps = [face for face in solid if face.name.startswith("+z")]
    assert len(caps) == 2
    assert sum(shapely.Polygon(f.points[:, :2]).area for f in caps) == pytest.approx(2)


def test_a_tracked_region_renders_like_a_static_region() -> None:
    outer = vv.outlines.rect((-3, -3), (3, 3))

    def footprint(t: float) -> vv.Region:
        return vv.Region(outer, [vv.outlines.regular(12, 0.5 + t)])

    scene = vv.Scene(vv.OrthographicCamera(35, 24, 40))
    scene.prism_walls(0, vv.animation.Track(footprint), 0, 1, fill="#888")
    static = vv.Scene(vv.OrthographicCamera(35, 24, 40))
    static.prism_walls(0, footprint(0.5), 0, 1, fill="#888")
    assert scene.at(0.5).to_svg_document() == static.to_svg_document()


def test_cut_accepts_an_empty_result_from_an_earlier_cut() -> None:
    removed = vv.cut(vv.box_faces((0, 0, 0), (1, 1, 1)), (0, 0, -2), (0, 0, 1))
    assert removed == []
    assert vv.cut(removed, (0, 0, 0), (1, 0, 0)) == []


def test_a_flat_hole_and_a_flat_region_get_errors_that_name_them() -> None:
    square = vv.outlines.rect((-2, -2), (2, 2))
    with pytest.raises(ValueError, match="hole encloses no area"):
        vv.prism_faces(vv.Region(square, [[(0, 0), (0.5, 0.5), (1, 1)]]), 0, 1)
    with pytest.raises(ValueError, match="to_plane"):
        vv.Scene().polygon(0, vv.Region(square, [vv.outlines.rect((-1, -1), (1, 1))]))
