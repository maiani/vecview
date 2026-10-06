"""Parts: objects recorded once and placed, moved, any number of times."""

from __future__ import annotations

import re

import numpy as np
import pytest
import svg

import vecview
from vecview import Camera, OrthographicCamera, Part, Scene

CAM = OrthographicCamera(35.0, 24.0, 40.0)
BOX = vecview.box_faces((0.5, 0.0, 0.0), (1.0, 0.4, 0.2))


def draw(target: Scene | Part) -> None:
    """One of nearly every world-space call, at the part's own origin."""
    target.polygon(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], fill="#9ab", id="tri")
    target.faces(0, BOX, cull=True, fill="#9ab", id="slab", class_="slab")
    gate = vecview.annulus_sector((0, 0), 0.6, 0.8, 0, 120, n=10)
    target.prism_walls(0, gate, 0.0, 0.2, fill="#456", id="gate")
    target.sphere(1, (0, 0, 0), 0.3, fill="#c33", highlight="#fcc", id="atom")
    target.cylinder(1, (0, 0, 0), (1, 0, 0), 0.08, fill="#888", id="bond")
    target.arrow3d(2, (0, 0, 0), (0, 0, 1), 1.0, shaft_r=0.04, head_r=0.1, head_len=0.25)
    target.arrow(
        2, (0, 0, 0), (1, 0, 0), 1.0, normal="camera", shaft_w=0.05, head_w=0.2, head_len=0.2
    )
    target.gaussian(2, (0, 0, 0.01), (1, 0, 0), (0, 1, 0), 0.5, 0.3, id="spot", color="red")
    target.edges(2, BOX, back={"stroke_dasharray": "2"}, back_layer=-1, stroke="#000", id="e")
    target.tube(3, vecview.helix((0, 0, 0), (0, 0, 1), 0.3, 0.2, 2), 0.03, stroke="#000")
    ring = vecview.circle_shape((0, 0, 0), 0.3, (0, 0, 1))
    target.sphere_curve(3, (0, 0, 0), ring, closed=True, back={}, stroke="#000", id="eq")
    target.plane(3, (0, 0, 0.3), (1, 0, 0), (0, 1, 0), id="plot")
    target.slot(4, (0, 0, 1.2), 10.0, 8.0, id="label", align="south")
    target.text(4, (0, 0, 1.3), "s", id="t")
    target.polyline(4, [(0, 0, 0), (0, 1, 1)], stroke="#000", id="ray")
    target.silhouette(4, BOX, fill="#9ab", id="hull")
    target.cone(4, (1, 0, 0), (1, 0, 0.5), 0.2, fill="#a5c", id="cone")


def polygons(scene: Scene) -> list[np.ndarray]:
    """The screen points of every rendered polygon, in order."""
    return [
        np.array([[p.x, p.y] for p in el.points or []])
        for el in scene.render().elements or []
        if isinstance(el, svg.Polygon)
    ]


class TestPlacement:
    def test_in_place_it_draws_what_drawing_directly_does(self) -> None:
        direct, part, placed = Scene(CAM), Part(), Scene(CAM)
        draw(direct)
        draw(part)
        placed.place(0, part)
        assert placed.to_svg_document() == direct.to_svg_document()

    def test_moved_and_scaled(self) -> None:
        direct = Scene(CAM)
        direct.sphere(0, (2, 3, 4), 0.6, fill="#c33")
        direct.cylinder(0, (2, 3, 4), (4, 3, 4), 0.2, fill="#888")
        part = Part()
        part.sphere(0, (0, 0, 0), 0.3, fill="#c33")
        part.cylinder(0, (0, 0, 0), (1, 0, 0), 0.1, fill="#888")
        placed = Scene(CAM)
        placed.place(0, part, at=(2, 3, 4), scale=2.0)
        assert placed.to_svg_document() == direct.to_svg_document()

    def test_rotated_about_z_like_a_rotated_footprint(self) -> None:
        footprint = vecview.annulus_sector((0, 0), 0.6, 0.8, 0, 120, n=10)
        turned = footprint @ np.array([[0.0, 1.0], [-1.0, 0.0]])  # +90 degrees
        direct = Scene(CAM)
        direct.prism_walls(0, turned, 0.0, 0.2, fill="#456")
        part = Part()
        part.prism_walls(0, footprint, 0.0, 0.2, fill="#456")
        placed = Scene(CAM)
        placed.place(0, part, rotate=((0, 0, 1), 90.0))
        assert placed.to_svg_document() == direct.to_svg_document()

    def test_rotation_is_counter_clockwise_about_the_axis(self) -> None:
        part = Part()
        part.polyline(0, [(0, 0, 0), (1, 0, 0)], stroke="#000")
        placed, direct = Scene(CAM), Scene(CAM)
        placed.place(0, part, rotate=((0, 0, 1), 90.0))
        direct.polyline(0, [(0, 0, 0), (0, 1, 0)], stroke="#000")
        assert placed.to_svg_document() == direct.to_svg_document()

    def test_mirror_reflects_and_keeps_faces_wound_outward(self) -> None:
        part = Part()
        part.faces(0, BOX, cull=True, fill="#9ab")
        placed, direct = Scene(CAM), Scene(CAM)
        placed.place(0, part, mirror=(1, 0, 0))
        direct.faces(0, vecview.box_faces((-0.5, 0.0, 0.0), (1.0, 0.4, 0.2)), cull=True)
        key = sorted(tuple(sorted(map(tuple, p))) for p in polygons(placed))
        assert key == sorted(tuple(sorted(map(tuple, p))) for p in polygons(direct))

    def test_mirror_is_applied_before_the_rotation(self) -> None:
        part = Part()
        part.polyline(0, [(0, 0, 0), (1, 0, 0)], stroke="#000")
        placed, direct = Scene(CAM), Scene(CAM)
        placed.place(0, part, mirror=(1, 0, 0), rotate=((0, 0, 1), 90.0))
        direct.polyline(0, [(0, 0, 0), (0, -1, 0)], stroke="#000")
        assert placed.to_svg_document() == direct.to_svg_document()

    def test_a_camera_facing_arrow_still_faces_the_camera(self) -> None:
        part = Part()
        part.arrow(
            0, (0, 0, 0), (0, 0, 1), 1.0, normal="camera", shaft_w=0.1, head_w=0.3, head_len=0.3
        )
        placed, direct = Scene(CAM), Scene(CAM)
        placed.place(0, part, rotate=((1, 0, 0), 90.0))
        direct.arrow(
            0, (0, 0, 0), (0, -1, 0), 1.0, normal="camera", shaft_w=0.1, head_w=0.3, head_len=0.3
        )
        assert np.allclose(polygons(placed)[0], polygons(direct)[0], atol=0.011)

    def test_the_copy_is_taken_when_placed(self) -> None:
        part = Part()
        part.sphere(0, (0, 0, 0), 0.3)
        scene = Scene(CAM)
        scene.place(0, part)
        part.sphere(0, (1, 0, 0), 0.3)
        assert sum(isinstance(el, svg.Circle) for el in scene.render().elements or []) == 1


class TestLayers:
    def test_the_placement_layer_offsets_every_layer(self) -> None:
        part = Part()
        part.polygon(5, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], id="upper")
        part.polygon(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], id="lower")
        scene = Scene(CAM)
        scene.polygon(12, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], id="between")
        scene.place(10, part, id="p")
        assert [el.id for el in scene.render().elements or []] == ["p-lower", "between", "p-upper"]

    def test_a_back_layer_moves_with_the_part(self) -> None:
        part = Part()
        part.edges(5, BOX, back={}, back_layer=0, stroke="#000", id="e")
        scene = Scene(CAM)
        scene.polygon(12, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], id="between")
        scene.place(10, part)
        ids = [el.id for el in scene.render().elements or []]
        assert ids == ["e-back", "between", "e-front"]

    def test_sorting_is_the_host_scene_s_choice(self) -> None:
        part = Part()
        part.sphere(0, (0, 0, 0), 0.3, id="atom")
        scene = Scene(CAM)
        scene.sort_by_depth(10)
        scene.place(10, part, at=-3.0 * CAM.view, id="far")
        scene.place(10, part, at=3.0 * CAM.view, id="near")
        scene.place(10, part, id="mid")
        assert [el.id for el in scene.render().elements or []] == [
            "far-atom",
            "mid-atom",
            "near-atom",
        ]


class TestNames:
    def test_ids_are_prefixed_and_nest(self) -> None:
        cell = Part()
        cell.sphere(0, (0, 0, 0), 0.3, id="atom")
        row = Part()
        row.place(0, cell, id="a")
        row.place(0, cell, at=(1, 0, 0), id="b")
        scene = Scene(CAM)
        scene.place(0, row, id="row")
        assert [el.id for el in scene.render().elements or []] == ["row-a-atom", "row-b-atom"]

    def test_derived_ids_follow_the_prefix(self) -> None:
        part = Part()
        part.faces(0, BOX, cull=True, id="slab")
        scene = Scene(CAM)
        scene.place(0, part, id="p")
        assert all(re.fullmatch(r"p-slab-[pm][xyz]", el.id) for el in scene.render().elements or [])

    def test_two_unnamed_placements_collide(self) -> None:
        part = Part()
        part.sphere(0, (0, 0, 0), 0.3, id="atom")
        scene = Scene(CAM)
        scene.place(0, part)
        scene.place(0, part, at=(1, 0, 0))
        with pytest.raises(ValueError, match="own id for each placement"):
            scene.render()

    def test_unnamed_objects_need_no_prefix(self) -> None:
        part = Part()
        part.sphere(0, (0, 0, 0), 0.3)
        scene = Scene(CAM)
        scene.place(0, part)
        scene.place(0, part, at=(1, 0, 0))
        scene.render()

    def test_classes_are_added_to_the_part_s_own(self) -> None:
        part = Part()
        part.sphere(0, (0, 0, 0), 0.3, class_="atom")
        part.polygon(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        scene = Scene(CAM)
        scene.place(0, part, class_="cell cell-0")
        classes = [el.class_ for el in scene.render().elements or []]
        assert classes == [["atom", "cell", "cell-0"], ["cell", "cell-0"]]


class TestReprojection:
    @pytest.mark.parametrize(
        "camera",
        [vecview.OrthographicCamera.isometric(40.0), vecview.ObliqueCamera.cabinet(40.0)],
        ids=["isometric", "cabinet"],
    )
    def test_matches_building_with_that_camera(self, camera: Camera) -> None:
        def build(cam: Camera) -> Scene:
            part = Part()
            draw(part)
            scene = Scene(cam)
            scene.place(0, part, rotate=((1, 1, 0), 40.0), mirror=(0, 1, 0), scale=1.5, id="p")
            return scene

        assert build(CAM).with_camera(camera).to_svg_document() == build(camera).to_svg_document()


class TestChecks:
    def test_only_a_part_can_be_placed(self) -> None:
        with pytest.raises(TypeError, match="takes a Part"):
            Scene().place(0, Scene())  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"scale": 0.0}, "scale"),
            ({"scale": float("nan")}, "scale"),
            ({"rotate": ((0, 0, 0), 30.0)}, "axis"),
            ({"mirror": (0, 0, 0)}, "normal"),
            ({"id": ""}, "empty"),
        ],
    )
    def test_bad_placements_are_refused_at_the_call(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Scene().place(0, Part(), **kwargs)

    def test_every_world_space_call_knows_how_to_move(self) -> None:
        from vecview._drawing import _Drawing
        from vecview._place import _MOVES

        calls = {name for name in vars(_Drawing) if not name.startswith("_")} - {"place"}
        assert calls == set(_MOVES)

    def test_a_part_holds_no_document_level_calls(self) -> None:
        for name in ("add", "add_def", "rect2d", "text2d", "sort_by_depth", "render"):
            assert not hasattr(Part(), name)
