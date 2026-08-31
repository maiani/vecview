"""The layer stack, bounding box, and document assembly."""

from __future__ import annotations

import re

import numpy as np
import pytest
import svg

import vecview
from vecview import Camera, OrthographicCamera, Scene

SQUARE = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)


@pytest.fixture
def cam() -> Camera:
    return OrthographicCamera(35.0, 24.0, 10.0)


@pytest.fixture
def scene(cam: Camera) -> Scene:
    return Scene(cam)


def tags(document: svg.SVG) -> list[str]:
    return [type(el).__name__ for el in (document.elements or [])]


class TestLayering:
    def test_lower_layers_are_emitted_first(self, scene: Scene) -> None:
        scene.polygon(30, SQUARE, id="top")
        scene.polygon(10, SQUARE, id="bottom")
        scene.polygon(20, SQUARE, id="middle")
        ids = [el.id for el in (scene.render().elements or [])]
        assert ids == ["bottom", "middle", "top"]

    def test_ties_within_a_layer_keep_insertion_order(self, scene: Scene) -> None:
        for i in range(5):
            scene.polygon(7, SQUARE, id=f"e{i}")
        ids = [el.id for el in (scene.render().elements or [])]
        assert ids == [f"e{i}" for i in range(5)]

    def test_negative_layers_sort_below_zero(self, scene: Scene) -> None:
        scene.polygon(0, SQUARE, id="zero")
        scene.polygon(-5, SQUARE, id="under")
        ids = [el.id for el in (scene.render().elements or [])]
        assert ids == ["under", "zero"]

    def test_add_accepts_a_ready_made_element(self, scene: Scene) -> None:
        scene.polygon(5, SQUARE)
        scene.add(1, svg.Circle(cx=0, cy=0, r=1, id="raw"))
        assert (scene.render().elements or [])[0].id == "raw"


class TestBoundingBox:
    def test_a_fresh_scene_is_empty(self, scene: Scene) -> None:
        assert scene.is_empty

    def test_geometry_makes_it_non_empty(self, scene: Scene) -> None:
        scene.polygon(0, SQUARE)
        assert not scene.is_empty

    def test_bounds_cover_the_projected_geometry(self, scene: Scene, cam: Camera) -> None:
        scene.polygon(0, SQUARE)
        lo, hi = scene.bbox()
        p = cam.project(SQUARE)
        assert np.allclose(lo, p.min(axis=0))
        assert np.allclose(hi, p.max(axis=0))

    def test_bbox_returns_a_copy(self, scene: Scene) -> None:
        scene.polygon(0, SQUARE)
        lo, _ = scene.bbox()
        lo[0] = 1e9
        assert scene.bbox()[0][0] != 1e9

    def test_rect2d_is_excluded_by_default(self, scene: Scene) -> None:
        """A soft glow must not inflate the fitted viewBox."""
        scene.polygon(0, SQUARE)
        before = scene.bbox()
        scene.rect2d(0, -500, -500, 1000, 1000, fill="red")
        assert np.allclose(np.array(before), np.array(scene.bbox()))

    def test_rect2d_can_opt_in(self, scene: Scene) -> None:
        scene.polygon(0, SQUARE)
        scene.rect2d(0, -500, -500, 1000, 1000, grow=True, fill="red")
        assert scene.bbox()[0][0] == pytest.approx(-500.0)

    def test_text_widens_the_bounds(self, scene: Scene) -> None:
        """Estimated metrics, but a long label must not be clipped by the viewBox."""
        scene.polygon(0, SQUARE)
        narrow = scene.bbox()[1][0]
        scene.text2d(0, 0, 0, "a very long label indeed", size=30)
        assert scene.bbox()[1][0] > narrow


class TestRender:
    def test_viewbox_is_the_content_plus_padding(self, cam: Camera) -> None:
        scene = Scene(cam)
        scene.polygon(0, SQUARE)
        lo, hi = scene.bbox()
        document = scene.render(pad=10.0)
        assert document.viewBox is not None
        assert document.viewBox.min_x == pytest.approx(round(lo[0] - 10.0, 1))
        assert document.width == pytest.approx(round(hi[0] - lo[0] + 20.0, 1))

    def test_constructor_defaults_are_used(self, cam: Camera) -> None:
        scene = Scene(cam, pad=40.0, background="#ff0000")
        scene.polygon(0, SQUARE)
        document = scene.render()
        assert "#ff0000" in str(document)
        assert document.viewBox is not None
        assert document.viewBox.min_x == pytest.approx(round(scene.bbox()[0][0] - 40.0, 1))

    def test_call_arguments_override_the_defaults(self, cam: Camera) -> None:
        scene = Scene(cam, pad=40.0)
        scene.polygon(0, SQUARE)
        assert scene.render(pad=0.0).width < scene.render().width

    def test_background_is_drawn_behind_everything(self, cam: Camera) -> None:
        scene = Scene(cam, background="#ffffff")
        scene.polygon(0, SQUARE)
        assert tags(scene.render()) == ["Rect", "Polygon"]

    def test_no_background_element_when_transparent(self, scene: Scene) -> None:
        scene.polygon(0, SQUARE)
        assert tags(scene.render()) == ["Polygon"]

    def test_defs_come_first(self, cam: Camera) -> None:
        scene = Scene(cam, background="#ffffff")
        scene.polygon(0, SQUARE)
        scene.add_def(svg.RadialGradient(id="g", elements=[svg.Stop(offset=0)]))
        assert tags(scene.render()) == ["Defs", "Rect", "Polygon"]

    def test_an_empty_scene_reports_why_it_cannot_render(self, scene: Scene) -> None:
        with pytest.raises(ValueError, match="empty scene"):
            scene.render()

    def test_render_does_not_consume_the_scene(self, scene: Scene) -> None:
        scene.polygon(0, SQUARE)
        assert str(scene.render()) == str(scene.render())


class TestPrimitives:
    def test_polyline_is_unfilled_by_default(self, scene: Scene) -> None:
        scene.polyline(0, SQUARE)
        assert (scene.render().elements or [])[0].fill == "none"

    def test_polyline_fill_can_be_overridden(self, scene: Scene) -> None:
        scene.polyline(0, SQUARE, fill="red")
        assert (scene.render().elements or [])[0].fill == "red"

    def test_points_are_emitted_as_rounded_xy_pairs(self, scene: Scene) -> None:
        """`x,y` pairs, not a flat coordinate run, and short enough to stay readable."""
        scene.polygon(0, [[1 / 3, 1 / 7, 1 / 11]] * 3)
        points = (scene.render().elements or [])[0].points
        assert all(isinstance(p, svg.Point) for p in points)
        for point in points:
            for coord in (point.x, point.y):
                assert len(str(coord).split(".")[-1]) <= 2

    def test_faces_draws_every_face_given(self, scene: Scene) -> None:
        faces = vecview.box_faces((0, 0, 0), (1, 1, 1))
        scene.faces(0, faces, fill="grey")
        assert len(scene.items) == 6

    def test_faces_does_not_cull_on_its_own(self, scene: Scene, cam: Camera) -> None:
        """Culling is explicit via Camera.visible, so styling stays predictable."""
        faces = vecview.box_faces((0, 0, 0), (1, 1, 1))
        scene.faces(0, cam.visible(faces), fill="grey")
        assert len(scene.items) == 3

    def test_text_offsets_are_in_screen_units(self, scene: Scene, cam: Camera) -> None:
        scene.text(0, (0, 0, 0), "label", dx=5.0, dy=-3.0, size=10)
        text = (scene.render().elements or [])[0]
        x0, y0 = cam.at((0, 0, 0))
        assert text.x == pytest.approx(round(x0 + 5.0, 2))
        assert text.y == pytest.approx(round(y0 - 3.0, 2))

    def test_text_has_a_default_font_and_fill(self, scene: Scene) -> None:
        scene.text2d(0, 0, 0, "label")
        text = (scene.render().elements or [])[0]
        assert text.font_family == vecview.scene.DEFAULT_FONT
        assert text.fill == vecview.scene.DEFAULT_TEXT_FILL

    def test_text_styling_can_be_overridden(self, scene: Scene) -> None:
        scene.text2d(0, 0, 0, "label", fill="#abcdef", font_family="Times")
        text = (scene.render().elements or [])[0]
        assert (text.fill, text.font_family) == ("#abcdef", "Times")

    @pytest.mark.parametrize("anchor", ["start", "middle", "end"])
    def test_bounds_follow_the_text_anchor(self, cam: Camera, anchor: str) -> None:
        scene = Scene(cam)
        scene.text2d(0, 0.0, 0.0, "wide label", size=20, text_anchor=anchor)
        lo, hi = scene.bbox()
        assert lo[0] <= 0.0 <= hi[0]
        if anchor == "start":
            assert lo[0] == pytest.approx(0.0)
        if anchor == "end":
            assert hi[0] == pytest.approx(0.0)


class TestDocumentProtocol:
    def test_to_svg_document_needs_no_arguments(self, scene: Scene) -> None:
        """A consumer calls this with no arguments, so the defaults must suffice."""
        scene.polygon(0, SQUARE)
        assert scene.to_svg_document().startswith("<svg")

    def test_document_carries_a_viewbox_and_size(self, cam: Camera) -> None:
        scene = Scene(cam, pad=5.0)
        scene.polygon(0, SQUARE)
        document = scene.to_svg_document()
        assert re.search(r'viewBox="[-\d. ]+"', document)
        assert re.search(r'width="[\d.]+"', document)

    def test_save_writes_the_document(self, scene: Scene, tmp_path) -> None:
        scene.polygon(0, SQUARE)
        out = scene.save(tmp_path / "s.svg")
        assert out.read_text(encoding="utf-8") == scene.to_svg_document()

    def test_repr_summarizes_the_scene(self, scene: Scene) -> None:
        scene.polygon(0, SQUARE)
        scene.add_def(svg.RadialGradient(id="g", elements=[]))
        assert repr(scene) == f"Scene({scene.cam!r}, 1 elements, 1 defs)"


class TestFaceIds:
    def test_ids_are_suffixed_per_face_not_repeated(self, scene: Scene, cam: Camera) -> None:
        """Duplicate ids are invalid SVG and break selection downstream."""
        walls = cam.visible(vecview.box_faces((0, 0, 0), (1, 1, 1)))
        scene.faces(0, walls, fill="grey", id="slab")
        ids = [el.id for el in (scene.render().elements or [])]
        assert sorted(ids) == ["slab-px", "slab-py", "slab-pz"]
        assert len(set(ids)) == len(ids)

    def test_generated_ids_are_valid_xml_names(self, scene: Scene) -> None:
        """`+` is not a legal XML name character, so the sign is spelled out."""
        scene.faces(0, vecview.box_faces((0, 0, 0), (1, 1, 1)), fill="grey", id="box")
        ids = [el.id for el in (scene.render().elements or [])]
        assert all(re.fullmatch(r"[A-Za-z_][\w.-]*", i) for i in ids), ids
        assert sorted(ids) == [
            "box-mx",
            "box-my",
            "box-mz",
            "box-px",
            "box-py",
            "box-pz",
        ]

    def test_no_id_attribute_when_none_given(self, scene: Scene) -> None:
        scene.faces(0, vecview.box_faces((0, 0, 0), (1, 1, 1)), fill="grey")
        assert all(el.id is None for el in (scene.render().elements or []))


class TestReprojection:
    """`with_camera` replays recorded world-space calls against a new camera."""

    @staticmethod
    def build(camera: Camera) -> Scene:
        scene = Scene(camera, pad=8.0, background="#ffffff")
        scene.faces(10, vecview.box_faces((0, 0, -0.45), (11, 9, 0.9)), cull=True, id="slab")
        scene.polyline(20, [[0, 0, 4], [0, 0, -3]], stroke="red")
        scene.text(30, (0, 0, 4.5), "beam", size=20, text_anchor="middle")
        scene.plane(15, (-3, 2, 0.01), (6, 0, 0), (0, -4, 0), id="plot")
        scene.add_def(svg.RadialGradient(id="glow", elements=[svg.Stop(offset=0)]))
        return scene

    @pytest.mark.parametrize(
        "camera",
        [
            vecview.OrthographicCamera.isometric(62.0),
            vecview.OrthographicCamera.dimetric(62.0),
            vecview.OrthographicCamera(12.0, 70.0, 62.0),
            vecview.ObliqueCamera.cavalier(62.0),
            vecview.ObliqueCamera.cabinet(62.0),
        ],
        ids=["isometric", "dimetric", "trimetric", "cavalier", "cabinet"],
    )
    def test_matches_building_from_scratch(self, camera: Camera) -> None:
        base = self.build(vecview.OrthographicCamera(35.0, 24.0, 62.0))
        assert base.with_camera(camera).to_svg_document() == self.build(camera).to_svg_document()

    def test_same_camera_round_trips_byte_identically(self) -> None:
        cam = vecview.OrthographicCamera(35.0, 24.0, 62.0)
        base = self.build(cam)
        assert base.with_camera(cam).to_svg_document() == base.to_svg_document()

    def test_leaves_the_original_untouched(self) -> None:
        base = self.build(vecview.OrthographicCamera(35.0, 24.0, 62.0))
        before = base.to_svg_document()
        base.with_camera(vecview.ObliqueCamera.cabinet(62.0))
        assert base.to_svg_document() == before

    def test_carries_pad_and_background(self) -> None:
        cam = vecview.OrthographicCamera(35.0, 24.0, 62.0)
        scene = Scene(cam, pad=40.0, background="#ff0000")
        scene.polygon(0, SQUARE)
        clone = scene.with_camera(vecview.OrthographicCamera.isometric(62.0))
        assert (clone.pad, clone.background) == (40.0, "#ff0000")

    def test_does_not_duplicate_delegating_calls(self, cam: Camera) -> None:
        """`faces` fans out to `polygon`; recording both would double every face."""
        scene = Scene(cam)
        scene.faces(0, vecview.box_faces((0, 0, 0), (1, 1, 1)))
        scene.text(0, (0, 0, 0), "label")
        clone = scene.with_camera(cam)
        assert len(clone.items) == len(scene.items)

    def test_bounding_box_is_recomputed(self) -> None:
        base = self.build(vecview.OrthographicCamera(35.0, 24.0, 62.0))
        flat = base.with_camera(vecview.OrthographicCamera(35.0, 85.0, 62.0))
        assert not np.allclose(np.array(base.bbox()), np.array(flat.bbox()))

    def test_screen_space_calls_stay_at_their_screen_position(self, cam: Camera) -> None:
        """Documented behaviour: screen space means screen space."""
        scene = Scene(cam)
        scene.polygon(0, SQUARE)
        scene.rect2d(0, 10.0, 20.0, 5.0, 5.0, grow=True, fill="red")
        clone = scene.with_camera(vecview.ObliqueCamera.cabinet(cam.scale))
        rect = [el for el in (clone.render().elements or []) if isinstance(el, svg.Rect)][-1]
        assert (rect.x, rect.y) == (10.0, 20.0)


class TestCulling:
    def test_cull_selects_the_walls_for_the_current_camera(self, cam: Camera) -> None:
        scene = Scene(cam)
        scene.faces(0, vecview.box_faces((0, 0, 0), (1, 1, 1)), cull=True, id="box")
        ids = [el.id for el in (scene.render().elements or [])]
        assert sorted(ids) == ["box-px", "box-py", "box-pz"]

    def test_cull_is_redone_on_reprojection(self, cam: Camera) -> None:
        """A cabinet camera sees -y where this one sees +y."""
        scene = Scene(cam)
        scene.faces(0, vecview.box_faces((0, 0, 0), (1, 1, 1)), cull=True, id="box")
        clone = scene.with_camera(vecview.ObliqueCamera.cabinet(cam.scale))
        ids = sorted(el.id for el in (clone.render().elements or []))
        assert ids == ["box-my", "box-px", "box-pz"]

    def test_caller_side_culling_is_frozen_by_comparison(self, cam: Camera) -> None:
        """Why `cull=True` exists: `Camera.visible` here bakes in this camera."""
        scene = Scene(cam)
        scene.faces(0, cam.visible(vecview.box_faces((0, 0, 0), (1, 1, 1))), id="box")
        clone = scene.with_camera(vecview.ObliqueCamera.cabinet(cam.scale))
        ids = sorted(el.id for el in (clone.render().elements or []))
        assert ids == ["box-px", "box-py", "box-pz"], "the original camera's walls"

    def test_cull_off_by_default(self, cam: Camera) -> None:
        scene = Scene(cam)
        scene.faces(0, vecview.box_faces((0, 0, 0), (1, 1, 1)))
        assert len(scene.items) == 6
