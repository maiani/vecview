"""The layer stack, bounding box, and document assembly."""

from __future__ import annotations

import re

import numpy as np
import pytest
import svg

import vecview
from vecview import Camera, OrthographicCamera, Scene
from vecview.scene import _Canvas

SQUARE = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)


@pytest.fixture
def cam() -> Camera:
    return OrthographicCamera(35.0, 24.0, 10.0)


@pytest.fixture
def scene(cam: Camera) -> Scene:
    return Scene(cam)


def drawn(scene: Scene) -> list[svg.Element]:
    """The rendered elements, without the definitions."""
    return [el for el in (scene.render().elements or []) if not isinstance(el, svg.Defs)]


def definitions(scene: Scene) -> list[svg.Element]:
    """The rendered ``<defs>`` children."""
    first = (scene.render().elements or [None])[0]
    return list(first.elements or []) if isinstance(first, svg.Defs) else []


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
        assert len(drawn(scene)) == 6

    def test_faces_does_not_cull_on_its_own(self, scene: Scene, cam: Camera) -> None:
        """Culling is explicit via Camera.visible, so styling stays predictable."""
        faces = vecview.box_faces((0, 0, 0), (1, 1, 1))
        scene.faces(0, cam.visible(faces), fill="grey")
        assert len(drawn(scene)) == 3

    def test_text_offsets_are_in_screen_units(self, scene: Scene, cam: Camera) -> None:
        scene.text(0, (0, 0, 0), "label", dx=5.0, dy=-3.0, size=10)
        text = (scene.render().elements or [])[0]
        x0, y0 = cam.at((0, 0, 0))
        assert text.x == pytest.approx(round(x0 + 5.0, 2))
        assert text.y == pytest.approx(round(y0 - 3.0, 2))

    def test_text_accepts_tspan_runs_for_subscripts(self, scene: Scene) -> None:
        runs = [svg.TSpan(text="k"), svg.TSpan(text="x", baseline_shift="sub")]
        scene.text(0, (0, 0, 0), runs, id="kx")
        doc = scene.to_svg_document()
        assert '<tspan>k</tspan><tspan baseline-shift="sub">x</tspan>' in doc

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
        assert repr(scene) == f"Scene({scene.camera!r}, cameras=[], 2 calls)"


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
        fin = vecview.prism_faces([(2, -1), (4, -0.5), (4, 0.5), (2, 1)], 0.0, 0.4)
        scene.silhouette(25, fin, fill="#b98a40", id="fin-walls")
        scene.slot(40, (4, 0, 0.4), 30.0, 12.0, id="fin-label", align="west", dx=4.0)
        scene.arrow(
            35, (0, 0, 0), (0, 0, 1), 2.0, normal="camera", shaft_w=0.1, head_w=0.3, head_len=0.3
        )
        scene.gaussian(12, (0, 0, 0.01), (1, 0, 0), (0, 1, 0), 1.5, 0.8, id="spot", color="red")
        gate = vecview.annulus_sector((0, 0), 2.6, 3.0, 20, 160, n=12)
        scene.prism_walls(30, gate, 0.0, 0.26, fill="#4a5059", id="gate-walls")
        scene.sort_by_depth(40)
        scene.sphere(40, (1, 1, 1), 0.5, fill="#c33", highlight="#fcc")
        scene.cylinder(40, (0, 0, 0), (1, 1, 1), 0.1, fill="#888", highlight="#eee", id="bond")
        scene.cone(40, (2, 0, 0), (2, 0, 1), 0.3, fill="#a5c")
        scene.cylinder(40, (-2, 0, 0), (-2, 3, 0), 0.4, slices=5, stroke="#000", id="core")
        scene.arrow3d(40, (-1, 0, 0), (1, -1, 2), 1.2, shaft_r=0.05, head_r=0.15, head_len=0.3)
        scene.tube(40, vecview.helix((3, 0, 0), (0, 0, 1), 0.5, 0.4, 2), 0.05, stroke="#000")
        cell = vecview.box_faces((0, 0, 3), (1, 1, 1))
        scene.edges(41, cell, back={"stroke_dasharray": "3 2"}, back_layer=39, stroke="#000")
        ring = vecview.circle_shape((1, 1, 1), 0.5, (0, 0, 1))
        scene.sphere_curve(42, (1, 1, 1), ring, closed=True, back={}, stroke="#000")
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
        assert len(drawn(scene)) == 7

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
        assert len(drawn(scene)) == 6


class TestSlot:
    """A reserved, screen-aligned group pinned to a projected world point."""

    @staticmethod
    def group(scene: Scene, id: str) -> svg.G:
        return next(el for el in (scene.render().elements or []) if el.id == id)

    def test_group_is_empty_and_translated_to_the_anchor(self, scene: Scene, cam: Camera) -> None:
        scene.slot(5, (1, 2, 3), 20.0, 10.0, id="label", dx=3.0, dy=-4.0)
        group = self.group(scene, "label")
        x, y = cam.at((1, 2, 3))
        assert not group.elements
        assert group.transform == [svg.Translate(round(x + 3.0, 2), round(y - 4.0, 2))]

    def test_alignment_is_recorded_for_the_consumer(self, scene: Scene) -> None:
        scene.slot(5, (0, 0, 0), 20.0, 10.0, id="label", align="southwest")
        assert self.group(scene, "label").data == {"align": "southwest"}

    @pytest.mark.parametrize(
        ("align", "corner"),
        [
            ("center", (-10.0, -5.0)),
            ("west", (0.0, -5.0)),
            ("east", (-20.0, -5.0)),
            ("north", (-10.0, 0.0)),
            ("southeast", (-20.0, -10.0)),
        ],
    )
    def test_reserved_box_is_aligned_on_the_anchor(
        self, scene: Scene, cam: Camera, align: str, corner: tuple[float, float]
    ) -> None:
        scene.slot(5, (0, 0, 0), 20.0, 10.0, id="label", align=align)  # type: ignore[arg-type]
        lo, hi = scene.bbox()
        x, y = cam.at((0, 0, 0))
        assert np.allclose(lo, [x + corner[0], y + corner[1]])
        assert np.allclose(hi - lo, [20.0, 10.0])

    def test_rejects_an_unknown_alignment(self, scene: Scene) -> None:
        with pytest.raises(ValueError, match="align"):
            scene.slot(5, (0, 0, 0), 1.0, 1.0, id="x", align="left")  # type: ignore[arg-type]

    def test_follows_the_geometry_under_reprojection(self, scene: Scene) -> None:
        scene.slot(5, (2, 0, 1), 20.0, 10.0, id="label")
        other = vecview.ObliqueCamera.cabinet(10.0)
        x, y = other.at((2, 0, 1))
        moved = self.group(scene.with_camera(other), "label")
        assert moved.transform == [svg.Translate(round(x, 2), round(y, 2))]


class TestSilhouette:
    def test_box_outline_is_its_six_outer_vertices(self, scene: Scene) -> None:
        scene.silhouette(0, vecview.box_faces((0, 0, 0), (2, 2, 2)), fill="grey", id="walls")
        (polygon,) = scene.render().elements or []
        assert isinstance(polygon, svg.Polygon)
        assert len(polygon.points or []) == 6, "a box seen from a generic angle is a hexagon"

    def test_outline_contains_every_projected_vertex(self, scene: Scene, cam: Camera) -> None:
        fin = vecview.prism_faces([(0, -1), (3, -0.3), (3.4, 0), (3, 0.3), (0, 1)], 0.0, 0.5)
        scene.silhouette(0, fin)
        lo, hi = scene.bbox()
        p = cam.project(np.vstack([f.points for f in fin]))
        assert np.allclose(lo, p.min(axis=0)) and np.allclose(hi, p.max(axis=0))

    def test_accepts_bare_vertices(self, scene: Scene) -> None:
        scene.silhouette(0, SQUARE)
        assert len(drawn(scene)) == 1

    def test_is_recomputed_on_reprojection(self, scene: Scene) -> None:
        scene.silhouette(0, vecview.box_faces((0, 0, 0), (2, 2, 2)))
        top_down = scene.with_camera(vecview.OrthographicCamera(0.0, 90.0, 10.0))
        (polygon,) = top_down.render().elements or []
        assert isinstance(polygon, svg.Polygon)
        assert len(polygon.points or []) == 4, "seen from straight above, a box is a square"


class TestArrow:
    def test_explicit_normal_matches_arrow_shape(self, scene: Scene, cam: Camera) -> None:
        args = ((0, 0, 0), (1, 0, 0), 2.0)
        scene.arrow(0, *args, normal=(0, 0, 1), shaft_w=0.1, head_w=0.3, head_len=0.4)
        expected = Scene(cam)
        expected.polygon(0, vecview.arrow_shape(*args, (0, 0, 1), 0.1, 0.3, 0.4))
        assert scene.to_svg_document() == expected.to_svg_document()

    def test_camera_normal_shows_the_full_head_width(self, scene: Scene, cam: Camera) -> None:
        """Facing the camera, the head's screen width equals its world width."""
        scene.arrow(
            0, (0, 0, 0), (0, 0, 1), 2.0, normal="camera", shaft_w=0.1, head_w=0.5, head_len=0.4
        )
        shape = vecview.arrow_shape(
            (0, 0, 0), (0, 0, 1), 2.0, _Canvas(cam)._facing_normal((0, 0, 1)), 0.1, 0.5, 0.4
        )
        head = cam.project(shape[[2, 4]])
        assert np.linalg.norm(head[0] - head[1]) == pytest.approx(0.5 * cam.scale)

    def test_camera_normal_contains_the_arrow(self, cam: Camera) -> None:
        n = _Canvas(cam)._facing_normal((0, 0, 1))
        assert np.dot(n, [0, 0, 1]) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "camera",
        [vecview.ObliqueCamera.cabinet(10.0), vecview.ObliqueCamera.cavalier(10.0)],
        ids=["cabinet", "cavalier"],
    )
    @pytest.mark.parametrize("direction", [(0, 0, 1), (1, 0, 0), (1, 2, 0.5)])
    def test_camera_normal_is_the_widest_face(
        self, camera: vecview.ParallelCamera, direction: tuple[float, float, float]
    ) -> None:
        """For an oblique camera no face is undistorted; the widest is chosen."""
        d = vecview.unit(direction)
        n = _Canvas(camera)._facing_normal(d)
        chosen = np.linalg.norm(camera.direction(np.cross(n, d)))
        e1 = vecview.unit(np.cross(d, [0.3, 0.5, 0.7]))
        e2 = np.cross(d, e1)
        widths = [
            np.linalg.norm(camera.direction(np.cos(t) * e1 + np.sin(t) * e2))
            for t in np.linspace(0, np.pi, 721)
        ]
        assert chosen == pytest.approx(max(widths), rel=1e-4)

    def test_rejects_an_arrow_along_the_projection_ray(self, scene: Scene, cam: Camera) -> None:
        assert isinstance(cam, vecview.ParallelCamera)
        scene.arrow(
            0, (0, 0, 0), cam.view, 1.0, normal="camera", shaft_w=0.1, head_w=0.3, head_len=0.3
        )
        with pytest.raises(ValueError, match="projection ray"):
            scene.render()

    def test_records_one_call(self, scene: Scene) -> None:
        scene.arrow(
            0, (0, 0, 0), (0, 0, 1), 1.0, normal="camera", shaft_w=0.1, head_w=0.3, head_len=0.3
        )
        assert len(drawn(scene.with_camera(scene.camera))) == 1


class TestGaussian:
    @pytest.fixture
    def spot(self, scene: Scene) -> Scene:
        scene.gaussian(0, (1, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0, id="spot", color="red")
        return scene

    def test_one_polygon_filled_by_its_own_gradient(self, spot: Scene) -> None:
        document = spot.render()
        defs, polygon = document.elements or []
        assert isinstance(defs, svg.Defs) and isinstance(polygon, svg.Polygon)
        assert polygon.fill == "url(#spot-profile)"
        assert [el.id for el in defs.elements or []] == ["spot-profile"]

    def test_gradient_maps_onto_the_projected_axes(self, spot: Scene, cam: Camera) -> None:
        (gradient,) = definitions(spot)
        assert isinstance(gradient, svg.RadialGradient)
        (matrix,) = gradient.gradientTransform or []
        assert isinstance(matrix, svg.Matrix)
        a, b, c, d, e, f = (matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f)
        # gradient (1, 0) is one half-width along u from the centre
        expected = cam.at((1 + 2.0, 0, 0))
        assert (a + e, b + f) == pytest.approx(expected, abs=1e-3)
        expected = cam.at((1, 1.0, 0))
        assert (c + e, d + f) == pytest.approx(expected, abs=1e-3)

    def test_profile_is_gaussian_and_reaches_zero_at_the_rim(self, spot: Scene) -> None:
        (gradient,) = definitions(spot)
        assert isinstance(gradient, svg.RadialGradient)
        stops = [s for s in gradient.elements or [] if isinstance(s, svg.Stop)]
        offsets = np.array([float(s.offset or 0) for s in stops])
        opacity = np.array([float(s.stop_opacity or 0) for s in stops])
        floor = np.exp(-4.0)
        assert opacity[0] == 1.0 and opacity[-1] == 0.0
        assert np.allclose(
            opacity, (np.exp(-((2.0 * offsets) ** 2)) - floor) / (1 - floor), atol=1e-4
        )

    def test_needs_a_parallel_camera(self) -> None:
        class Pinhole(vecview.Camera):
            def project(self, pts):  # type: ignore[no-untyped-def]
                return np.zeros((len(pts), 2))

            def depth(self, pts):  # type: ignore[no-untyped-def]
                return np.zeros(len(pts))

            def visible(self, faces, *, tol=0.0):  # type: ignore[no-untyped-def]
                return list(faces)

        scene = Scene(Pinhole(1.0))
        scene.gaussian(0, (0, 0, 0), (1, 0, 0), (0, 1, 0), 1, 1, id="s", color="red")
        with pytest.raises(TypeError, match="parallel projection"):
            scene.render()


class TestPrismWalls:
    from vecview.scene import _runs as runs

    @pytest.mark.parametrize(
        ("visible", "expected"),
        [
            ([True, True, False, True], [(3, 3)]),  # wraps past the end
            ([False, True, True, False, True, False], [(1, 2), (4, 1)]),
            ([True, True, True], [(0, 3)]),
            ([False, False], []),
            ([True, False, False, False], [(0, 1)]),
        ],
    )
    def test_runs_are_maximal_and_cyclic(
        self, visible: list[bool], expected: list[tuple[int, int]]
    ) -> None:
        assert TestPrismWalls.runs(visible) == expected

    @staticmethod
    def subpaths(scene: Scene) -> list[list[tuple[float, float]]]:
        (path,) = scene.render().elements or []
        assert isinstance(path, svg.Path)
        paths: list[list[tuple[float, float]]] = []
        for command in path.d or []:
            if isinstance(command, svg.M):
                paths.append([(float(command.x), float(command.y))])
            elif isinstance(command, svg.L):
                paths[-1].append((float(command.x), float(command.y)))
        return paths

    def test_one_strip_per_run_of_facing_walls(self, scene: Scene, cam: Camera) -> None:
        gate = vecview.annulus_sector((0, 0), 2.6, 3.0, 20, 160, n=12)
        scene.prism_walls(0, gate, 0.0, 0.26, id="walls")
        walls = vecview.prism_faces(gate, 0.0, 0.26)[2:]
        facing = [any(w is v for v in cam.visible(walls)) for w in walls]
        expected = TestPrismWalls.runs(facing)
        strips = self.subpaths(scene)
        assert len(strips) == len(expected)
        # A run of k walls is k + 1 base points forward and k + 1 top points back.
        assert sorted(len(s) for s in strips) == sorted(2 * (k + 1) for _, k in expected)

    def test_strip_traces_base_then_top(self, scene: Scene, cam: Camera) -> None:
        square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        scene.prism_walls(0, square, 0.0, 0.5)
        (strip,) = self.subpaths(scene)
        walls = vecview.prism_faces(square, 0.0, 0.5)[2:]
        facing = cam.visible(walls)
        assert len(facing) == 2, "a generic view of a box shows two walls"
        start = next(i for i, w in enumerate(walls) if w is facing[0])
        first, second = (walls[(start + k) % 4] for k in range(2))
        if second is not facing[1]:
            first, second = second, walls[(start + 1) % 4]
        expected = cam.project(
            [
                first.points[0],
                second.points[0],
                second.points[1],
                second.points[2],
                second.points[3],
                first.points[3],
            ]
        )
        assert np.allclose(strip, expected, atol=0.01)

    def test_walls_are_reculled_on_reprojection(self, scene: Scene) -> None:
        scene.prism_walls(0, [(-1, -1), (1, -1), (1, 1), (-1, 1)], 0.0, 0.5)
        clone = scene.with_camera(vecview.ObliqueCamera.cabinet(10.0))
        assert self.subpaths(clone) != self.subpaths(scene)


class TestObjectsAndCamera:
    """A scene records objects; a camera is only needed to render it."""

    def build(self) -> Scene:
        scene = Scene(pad=4.0)
        scene.sort_by_depth(1)
        scene.sphere(1, (0, 0, 0), 1.0, fill="#c33")
        scene.cylinder(1, (0, 0, 0), (2, 0, 0), 0.2, fill="#888")
        return scene

    def test_drawing_needs_no_camera(self) -> None:
        scene = self.build()
        assert scene.camera is None and not scene.is_empty

    def test_rendering_without_a_camera_says_so(self) -> None:
        with pytest.raises(ValueError, match="no camera"):
            self.build().render()

    def test_a_camera_can_be_passed_or_set(self, cam: Camera) -> None:
        scene = self.build()
        passed = str(scene.render(cam))
        scene.camera = cam
        assert scene.to_svg_document() == passed

    def test_rendering_is_pure(self, cam: Camera) -> None:
        scene = self.build()
        other = vecview.ObliqueCamera.cabinet(10.0)
        first = str(scene.render(cam))
        assert str(scene.render(other)) != first
        assert str(scene.render(cam)) == first

    def test_save_takes_a_camera(self, cam: Camera, tmp_path) -> None:
        out = self.build().save(tmp_path / "s.svg", cam)
        assert out.read_text(encoding="utf-8") == str(self.build().render(cam))

    def test_bbox_follows_the_camera(self, cam: Camera) -> None:
        scene = self.build()
        top = vecview.OrthographicCamera(0.0, 90.0, 10.0)
        assert not np.allclose(np.array(scene.bbox(cam)), np.array(scene.bbox(top)))

    def test_jupyter_needs_a_camera(self, cam: Camera) -> None:
        scene = self.build()
        assert scene._repr_svg_() is None
        scene.camera = cam
        assert scene._repr_svg_() == scene.to_svg_document()

    def test_with_camera_copies_the_record(self, cam: Camera) -> None:
        scene = self.build()
        view = scene.with_camera(cam)
        scene.sphere(1, (5, 0, 0), 1.0)
        assert len(drawn(view)) == 2 and len(drawn(scene.with_camera(cam))) == 3


class TestEagerChecks:
    """Mistakes that do not depend on the camera are reported where they are made."""

    def test_a_tube_needs_two_points(self) -> None:
        with pytest.raises(ValueError, match="two points"):
            Scene().tube(0, [(0, 0, 0)], 0.1)

    def test_an_arrow_normal_is_a_vector_or_camera(self) -> None:
        with pytest.raises(ValueError, match="camera"):
            Scene().arrow(
                0, (0, 0, 0), (1, 0, 0), 1, normal="viewer", shaft_w=0.1, head_w=0.2, head_len=0.2
            )  # type: ignore[arg-type]

    def test_a_prism_footprint_must_be_simple(self) -> None:
        bowtie = [(0, 0), (1, 1), (1, 0), (0, 1)]
        with pytest.raises(ValueError, match="footprint"):
            Scene().prism_walls(0, bowtie, 0.0, 1.0)

    def test_a_solid_needs_an_axis(self) -> None:
        with pytest.raises(ValueError, match="coincide"):
            Scene().cone(0, (1, 1, 1), (1, 1, 1), 0.3)


class TestNamedCameras:
    """A scene holds named cameras, one active, as a 3D application does."""

    @pytest.fixture
    def scene(self) -> Scene:
        scene = Scene(
            "main",
            cameras={
                "main": OrthographicCamera(35.0, 24.0, 10.0),
                "cabinet": vecview.ObliqueCamera.cabinet(10.0),
            },
        )
        scene.sphere(0, (0, 0, 0), 1.0)
        scene.faces(0, vecview.box_faces((2, 0, 0), (1, 1, 1)), cull=True)
        return scene

    def test_render_takes_a_name(self, scene: Scene) -> None:
        by_name = str(scene.render("cabinet"))
        assert by_name == str(scene.render(scene.cameras["cabinet"]))
        assert by_name != str(scene.render())

    def test_the_active_camera_is_chosen_by_name(self, scene: Scene) -> None:
        assert scene.camera is scene.cameras["main"]
        scene.camera = "cabinet"
        assert scene.to_svg_document() == str(scene.render("cabinet"))

    def test_an_active_name_follows_its_entry(self, scene: Scene) -> None:
        replacement = OrthographicCamera.isometric(10.0)
        scene.cameras["main"] = replacement
        assert scene.camera is replacement

    def test_an_unknown_name_is_refused(self, scene: Scene) -> None:
        with pytest.raises(KeyError, match="no camera named 'top'"):
            scene.camera = "top"
        with pytest.raises(KeyError, match="'cabinet', 'main'"):
            scene.render("top")

    def test_with_camera_keeps_the_named_cameras(self, scene: Scene) -> None:
        view = scene.with_camera("cabinet")
        assert set(view.cameras) == {"main", "cabinet"}
        assert view.to_svg_document() == str(scene.render("cabinet"))
        view.cameras["top"] = OrthographicCamera(0.0, 90.0, 10.0)
        assert "top" not in scene.cameras
