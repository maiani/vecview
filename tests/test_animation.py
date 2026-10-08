import math
import xml.etree.ElementTree as ET

import pytest
import svg

import vecview
from vecview.animation import Animation, Track, hold, linear, rotate, scale, smoothstep

SVG = "{http://www.w3.org/2000/svg}"
CAMERA = vecview.OrthographicCamera.isometric(20)


def make_scene(x: float = 0.0) -> vecview.Scene:
    scene = vecview.Scene(CAMERA, background="#f00", pad=50)
    scene.sphere(0, (x, 0, 0), 0.5, id="ball", class_="particle", fill="#369")
    return scene


def document(frame, **options) -> ET.Element:
    options = {"duration": 1, "fps": 1, "view_box": (0, 0, 10, 10), **options}
    return ET.fromstring(Animation(frame, **options).to_svg_document())


def groups(root: ET.Element) -> list[ET.Element]:
    return root.findall(f".//{SVG}g[@data-vecview-frame]")


def timings(root: ET.Element) -> list[tuple[str, str]]:
    """Each drawn ``<use>``'s ``href`` values and key times, in paint order."""
    return [
        (use.find(f"{SVG}animate").get("values"), use.find(f"{SVG}animate").get("keyTimes"))
        for use in root.findall(f"{SVG}use")
    ]


def test_animation_evaluates_in_closed_time_interval_without_calling_constructor_callback():
    calls = []
    animation = Animation(
        lambda t: calls.append(t) or make_scene(t), duration=2, view_box=(0, 0, 10, 10)
    )
    assert calls == []
    assert animation.frame(0).camera is not None
    assert animation.frame(2).camera is not None
    assert calls == [0.0, 2.0]
    for t in (-0.1, 2.1, math.inf, math.nan):
        with pytest.raises(ValueError):
            animation.frame(t)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"duration": 0, "view_box": (0, 0, 1, 1)}, ValueError),
        ({"duration": 1, "view_box": (0, 0, 0, 1)}, ValueError),
        ({"duration": True, "view_box": (0, 0, 1, 1)}, TypeError),
        ({"duration": 1, "view_box": (0, 0, 1, 1), "repeat": True}, TypeError),
        ({"duration": 1, "view_box": (0, 0, 1, 1), "repeat": 0}, ValueError),
        ({"duration": 1, "view_box": (0, 0, 1, 1), "fps": -1}, ValueError),
    ],
)
def test_animation_validates_options(kwargs, error):
    with pytest.raises(error):
        Animation(lambda _: make_scene(), **kwargs)


def test_non_scene_callback_result_is_rejected():
    with pytest.raises(TypeError, match="return Scene"):
        Animation(lambda _: None, duration=1, view_box=(0, 0, 1, 1)).frame(0)


def test_frames_need_an_active_camera():
    with pytest.raises(ValueError, match="active camera"):
        Animation(lambda _: vecview.Scene(), duration=1, view_box=(0, 0, 1, 1)).render()


def test_samples_exact_interval_times_and_holds_the_terminal_frame():
    sampled = []
    root = document(
        lambda t: sampled.append(t) or make_scene(t),
        fps=2.5,
        repeat=2,
        view_box=(-3, -4, 25, 30),
    )
    assert sampled == [0.0, 1 / 3, 2 / 3, 1.0]
    assert (root.get("width"), root.get("height")) == ("25", "30")
    assert root.get("viewBox") == "-3 -4 25 30"
    assert root.find(f"{SVG}rect") is None  # a frame's own background does not leak in
    use = root.find(f"{SVG}use")
    assert use.get("href") == "#frame0-ball"  # what a viewer without animation shows
    timing = use.find(f"{SVG}animate")
    assert (timing.get("dur"), timing.get("repeatCount")) == ("1s", "2")
    assert (timing.get("attributeName"), timing.get("calcMode")) == ("href", "discrete")
    assert timing.get("fill") == "freeze"
    # The value at key time 1 is the terminal sample, held once playback ends.
    assert timings(root) == [
        (
            "#frame0-ball;#frame1-ball;#frame2-ball;#frame3-ball",
            "0;0.333333333333;0.666666666667;1",
        )
    ]


def test_indefinite_playback_loops_the_cycle_without_a_terminal_sample():
    sampled = []
    root = document(lambda t: sampled.append(t) or make_scene(t), fps=2, repeat=None)
    assert sampled == [0, 0.5]
    assert root.find(f".//{SVG}animate").get("repeatCount") == "indefinite"
    assert timings(root) == [("#frame0-ball;#frame1-ball;#frame1-ball", "0;0.5;1")]


def test_each_sample_is_shown_for_its_own_interval():
    root = document(lambda t: make_scene(t), fps=4, repeat=None)
    values = ";".join(f"#frame{i}-ball" for i in (0, 1, 2, 3, 3))
    assert timings(root) == [(values, "0;0.25;0.5;0.75;1")]


def test_what_every_frame_draws_alike_is_written_once_with_its_own_id():
    def frame(t):
        scene = vecview.Scene(CAMERA)
        scene.sphere(0, (0, 0, -1), 0.5, id="floor", fill="#ccc")
        scene.sphere(1, (t, 0, 0), 0.5, id="ball", fill="#369")
        scene.text2d(2, 0, 0, "caption", id="caption")
        return scene

    root = document(frame)
    assert [node.get("id") for node in root.find(f"{SVG}defs")] == ["frame0-ball", "frame1-ball"]
    drawn = [(node.tag.removeprefix(SVG), node.get("id")) for node in root][1:]
    assert drawn == [("circle", "floor"), ("use", None), ("text", "caption")]


def test_what_does_not_move_is_written_once_even_between_what_does():
    def frame(t):
        scene = vecview.Scene(CAMERA)
        scene.sphere(0, (t, 0, 0), 0.5, id="low")
        scene.text2d(1, 0, 0, "between", id="between")
        scene.sphere(2, (0, t, 0), 0.5, id="high")
        return scene

    root = document(frame, fps=2, repeat=None)
    drawn = [(node.tag.removeprefix(SVG), node.get("id")) for node in root][1:]
    assert drawn == [("use", None), ("text", "between"), ("use", None)]
    assert [values for values, _ in timings(root)] == [
        "#frame0-low;#frame1-low;#frame1-low",
        "#frame0-high;#frame1-high;#frame1-high",
    ]


def test_an_element_is_written_once_per_change_not_once_per_frame():
    def frame(t):
        scene = make_scene(t)
        scene.text2d(1, 0, 0, "before" if t < 0.6 else "after", id="state")
        return scene

    root = document(frame, fps=10, repeat=None)
    states = [node.text for node in root.iter(f"{SVG}text")]
    assert states == ["before", "after"]
    assert timings(root)[-1] == ("#frame0-state;#frame6-state;#frame6-state", "0;0.6;1")


def test_identical_samples_share_one_content_wherever_they_fall():
    def frame(t):
        return make_scene(abs(t - 0.5))  # there and back

    root = document(frame, fps=4, repeat=None)
    assert len(root.find(f"{SVG}defs")) == 3
    values = ";".join(f"#frame{i}-ball" for i in (0, 1, 2, 1, 1))
    assert timings(root) == [(values, "0;0.25;0.5;0.75;1")]


def test_consecutive_identical_samples_are_one_step():
    def frame(t):
        return make_scene(min(t, 0.5))  # still from t = 0.5 on

    root = document(frame, fps=4, repeat=1)
    assert len(root.find(f"{SVG}defs")) == 3
    values = ";".join(f"#frame{i}-ball" for i in (0, 1, 2, 2))
    assert timings(root) == [(values, "0;0.25;0.5;1")]


def test_an_animation_that_never_changes_is_a_static_document():
    root = document(lambda _: make_scene(), fps=10, repeat=None)
    assert groups(root) == []
    assert root.find(f".//{SVG}animate") is None
    assert root.find(f"{SVG}circle").get("id") == "ball"


def test_frames_with_different_element_counts_render():
    def frame(t):
        scene = make_scene()
        if t > 0:
            scene.sphere(1, (1, 0, 0), 0.2, id="extra")
        return scene

    root = document(frame)
    assert root.find(f"{SVG}circle").get("id") == "ball"
    empty, extra = root.find(f"{SVG}defs")
    assert (empty.get("id"), len(empty)) == ("frame0", 0)  # nothing to draw is a target too
    assert extra.get("id") == "frame1-extra"
    assert timings(root) == [("#frame0;#frame1-extra", "0;1")]


def test_content_pointing_into_another_slot_is_drawn_with_it():
    def frame(t):
        scene = make_scene(t)
        scene.add(1, svg.Use(href="#ball", x=t))  # a moving copy of the moving ball
        return scene

    root = document(frame)
    for group in groups(root):
        ball, copy = group
        assert copy.get("href") == "#" + ball.get("id")


def test_references_follow_their_targets():
    def frame(t):
        scene = make_scene()
        scene.add_def(svg.LinearGradient(id="still", x1="0%", x2="100%"))
        scene.add_def(svg.LinearGradient(id="paint", x1=f"{100 * t:g}%", x2="100%"))
        scene.rect2d(1, 0, 0, 1, 1, fill="url(#paint)", stroke="url(#still)", id="wash")
        return scene

    root = document(frame)
    defs = [d.get("id") for d in root.find(f"{SVG}defs")]
    assert defs == ["still", "frame0-paint", "frame0-wash", "frame1-paint", "frame1-wash"]
    washes = [node for node in root.iter() if node.get("data-vecview-id") == "wash"]
    assert [(w.get("fill"), w.get("stroke")) for w in washes] == [
        ("url(#frame0-paint)", "url(#still)"),
        ("url(#frame1-paint)", "url(#still)"),
    ]


def test_a_shared_element_pointing_into_a_frame_is_not_shared():
    def frame(t):
        scene = vecview.Scene(CAMERA)
        scene.add_def(svg.LinearGradient(id="paint", x1=f"{100 * t:g}%"))
        scene.rect2d(0, 0, 0, 1, 1, fill="url(#paint)", id="wash")  # alike in every frame
        return scene

    root = document(frame)
    assert root.find(f"{SVG}rect") is None
    fills = [node.get("fill") for node in root.iter() if node.get("data-vecview-id") == "wash"]
    assert fills == ["url(#frame0-paint)", "url(#frame1-paint)"]


def test_elements_are_compared_by_their_markup():
    class Raw(svg.Element):
        """Content svg.py's dataclass comparison does not see, as a VecTeX fragment."""

        element_name = "g"

        def __init__(self, content: str) -> None:
            self._content = content

        def as_str(self) -> str:
            return self._content

    def frame(t):
        scene = make_scene(t)
        scene.add(1, Raw(f'<g id="readout-{t:g}"/>'))  # == to the others, unlike in markup
        scene.add(2, Raw('<g id="label"/>'))  # a new object each frame, alike in markup
        return scene

    text = Animation(frame, duration=1, fps=1, view_box=(0, 0, 9, 9)).to_svg_document()
    assert text.count('id="label"') == 1
    assert 'id="readout-0"' in text and 'id="readout-1"' in text


def test_a_stylesheet_that_changes_is_scoped_to_its_frame():
    def frame(t):
        scene = make_scene(t)
        scene.add_def(svg.LinearGradient(id="paint", x1=f"{100 * t:g}%"))
        scene.add(8, svg.Style(text=f"#ball, .particle {{ fill: url(#paint); opacity: {t}; }}"))
        return scene

    styles = [node.text for node in document(frame).iter(f"{SVG}style")]
    assert styles == [
        '[data-vecview-frame="0"] #frame0-ball, [data-vecview-frame="0"] .particle '
        "{fill: url(#frame0-paint); opacity: 0.0;}",
        '[data-vecview-frame="1"] #frame1-ball, [data-vecview-frame="1"] .particle '
        "{fill: url(#frame1-paint); opacity: 1.0;}",
    ]


def test_a_stylesheet_alike_in_every_frame_is_left_alone():
    def frame(t):
        scene = make_scene(t)
        scene.add(8, svg.Style(text=".particle { stroke: #123; }"))
        return scene

    styles = [node.text for node in document(frame).iter(f"{SVG}style")]
    assert styles == [".particle { stroke: #123; }"]


def test_a_changing_stylesheet_with_at_rules_is_refused():
    def frame(t):
        scene = make_scene()
        scene.add(0, svg.Style(text=f"@media print {{ .particle {{ opacity: {t}; }} }}"))
        return scene

    with pytest.raises(ValueError, match="at-rules"):
        Animation(frame, duration=1, view_box=(0, 0, 1, 1)).render()


def test_shared_raw_svg_is_not_mutated_and_output_is_deterministic():
    gradient = svg.LinearGradient(id="shared-gradient", x1="0%", x2="100%")
    rectangle = svg.Rect(id="shared-rect", fill="url(#shared-gradient)")

    def frame(t):
        scene = make_scene(t)
        scene.add_def(gradient)
        scene.add(1, rectangle)
        return scene

    animation = Animation(frame, duration=1, fps=3, view_box=(0, 0, 10, 10))
    assert animation.to_svg_document() == animation.to_svg_document()
    assert (rectangle.id, rectangle.fill, gradient.id) == (
        "shared-rect",
        "url(#shared-gradient)",
        "shared-gradient",
    )


def test_fixed_viewbox_and_background():
    root = document(lambda t: make_scene(t), view_box=(-2, -3, 12, 13), background="#fff")
    assert root.get("viewBox") == "-2 -3 12 13"
    rect = root.find(f"{SVG}rect")
    assert (rect.get("fill"), rect.get("width")) == ("#fff", "12")


def test_render_returns_an_svg_tree_and_empty_scenes_render():
    assert isinstance(
        Animation(lambda _: make_scene(), duration=1, view_box=(0, 0, 5, 6)).render(), svg.SVG
    )
    root = document(lambda _: vecview.Scene(CAMERA), view_box=(0, 0, 2, 3))
    assert root.get("viewBox") == "0 0 2 3"
    assert groups(root) == []


def test_track_keyframes_endpoints_exact_keys_interpolation_and_map():
    called = []
    track = Track.keyframes(
        [(1, 0.0), (3, 4.0)],
        interpolate=lambda a, b, u: called.append(u) or linear(a, b, u),
        ease=smoothstep,
    )
    assert track(0) == 0
    assert track(1) == 0
    assert called == []
    assert track(2) == 2
    assert called == [0.5]
    assert track(3) == 4
    assert track(4) == 4
    assert track.map(lambda value: value + 1)(2) == 3
    assert Track.keyframes([(1, "a"), (2, "b")], interpolate=hold)(1.5) == "a"
    points = Track.keyframes([(0, (0, 0, 0)), (2, (2, 4, -2))], interpolate=linear)
    assert points(0.5) == (0.5, 1.0, -0.5)


@pytest.mark.parametrize("keys", [[], [(1, 0), (1, 2)], [(2, 0), (1, 2)], [(math.inf, 1)]])
def test_track_keyframe_invalid_times_raise(keys):
    with pytest.raises(ValueError):
        Track.keyframes(keys, interpolate=linear)


def test_track_rejects_nonfinite_sample_and_ease_result():
    with pytest.raises(ValueError):
        Track(lambda _: 0)(math.inf)
    track = Track.keyframes([(0, 0), (1, 1)], interpolate=linear, ease=lambda _: math.inf)
    with pytest.raises(ValueError):
        track(0.5)


def test_interpolation_helpers_validate_and_compute():
    assert linear(0, 4, 0.25) == 1
    assert linear((0, 1, 2), (4, 5, 6), 0.5) == (2, 3, 4)
    assert hold("early", "late", 0.999) == "early"
    assert hold("early", "late", 1) == "late"
    assert smoothstep(0.5) == 0.5
    with pytest.raises(ValueError):
        linear(math.inf, 0, 0)
    with pytest.raises(ValueError):
        linear((0, 1), (0, 1, 2), 0.5)


def test_part_rotation_and_scaling_act_about_the_pivot():
    bead = vecview.Part()
    bead.sphere(2, (2, 0, 0), 1, id="body", class_="atom")
    turned = rotate(bead, (0, 0, 2), 90, pivot=(1, 0, 0))
    grown = scale(bead, 2, pivot=(1, 0, 0))
    assert len(bead._log) == len(turned._log) == len(grown._log) == 1
    scene = vecview.Scene(CAMERA)
    scene.place(0, turned)
    scene.place(0, grown)
    (_, turned_args, _, classes), (_, grown_args, _, _) = scene._log
    assert turned_args[0] == 2  # the part's own layer
    assert turned_args[1] == pytest.approx((1, 1, 0))
    assert classes == ("atom",)
    assert grown_args[1] == pytest.approx((3, 0, 0))
    assert grown_args[2] == pytest.approx(2)
    with pytest.raises(ValueError):
        rotate(bead, (0, 0, 0), 30)
    with pytest.raises(ValueError):
        scale(bead, 0)
    with pytest.raises(TypeError):
        rotate("bead", (0, 0, 1), 30)


def test_a_lone_element_without_an_id_is_its_own_target():
    def frame(t):
        scene = make_scene()
        scene.text2d(1, 0, 0, f"{t:.1f}")  # no id
        return scene

    root = document(frame, fps=2, repeat=None)
    texts = root.find(f"{SVG}defs").findall(f"{SVG}text")
    assert [t.get("id") for t in texts] == ["frame0", "frame1"]
    assert root.find(f"{SVG}defs").find(f"{SVG}g") is None


def test_what_changes_rarely_is_cut_from_what_changes_often_at_a_layer_boundary():
    def frame(t):
        scene = make_scene(t)
        if t > 0.45:
            scene.sphere(0, (0, t, 0), 0.2, id="late")  # so the stretch varies in size
        scene.text2d(1, 0, 0, "early" if t < 0.5 else "late", id="caption")
        return scene

    root = document(frame, fps=10, repeat=None)
    assert [t.text for t in root.iter(f"{SVG}text")] == ["early", "late"]


def test_a_breakdown_says_where_the_bytes_go():
    def frame(t):
        scene = vecview.Scene(CAMERA)
        scene.sphere(0, (0, 0, -1), 0.5, id="floor")
        scene.sphere(1, (t, 0, 0), 0.5, id="ball")
        return scene

    animation = Animation(frame, duration=1, fps=4, repeat=None, view_box=(0, 0, 9, 9))
    breakdown = animation.breakdown()
    assert breakdown.frames == 4
    assert breakdown.bytes == len(animation.to_svg_document())
    floor, ball = breakdown.stretches
    assert (floor.names, floor.contents, floor.layers) == (("floor",), 1, (0,))
    assert (ball.names, ball.contents, ball.elements) == (("ball",), 4, (1, 1))
    assert ball.bytes > floor.bytes
    report = str(breakdown)
    assert report.startswith("4 frames,") and "4 contents" in report
