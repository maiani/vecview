import xml.etree.ElementTree as ET

import pytest

import vecview
from vecview.animation import Animation, Track

SVG = "{http://www.w3.org/2000/svg}"
CAMERA = vecview.OrthographicCamera.isometric(20)
ALONG = Track(lambda t: (t, 0.0, 0.0))


def ball(scene: vecview.Scene) -> ET.Element:
    circle = ET.fromstring(scene.to_svg_document()).find(f".//{SVG}circle[@id='ball']")
    assert circle is not None
    return circle


def test_a_track_is_read_wherever_a_value_goes():
    scene = vecview.Scene(CAMERA)
    scene.sphere(0, ALONG, Track(lambda t: 0.5 + t), id="ball")
    early, late = ball(scene.at(0.0)), ball(scene.at(1.0))
    assert early.get("cx") != late.get("cx")
    assert float(late.get("r")) == pytest.approx(3 * float(early.get("r")))


def test_tracks_are_read_inside_lists_and_styles():
    scene = vecview.Scene(CAMERA)
    scene.polyline(0, [(0, 0, 0), ALONG], stroke=Track(lambda t: "#000" if t < 1 else "#f00"))
    line = ET.fromstring(scene.at(1.0).to_svg_document()).find(f".//{SVG}polyline")
    assert line.get("stroke") == "#f00"


def test_calls_without_a_track_are_shared_not_made_again():
    scene = vecview.Scene(CAMERA)
    scene.sphere(0, (0, 0, 0), 0.5, id="still")
    scene.sphere(1, ALONG, 0.5, id="ball")
    still = scene._log[0]
    assert scene.at(0.3)._log[0] is still
    assert scene.at(0.3)._calls()[1] != scene.at(0.6)._calls()[1]


def test_a_track_is_checked_when_the_call_is_made():
    scene = vecview.Scene(CAMERA)
    with pytest.raises(ValueError, match="coincide"):
        scene.cylinder(0, (0, 0, 0), ALONG, 0.5)  # no axis at t = 0


def test_a_scene_with_tracks_renders_as_it_stands_at_zero():
    scene = vecview.Scene(CAMERA)
    scene.sphere(0, ALONG, 0.5, id="ball")
    assert ball(scene).get("cx") == ball(scene.at(0.0)).get("cx")
    assert scene.to_svg_document() == scene.at(0.0).to_svg_document()


def test_a_part_can_be_a_track_and_can_hold_tracks():
    moving = vecview.Part()
    moving.sphere(0, ALONG, 0.5, id="ball")
    grown = Track(lambda t: [vecview.Part() for _ in range(1 + round(t))])
    scene = vecview.Scene(CAMERA)
    scene.place(0, moving)
    scene.place(1, Track(lambda t: moving.at(t)), id="copy")
    scene.place(2, grown.map(lambda parts: parts[-1]))
    assert ball(scene.at(0.0)).get("cx") != ball(scene.at(1.0)).get("cx")
    copies = ET.fromstring(scene.at(1.0).to_svg_document()).findall(f".//{SVG}circle")
    assert [c.get("id") for c in copies] == ["ball", "copy-ball"]


def test_an_animation_of_a_scene_matches_one_of_a_callback():
    def frame(t):
        scene = vecview.Scene(CAMERA)
        scene.sphere(0, (0, 0, -1), 0.5, id="floor")
        scene.sphere(1, (t, 0.0, 0.0), 0.5, id="ball")
        return scene

    scene = vecview.Scene(CAMERA)
    scene.sphere(0, (0, 0, -1), 0.5, id="floor")
    scene.sphere(1, ALONG, 0.5, id="ball")
    options = {"duration": 1, "fps": 4, "view_box": (0, 0, 10, 10)}
    assert Animation(scene, **options).to_svg_document() == (
        Animation(frame, **options).to_svg_document()
    )
    assert Animation(scene, **options).frame(0.5)._calls() == frame(0.5)._log


def test_the_camera_can_be_a_track():
    turntable = Track(lambda t: vecview.OrthographicCamera(360 * t, 30, 20))
    scene = vecview.Scene(turntable)
    scene.sphere(0, (1, 0, 0), 0.5, id="ball")
    assert scene.camera is not None and scene.camera.azim_deg == 0
    assert scene.at(0.25).camera.azim_deg == 90
    assert ball(scene.at(0.0)).get("cx") != ball(scene.at(0.25)).get("cx")
    root = ET.fromstring(
        Animation(scene, duration=1, fps=4, view_box=(0, 0, 9, 9)).to_svg_document()
    )
    assert len(root.find(f"{SVG}defs")) == 4  # four sides; a whole turn ends where it began


def test_a_track_of_cameras_must_give_cameras():
    with pytest.raises(TypeError, match="Camera"):
        vecview.Scene(Track(lambda t: 3.0))
    with pytest.raises(KeyError):
        vecview.Scene(Track(lambda t: "missing"))


def test_rotate_and_scale_take_tracks():
    from vecview.animation import rotate, scale

    bead = vecview.Part()
    bead.sphere(0, (1, 0, 0), 0.2, id="ball")
    spin = Track(lambda t: 360.0 * t)
    scene = vecview.Scene(CAMERA)
    scene.place(0, scale(rotate(bead, (0, 0, 1), spin), Track(lambda t: 1.0 + t)))
    still = vecview.Scene(CAMERA)
    still.place(0, scale(rotate(bead, (0, 0, 1), 90.0), 1.25))
    assert ball(scene.at(0.25)).attrib == ball(still).attrib
