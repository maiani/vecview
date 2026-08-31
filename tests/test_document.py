"""The document contract.

``Scene.to_svg_document()`` is the whole surface a consumer needs, so these tests
pin the properties one relies on: a parseable standalone document, a viewBox that
agrees with the declared size, geometry inside it, and deterministic bytes.

Nothing here imports a consumer. This package has no downstream dependencies and
a test must not invent one.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import vecview
from vecview import OrthographicCamera, Scene

SVG_NS = "http://www.w3.org/2000/svg"


@pytest.fixture
def scene() -> Scene:
    cam = OrthographicCamera(35.0, 24.0, 62.0)
    sc = Scene(cam, pad=28.0)
    sc.faces(10, cam.visible(vecview.box_faces((0, 0, -0.45), (11, 9, 0.9))), fill="#cfd6e0")
    sc.text(20, (0, 0, 2.0), "label", size=24, text_anchor="middle")
    return sc


class TestDocumentShape:
    def test_is_a_complete_standalone_document(self, scene: Scene) -> None:
        root = ET.fromstring(scene.to_svg_document())
        assert root.tag == f"{{{SVG_NS}}}svg"

    def test_declares_a_viewbox_with_width_and_height(self, scene: Scene) -> None:
        root = ET.fromstring(scene.to_svg_document())
        assert root.get("viewBox") is not None
        assert root.get("width") is not None
        assert root.get("height") is not None

    def test_viewbox_origin_is_generally_non_zero(self, scene: Scene) -> None:
        """A fitted viewBox rarely starts at 0,0 -- consumers must honour the origin."""
        root = ET.fromstring(scene.to_svg_document())
        assert root.get("viewBox") is not None
        min_x, min_y, _, _ = (float(v) for v in root.get("viewBox", "").split())
        assert (min_x, min_y) != (0.0, 0.0)

    def test_viewbox_matches_the_declared_size(self, scene: Scene) -> None:
        """Otherwise a consumer scaling by width/height distorts the content."""
        root = ET.fromstring(scene.to_svg_document())
        _, _, w, h = (float(v) for v in root.get("viewBox", "").split())
        assert float(root.get("width", "0")) == pytest.approx(w)
        assert float(root.get("height", "0")) == pytest.approx(h)

    def test_geometry_lies_inside_the_viewbox(self, scene: Scene) -> None:
        root = ET.fromstring(scene.to_svg_document())
        min_x, min_y, w, h = (float(v) for v in root.get("viewBox", "").split())
        for polygon in root.iter(f"{{{SVG_NS}}}polygon"):
            pairs = polygon.get("points", "").split()
            pts = np.array([[float(c) for c in pair.split(",")] for pair in pairs])
            assert pts[:, 0].min() >= min_x and pts[:, 0].max() <= min_x + w
            assert pts[:, 1].min() >= min_y and pts[:, 1].max() <= min_y + h

    def test_is_deterministic(self, scene: Scene) -> None:
        """Byte-identical output is what makes a figure diffable in version control."""
        assert scene.to_svg_document() == scene.to_svg_document()

    def test_definitions_are_inline(self) -> None:
        """A consumer that hoists `<defs>` must find them in the document itself."""
        import svg

        cam = OrthographicCamera(35.0, 24.0, 62.0)
        sc = Scene(cam)
        sc.add_def(svg.RadialGradient(id="glow", elements=[svg.Stop(offset=0, stop_color="red")]))
        sc.rect2d(0, -10, -10, 20, 20, grow=True, fill="url(#glow)")

        root = ET.fromstring(sc.to_svg_document())
        defs = root.find(f"{{{SVG_NS}}}defs")
        assert defs is not None
        assert [child.get("id") for child in defs] == ["glow"]

    def test_has_no_external_references(self, scene: Scene) -> None:
        """No fetched fonts, images, or scripts: the document must stand alone."""
        document = scene.to_svg_document()
        for marker in ("<image", "<script", "xlink:href", "http://", "https://"):
            if marker == "http://":
                # the SVG namespace declaration is the one permitted URL
                assert document.count(marker) == document.count(SVG_NS)
                continue
            assert marker not in document

    def test_ids_are_emitted_verbatim(self) -> None:
        """Ids must survive to the document, or post-placement selection breaks."""
        cam = OrthographicCamera(35.0, 24.0, 62.0)
        sc = Scene(cam)
        sc.faces(10, cam.visible(vecview.box_faces((0, 0, 0), (2, 2, 1))), id="slab")

        root = ET.fromstring(sc.to_svg_document())
        ids = [node.get("id") for node in root.iter() if node.get("id")]
        assert sorted(ids) == ["slab-px", "slab-py", "slab-pz"]
        assert len(set(ids)) == len(ids), "duplicate ids are invalid SVG"
