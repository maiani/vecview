# VecView

Layered 3D scenes that render to SVG, for scientific schematics.

A small projection layer on top of [`svg.py`](https://pypi.org/project/svg.py/).
`svg.py` builds the elements; VecView supplies what it has no notion of — a
camera, world-space glyph geometry, and an explicit layer stack.

## Install

```bash
python -m pip install vecview
```

Python 3.12 or newer is required; 3.12–3.14 are tested. VecView is alpha and its
API may change before 1.0.

## The shape of the package

Three concerns in four modules, with a hard line between them:

| Module | Knows about |
| --- | --- |
| [`shapes`](shapes.md) | Numbers only. Returns world-space `(n, 3)` arrays and `Face` records. No camera, no style, no SVG. |
| [`camera`](cameras.md) | What a camera *is*: the projection contract, and the affine machinery every parallel projection shares. |
| [`projections`](cameras.md#orthographiccamera) | The projections shipped: orthographic (isometric, dimetric, trimetric) and oblique (cavalier, cabinet). |
| [`scene`](scenes.md) | Objects, parts, named cameras, the layer stack, and document assembly. With its private rendering modules, the only part that touches `svg.py`. |

That split is what keeps geometry testable as plain numbers, and lets one polygon
be drawn twice with different fills or reused as a clip path.

## Conventions

- World coordinates are **right-handed `(x, y, z)` with `z` up**.
- Every camera is a **parallel projection**, so parallel edges stay parallel and
  no perspective distortion creeps into a lattice or a repeated texture. Pick from
  isometric, dimetric, trimetric, cavalier, and cabinet.
- Screen coordinates are **SVG user units with `y` growing downward**.
  `Camera.project` has already applied that flip.
- Angles are in **degrees** at the API surface, radians nowhere.

## A first scene

```python
import vecview

cam = vecview.OrthographicCamera(azim_deg=35, elev_deg=24, scale=62)
scene = vecview.Scene(cam, pad=28, background="#ffffff")

# The slab: a box, drawing only the walls the rendering camera can see.
slab = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
scene.faces(10, slab, cull=True, fill="#cfd6e0", stroke="#8b96a6", stroke_width=1.6)

# Two axes on its face, drawn over the slab.
for angle, color in ((0.0, "#d62828"), (67.5, "#6a2fb5")):
    scene.polygon(
        20,
        vecview.double_arrow_shape(
            (0, 0, 0.02),
            vecview.in_plane_dir(angle),
            length=4.0,
            normal=(0, 0, 1),
            shaft_w=0.10,
            head_w=0.36,
            head_len=0.34,
        ),
        fill=color,
    )

scene.text(30, (0, 0, 2.0), "polarizer", size=26, text_anchor="middle")
scene.save("slab.svg")
```

The first argument of every drawing call is its layer: lower layers are drawn
first, so the arrows at 20 sit on the slab at 10. Style keywords such as
`stroke_width` pass straight to `svg.py`. The viewBox is fitted to the content,
so nothing needs centring by hand.

## Objects and cameras

A scene records objects, and holds any number of named cameras, one of them
active — as a 3D application does. A camera is only needed to render, and any
camera will do. Rendering is exact — it even redoes which walls `cull=True`
selects, which a cabinet camera answers differently from an orthographic one:

```python
scene.cameras.update(
    isometric=vecview.OrthographicCamera.isometric(62),
    dimetric=vecview.OrthographicCamera.dimetric(62),
    cabinet=vecview.ObliqueCamera.cabinet(62),
)
for name in scene.cameras:
    scene.save(f"slab_{name}.svg", name)
```

The camera given to `Scene(...)`, or set later as `scene.camera` — a camera, or a
name from `scene.cameras` — is the active one, which `to_svg_document` uses.

## Flat content inside the scene

`Scene.plane` reserves a rectangle of a world plane for a plot, an equation, or a
bitmap, so it lies *in* the picture rather than on top of it. The embedding is
exact: a parallel projection is affine, so restricted to a plane it is still
affine — precisely what an SVG `matrix` expresses. `Scene.slot` is its upright
sibling, pinning a label to a world point. Both reserve an empty group that the
tool composing the page fills; see [Embedding a scene](embedding.md).

## Layers first, depth sorting by request

Draw order is an explicit layer stack. For a schematic with a beam passing
through a translucent slab, deciding what occludes what by hand is worth more
than getting it automatically and almost right: the beam above the slab, the
attenuated segment inside it, and the emerging beam below are three draw calls at
three layers, and no automatic rule orders them correctly against a partially
transparent face.

A lattice of hundreds of atoms is the opposite case, and a layer can opt in to
[sorting by depth](scenes.md#sorting-by-depth) for it — or to
[exact visibility](scenes.md#exact-visibility), which clips every
element to what shows of it and dashes lines where they pass behind. The curved solids —
spheres, cylinders, cones, solid arrows, tubes — have exact outlines under every
parallel projection, so a sorted layer of them stays a small, editable file. See
the [gallery](gallery.md).

## Related projects

VecView is developed alongside [FigWorks](https://github.com/maiani/figworks),
which composes multi-panel figures, and two other producers of editable SVG:
[VecTeX](https://github.com/maiani/vectex) (TeX equations) and
[VecWire](https://github.com/maiani/vecwire) (circuit schematics). The four form
a suite for publication figures, and each is usable on its own.

VecView depends on none of them and contains no code specific to any of them. A
composition layer needs only [`Scene.to_svg_document()`](embedding.md), so the
integration costs no import in either direction.

## Next

- [Cameras](cameras.md) — the hierarchy, the five projections, `screen_basis`, culling
- [Scenes and layers](scenes.md) — objects and cameras, the layer stack, embedded planes
- [Shapes](shapes.md) — the world-space geometry catalogue
- [Gallery](gallery.md) — the figures everyone draws, each from one script
- [Embedding a scene](embedding.md) — handing the output to a larger document
- [Development](development.md) — toolchain, and the name situation
