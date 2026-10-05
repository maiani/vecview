# VecView

Layered 3D scenes that render to SVG, for scientific schematics.

A small projection layer on top of [`svg.py`](https://pypi.org/project/svg.py/).
`svg.py` builds the elements; `vecview` supplies what it has no notion of — a
camera, world-space glyph geometry, and an explicit layer stack.

## The shape of the package

Three pieces, with a hard line between them:

| Module | Knows about |
| --- | --- |
| [`shapes`](shapes.md) | Numbers only. Returns world-space `(n, 3)` arrays and `Face` records. No camera, no style, no SVG. |
| [`camera`](cameras.md) | What a camera *is*: the projection contract, and the affine machinery every parallel projection shares. |
| [`projections`](cameras.md#orthographiccamera) | The projections shipped: orthographic (isometric, dimetric, trimetric) and oblique (cavalier, cabinet). |
| [`scene`](scenes.md) | The layer stack and document assembly. The only part that touches `svg.py`. |

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

# The slab: a box, with only the walls this camera can see.
slab = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
scene.faces(10, cam.visible(slab), fill="#cfd6e0", stroke="#8b96a6", stroke_width=1.6)

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

The viewBox is fitted to the content, so nothing needs centring by hand.

## One scene, several projections

A finished scene can be re-rendered under any camera. The replay is exact — it
even redoes which walls `cull=True` selects, which a cabinet camera answers
differently from an orthographic one:

```python
for name, cam in {
    "isometric": vecview.OrthographicCamera.isometric(62),
    "dimetric": vecview.OrthographicCamera.dimetric(62),
    "cabinet": vecview.ObliqueCamera.cabinet(62),
}.items():
    scene.with_camera(cam).save(f"slab_{name}.svg")
```

## Flat content inside the scene

`Scene.plane` reserves a rectangle of a world plane for a plot, an equation, or a
bitmap, so it lies *in* the picture rather than on top of it. The embedding is
exact: a parallel projection is affine, so restricted to a plane it is still
affine — precisely what an SVG `matrix` expresses.

## Why layers, not a depth sort

There is no z-buffer and no painter's-algorithm depth sort. For a schematic with
a beam passing through a translucent slab, deciding what occludes what by hand is
worth more than getting it automatically and almost right: the beam above the
slab, the attenuated segment inside it, and the emerging beam below are three draw
calls at three layers, and no automatic rule orders them correctly against a
partially transparent face.

`Camera.visible()` covers the one case where the answer *is* unambiguous — the
back faces of a convex solid, culled by their outward normals.
`Camera.depth()` is available if you want to order something by depth yourself.

## Related projects

`vecview` is developed alongside [FigWorks](https://github.com/maiani/figworks),
which composes multi-panel figures, and
[VecTeX](https://github.com/maiani/vectex), which renders TeX equations to SVG
fragments. The three form a suite for publication figures, and each is usable on
its own.

`vecview` depends on neither and contains no code specific to either. A
composition layer needs only [`Scene.to_svg_document()`](embedding.md), so the
integration costs no import in either direction.

## Next

- [Cameras](cameras.md) — the hierarchy, the five projections, `screen_basis`, culling
- [Scenes and layers](scenes.md) — the layer stack, reprojection, embedded planes
- [Shapes](shapes.md) — the world-space geometry catalogue
- [Embedding a scene](embedding.md) — handing the output to a larger document
- [Development](development.md) — toolchain, and the name situation
