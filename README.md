# vecview

Layered 3D scenes that render to SVG, for scientific schematics.

A small projection layer on top of [`svg.py`](https://pypi.org/project/svg.py/).
`svg.py` builds the elements; `vecview` supplies what it has no notion of — a
camera, world-space glyph geometry, and an explicit layer stack.

```python
import vecview

cam = vecview.OrthographicCamera(azim_deg=35, elev_deg=24, scale=62)
scene = vecview.Scene(cam, pad=28, background="#ffffff")

slab = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
scene.faces(10, slab, cull=True, fill="#cfd6e0", stroke="#8b96a6", stroke_width=1.6)

for angle in (0.0, 67.5):
    axis = vecview.in_plane_dir(angle)
    scene.polygon(
        20,
        vecview.double_arrow_shape(
            (0, 0, 0.02), axis, 4.0, (0, 0, 1), shaft_w=0.1, head_w=0.36, head_len=0.34
        ),
        fill="#d62828",
    )

scene.save("slab.svg")
```

## Projections

Five, all parallel, so parallel edges stay parallel:

| Camera | Foreshortening |
| --- | --- |
| `OrthographicCamera.isometric(scale)` | all three axes equal, `0.8165` |
| `OrthographicCamera.dimetric(scale, ratio=0.5)` | two equal, third by `ratio` |
| `OrthographicCamera(azim, elev, scale)` | trimetric — the general case |
| `ObliqueCamera.cavalier(scale)` | front face true, depth `1.0` |
| `ObliqueCamera.cabinet(scale)` | front face true, depth `0.5` |

`foreshortening()` reports the ratios for any camera and `axonometry()` names the
class an orthographic one falls into — both read the projection rather than how it
was built.

A finished scene re-renders under any of them, exactly:

```python
scene.with_camera(vecview.ObliqueCamera.cabinet(62)).save("cabinet.svg")
```

The hierarchy keeps that honest. `Camera` promises only `project`, `at`, `depth`,
and `visible`; `direction`, `screen_basis`, `foreshortening`, and `plane_matrix`
live on `ParallelCamera` because each assumes a screen offset that does not depend
on where in the scene you are. A future `PerspectiveCamera` would subclass `Camera`
directly and inherit none of them, which is the point.

## Flat content in a world plane

`Scene.plane` reserves a rectangle of a world plane, so a Matplotlib plot or a TeX
equation lies *in* the scene — foreshortened and sheared with the geometry — rather
than pasted on top:

```python
scene.plane(15, origin, u_edge, v_edge, id="plot-plane")
```

The group is left empty for a consumer to fill; this package never parses foreign
SVG. The embedding is exact rather than approximate, because a parallel projection
is affine and stays affine when restricted to a plane — which is what an SVG
`matrix` expresses. (A perspective camera would give a homography, which `matrix`
cannot represent; that is one concrete cost of ever adding one.)

## Why layers, not a depth sort

There is no z-buffer and no painter's-algorithm depth sort. For a schematic with
a beam passing through a translucent slab, deciding what occludes what by hand is
worth more than getting it automatically and almost right — the beam above the
slab, the attenuated segment inside it, and the emerging beam below are three
draw calls at three layers, and no automatic rule orders them correctly against a
partially transparent face.

`Camera.visible()` covers the one case where the answer *is* unambiguous: the
back faces of a convex solid, culled by their outward normals.

## Install

Not published to PyPI (the name is taken by an unrelated project — see
[Development](docs/development.md)). Install from a checkout:

```bash
pip install -e /path/to/vecview
```

## The pieces

| Piece | What it does |
| --- | --- |
| `Camera` | The projection contract. `ParallelCamera` adds the affine frames: `screen_basis()`, `direction()`, `foreshortening()`, `plane_matrix()` |
| projections | `OrthographicCamera` (isometric / dimetric / trimetric) and `ObliqueCamera` (cavalier / cabinet) |
| `Scene` | The layer stack. World-space `polygon`/`polyline`/`text`/`faces`/`plane`, screen-space `rect2d`/`text2d`, `<defs>`, a fitted viewBox, and `with_camera()` |
| shapes | `box_faces`, `rect_shape`, `circle_shape`, `arrow_shape`, `double_arrow_shape`, `sine_ribbon`, `in_plane_dir` — all returning plain world-space arrays |

Geometry functions know nothing about cameras, styles, or SVG. That split keeps
them testable as numbers, and lets one polygon be drawn twice with different
fills.

### Placement without guesswork

At an azimuth of 35°, neither `+x` nor `+y` moves a point horizontally across the
picture, so positioning things inside a projected plane by eye is guesswork.
`screen_basis()` returns the two in-plane world directions that *do*:

```python
horizontal, down = cam.screen_basis()  # in the z = 0 plane
spots = [-3.0 * horizontal + 1.7 * down, +3.0 * horizontal + 1.7 * down]
```

## Embedding in a larger document

`Scene.to_svg_document()` returns a complete, standalone SVG document string.
That is the whole contract: any tool that accepts an object exposing that method
can place a scene, without `vecview` knowing about it or depending on it.

```python
document = scene.to_svg_document()
```

The document carries a `viewBox` fitted to the content, so its origin is usually
*not* `0, 0` — a consumer has to honour that. See
[Embedding a scene](docs/embedding.md).

Rasterizing and PDF export are deliberately out of scope: they belong to whatever
assembles the final page, and keeping them out holds the dependency set to `numpy`
and `svg.py`.

## Documentation

- [Overview](docs/index.md)
- [Cameras](docs/cameras.md)
- [Scenes and layers](docs/scenes.md)
- [Shapes](docs/shapes.md)
- [Embedding a scene](docs/embedding.md)
- [Development](docs/development.md)

## Examples

```bash
uv run python examples/slab_polarizer.py                      # one projection
uv run python examples/slab_polarizer.py --projection all     # all five
```

## License

MIT
