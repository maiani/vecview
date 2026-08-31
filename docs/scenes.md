# Scenes and layers

```python
Scene(cam, pad=26.0, background=None)
```

A `Scene` collects SVG elements built from world-space geometry and assembles them
into one document. `pad` and `background` are the defaults `render` and
`to_svg_document` use; both can be overridden per `render` call.

## The layer stack

Every drawing call takes an integer `layer` as its first argument. Lower layers
are emitted first, so they sit *behind*. Ties within a layer resolve by insertion
order, which makes output stable and diffable.

```python
scene.polygon(10, slab_top, fill="#eef1f5")  # behind
scene.polyline(20, beam_above, stroke="#d62828")
scene.text(30, tip, "low frequency", size=27)  # in front
```

Layers are integers rather than names so you can leave gaps and slot something in
later without renumbering. A common habit is decades — 10 for the solid, 20 for
what sits on it, 30 for labels.

### Why not a depth sort

There is no z-buffer and no painter's algorithm. A beam passing through a
translucent slab has three parts — above, attenuated inside, emerging below — and
no automatic depth rule orders those correctly against a partially transparent
face. Deciding it by hand is worth more than getting it automatically and almost
right. Back-face culling of a convex solid, the one unambiguous case, is
available as [`Camera.visible`](cameras.md#back-face-culling).

`Camera.depth()` is there if you want to order something by depth yourself.

## World-space calls

```python
scene.polygon(layer, points3, **style)  # filled polygon
scene.polyline(layer, points3, **style)  # open path, fill="none" by default
scene.faces(layer, faces, **style)  # one polygon per Face, one style
scene.text(layer, point3, s, dx=0, dy=0, size=22, **style)
```

Style keyword arguments pass straight through to `svg.py`, so Python's
underscores map to SVG's hyphens: `stroke_width` → `stroke-width`,
`fill_opacity` → `fill-opacity`, `text_anchor` → `text-anchor`.

`text` anchors at a projected world point and then offsets by `dx`/`dy` in
*screen* units — the right frame for "just above the label's anchor", which
should not shift when the camera turns.

`faces` draws exactly the faces it is handed. Pass `cull=True` to drop back faces
at draw time using the scene's own camera:

```python
box = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
scene.faces(10, box, cull=True, fill="#cfd6e0")
```

Prefer that over filtering with `Camera.visible` yourself whenever the scene might
be [reprojected](#rendering-one-scene-several-ways): culling done by the caller
bakes in *that* camera's answer, and a cabinet camera sees `-y` where a 35°
orthographic one sees `+y`. With `cull=True` the full set is recorded and the new
camera decides.

An `id` passed to `faces` is **suffixed per face** rather than repeated on each —
duplicate ids are invalid SVG and break selection downstream. `+` is not a legal
XML name character, so the sign is spelled out:

```python
scene.faces(10, cam.visible(slab), fill="#cfd6e0", id="slab")
# -> slab-pz, slab-px, slab-py
```

## Screen-space calls

```python
scene.rect2d(layer, x, y, w, h, grow=False, **style)
scene.text2d(layer, x, y, s, size=22, grow=True, **style)
```

For things that belong to the picture rather than the world: a backdrop, a
gradient wash behind a beam, a corner annotation.

`rect2d` is **excluded from the bounding box by default** (`grow=False`) — that is
the point of it. A soft glow deliberately extends past the geometry, and letting
it inflate the fitted viewBox would leave a wide dead margin. Pass `grow=True`
when the rectangle really is part of the content.

## Bounds and the fitted viewBox

`render` fits the viewBox to everything added, plus `pad`. Nothing needs centring
by hand, and the document size follows the content:

```python
lo, hi = scene.bbox()  # screen-space min/max corners
scene.is_empty  # nothing contributing to the box yet
```

Text bounds are **estimated** from a nominal glyph width, not measured — real
advance widths would mean loading the font. The estimate is generous enough to
keep a label from being clipped, but do not treat `bbox` as exact where text is
involved.

Rendering an empty scene raises `ValueError`: there is no content to fit a
viewBox to, and a zero-size document is never what was wanted.

## Definitions

```python
scene.add_def(svg.RadialGradient(id="glow", ...))
scene.rect2d(6, x, y, w, h, fill="url(#glow)")
```

`add_def` puts an element in `<defs>`, emitted before everything else. Gradients,
markers, and clip paths go here.

One hard-won caveat: a **gradient-filled `<mask>` does not survive `cairosvg`
rasterization** — the effect vanishes silently, with no warning. Where you would
reach for a masked shape, use a gradient *fill* on a plain rectangle instead.

## Escape hatch

```python
scene.add(layer, svg.Circle(cx=0, cy=0, r=4, fill="red"))
```

`add` takes any `svg.py` element at a layer, bypassing projection and the
bounding box. Use it for anything the primitives do not cover; you are not
fenced in by them.

## Embedding flat content

```python
scene.plane(layer, origin, u_edge, v_edge, id="plot-plane")
```

Reserves a rectangle of a world plane as an **empty group** carrying the affine
transform that maps content coordinates in `[0, 1]²` onto that rectangle. Flat SVG
content — a plot, an equation, a bitmap — then sits *in* the plane, foreshortened
and sheared with the geometry, rather than pasted on top of the picture.

Nothing is drawn: this package does not parse or embed foreign SVG. A consumer
fills the group by `id`, normalizing its content to the unit square. See
[Embedding a scene](embedding.md) for that side, and
[`Camera.plane_matrix`](cameras.md) for why the transform is exact.

- `origin` is the content's **top-left** corner, since SVG `y` grows downward.
- `u_edge` runs along content `+x`, `v_edge` along content `+y` (downward). Their
  **lengths are the rectangle's extents** — they are not normalized.
- The group joins the layer stack like anything else, so content in it can be
  drawn over the slab it lies on and under the beam that crosses it.
- The rectangle grows the fitted viewBox, so inserted content is never clipped.

Choosing the edges is a real decision, and getting it wrong mirrors or inverts
every label. The rule is in [Orientation](embedding.md#orientation).

## Rendering one scene several ways

```python
scene = build(OrthographicCamera(35, 24, 62))
scene.with_camera(OrthographicCamera.isometric(62)).save("iso.svg")
scene.with_camera(ObliqueCamera.cabinet(62)).save("cabinet.svg")
```

`with_camera` returns the same scene projected by a different camera. Every call
made through the scene's own methods is recorded and replayed, so the result is
byte-identical to building from scratch with that camera — including which walls
`cull=True` selects.

Two things behave as their names promise rather than as a reprojection might
suggest:

- **Screen-space calls stay put.** `rect2d`, `text2d`, and anything handed to
  `add` are replayed at the same screen coordinates, because that is what screen
  space means. They do not follow the geometry.
- **Caller-side camera math is already baked in.** A point placed via `cam.at()`
  or a face list filtered by `cam.visible()` was resolved before the scene saw it.
  A scene meant to be reprojected should take its camera as an argument and derive
  such positions inside — and use `cull=True` rather than culling itself.

Where a scene mixes in screen-space work, rebuild it per camera instead. The
`slab_polarizer` example does exactly that, and says why: its soft beam glows are
`rect2d` columns positioned from `cam.at(...)`, so a replay would leave them where
the first camera put them.

World-space arrays are held by reference, not copied, so do not mutate them after
adding.

## Output

```python
document = scene.render(pad=None, background=None)  # an svg.SVG object
text = scene.to_svg_document()  # a complete SVG document string
path = scene.save("scene.svg")  # writes it, returns the Path
```

Rendering does not consume the scene — call it as often as you like, and keep
adding afterwards.

Only SVG is written. PNG and PDF export are left to the consumer, which is what
holds the runtime dependencies to `numpy` and `svg.py`. Run `cairosvg` over the
file, or hand the document to whatever assembles the final page — see
[Embedding a scene](embedding.md).
