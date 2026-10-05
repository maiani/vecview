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

`arrow` draws [`arrow_shape`](shapes.md#arrows) as a polygon, with one addition:
`normal="camera"` turns the arrow about its own axis to show the widest face it
can, so a spin along `z` reads from any viewpoint:

```python
scene.arrow(22, base, (0, 0, 1), 1.05, normal="camera", shaft_w=0.1, head_w=0.34, head_len=0.34)
```

The normal is resolved from the scene's camera when the arrow is drawn, so a
[reprojection](#rendering-one-scene-several-ways) turns it too. Computing it
yourself from `cam.view` bakes in the original camera — and for an oblique
camera, `view` does not even give the widest face. An arrow pointing along the
projection ray has no face to show and raises `ValueError`.

`gaussian` draws a soft spot lying in a world plane, with opacity
`exp(-(s/a)² - (t/b)²)` along the in-plane axes `u` and `v`:

```python
scene.gaussian(
    20, (0, 0, 0.01), (1, 0, 0), (0, 1, 0), 1.3, 0.8, id="density-up", color="#d62828", opacity=0.8
)
```

It is one polygon filled by a radial gradient mapped through the plane's affine
transform, so it foreshortens with the plane — where nested translucent ellipses
would take many elements and show their steps. The profile is shifted to reach
exactly zero at `extent` half-widths (default `2`), where the polygon ends, so
there is no rim. The gradient lands in `<defs>` as `{id}-profile`. Like `plane`,
it needs a parallel camera.

The profile is faithful, so it reads more compact than a stack of nested
translucent contours, which overweights the tails. Where two spots overlap, the
one drawn later covers the earlier at its centre in proportion to its opacity: at
`opacity=1` it hides it entirely. To show both densities through each other, keep
both opacities well below 1 (around `0.5`–`0.65`) rather than raising the later
one.

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

## Anchoring upright content

```python
scene.slot(layer, point3, w, h, id="label-x", align="west", dx=1.6, dy=0.5)
```

The screen-aligned sibling of `plane`. Where a plane makes content lie *in* the
scene, a slot keeps it upright and unforeshortened — a TeX label, an inset —
while pinning it to a point of the geometry.

It reserves an **empty group** translated to the anchor, the projected point
offset by `(dx, dy)` in screen units, and records `align` as `data-align`.
`align` names the point of the content's box that sits on the anchor: `"west"`
puts the anchor at the middle of the box's left edge, so the content extends to
the right. The nine values are `"center"` and the eight compass points.

A `w` by `h` box, aligned the same way, grows the fitted viewBox so content of
that size is not clipped. Nothing is drawn; a consumer fills the group by `id`
and lines its content up against the anchor using `data-align` — see
[Embedding a scene](embedding.md#filling-a-slot). Because the anchor is a world
point, a [reprojection](#rendering-one-scene-several-ways) moves the slot with
the geometry, which a hand-placed `rect2d` would not.

## Seamless solids

```python
gate = vecview.annulus_sector((0, 0), 2.6, 3.0, 20, 160)
scene.prism_walls(30, gate, 0.0, 0.26, fill="#4a5059", id="gate-walls")
scene.faces(30, vecview.prism_faces(gate, 0.0, 0.26)[:1], fill="#737a84", id="gate")
```

Drawing a solid's visible walls one polygon each leaves hairline seams where
neighbouring walls meet, in `cairosvg` and Inkscape alike: each shared edge is
anti-aliased against the background twice. That is worst on a curved footprint,
which is many thin facets.

`prism_walls` culls the walls of an extruded footprint with the scene's own
camera, merges each run of consecutive facing walls into one strip — along the
base, back along the top — and emits all strips as one `<path>`. Draw the cap
over it. It handles non-convex footprints.

Walls-then-cap is **exact at any height** when the walls are unstroked and the
cap faces the camera:

- along any view ray, the cap is never *behind* a wall, so painting it last is
  always right where it shows;
- where one facing wall hides another of the same solid, both share one fill, so
  which is painted on top cannot be seen.

Only a **stroke** breaks this: the outline of a wall hidden by another wall of
the same solid shows through. For a low extrusion that hidden part is a sliver
and does not matter; for a tall non-convex prism it can. There is no depth sort
to fix it, by design — leave such walls unstroked and stroke the cap instead.

`silhouette` is the older, simpler tool for a *convex* solid given as faces or
points: it fills the convex hull of the projected vertices. For a non-convex
outline the hull is wrong; use `prism_walls`.

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
the first camera put them. The `altermagnetic_dot` example is the opposite case:
world-space throughout, it is built once and replayed with `with_camera`.

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
