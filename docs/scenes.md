# Scenes and layers

```python
Scene(camera=None, *, cameras=None, pad=26.0, background=None)
```

A `Scene` holds objects in world space and the cameras that look at them, the way
a 3D application does. Every drawing call only *records* an object; nothing is
projected until the scene is rendered, and then the record is replayed against a
camera:

```python
scene = vecview.Scene(pad=28, background="#ffffff")
scene.sphere(10, (0, 0, 0), 1.0, fill="#c33")

scene.cameras["main"] = vecview.OrthographicCamera(35, 24, 62)
scene.cameras["cabinet"] = vecview.ObliqueCamera.cabinet(62)
scene.camera = "main"  # the active camera

scene.save("main.svg")  # the active camera
scene.save("cabinet.svg", "cabinet")  # another, by name
scene.render(vecview.OrthographicCamera.isometric(62))  # or any camera at all
```

`cameras` is a plain dict of named cameras, and `camera` is the **active** one:
`render`, `save`, and `bbox` use it unless given another camera or a name, and
`to_svg_document` — the zero-argument embedding contract — always uses it. Set by
name, the active camera follows its entry if the entry is replaced; it can also be
set to a camera directly, or left `None` while the scene is built. `pad` and
`background` are the defaults `render` uses.

Mistakes that do not depend on the camera — a footprint that crosses itself, a
tube of one point, a highlight with no fill — raise where the call is made.
Those that do — a plane seen edge-on, an arrow along the projection ray — raise
when that camera renders.

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

### Sorting by depth

```python
scene.sort_by_depth(30)
```

Layers are the model, and a scene never reorders a layer on its own. A layer
passed to `sort_by_depth` opts in to the painter's algorithm: its world-space
elements are drawn farthest first, each keyed by the mean depth of the points
that made it — a sphere by its centre, a cylinder by its axis midpoint, a polygon
by its vertices. Equal depths keep insertion order, so the output stays
deterministic, and screen-space elements, which have no depth, go on top. Other
layers are untouched, and the setting is recorded, so
[another camera](#rendering-one-scene-several-ways) re-sorts.

It is for **many separate objects that do not interpenetrate**: the atoms and
bonds of a lattice, the arrows of a spin texture, the quads of a surface. There,
assigning layers by hand is not an option, and sorting is close to exact — exact
for non-overlapping spheres of one radius.

It is a heuristic, and two cases defeat it:

- **A line or face running into a sphere's centre.** Part of it is inside the
  ball, so no order is right. Cut it back to the surface — `edges(trim=r)`,
  [`trim_corners`](shapes.md#polyhedra), or a bond whose ends you move to the atom
  surfaces — and it sorts exactly.
- **Long objects that each cover part of the other**, like a coil round a core.
  A single key cannot describe either. Cut the long one into
  [slices](#long-objects).

Both have a fix that keeps the layer fast and dependency-free, and exact
visibility removes the need for either.

### Exact visibility

```python
scene.sort_by_depth(10, exact=True)  # pip install 'vecview[occlusion]'
```

With `exact=True` visibility is decided point by point rather than element by
element. Every surface keeps its native SVG element — a `<circle>` stays a circle
— clipped to the part of it that no opaque surface hides, so a bond can run into
an atom's centre, two planes can cut through each other, and a coil can wrap an
unsliced core. Every line is cut where an opaque surface hides it, and the hidden
part is dropped, or drawn in the `back` style that `polyline`, `edges`, and
`sphere_curve` take — a ray dashed where it passes behind an atom:

```python
scene.polyline(10, [start, end], back={"stroke_dasharray": "4 3"}, stroke="#000")
```

How exact is exact:

- Between two **planar** surfaces — faces, polygons, quads of a mesh — the
  boundary is a straight line, computed exactly.
- Wherever a **sphere, cylinder, cone, arrow, or tube** is involved, the depth of
  each surface is known in closed form at every point of the screen, and the
  boundary is the zero contour of the difference, traced to half a screen unit
  and then simplified to a twentieth of one.
- **Lines** are split where they cross behind a surface, refined by bisection.
- **Translucent** surfaces (`fill_opacity` or `opacity` below 1) hide nothing, but
  are clipped by what is in front of them. Soft `gaussian` spots never hide.
- Opaque surfaces are painted so that whatever hides another comes after it, and
  each hidden one runs on a little under the edge in front of it, so their
  anti-aliased edges never leave a hairline of background between them.

It costs a little: every partly hidden element gains a `<clipPath>`, and a layer
of a few hundred solids takes a second or so to render, against milliseconds for
plain sorting. It needs `shapely` and `contourpy`, which only exact layers import.

What no visibility rule can do is order a beam inside a translucent slab: a
translucent face hides nothing, so the beam above, the attenuated segment
inside, and the emerging beam below are three draw calls at three layers,
decided by hand.

`Camera.depth()` is there if you want to order something yourself, and
back-face culling of a convex solid is
[`Camera.visible`](cameras.md#back-face-culling).

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

`text` and `text2d` take either a string or a list of `svg.TSpan` runs, for a
subscript or a mixed style:

```python
k_x = [svg.TSpan(text="k"), svg.TSpan(text="x", baseline_shift="sub", font_size=14)]
scene.text(40, tip, k_x, size=20, font_style="italic")
```

For real mathematics, reserve a [slot](#anchoring-upright-content) and fill it
with a TeX fragment in the tool that composes the page.

## Curved solids

```python
scene.sphere(layer, center, radius, highlight=None, **style)
scene.cylinder(
    layer, p0, p1, radius, r1=None, ends=True, end_style=None, highlight=None, slices=1, **style
)
scene.cone(layer, base, apex, radius, end=True, end_style=None, highlight=None, **style)
scene.arrow3d(
    layer, origin, direction, length, shaft_r=..., head_r=..., head_len=..., pivot="tail", **style
)
scene.tube(layer, points3, radius, chunk=4, **style)
```

A parallel projection maps a sphere onto an ellipse and a circle onto an ellipse,
so curved solids have **exact, closed-form outlines**, and each is drawn as one:

- `sphere` is a single `<circle>` — or, under an oblique camera, a rotated
  `<ellipse>`. An editor sees a circle, not a polygon.
- `cylinder` is one `<path>` of two straight sides and two elliptical arcs: the
  outer common tangents of the projected end circles, which have a closed form.
  The end disk that faces the camera, if any, is drawn over the body as a native
  ellipse. `r1` makes a frustum and `ends=False` an open tube — what a bond hidden
  inside two atoms wants. `end_style` restyles the disks, typically a lighter
  `fill`.
- `cone` is `cylinder` with `r1=0`: two tangents from the apex and one arc.
- `arrow3d` is a cylindrical shaft and a conical head in one `<g>`, ordered within
  the group by which end is nearer, so it reads as a solid from every side. The
  flat [`arrow`](#world-space-calls) is still the better choice when an arrow
  seen end-on must stay legible.
- `tube` follows a world-space curve — a coil, a field line. It is drawn as wide
  round-joined strokes, which is exact for an orthographic camera, since a tube
  projects to its centre line thickened by its radius. `fill` is the tube's colour
  and `stroke` its outline. It is cut into pieces of `chunk` segments so that in a
  sorted layer it passes over and under itself; neighbouring pieces overlap and
  each outline stops short of its body, so no seam shows.

Each solid is one `<g>` (or, for a sphere, one element), so it is one object in
an editor and one key for [`sort_by_depth`](#sorting-by-depth). An `id` goes on
the group and is suffixed for the parts: `{id}-body`, `{id}-body-end0`,
`{id}-head`, `{id}-shaft`, `{id}-3` for the fourth tube piece.

### Highlights

`highlight="#ffffff"` shades a solid: a sphere with a radial gradient from the
highlight near its upper left to its `fill` at the rim, a cylinder or cone with a
linear gradient across its width. This is a **fill style, not a lighting model**
— the highlight sits in the same place on screen whatever the camera, and there
is no light direction, material, or shading per face.

Spheres of one colour pair share one gradient, `ball-{fill}-{highlight}`, so a
lattice of a thousand atoms in two colours adds two definitions. A cylinder's
gradient depends on its geometry, so it needs an `id` and is named `{id}-shade`
(or `{id}-body-shade` through the group).

### Long objects

```python
scene.cylinder(10, (-2, 0, 0), (2, 0, 0), 0.8, slices=20, id="core")
```

Keyed by its centre alone, a long cylinder sorts wholly in front of everything
on its far half and wholly behind everything on its near half — a coil wound
round it comes out wrong at both ends. `slices` cuts it into lengths along its
axis, each its own `<g>` keyed by its own midpoint, so each turn meets the slice
it wraps. The slices overlap a little and the outline is stroked along the sides
only, so the result still looks like one solid — except with a translucent fill,
where the overlaps show.

## Hidden lines

```python
scene.edges(layer, faces, back=None, back_layer=None, separate=False, trim=0.0, **style)
scene.sphere_curve(layer, center, points3, closed=False, back=None, back_layer=None, **style)
```

`edges` draws the edges of a convex solid, visible ones with `style`. An edge is
visible when either face it bounds faces the camera, which is exact for a convex
solid. Hidden edges are dropped unless `back` is given, in which case they get
`{**style, **back}` — the crystallographer's dashed back edges — on `back_layer`,
which you can put under a translucent solid's faces so they veil it:

```python
zone = vecview.convex_polyhedron(corners)
scene.edges(30, zone, back={"stroke_dasharray": "5 4"}, back_layer=5, stroke="#222")
scene.faces(20, zone, cull=True, fill="#a9c8ea", fill_opacity=0.2)
```

`separate=True` emits one `<path>` per edge, keyed by its own midpoint, so cell
edges interleave with atoms in a sorted layer; `trim` shortens each edge at both
ends, to stop at the surface of an atom on each corner.

`sphere_curve` draws a curve lying on a sphere — an equator, a meridian — split
exactly where it passes behind the sphere, with the same `back` and `back_layer`.
A point on the sphere faces the camera when `(p - center) · view ≥ 0`, and the
curve is cut where that changes sign.

Both re-split for [each camera](#rendering-one-scene-several-ways).

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
scene = build()  # objects, and named cameras
for name in scene.cameras:
    scene.save(f"device_{name}.svg", name)
```

Rendering is a pure function of the recorded objects and the camera, so a scene
renders under any number of cameras, and each result is byte-identical to building
the scene with that camera from the start — including which walls `cull=True`
selects. `with_camera(camera)` returns a copy with a different active camera — a
name or a camera — for handing one scene, seen two ways, to a tool that only
calls `to_svg_document`.

Two things behave as their names promise rather than as a new camera might
suggest:

- **Screen-space calls stay put.** `rect2d`, `text2d`, and anything handed to
  `add` are replayed at the same screen coordinates, because that is what screen
  space means. They do not follow the geometry.
- **Caller-side camera math is already baked in.** A point placed via `cam.at()`
  or a face list filtered by `cam.visible()` was resolved before the scene saw it.
  Prefer what the scene resolves itself — `cull=True`, `normal="camera"`, a
  `slot` — and a scene needs no camera until it is rendered.

Where a scene mixes in screen-space work, rebuild it per camera instead. The
`slab_polarizer` example does exactly that, and says why: its soft beam glows are
`rect2d` columns positioned from `cam.at(...)`, so another camera would leave them
where the first one put them. The `altermagnetic_dot` example is the opposite
case: world-space throughout, it is built once, holds its four projections as
named cameras, and renders each by name.

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

In a Jupyter notebook a scene displays itself: it implements `_repr_svg_`, which
returns the document, or nothing while the scene is still empty.

Only SVG is written. PNG and PDF export are left to the consumer, which is what
holds the runtime dependencies to `numpy` and `svg.py`. Run `cairosvg` over the
file, or hand the document to whatever assembles the final page — see
[Embedding a scene](embedding.md).
