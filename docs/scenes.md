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
set to a camera directly, or left `None` while the scene is built. Setting it to
a name the scene does not hold raises `KeyError`, as does rendering by one;
rendering with no camera at all raises `ValueError`. `pad` and `background` are
the defaults `render` uses, and plain attributes you can change later.

Mistakes that do not depend on the camera — a footprint that crosses itself, a
tube of one point, a highlight with no fill — raise where the call is made.
Those that do — a plane seen edge-on, an arrow along the projection ray, a
sphere under a camera that is not a `ParallelCamera` — raise when that camera
renders.

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
scene.sort_by_depth(10)
```

Layers are the model, and a scene never reorders a layer on its own. A layer
passed to `sort_by_depth` has its visibility decided by depth instead, point by
point rather than element by element. Every surface keeps its native SVG element
— a `<circle>` stays a circle — clipped to the part of it that no opaque surface
hides, so a bond can run into an atom's centre, two planes can cut through each
other, and a coil can wrap its core. A surface hidden entirely is dropped. Every
line is cut where an opaque surface hides it, and the hidden part is dropped, or
drawn in the `back` style that `polyline`, `edges`, and `sphere_curve` take — a
ray dashed where it passes behind an atom:

```python
scene.polyline(10, [start, end], back={"stroke_dasharray": "4 3"}, stroke="#000")
```

It is for anything whose overlaps no fixed order gets right: the atoms and bonds
of a lattice, the arrows of a spin texture, the quads of a surface, a coil round
a core, crossing planes. Other layers are untouched, and the setting is recorded,
so [another camera](#rendering-one-scene-several-ways) decides afresh.

How exact it is:

- Between two **planar** surfaces — faces, polygons, quads of a mesh — the
  boundary is a straight line, computed exactly.
- Wherever a **sphere, cylinder, cone, arrow, or tube** is involved, the depth of
  each surface is known in closed form at every point of the screen, and the
  boundary is the zero contour of the difference, traced on a grid of half a
  screen unit (at most 160 steps across an overlap) and simplified to a
  twentieth of one.
- **Lines** are split where they cross behind a surface, refined by bisection.
  Text, slots, and planes are never clipped or split.
- **Translucent** surfaces (`fill_opacity` or `opacity` below 1) hide nothing, but
  are clipped by what is in front of them, and are painted back to front over
  the opaque ones. Soft `gaussian` spots never hide. Screen-space elements go on
  top, in the order they were drawn.
- Opaque surfaces are painted so that whatever hides another comes after it, and
  each hidden one runs on a little under the edge in front of it, so their
  anti-aliased edges never leave a hairline of background between them.

It costs something: every partly hidden element gains a `<clipPath>`, named
`{id}-visible` after the element (or `visible-{n}` for one without an id), and
the hidden part of a line with an id becomes a second path, `{id}-hidden`. A
layer of a few hundred solids takes a second or two to render, where a layer
drawn in order takes milliseconds, and its SVG can be twice the size.
`shapely` and `contourpy`, which do the clipping, load only when a sorted layer
first renders.

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
scene.polyline(layer, points3, back=None, **style)  # open path, fill="none" by default
scene.faces(layer, faces, cull=False, **style)  # one polygon per Face, one style
scene.text(layer, point3, s, dx=0, dy=0, size=22, **style)
scene.arrow(layer, origin, direction, length, normal=..., shaft_w=..., head_w=..., head_len=...)
scene.gaussian(layer, center, u, v, a, b, id=..., color=..., opacity=1.0, extent=2.0, stops=9)
```

Style keyword arguments pass straight through to `svg.py`, so Python's
underscores map to SVG's hyphens: `stroke_width` → `stroke-width`,
`fill_opacity` → `fill-opacity`, `text_anchor` → `text-anchor`.

`arrow` draws [`arrow_shape`](shapes.md#arrows) as a polygon, taking the same
arguments (including `pivot="tail"`) with one addition:
`normal="camera"` turns the arrow about its own axis to show the widest face it
can, so a spin along `z` reads from any viewpoint:

```python
scene.arrow(22, base, (0, 0, 1), 1.05, normal="camera", shaft_w=0.1, head_w=0.34, head_len=0.34)
```

The normal is resolved from the camera that renders, so
[every camera](#rendering-one-scene-several-ways) sees the arrow turned toward it.
Computing it yourself from `cam.view` bakes in one camera's answer — and for an oblique
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
there is no rim. `u` and `v` must be perpendicular; `stops` samples the profile,
and more is smoother. The `id` is required, because the gradient lands in
`<defs>` as `{id}-profile`. Like `plane`, it needs a parallel camera.

The profile is faithful, so it reads more compact than a stack of nested
translucent contours, which overweights the tails. Where two spots overlap, the
one drawn later covers the earlier at its centre in proportion to its opacity: at
`opacity=1` it hides it entirely. To show both densities through each other, keep
both opacities well below 1 (around `0.5`–`0.65`) rather than raising the later
one.

`text` anchors at a projected world point and then offsets by `dx`/`dy` in
*screen* units — the right frame for "just above the label's anchor", which
should not shift when the camera turns. `size` is the font size in screen units.
Unless the style says otherwise, text is set in `DejaVu Sans, Verdana,
sans-serif` with fill `#222222`.

`faces` draws exactly the faces it is handed. Pass `cull=True` to drop back faces
when rendering, using whichever camera renders:

```python
box = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
scene.faces(10, box, cull=True, fill="#cfd6e0")
```

Prefer that over filtering with `Camera.visible` yourself: culling done by the
caller bakes in *that* camera's answer, and a cabinet camera sees `-y` where a 35°
orthographic one sees `+y`, so [another camera](#rendering-one-scene-several-ways)
would draw the wrong walls. With `cull=True` the full set is recorded and the
camera decides.

An `id` passed to `faces` is **suffixed per face** rather than repeated on each —
duplicate ids are invalid SVG and break selection downstream. `+` is not a legal
XML name character, so the sign is spelled out:

```python
scene.faces(10, slab, cull=True, fill="#cfd6e0", id="slab")
# -> slab-px, slab-py, slab-pz under the 35° camera; slab-px, slab-my, slab-pz under cabinet
```

Ids must be unique across the whole document, and rendering checks it: an id
used twice raises `ValueError` naming it. That includes the ids a call derives —
`slab-pz` from `faces`, `{id}-body` inside a cylinder, `{id}-profile` for a
Gaussian's gradient — so a hand-written `slab-pz` next to
`faces(..., id="slab")` is caught rather than silently selected in its place.

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
    layer,
    origin,
    direction,
    length,
    shaft_r=...,
    head_r=...,
    head_len=...,
    pivot="tail",
    end_style=None,
    highlight=None,
    **style,
)
scene.tube(layer, points3, radius, chunk=4, **style)
```

```python
scene.sort_by_depth(10)
scene.sphere(10, (0, 0, 0), 0.4, fill="#c33", highlight="#f4b6b6", id="o1")
scene.sphere(10, (1.5, 0, 0), 0.3, fill="#ccc", highlight="#fff", id="h1")
scene.cylinder(10, (0.4, 0, 0), (1.2, 0, 0), 0.1, ends=False, fill="#999", id="bond")
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
- `tube` follows a world-space curve of at least two points — a coil, a field
  line. It is drawn as wide round-joined strokes, which is exact for an
  orthographic camera, since a tube projects to its centre line thickened by its
  radius; under an oblique camera the width is an average, an approximation.
  `fill` is the tube's colour (default black), `stroke` its outline (default
  none), and `stroke_width` the outline's width (default 1); its two ends are
  cut square. It is cut into pieces of `chunk` segments, each a surface of its own
  in a sorted layer, so that it passes over and under itself; neighbouring pieces overlap and each outline
  stops short of its body, so no seam shows.

A sphere is one element and every other solid one `<g>`, so it is one object in
an editor and is clipped as one by [`sort_by_depth`](#sorting-by-depth). The
exceptions are the pieces of a tube, each its own surface so that a tube can
hide part of itself, and the slices of a [sliced cylinder](#long-objects).

An `id` goes on the sphere or the group, and the parts are named from it:

| Call | Parts |
| --- | --- |
| `cylinder`, `cone` | `{id}-body`, and the end disk `{id}-end0` at `p0` or `base`, or `{id}-end1` at `p1` |
| `arrow3d` | `{id}-shaft` and `{id}-head`, with their disks `{id}-shaft-end0` and `{id}-head-end0` |
| `tube` | one group per piece: `{id}-0`, `{id}-1`, … |

An end disk is drawn only when it faces the camera that renders, so which of
them exist depends on the camera.

### Highlights

`highlight="#ffffff"` shades a solid: a sphere with a radial gradient from the
highlight near its upper left to its `fill` at the rim, a cylinder, cone, or solid
arrow with a linear gradient across its width. It needs a `fill` colour to shade
toward, and raises `ValueError` at the call without one. This is a **fill style,
not a lighting model**
— the highlight sits in the same place on screen whatever the camera, and there
is no light direction, material, or shading per face.

Spheres of one colour pair share one gradient, `ball-{fill}-{highlight}`, so a
lattice of a thousand atoms in two colours adds two definitions. Every other
solid's gradient depends on its geometry, so it needs an `id` and is named after
the solid it shades: `{id}-shade` for a cylinder or cone, sliced or not — the
slices share one — and `{id}-shaft-shade` and `{id}-head-shade` for a solid
arrow's two parts.

### Long objects

```python
scene.cylinder(10, (-2, 0, 0), (2, 0, 0), 0.8, slices=20, id="core")
```

`slices` cuts a cylinder into lengths along its axis, each its own `<g>`.
[Depth sorting](#sorting-by-depth) does not need it — a coil wraps an unsliced
core exactly — so it is only for a solid wanted in separately selectable lengths.
The slices overlap a little and the outline is stroked along the sides only, so
the result still looks like one solid — except with a translucent fill, where
the overlaps show.

Sliced, the groups are `{id}-0`, `{id}-1`, …, each holding `{id}-{k}-body` and,
when there is a stroke, the outline `{id}-{k}-edge`; the end disks keep their
unsliced names, `{id}-end0` and `{id}-end1`, so slicing renames nothing a
consumer selects.

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

`separate=True` emits one `<path>` per edge, each with its own id, so edges can
be selected one by one. `trim` shortens each edge at both ends, to stop at the
surface of an atom on each corner; otherwise, in a sorted layer, the part of the
edge inside the atom counts as hidden and is drawn in the `back` style — a dash
across the atom. The faces must share
vertices exactly where they meet, as every solid from
[`shapes`](shapes.md) does.

`sphere_curve` draws a curve lying on a sphere — an equator, a meridian — split
exactly where it passes behind the sphere, with the same `back` and `back_layer`.
A point on the sphere faces the camera when `(p - center) · view ≥ 0`, and the
curve is cut where that changes sign. `closed=True` joins the last point back to
the first, as for a circle:

```python
scene.sphere(10, (0, 0, 0), 1.0, fill="#dfe8f3", fill_opacity=0.4)
equator = vecview.circle_shape((0, 0, 0), 1.0, normal=(0, 0, 1))
dashed = {"stroke_dasharray": "3 3"}
scene.sphere_curve(20, (0, 0, 0), equator, closed=True, back=dashed, back_layer=5, stroke="#333")
```

An `id` becomes `{id}-front` and `{id}-back` for both calls, suffixed `-0`, `-1`,
… per edge when `separate`. Both re-split for
[each camera](#rendering-one-scene-several-ways).

## Classes

An `id` names one object. A class names a *kind* of object, so a stylesheet, a
selector in the tool that composes the page, or Inkscape's *Select Same* can
reach all of them at once:

```python
for k, x in enumerate((-3, -1, 1, 3)):
    gate = vecview.box_faces((x, 0, 0), (1.2, 4, 0.2))
    scene.faces(10, gate, cull=True, fill="#9aa3ad", id=f"gate-{k}", class_="gate")
scene.sphere(20, (0, 0, 0.6), 0.3, fill="#c33", class_=["atom", "oxygen"])
```

Every drawing call that takes style keywords also takes `class_`: one string,
space-separated as in SVG (`"atom oxygen"`), or a sequence of names. `None` means
none. Repeated names are dropped, and anything other than strings raises
`TypeError` at the call, not at render.

The rule for where the classes land is the same for every call: **each
top-level element the call emits carries them, and nothing inside it does.**

| Call | Elements that carry the classes |
| --- | --- |
| `polygon`, `polyline`, `arrow`, `gaussian`, `sphere`, `silhouette`, `prism_walls`, `text`, `text2d`, `rect2d` | the one element |
| `faces` | every face polygon |
| `edges`, `sphere_curve` | the front path and the back path |
| `cylinder`, `cone`, `arrow3d` | the solid's `<g>`, not the body and end disks inside it |
| `cylinder(slices=n)`, `tube` | every slice or chunk `<g>` |
| `plane`, `slot` | the reserved group, around its content or whatever a consumer fills it with |
| a line split by an [depth-sorted layer](#sorting-by-depth) | both the visible and the hidden part |

So `.gate` selects four gates' faces, and `.atom` selects each sphere once,
however many parts a solid is drawn with. Elements handed to `add` keep whatever
classes you built them with, and classes survive
[rendering under another camera](#rendering-one-scene-several-ways).

## Parts

A `Part` is drawn once, in its own coordinates, and placed any number of times:
a unit cell tiled into a lattice, one gate turned into four quadrants, a lens
repeated along a bench.

```python
cell = vecview.Part()
cell.sphere(0, (0, 0, 0), 0.15, fill="#3b6fb6", highlight="#cfe0f7", id="atom", class_="atom")
cell.edges(1, vecview.box_faces((0.5, 0.5, 0.5), (1, 1, 1)), stroke="#222", id="edge")

scene = vecview.Scene(cam)
scene.sort_by_depth(10)
for i, j in itertools.product(range(3), repeat=2):
    scene.place(10, cell, at=(i, j, 0), id=f"cell-{i}{j}", class_="cell")
```

```python
scene.place(layer, part, *, at=(0, 0, 0), rotate=None, mirror=None, scale=1.0, id=None, class_=None)
```

A point `p` of the part lands at `at + scale * R @ M @ p`. `M` reflects through
the plane with normal `mirror`, and `R` turns by `rotate=(axis, angle_deg)`,
counter-clockwise looking down `axis`; both act about the part's origin, mirror
first. Only rigid motions and one uniform `scale` are offered, so a sphere stays
a sphere and every solid keeps its [exact outline](#curved-solids). `scale`
multiplies world lengths — positions, radii, arrow widths — and leaves screen
units alone: stroke widths, text size, and `dx`/`dy` offsets.

Under a mirror, faces keep their winding true to their outward normals, so
`cull=True` still picks the walls the camera sees. A mirrored helix turns the
other way, as a mirror image should.

**Layers.** The placement's `layer` is added to every layer the part draws on:
a part drawn on layers `0` and `1` and placed at `10` lands on `10` and `11`,
and an `edges` `back_layer` moves with it. Depth sorting stays the scene's
choice — a part has no `sort_by_depth` — so a lattice cell needs
`scene.sort_by_depth(10)` in the scene that places it. Sorting is a property of
the whole layer, and a part that switched it on would re-sort everything else
drawn there.

**Ids and classes.** `id=` is prefixed to every id in the part, derived ones
included, and placements nest: a cell placed as `a` in a row placed as `row`
gives `row-a-atom`. [Ids must be unique](#world-space-calls), so a part with ids
placed twice needs a different `id` for each placement; without one, rendering
raises `ValueError`. Objects without ids need no prefix. `class_=` is added to
the classes of everything the part draws, so `.cell` selects whole placements
while `.atom` still selects every atom.

**What a part holds.** The world-space calls, and `place` itself, so parts nest.
It has no camera, no `<defs>`, no screen-space calls, and no `sort_by_depth`:
each belongs to the document the part ends up in. `scene.place(other_scene)`
raises `TypeError` for the same reason. To look at a part on its own, place it
in a scene.

The copy is taken when `place` is called: drawing into the part afterwards
changes later placements, not earlier ones. Placed calls are recorded like any
other, so they re-render under [every camera](#rendering-one-scene-several-ways).

## Screen-space calls

```python
scene.rect2d(layer, x, y, w, h, grow=False, **style)
scene.text2d(layer, x, y, s, size=22, grow=True, **style)
```

For things that belong to the picture rather than the world: a backdrop, a
gradient wash behind a beam, a corner annotation.

Coordinates are SVG user units in the frame the camera projects into, so `x` and
`y` usually come from `cam.at(...)` or `bbox()`. That is caller-side camera math,
which another camera [does not redo](#rendering-one-scene-several-ways).

`rect2d` is **excluded from the bounding box by default** (`grow=False`) — that is
the point of it. A soft glow deliberately extends past the geometry, and letting
it inflate the fitted viewBox would leave a wide dead margin. Pass `grow=True`
when the rectangle really is part of the content. `text2d` grows the box by
default, as a label should, and takes `s` and the font defaults as `text` does.

## Bounds and the fitted viewBox

`render` fits the viewBox to everything added, plus `pad`. Nothing needs centring
by hand, and the document size follows the content:

```python
lo, hi = scene.bbox()  # screen-space min/max corners, under the active camera
lo, hi = scene.bbox("cabinet")  # or under another, by name or directly
scene.is_empty  # True until something is drawn
```

`bbox` projects the scene just as `render` does, but leaves out `pad`. Text
bounds are **estimated** from a nominal glyph width, not measured — real advance
widths would mean loading the font. The estimate is generous enough to keep a
label from being clipped, but do not treat `bbox` as exact where text is
involved.

Rendering raises `ValueError` when nothing grows the box: there is no content to
fit a viewBox to, and a zero-size document is never what was wanted. `is_empty`
says so in advance: it is true while the scene holds nothing that grows the box
— only settings, definitions, `add`, or `rect2d` and `text2d` with `grow=False`.

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
fenced in by them. Its coordinates are screen coordinates, like those of
`rect2d`, and in a sorted layer it has no depth, so it goes on top.

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
[`plane_matrix`](cameras.md#plane_matrix-flat-content-in-a-world-plane) for why
the transform is exact. It needs a parallel camera.

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
scene.slot(layer, point3, w, h, id="label-x", align="west", dx=1.6, dy=0.5, content=None)
```

The screen-aligned sibling of `plane`. Where a plane makes content lie *in* the
scene, a slot keeps it upright and unforeshortened — a TeX label, an inset —
while pinning it to a point of the geometry.

It reserves a **group** translated to the anchor, the projected point
offset by `(dx, dy)` in screen units, and records `align` as `data-align`.
`align` names the point of the content's box that sits on the anchor: `"west"`
puts the anchor at the middle of the box's left edge, so the content extends to
the right. The nine values are `"center"` (the default) and the eight compass
points; anything else raises `ValueError` at the call. `w`, `h`, `dx`, and `dy`
are screen units, and the offsets default to zero.

A `w` by `h` box, aligned the same way, grows the fitted viewBox so content of
that size is not clipped. Because the anchor is a world point,
[each camera](#rendering-one-scene-several-ways) moves the slot with the
geometry, which a hand-placed `rect2d` would not.

Left empty, the group is for a consumer to fill by `id`, lining its content up
against the anchor using `data-align` — see
[Embedding a scene](embedding.md#filling-a-slot). Given `content`, an svg.py
element `w` by `h` in its own coordinates from `(0, 0)` at its top-left, the
slot holds it with its box aligned on the anchor. VecView places the element as
it is given, without parsing, copying, or changing it, so the same element can
sit in any number of slots. A label typeset by
[VecTeX](https://github.com/maiani/vectex) comes with its size in px:

```python
label = vectex.render(r"$\gamma_1$", size_pt=18, color="#c0392b")
scene.slot(
    40,
    (x, 0, top),
    label.width_px,
    label.height_px,
    id="label-gamma-1",
    align="south",
    dy=-21,
    content=label.to_svg_py(),
)
```

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

`prism_walls(layer, footprint, z0, z1, **style)` culls the walls of an extruded
footprint with the camera that renders, merges each run of consecutive facing
walls into one strip — along the base, back along the top — and emits all strips
as one `<path>`. Draw the cap over it. It handles non-convex footprints, and
rejects one that is not a simple polygon at the call, as
[`prism_faces`](shapes.md#prisms) does.

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

`silhouette(layer, solid, **style)` is the older, simpler tool for a *convex*
solid given as faces or as world points: it fills the convex hull of the
projected vertices as one polygon, recomputed by each camera.

```python
fin = vecview.prism_faces(footprint, 0.0, 0.3)
scene.silhouette(30, fin, fill="#b98a40", id="lead-walls")
scene.faces(30, fin[:1], fill="#e2b56a", id="lead")
```

For a non-convex outline the hull is wrong; use `prism_walls`.

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
calls `to_svg_document`. The copy has its own record and its own dict of named
cameras, so drawing into either scene afterwards leaves the other alone.

Two things behave as their names promise rather than as a new camera might
suggest:

- **Screen-space calls stay put.** `rect2d`, `text2d`, and anything handed to
  `add` stay at the same screen coordinates, because that is what screen
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

## Animation

A scene is also one frame of an [animation](animation.md): an `Animation`
calls a function of time for a whole scene per sample, renders each exactly as
this page describes -- its active camera, its layers, its depth sorting -- and
writes them all into one SVG that plays itself.

## Output

```python
document = scene.render(camera=None, pad=None, background=None)  # an svg.SVG object
text = scene.to_svg_document()  # a complete SVG document string, active camera
path = scene.save("scene.svg", camera=None)  # writes it, returns the Path
```

`render` and `save` take a camera or a name, and default to the active one;
`render`'s `pad` and `background` default to the scene's. `to_svg_document`
takes nothing, which is the point of it: a consumer calls it without knowing
anything about the scene. Rendering does not consume the scene — call it as often as you like, and keep
adding afterwards.

In a Jupyter notebook a scene displays itself: it implements `_repr_svg_`, which
returns the document, or nothing while the scene has no active camera or nothing
drawn.

Only SVG is written. PNG and PDF export are left to the consumer, which is what
keeps rasterizers out of the runtime dependencies. Run `cairosvg` over the
file, or hand the document to whatever assembles the final page — see
[Embedding a scene](embedding.md).
