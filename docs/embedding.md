# Embedding a scene

A scene is usually not the whole figure. `vecview` exposes one method for handing
its output to whatever assembles the final page:

```python
document = scene.to_svg_document()  # a complete, standalone SVG document string
```

That is the entire contract. Any tool that accepts an object with a
`to_svg_document()` method can place a scene without `vecview` knowing anything
about it, and without `vecview` gaining a dependency.

## What a consumer needs from the document

The document is standalone and self-describing:

- a root `<svg>` with `width`, `height`, and a `viewBox`
- `width`/`height` always agree with the `viewBox` extents, so scaling by either
  gives the same result
- every `<defs>` element the content references, inline
- no external references, fonts, or scripts

## Honour the viewBox origin

A fitted viewBox generally does **not** start at `0, 0`. `vecview` fits the box to
the content, so `min_x` and `min_y` are usually negative:

```xml
<svg width="904.0" height="808.0" viewBox="-430.1 -184.7 904.0 808.0">
```

A consumer placing the scene must compensate, or the drawing lands outside the
target box by roughly half its size:

```python
min_x, min_y, width, height = (float(v) for v in root.get("viewBox").split())
scale = min(box_w / width, box_h / height)
transform = f"translate({dx:g} {dy:g}) scale({scale:g}) translate({-min_x:g} {-min_y:g})"
```

Scale **uniformly**. A per-axis scale skews the projection and breaks the
parallel-edge guarantee that made an orthographic camera worth having.

## Orientation

Which world directions to use for the content's axes is a genuine choice, and the
failure mode is mirrored or upside-down text.

**Content stays upright and unmirrored exactly when `a > 0` and `d > 0`** — its
`+x` must project rightward and its `+y` downward:

```python
a, b, c, d, e, f = cam.plane_matrix(origin, u_edge, v_edge)
assert a > 0 and d > 0
```

A positive determinant is **not** sufficient. `ad − bc > 0` rules out mirroring,
but a 180° rotation has a positive determinant too, which is the easy way to get a
plot with every label upside down.

Note that world axes are not screen axes: at an azimuth of 35°, world `+x`
projects to screen-*left* (`cam.right = [-0.57, 0.82, 0]`), so using it for content
`+x` inverts the content. At that azimuth world `+y` projects rightward and world
`+x` downward, so those are the axes to use.

`Camera.screen_basis()` also satisfies the rule, but it is usually the wrong tool
here: its edges project to *pure* screen axes by construction, so the content lands
as an upright screen rectangle — in the plane geometrically, yet visually pasted on,
with no foreshortening cue. Use it to *place* things in a plane, not to embed
content in one.

## Filling a slot

[`Scene.slot`](scenes.md#anchoring-upright-content) reserves an empty group for
upright content pinned to a world point. What a consumer finds:

- `<g id="..." transform="translate(x y)" data-align="west"/>` — the translation
  is the **anchor**, not a corner of the box.
- `data-align` names the point of the content's box to put on the anchor:
  `center`, or a compass point from `north` round to `northwest`.

Align against the anchor at the content's *final* size: offset the content by
`-fx * width, -fy * height`, where `(fx, fy)` is `(0, 0.5)` for `west`,
`(0.5, 0.5)` for `center`, `(1, 1)` for `southeast`, and so on.

The `w` by `h` box passed to `slot` only reserves room in the fitted viewBox, in
scene units. If the consumer scales the scene but keeps the content at its own
size — usually right for a label set in the document's font size — the reserved
room matches the content exactly only when the scene is placed at 1:1.

## Settings that matter when embedding

`to_svg_document()` takes no arguments, so a consumer cannot pass render options.
Configure them on the constructor rather than at a `render` call that will never
happen:

```python
scene = vecview.Scene(cam, pad=6, background=None)
```

- **`pad`** is in scene units and survives into the target box as margin. Since
  the scene gets scaled to fit, padding shrinks the drawing within its box. Keep
  it small, or zero, when the surrounding layout already provides spacing.
- **`background`** is best left `None` when the scene is one part of a larger
  page. An opaque rectangle covers whatever the scene overhangs, and defeats a
  figure meant to sit on a coloured ground.
- **`scale`** does not affect fit — a consumer normalizes it away. It does set
  stroke widths and font sizes *relative* to the geometry, so keep it consistent
  across scenes sharing a page, or one will come out visibly heavier.

## Ids survive

Ids assigned while building a scene are emitted verbatim, so they stay available
for selection or restyling after placement:

```python
scene.polygon(20, marker, id="absorption-axis", fill="#d62828")
```

Give `<defs>` elements distinct ids across scenes that will share a page. A
consumer that hoists definitions into one shared `<defs>` may not namespace them,
and two scenes both defining `#glow` leave `url(#glow)` resolving to whichever
landed first — a silent wrong colour rather than an error.

## Export

`vecview` writes SVG only. Rasterizing and PDF are left to the consumer, which is
what holds the runtime dependencies to `numpy` and `svg.py`. For a standalone
scene, run a converter over the file yourself:

```python
import cairosvg

path = scene.save("scene.svg")
cairosvg.svg2png(url=str(path), write_to="scene.png", scale=2.0)
```

One caveat worth knowing before designing around it: a gradient-filled `<mask>`
does not survive `cairosvg` rasterization — the effect vanishes silently, with no
warning. Use a gradient *fill* on a plain rectangle instead.
