# Outlines

A device is mostly layers with a cross-section: a wire with a film on three of
its facets, a contact draped over it, a gate with an opening. `vecview.outlines`
builds those cross-sections in 2D — where offsetting and combining shapes is
easy to think about — and [`extrude`](shapes.md#extrusions) turns them into
solids.

```python
import vecview
from vecview import outlines

wire = outlines.regular(6, 0.4)  # a hexagonal nanowire, flat side down
al = outlines.film(wire, [0, 1, 2], thickness=0.07)  # Al on its three top facets

# The sections lie across a wire running along x, resting on z = 0.
across = dict(origin=(-1, 0, -wire[:, 1].min()), u=(0, 1, 0), v=(0, 0, 1))
cam = vecview.OrthographicCamera(azim_deg=-60, elev_deg=30, scale=120)
scene = vecview.Scene(cam, background="#ffffff")
scene.sort_by_depth(10)
for name, section, fill in (("wire", wire, "#7fae8c"), ("al", al, "#9bb8e0")):
    solid = vecview.extrude(outlines.to_plane(section, **across), (2, 0, 0))
    scene.faces(10, solid, cull=True, fill=fill, stroke="#33475b", stroke_width=0.8, id=name)
```

Like [shapes](shapes.md), these are numbers only: every function takes and
returns `(n, 2)` arrays, knows no camera or style, and returns outlines wound
counter-clockwise. `shapely` does the clipping.

## Building

```python
outlines.rect(lo, hi)  ->  (4, 2) array
outlines.regular(n, radius, *, center=(0, 0), rotate_deg=0)  ->  (n, 2) array
```

`rect` is the axis-aligned rectangle with corners `lo` and `hi`. `regular` is a
regular `n`-gon of circumradius `radius`; vertex `k` sits at angle
`rotate_deg + 360 k / n`, so edge `k` runs from vertex `k` to `k + 1`. With no
rotation a hexagon has a vertex on `+x` and flat edges top and bottom — edge `1`
on top, edge `4` underneath — which is how a wire lies on a substrate.

## Growing

```python
outlines.offset(outline, distance, *, join="mitre")  ->  array
outlines.film(outline, edges, thickness, *, join="mitre")  ->  array
```

`offset` grows an outline outward by `distance`, or shrinks it for a negative
one. `join` is how the offset meets itself at a corner: `"mitre"` keeps it
sharp, `"round"` rounds it, `"bevel"` cuts it straight. Shrinking an outline so
far that it vanishes or splits raises `ValueError`.

`film` is a layer of `thickness` deposited on some edges of a body: the Al shell
on a nanowire, an oxide on a mesa's side walls. Edge `i` runs from vertex `i` to
`i + 1` as given — the same numbering as the walls of
[`extrude`](shapes.md#extrusions), so the facet a film covers and the wall it
lies on carry the same index. The edges must form one unbroken run, which may
wrap past the last vertex (`[5, 0, 1]`). The film lies outside the body
whichever way the outline winds, its ends are cut square to the first and last
edge, and it never reaches into the body, so a film on a concave run stays out of
the notch.

## Combining

```python
outlines.union(*outlines)  ->  list of arrays
outlines.difference(outline, *cuts)  ->  list of arrays
outlines.intersection(*outlines)  ->  list of arrays
```

```python
plate = outlines.rect((-2, -1), (2, 1))
(gate,) = outlines.difference(plate, outlines.rect((-0.3, 0), (0.3, 2)))  # a notch for the wire
```

Each returns a list, because combining can split a shape: a cut through the
middle of a gate leaves two pieces, and a cut that removes everything leaves
none. The list is ordered largest first, then by position, so it never depends
on the clipping library's internals. Unpack it when you know how many pieces to
expect — `(gate,) = outlines.difference(...)` fails loudly if the notch cut the
plate in two.

An outline is a simple polygon, so it cannot hold a hole. A result with one — a
square with a smaller square cut out of its middle — raises `ValueError` rather
than silently dropping the hole. An outline that crosses or touches itself raises
`ValueError` too.

## Lifting into 3D

```python
outlines.to_plane(outline, origin=(0, 0, 0), u=(1, 0, 0), v=(0, 1, 0))  ->  (n, 3) array
```

The point `(x, y)` lands on `origin + x u + y v`, with `u` and `v` used as given:
for a section across a wire that runs along `x`, `u = (0, 1, 0)` and
`v = (0, 0, 1)`. The result is what [`extrude`](shapes.md#extrusions) sweeps and
what `Scene.polygon` draws. Parallel `u` and `v` raise `ValueError`.
