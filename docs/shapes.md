# Shapes

Every function here returns plain world-space points — an `(n, 3)` array, or a
list of `Face` for solids. Nothing in this module knows about a camera, a style,
or SVG.

That is worth stating plainly because it is what makes the geometry testable as
numbers, and it means one polygon can be drawn twice with different fills, or
offset and drawn again as a shadow. Draw an array with `Scene.polygon` or
`Scene.polyline`, and a list of faces with `Scene.faces`:

```python
ring = vecview.circle_shape((0, 0, 0), 1.5, normal=(0, 0, 1))
scene.polygon(20, ring, fill="none", stroke="#333")
```

## Directions

```python
vecview.unit(v)  # normalize; a zero vector passes through
vecview.in_plane_dir(angle_deg, u=(1, 0, 0), v=(0, 1, 0))
```

`in_plane_dir` gives a unit direction at `angle_deg` from `u`, rotating toward
`v`. The default axes cover the common case — an angle measured in the `xy` plane
from `+x`, as crystal and polarization angles usually are:

```python
q = vecview.in_plane_dir(22.5)  # 22.5 degrees off +x, in-plane
z_tilt = vecview.in_plane_dir(30, u=(1, 0, 0), v=(0, 0, 1))  # tilted out of plane
```

`unit` returns a zero vector unchanged rather than raising. Degenerate directions
turn up naturally in generated geometry — a wave with zero amplitude, a box with
zero thickness — and a silent zero draws as nothing, which is the intended result.

## Rectangles and boxes

```python
vecview.rect_shape(center, u, v, du, dv)
vecview.box_faces(center, size)  ->  list[Face]
```

`rect_shape` is a rectangle centred on `center` in the plane spanned by `u` and
`v`, with full extents `du` and `dv`. Both axes are normalized for you, so passing
an unnormalized direction does not silently stretch the rectangle.

`box_faces` returns all six faces of an axis-aligned box, each wound
counter-clockwise about its outward normal. **`size` is the full extent, not the
half-extent.**

```python
class Face(NamedTuple):
    name: str  # a box's are "+x", "-x", "+y", "-y", "+z", "-z", in that order
    points: Array  # (n, 3), wound CCW about the normal; (4, 3) for a box
    normal: Array  # outward unit normal
```

Every function below that returns solids returns `Face`s. Carrying the normal
alongside the points is what lets back-face culling —
[`Camera.visible`](cameras.md#back-face-culling), or `Scene.faces(..., cull=True)`
— work without the caller reasoning about which octant the camera sits in, and
`name` is what lets you style the top differently from the sides:

```python
slab = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
top = [f for f in slab if f.name == "+z"]
walls = [f for f in slab if f.name != "+z"]
scene.faces(10, walls, cull=True, fill="#cfd6e0")
scene.faces(11, top, cull=True, fill="#eef1f5", fill_opacity=0.86)
```

A zero-thickness box is legitimate — a bare plane you want to draw with the box
machinery — and still returns six faces.

## Prisms

```python
vecview.prism_faces(footprint, z0, z1)  ->  list[Face]
```

A footprint in the `xy` plane, extruded from `z0` to `z1` — a tapered electrode,
a hexagonal pillar, an arc-shaped gate. The faces come in a fixed order: the cap
`"+z"`, the base `"-z"`, as for a box, then the walls, wall `i` spanning
footprint vertices `i` and `i + 1` and named `"side-{i}"`. So `faces[0]` is the
top, and `faces[2:]` the walls. Every face is wound counter-clockwise about its
outward normal, so `faces(..., cull=True)` works. `z1` must exceed `z0`.

The footprint may come in either winding and may be non-convex, but it must be
a **simple polygon**: an outline that crosses or touches itself has no
well-defined outside, and raises `ValueError`. Collinear vertices are fine;
repeated ones are not. A clockwise footprint is reversed before the walls are
numbered, so for one, wall `i` spans vertices `i` and `i + 1` of the *reversed*
outline; [`extrude`](#extrusions) numbers walls by the outline as given.

For a convex footprint, culling leaves exactly the visible walls. For a
non-convex one it leaves the walls that *face* the camera, and one of those can
still sit behind another wall of the same solid.

Either way, walls drawn one polygon each show hairline seams where their
anti-aliased edges meet. Draw them with
[`Scene.prism_walls`](scenes.md#seamless-solids) and the cap over them instead.

### Extrusions

```python
vecview.extrude(section, along)  ->  list[Face]
```

A planar cross-section in any plane, swept along any vector: a nanowire along
`x`, a waveguide, a fin. `prism_faces` is the special case of a section in a
horizontal plane swept up `z`. The section is the `"start"` face, the section
moved by `along` is the `"end"` face, and wall `i` spans section vertices `i`
and `i + 1` **as given**, whichever way the section winds, and is named
`"side-{i}"`. So a wall can be picked out by the edge it was built on, and its
id — `wire-side-3` — keeps naming that edge.

```python
hexagon = vecview.circle_shape((-4, 0, 0.4), 0.4, (1, 0, 0), n=6)
wire = vecview.extrude(hexagon, (8, 0, 0))
scene.faces(20, wire, cull=True, fill="#7fae8c", id="wire", class_="semiconductor")
top_facets = [f for f in wire[2:] if f.normal[2] > 0.1]
```

Every face is wound counter-clockwise about its outward normal, so it culls
like a box. `along` need not be perpendicular to the section, which gives a
slanted prism, but it must leave the section's plane. The section must be a
simple polygon, convex or not, lying in one plane; anything else raises
`ValueError`. A regular cross-section is `circle_shape` with a small `n`, as
above. To build a section in 2D first — a wire with a film on some facets, a
gate with a notch — see [Outlines](outlines.md), whose `to_plane` lifts it into
the plane to sweep.

```python
vecview.annulus_sector(center, r_in, r_out, theta0_deg, theta1_deg, n=32)
```

The footprint of an annular sector about the 2D point `center`, as an `(m, 2)`
array, counter-clockwise: the outer arc from
`theta0_deg` to `theta1_deg` in `n` segments, then the inner arc back. With
`r_in=0` it is a pie wedge. The span must lie strictly between 0° and 360°, since
a full ring is not a simple polygon. Join it to other outlines yourself to build
a compound footprint.

## Circles

```python
vecview.circle_shape(center, radius, normal, n=64)
```

A circle in the plane through `center`, as an `n`-gon. There is no duplicated
closing point, so `Scene.polygon` closes it cleanly and `n` is exactly the vertex
count. The in-plane rotation is arbitrary but deterministic, which is what matters
for reproducible output.

```python
vecview.ellipse_shape(center, u, v, a, b, n=64)
```

The generalization: semi-axis `a` along `u` and `b` along `v`, which must be
perpendicular. The first vertex is `center + a * u`, and the points run
counter-clockwise about `u × v`, again with no duplicated closing point.

## Arcs and helices

```python
vecview.arc_shape(center, u, v, radius, theta0_deg, theta1_deg, n=32)
vecview.helix(start, axis, radius, pitch, turns, n_per_turn=48, phase_deg=0.0)
```

`arc_shape` is a circular arc in the plane of `u` and `v`, with angles measured
from `u` toward `v`. `v` need not be perpendicular to `u`, only not parallel, so
the arc marking the angle between two vectors is
`arc_shape(origin, a, b, r, 0, angle_between)`.

`helix` winds about `axis` from `start`, which is on the axis, advancing `pitch`
along it per turn: right-handed for a positive pitch, left-handed for a negative
one. Draw it with [`Scene.tube`](scenes.md#curved-solids) for a coil, or
`polyline` for a spin spiral's envelope.

## Arrows

```python
vecview.arrow_shape(origin, direction, length, normal, shaft_w, head_w, head_len, pivot="tail")
vecview.double_arrow_shape(center, direction, length, normal, shaft_w, head_w, head_len)
```

Both are **flat polygons**, not strokes — they lie in the plane through their
anchor point with the given `normal`, and so foreshorten with the geometry they
label. That is the reason to prefer them over a stroked line with a marker: an
in-plane arrow on a slab face should compress as the slab does.

`length` is measured tail-to-tip, and `head_len` back from the tip. `pivot="mid"`
centres the arrow on `origin` instead of starting there — the natural choice for a
texture of spins, where the anchor is a lattice site:

```python
scene.polygon(
    15,
    vecview.arrow_shape(
        (x, y, 0.01),
        (u, v, 0.0),
        0.64,
        normal=(0, 0, 1),
        shaft_w=0.08,
        head_w=0.25,
        head_len=0.25,
        pivot="mid",
    ),
    fill="#2f6fb0",
)
```

`double_arrow_shape` is the axis or polarization marker: two heads, symmetric
about `center`, no direction implied.

A `head_len` too long for the arrow is clamped rather than producing a
self-crossing polygon: to the whole arrow in `arrow_shape`, and to half of it in
`double_arrow_shape`, where the two heads then meet at the centre.

## Surfaces

```python
vecview.surface_faces(x, y, z)  ->  list[Face]
```

The quads of a parametric surface sampled on a grid: `x`, `y` and `z` are 2D
arrays of one shape, from `np.meshgrid` for a height field or from any
parametrization. Quad `(i, j)` is named `"q-{i}-{j}"` and wound
counter-clockwise about its normal, `∂/∂i × ∂/∂j` — upward for a height field
from `meshgrid(..., indexing="ij")`. A polar grid gives a cone or a band surface:

```python
k, t = np.meshgrid(np.linspace(0, 1, 13), np.linspace(0, 2 * np.pi, 57), indexing="ij")
band = vecview.surface_faces(k * np.cos(t), k * np.sin(t), np.sqrt(k**2 + 0.25**2))
scene.sort_by_depth(10)
for face in band:
    scene.polygon(10, face.points, fill=colour(face), stroke=colour(face), stroke_width=0.4)
```

A surface is two-sided, so do not cull it; sort it, and use the normal to colour
a fold that turns toward the camera. A stroke in the fill colour hides the
hairline seams between quads.

## Polyhedra

```python
vecview.convex_polyhedron(vertices, tol=1e-9)  ->  list[Face]
vecview.trim_corners(faces, radius, n=8)  ->  list[Face]
```

`convex_polyhedron` returns the faces of the convex hull of `vertices`, each wound
counter-clockwise about its outward normal and named `"face-{k}"` in a
deterministic order. Coplanar vertices merge into one polygonal face, so the
truncated octahedron of an fcc Brillouin zone comes out as eight hexagons and six
squares rather than triangles. Interior points are ignored. The search tests
every vertex triple against every vertex — instant for the tens of vertices of a
zone or a coordination polyhedron, and not meant for a mesh of thousands.

`trim_corners` cuts a disk of `radius` out of every corner of every face, for a
polyhedron with an atom on each vertex whose faces must themselves stop at the
atoms. A [depth-sorted layer](scenes.md#sorting-by-depth) does not need it: it
clips each face exactly where an atom hides it. Pair it with
`Scene.edges(trim=radius)`:

```python
octahedron = vecview.convex_polyhedron(oxygens)
scene.faces(10, vecview.trim_corners(octahedron, r_o), fill="#6f9fd8", fill_opacity=0.35)
scene.edges(10, octahedron, trim=r_o, stroke="#2d5c94")
```

## Cutaways

```python
vecview.cut(faces, origin, normal)  ->  list[Face]
```

A closed solid cut open by a plane, to show what is inside: a nanowire's
hexagonal core within its shell, a heterostructure's layers from within. The
plane passes through `origin`, and everything on the side `normal` points to is
cut away. Each remaining face is clipped and keeps its name, so ids stay stable;
the cut is capped by faces lying in the plane and facing along `normal`, named
`"cut"` — or `"cut-0"`, `"cut-1"`, … where the plane crosses the solid in
several places — so a cap can take a style of its own:

```python
wire = vecview.extrude(vecview.circle_shape((-2, 0, 0.4), 0.4, (1, 0, 0), n=6), (4, 0, 0))
opened = vecview.cut(wire, origin=(1, 0, 0), normal=(1, 0, -0.3))
body = [f for f in opened if not f.name.startswith("cut")]
section = [f for f in opened if f.name.startswith("cut")]
scene.faces(10, body, cull=True, fill="#7fae8c", id="wire")
scene.faces(10, section, cull=True, fill="#cfe6d5", id="wire-section")
```

The faces must close a solid, each wound counter-clockwise about its outward
normal — as `box_faces`, `prism_faces`, `extrude`, and `convex_polyhedron` return
them — and the solid may be non-convex. Curved solids drawn by the scene (spheres,
cylinders) are not faces and cannot be cut. A cut whose cap would have a hole in
it, such as a plane across a hollow tube, raises `ValueError`.

## Waves

```python
vecview.sine_ribbon(start, axis, length, transverse, amplitude, wavelength, n=400, phase=0.0)
```

A transverse wave running `length` along `axis` from `start`, displaced along
`transverse`. Draw it with `Scene.polyline`.

`amplitude` may be a scalar or an array of length `n`, which is how a component
absorbed inside a medium is drawn — pass its envelope:

```python
t = np.linspace(0.0, total, n)
depth = np.clip(-(z_top - t), 0.0, thickness)
envelope = a0 * np.exp(-depth / 0.30)  # decays only inside the slab
pts = vecview.sine_ribbon(start, axis, total, e_abs, envelope, wavelength=1.7, n=n)
```

### Keeping a pair of waves legible

Two components of the same beam are each an exact sine, but the *pair* has to stay
visually separable. If their projected screen amplitudes come out near-equal with
opposite sign, the traces cross at every node and chain into lens shapes instead
of reading as two waves. Using the physically correct component amplitudes
usually avoids this; if it does not, change the incoming polarization angle rather
than fudging the amplitudes.
