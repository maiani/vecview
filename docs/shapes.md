# Shapes

Every function here returns plain world-space points — an `(n, 3)` array, or a
list of `Face` for solids. Nothing in this module knows about a camera, a style,
or SVG.

That is worth stating plainly because it is what makes the geometry testable as
numbers, and it means one polygon can be drawn twice with different fills, offset
and drawn again as a shadow, or handed to `Scene.add` inside a clip path.

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
    name: str  # "+x", "-x", "+y", "-y", "+z", "-z"
    points: Array  # (4, 3), wound CCW about the normal
    normal: Array  # outward unit normal
```

Carrying the normal alongside the points is what lets
[`Camera.visible`](cameras.md#back-face-culling) cull back
faces without the caller reasoning about which octant the camera sits in, and
`name` is what lets you style the top differently from the sides:

```python
slab = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
walls = cam.visible(slab)
scene.faces(10, [f for f in walls if f.name == "+z"], fill="#eef1f5", fill_opacity=0.86)
scene.faces(11, [f for f in walls if f.name != "+z"], fill="#cfd6e0")
```

A zero-thickness box is legitimate — a bare plane you want to draw with the box
machinery — and still returns six faces.

## Circles

```python
vecview.circle_shape(center, radius, normal, n=64)
```

A circle in the plane through `center`, as an `n`-gon. There is no duplicated
closing point, so `Scene.polygon` closes it cleanly and `n` is exactly the vertex
count. The in-plane rotation is arbitrary but deterministic, which is what matters
for reproducible output.

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

A `head_len` longer than the whole arrow is clamped rather than producing a
self-crossing polygon.

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
