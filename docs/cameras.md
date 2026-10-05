# Cameras

Every camera here is a **parallel projection**: the rays are mutually parallel, so
parallel world edges stay parallel on screen and no perspective distortion creeps
into a lattice or a repeated texture. That single property is what the rest of the
package leans on.

## The hierarchy

```
Camera                    the projection contract, and nothing more
├── ParallelCamera        affine: screen offsets are position-independent
│   ├── OrthographicCamera    axonometric, aimed by azimuth and elevation
│   └── ObliqueCamera         cavalier and cabinet
└── PerspectiveCamera     not implemented; would subclass Camera directly
```

`Camera` is abstract and promises only four things: `project`, `at`, `depth`, and
`visible`. Everything else lives on `ParallelCamera`, and the split is load-bearing
rather than tidy-minded. `direction`, `screen_basis`, `foreshortening`, and
`plane_matrix` all assume that a screen offset does not depend on *where* in the
scene you are — true for a parallel projection, false for a perspective one. A
future `PerspectiveCamera` would therefore hang off `Camera` directly, and keeping
those four one level down is what stops them from becoming a silent wrong answer.

`visible` is abstract for the same reason: a parallel camera decides from a face's
outward normal alone, while a perspective camera has to ask where the face *is*.

## Which projection to choose

| Projection | Foreshortening | Character |
| --- | --- | --- |
| **Isometric** | all three axes equal, `0.8165` | Symmetric and neutral. No axis is privileged, which is either fair or bland depending on the picture. |
| **Dimetric** | two equal, third by `ratio` | The workhorse. `ratio=0.5` reads as a natural "looking down at it" view. |
| **Trimetric** | all three differ | The general case, and what you get from arbitrary angles. Most freedom, least convention. |
| **Cavalier** | front face true, depth `1.0` | Every edge measures true, which is convenient and looks too deep. |
| **Cabinet** | front face true, depth `0.5` | Halving the depth is what makes cabinet look right where cavalier looks stretched. |

`foreshortening()` reports the three ratios for any camera, and
`OrthographicCamera.axonometry()` names the class it falls into. Both read the
projection itself rather than how it was constructed:

```python
OrthographicCamera(45, 35.264, 62).axonometry()  # "isometric"
OrthographicCamera(35, 24, 62).foreshortening()  # (0.663, 0.852, 0.914)
```

## OrthographicCamera

```python
OrthographicCamera(azim_deg, elev_deg, scale, origin=(0, 0, 0))
```

`azim_deg` and `elev_deg` give the direction the camera looks *from*, `scale` is
SVG user units per world unit, and `origin` is the world point that lands on the
SVG origin. Because `right` and `up` are genuinely orthonormal here, this class
exposes them:

| Attribute | Meaning |
| --- | --- |
| `cam.right` | World direction projecting to screen `+x` |
| `cam.up` | World direction projecting to screen *up* (SVG `−y`) |
| `cam.view` | Unit vector from the scene **toward** the camera |

### Isometric

```python
OrthographicCamera.isometric(scale, azim_deg=45)
```

All three axes foreshorten to `0.8165`, and the two horizontal axes land at
exactly 30° below the horizon — which is why a 30-60 set square draws an
isometric view.

**Both angles are forced, not just the elevation.** Requiring the horizontal axes
to share a ratio gives

$$(\sin^2 a - \cos^2 a)(1 - \sin^2 e) = 0,$$

so the azimuth must be an odd multiple of 45°. The four such azimuths are all
isometric and differ only in which octant you view from; anything else raises.

Draftsmen usually scale an "isometric drawing" back up by `1 / 0.8165` so the axes
measure true length. Pass a correspondingly larger `scale` if you want that.

### Dimetric

```python
OrthographicCamera.dimetric(scale, ratio=0.5)
```

The azimuth is fixed at 45°, which is what makes `x` and `y` share a ratio at any
elevation; `ratio` — the vertical foreshortening as a fraction of the horizontal
one — then picks the elevation:

$$\sin^2 e = \frac{2 - r^2}{2 + r^2}.$$

`ratio=1` is isometric, and the default `0.5` is the drafting standard `1:1:0.5`,
putting the receding axes 41.4° below the horizon. `ratio` must lie in
`(0, √2]`; at `√2` the elevation reaches zero and the ground plane is edge-on.

### Choosing angles by hand

Elevation is the real trade-off in a slab-and-beam schematic:

- **Too low** (~24° and below) and the slab plane is so foreshortened that
  distinct in-plane directions collapse to similar shallow shapes. If the
  picture's argument *is* an in-plane angle, that destroys it.
- **Too high** (~40° and above) and the slab reads as a flat plan view, with no
  visible underside to hang anything below.

If you need to go lower than the plane can carry, move the in-plane information
into face-on 2D insets rather than fighting the projection.

## ObliqueCamera

```python
ObliqueCamera.cavalier(scale, angle_deg=45)
ObliqueCamera.cabinet(scale, angle_deg=45)
ObliqueCamera(scale, depth_ratio=0.5, angle_deg=45)
```

The `xz` plane is drawn at true shape and true angle — world `x` along screen
`+x`, world `z` straight up — while world `y` recedes at `angle_deg` above the
horizon, foreshortened by `depth_ratio`.

Oblique is **not** axonometric: its rays are not perpendicular to the projection
plane, so no choice of azimuth and elevation reproduces it. It is still parallel,
so every guarantee this package rests on continues to hold.

## Projecting

```python
cam.project(points)  # (n, 3) world -> (n, 2) SVG user units
cam.at(point)  # one point -> (x, y) floats
cam.direction(vector)  # a displacement -> a screen displacement
cam.depth(points)  # signed distance along the view axis; larger is nearer
```

`project` subtracts `origin`; `direction` does not, because a displacement has no
position and moving the camera must not change it. Reach for `direction` when
asking *how far does this move on screen*, and `project` when asking *where does
this land*.

## `screen_basis` — placing things inside a projected plane

At an azimuth of 35°, neither `+x` nor `+y` moves a point horizontally across the
picture, so nudging something "a bit to the right and forward" by eye is guesswork.
`screen_basis` returns the two unit world vectors in a plane that *do* move a
point purely right and purely down on screen:

```python
horizontal, down = cam.screen_basis()  # in the z = 0 plane
horizontal, down = cam.screen_basis(normal=(0, 1, 0))  # in some other plane

spots = [-3.0 * horizontal + 1.7 * down, +3.0 * horizontal + 1.7 * down]
```

Both vectors lie in the plane and have unit length, so the coefficients are in
world units — usually what you want, since the geometry they position is in world
units too.

A plane seen exactly edge-on projects to a line and has no such basis; asking for
one raises `ValueError` rather than returning something arbitrary.

!!! warning "Not the right choice for embedding content"

    Because these edges project to *pure* screen axes, using them for
    [`Scene.plane`](scenes.md#embedding-flat-content) gives an upright screen
    rectangle — in the plane geometrically, yet visually pasted on, with no
    foreshortening cue. See [Embedding a scene](embedding.md#orientation).

## Back-face culling

`box_faces` returns all six faces of a box. `visible` keeps the ones whose outward
normals point at the camera:

```python
slab = vecview.box_faces(center=(0, 0, -0.45), size=(11, 9, 0.9))
scene.faces(10, cam.visible(slab), fill="#cfd6e0")
```

For a convex solid this is exactly right, and always leaves three walls of a box.
It replaces the kind of comment that goes stale the moment the azimuth moves —
*"the camera sits in the (+x, +y) octant, so these two walls are the visible
ones"*.

What it does **not** do is resolve one object occluding another. That is the
[layer stack's](scenes.md) job, deliberately.

!!! tip "Prefer `cull=True` in a scene"

    Culling here bakes in *this* camera's answer. A cabinet camera sees `-y`
    where a 35° orthographic one sees `+y`, so
    [rendering with another camera](scenes.md#rendering-one-scene-several-ways)
    would keep the original walls and quietly draw the wrong ones. Pass the full
    set to `Scene.faces(..., cull=True)` instead and let each camera decide when
    it renders.

`faces_camera(normal)` is the single-normal form, and takes a `tol` above `0` to
also drop faces seen so nearly edge-on that they project to slivers.
