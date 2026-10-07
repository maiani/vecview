# Animation

```python
Animation(frame, *, duration, view_box, fps=30.0, repeat=1, background=None)
```

An `Animation` is a function from seconds to a whole [`Scene`](scenes.md). It
samples that function at evenly spaced times, renders every sample with the same
renderer a still figure uses, and writes one standalone SVG that plays itself
with native SVG timing: no script, no video, every frame still vector. Anything a
scene can say can change from frame to frame — positions, shapes, styles, labels,
which objects exist, which layers are depth-sorted, the camera.

```python
from math import cos, sin

import vecview

CAMERA = vecview.OrthographicCamera(35, 24, 62)


def frame(t: float) -> vecview.Scene:
    scene = vecview.Scene(CAMERA)
    scene.sphere(0, (0, 0, 0), 0.6, id="sun", fill="#e9a23b")
    scene.sphere(1, (2 * cos(t), 2 * sin(t), 0), 0.25, id="planet", fill="#3b6fb6")
    return scene


animation = vecview.Animation(frame, duration=6.283, view_box=(-150, -110, 300, 220))
animation.save("orbit.svg")
animation.frame(1.5).save("still.svg")  # any moment, as an ordinary still
```

| Call | Returns |
| --- | --- |
| `animation.frame(t)` | The `Scene` at `t` seconds, for `0 <= t <= duration`. |
| `animation.render()` | The animated document as an `svg.SVG` tree. |
| `animation.to_svg_document()` | The same document as a string — the embedding contract. |
| `animation.save(path)` | Writes it as UTF-8 and returns the `Path`. |

The callback must be **pure**: the same `t` gives the same scene, whatever was
sampled before and in whatever order. Build a fresh scene each time — sharing
immutable geometry, parts, and rendered labels between frames is fine — and
compute the state at `t` rather than stepping it: a simulation is read at `t`,
not advanced. Ordinary Python describes the motion, so there is no animation
object model to learn; [tracks](#tracks-and-easing) and [part
helpers](#moving-parts) are conveniences on top.

`frame(t)` is the plain callback result, with the scene's own camera, padding,
and background; a time outside `[0, duration]` raises `ValueError`, and a
callback that returns anything but a `Scene` raises `TypeError`. Every frame
needs an active camera when the animation renders.
[`examples/pendulum.py`](https://github.com/maiani/vecview/blob/main/examples/pendulum.py)
is a complete example: a pendulum swinging through one seamless period, with
its equations typeset by TeX.

## Timing and playback

- **Sampling.** Each cycle of `duration` seconds is cut into
  `N = max(1, ceil(duration * fps))` equal intervals, sampled at
  `t_i = i * duration / N` and shown on `[t_i, t_(i+1))` with no tweening in
  between. `fps` is the least sample rate asked for; the rate used is
  `N / duration`.
- **Repeating.** `repeat` is the total number of cycles, `1` by default, or
  `None` to loop forever. Finite playback also samples the exact frame at
  `duration` and holds it once the last cycle ends; a loop never shows it. A
  loop only looks seamless if its first and last frames match — the pendulum
  runs exactly one period — since repetition never rewrites the motion.
- **Framing.** `view_box = (min_x, min_y, width, height)` is fixed, in projected
  SVG units, and shared by every frame; the document is `width` by `height`
  px and clips at the box. It is not fitted, so leave room for the motion's
  extremes, strokes, and labels. The scenes' own `pad` and `background` play no
  part; `background` fills the box for the whole animation.

Browsers play the file. A viewer without SVG animation, such as Inkscape or
CairoSVG, shows the first frame, so a still made from the animation is
`animation.frame(t)` rendered on its own.

## What the file holds

A frame is a whole scene, but most of it is usually the same in every frame.
What every sample draws alike is written **once**, outside the timed groups, and
keeps its own id:

- definitions — gradients, clip paths — with the same id and content in every
  frame, and
- the elements at the bottom and at the top of the paint order that are the
  same in every frame.

So draw what does not move on the lowest layers, or the highest, and it costs
nothing per frame: the pendulum's track, captions, and equations are written
once, and each of its frames holds only the rod, the bob, and two readouts.
Elements are compared by their markup, so a label rendered once and added to
every frame is shared even if each frame wraps it in a new object.

The rest of each sample goes in its own `<g data-vecview-frame="i">`, shown
during its interval. Consecutive samples with the same rest share one group, so
a pause costs one frame, and an animation that never changes comes out as a
static document. Ids inside a frame group are renamed `frame{i}-{id}`, the
element keeping its own as `data-vecview-id`, and the frame's `url(#...)` and
`href="#..."` references follow. An embedded stylesheet that changes between
frames is scoped to its frame's group, and must then be plain rules, without
at-rules or comments.

Content svg.py cannot see inside — a [VecTeX](https://github.com/maiani/vectex)
fragment is one — keeps its inner ids as they are. Render such a label once
and add it to every frame, as the pendulum does, or give each frame's render its
own `id_prefix`.

Rendering the same callback with the same options produces byte-identical
output.

## Tracks and easing

A track is a function from seconds to a value. Plain functions work anywhere a
track does; `Track` adds keyframes and composition:

```python
from vecview.animation import Track, hold, linear, smoothstep

height = Track.keyframes([(0.0, 0.0), (1.0, 2.0), (3.0, 2.0)], interpolate=linear)
where = Track.keyframes(
    [(0.0, (0.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 1.0))],
    interpolate=linear,
    ease=smoothstep,
)
label = Track.keyframes([(0.0, "before"), (1.5, "after")], interpolate=hold)
radius = height.map(lambda h: 0.2 + 0.1 * h)
```

| Call | Contract |
| --- | --- |
| `Track(sample)` | Wraps any function of finite seconds. |
| `Track.keyframes(keys, *, interpolate, ease=None)` | Interpolates `(seconds, value)` keys. |
| `track.map(function)` | The track `function(track(t))`. |
| `linear(a, b, u)` | `(1-u)*a + u*b` for numbers or 3D points, unclamped. |
| `hold(a, b, u)` | `a` until `u` reaches 1, then `b`: labels, discrete states. |
| `smoothstep(u)` | The easing `3u² - 2u³`, unclamped. |

Keys need finite, strictly increasing times — duplicates and unsorted keys raise
`ValueError` rather than being repaired — and one key is a constant. Before the
first key and after the last, the track holds the nearest value; at a key it
returns that key's value without interpolating. Between keys `a` and `b` it
returns `interpolate(value_a, value_b, ease(u))` with `u = (t - a) / (b - a)`,
easing each interval on its own; an ease may overshoot, but must return a finite
number.

Interpolation is always named, because an angle, a colour, and an orientation
need different rules even when they are the same Python type. Angles are plain
degrees, so `0` to `720` is two turns. Time is plain too: run tracks together by
reading the same `t`, and delay, reverse, or speed one up by changing what it is
given — `lambda t: where(2.0 - t)`.

## Moving parts

[`place`](scenes.md#parts) already moves, turns, and scales a part about its own
origin, and a track can drive each argument:

```python
spin = Track.keyframes([(0.0, 0.0), (2.0, 720.0)], interpolate=linear)


def frame(t: float) -> vecview.Scene:
    scene = vecview.Scene(CAMERA)
    scene.place(10, bead, at=where(t), rotate=((0, 0, 1), spin(t)), id="bead")
    return scene
```

For a turn or a scale about any other point, `rotate` and `scale` return a new
part and leave theirs untouched:

```python
from vecview.animation import rotate, scale

rotate(part, axis, angle_deg, *, pivot=(0, 0, 0))  # p -> pivot + R @ (p - pivot)
scale(part, factor, *, pivot=(0, 0, 0))  # p -> pivot + factor * (p - pivot)
```

Both keep the part's layers, ids, and classes and follow `place` in everything
else: a right-handed turn in degrees about a non-zero axis, one positive uniform
factor so every solid keeps its exact outline, screen sizes unscaled. They nest
inside first: `rotate(scale(arm, 2, pivot=hinge), (0, 0, 1), 30, pivot=hinge)`
grows the arm about its hinge, then swings it.

## Cost

Every sample is a full render, so an animation costs `N` times the still, plus
one terminal frame for finite playback. A layer under
[`sort_by_depth`](scenes.md#sorting-by-depth) resolves exact visibility in every
frame, and that is where the time goes: a depth-sorted figure that takes a
second to render takes a second per frame. Choose `fps` and `duration`
deliberately for those.
