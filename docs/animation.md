# Animation

```python
Animation(scene, *, duration, view_box, fps=30.0, repeat=1, background=None)
```

An `Animation` is a [`Scene`](scenes.md) that moves. Draw the scene once, and
wherever a still takes a value, give what moves a **track** — a function of
time. The animation reads every track at evenly spaced times, renders each
moment with the same renderer a still figure uses, and writes one standalone SVG
that plays itself with native SVG timing: no script, no video, every frame still
vector.

```python
from math import cos, sin

import vecview
from vecview.animation import Track

orbit = Track(lambda t: (2 * cos(t), 2 * sin(t), 0))

scene = vecview.Scene(vecview.OrthographicCamera(35, 24, 62))
scene.sphere(0, (0, 0, 0), 0.6, id="sun", fill="#e9a23b")
scene.sphere(1, orbit, 0.25, id="planet", fill="#3b6fb6")

animation = vecview.Animation(scene, duration=6.283, view_box=(-150, -110, 300, 220))
animation.save("orbit.svg")
animation.frame(1.5).save("still.svg")  # any moment, as an ordinary still
```

| Call | Returns |
| --- | --- |
| `animation.frame(t)` | The `Scene` at `t` seconds, for `0 <= t <= duration`. |
| `animation.render()` | The animated document as an `svg.SVG` tree. |
| `animation.to_svg_document()` | The same document as a string — the embedding contract. |
| `animation.save(path)` | Writes it as UTF-8 and returns the `Path`. |

A time outside `[0, duration]` raises `ValueError`. Every frame needs an active
camera when the animation renders.
[`examples/pendulum.py`](https://github.com/maiani/vecview/blob/main/examples/pendulum.py)
is a complete example: a pendulum swinging through one seamless period, with
its equations typeset by TeX.
[`examples/rotating_dipole.py`](https://github.com/maiani/vecview/blob/main/examples/rotating_dipole.py)
is a larger one: a rotating dipole and the exact retarded field it radiates,
a few hundred polygons a frame.

## Tracks in a scene

A track can stand for any argument of any drawing call — a point, a radius, a
direction, a colour, the text of a label — and anywhere inside a list or a
dict of them, as one end of a line:

```python
theta = Track(lambda t: AMPLITUDE * cos(OMEGA * t))
bob = theta.map(position)  # a track built from another

scene.polyline(1, [(0, 0, 0), bob], stroke=INK, id="rod")
scene.sphere(1, bob, 0.135, fill=ACCENT, id="bob")
scene.text2d(0, 96, 141, theta.map(lambda a: f"{degrees(a):+.1f}°"), id="angle")
scene.text2d(0, 96, 177, Track(lambda t: f"t = {t:.2f} s"), id="clock")
```

`scene.at(t)` is the scene as it stands at `t`: a copy with every track read
there, sharing the calls that hold none. `animation.frame(t)` is `scene.at(t)`.
A scene that holds tracks renders, saves, and embeds as it stands at `t = 0`,
which is also what a viewer without SVG animation shows.

- **Checked when the call is made.** A call holding a track is checked at
  `t = 0`, so a mistake raises where it is written. A value that goes wrong only
  later — a radius that turns negative — raises when that frame renders.
- **Only a `Track` is read as one.** A plain function passed to a drawing call
  is a value like any other; wrap it as `Track(f)`. Ids and classes are plain,
  so an object is the same object in every frame.
- **Pure.** A track gives the same value for the same `t`, whatever was read
  before: compute the state at `t` rather than stepping it, so a simulation is
  read at `t`, not advanced.

What moves need not be one object with changing values. When the number of
things changes — arrows too short to draw are left out, say — make a function
from the moment to a [part](scenes.md#parts), and place a track of parts:

```python
def electric(turn: float) -> vecview.Part:
    arrows = vecview.Part()
    for point, e in field_at(turn):
        arrows.arrow(0, point, e, length(e), normal=(0, 0, 1))
    return arrows


scene.place(2, turn.map(electric))  # a new part at every frame
```

The camera can be a track too, for a view that turns:

```python
turntable = Track(lambda t: vecview.OrthographicCamera(azim_deg=36 * t, elev_deg=24, scale=62))
scene = vecview.Scene(turntable)
```

Type hints do not say all this yet: apart from `place`, each call's signature
names the plain value, so a static type checker flags a track the call accepts.

## Frame callbacks

`Animation` also takes a function from seconds to a whole scene, for motion
that is easier said as code that builds each moment — a scene assembled from a
simulation's state, or one whose layers or depth sorting change:

```python
def frame(t: float) -> vecview.Scene:
    scene = vecview.Scene(CAMERA)
    scene.sphere(0, (0, 0, 0), 0.6, id="sun", fill="#e9a23b")
    scene.sphere(1, (2 * cos(t), 2 * sin(t), 0), 0.25, id="planet", fill="#3b6fb6")
    return scene


animation = vecview.Animation(frame, duration=6.283, view_box=(-150, -110, 300, 220))
```

The callback must be pure, as a track must, and build a fresh scene each time;
sharing immutable geometry, parts, and rendered labels between frames is fine.
`frame(t)` is then the plain callback result, with the scene's own camera,
padding, and background, and a callback that returns anything but a `Scene`
raises `TypeError`. A scene of tracks is this callback underneath — `scene.at`
— so the two are drawn alike.

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

A frame is a whole scene, but most of it is usually the same in every frame,
and the file stores each thing once per change rather than once per frame.
Elements are compared by their markup, so a label rendered once and added to
every frame is shared even if each frame wraps it in a new object.

What every sample draws alike is written **once**, in place, and keeps its own
id: definitions — gradients, clip paths — with the same content in every frame,
and every element drawn in every frame in the same paint order, wherever it
falls in the layer stack. The pendulum's track, captions, and equations are
written once, and so would be a caption between its rod and its bob.

What changes sits between those elements, and each stretch of it is drawn by
one `<use>` whose `href` steps through that stretch's contents with discrete
SMIL timing. Each distinct content is written once in `<defs>`, however many
frames show it and in whatever order: a pause, a motion that comes back, and a
label that changes once all cost one copy per distinct content. A stretch is
cut at the boundaries between its layers, and split into one `<use>` per
element where every frame draws as many there, wherever that is estimated to
be smaller: a caption on a layer of its own that changes four times is stored
four times, however often the electrons beside it move. So give what changes
rarely a layer of its own. An animation that never changes comes out as a
static document.

Ids inside a stretch's contents are renamed `frame{i}-{id}`, after the first
sample `i` that drew that content, the element keeping its own as
`data-vecview-id`, and the content's `url(#...)` and `href="#..."` references
follow. A content of several elements, or of none, is a
`<g id="frame{i}" data-vecview-frame="i">`, with `.{n}` after the frame number
when several stretches change; a lone element is its own target, given that id
if it has none. Content that points into another stretch is
drawn with it. An embedded stylesheet that changes between frames is scoped to
its frame's group, and must then be plain rules, without at-rules or comments;
all that changes is then drawn as one stretch, since the rules may reach any of
it.

Content svg.py cannot see inside — a [VecTeX](https://github.com/maiani/vectex)
fragment is one — keeps its inner ids as they are. Render such a label once
and add it once, as the pendulum does, or give each moment's render its own
`id_prefix`.

Rendering the same scene or callback with the same options produces
byte-identical output.

## Tracks and easing

A track is a function from seconds to a value. `Track` wraps one, so a drawing
call knows to read it, and adds keyframes and composition; in a frame callback a
plain function works as well:

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
scene.place(10, bead, at=where, rotate=((0, 0, 1), spin), id="bead")
```

For a turn or a scale about any other point, `rotate` and `scale` return a new
part and leave theirs untouched:

```python
from vecview.animation import rotate, scale

rotate(part, axis, angle_deg, *, pivot=(0, 0, 0))  # p -> pivot + R @ (p - pivot)
scale(part, factor, *, pivot=(0, 0, 0))  # p -> pivot + factor * (p - pivot)
```

Any of their arguments may be a track too, and the copy then moves as it
changes. Both keep the part's layers, ids, and classes and follow `place` in
everything else: a right-handed turn in degrees about a non-zero axis, one positive uniform
factor so every solid keeps its exact outline, screen sizes unscaled. They nest
inside first: `rotate(scale(arm, 2, pivot=hinge), (0, 0, 1), 30, pivot=hinge)`
grows the arm about its hinge, then swings it.

## Where the bytes go

`animation.breakdown()` says what the file stores, stretch by stretch in paint
order: each `Stretch` has its `layers`, the fewest and most `elements` a frame
draws there, how many distinct `contents` are stored, and the `bytes` they
take. Printed, it gives the total, what is written once, and the heaviest
stretches that change:

```text
360 frames, 1.93 MB
written once: 104 elements, 71.6 kB
what changes, heaviest first:
   465.5 kB  24%    61 contents     49 elements  layers 15       polygon, polygon, polygon, polygon, ...
   283.0 kB  15%   240 contents    1-2 elements  layers 20       polygon, polygon
     2.1 kB   0%     4 contents      3 elements  layers 26       text, text, text
```

A stretch with as many contents as there are frames changes in every frame,
and its bytes are what each frame redraws: fewer points in a shape, fewer
frames, or moving what changes rarely to a layer of its own are the remedies.
The breakdown renders the animation to find out, as `render` does. Most hosts
serve SVG gzipped, which shrinks repeated markup several times over.

## Cost

Every sample is a full render, so an animation costs `N` times the still, plus
one terminal frame for finite playback. A layer under
[`sort_by_depth`](scenes.md#sorting-by-depth) resolves exact visibility in every
frame, and that is where the time goes: a depth-sorted figure that takes a
second to render takes a second per frame. Choose `fps` and `duration`
deliberately for those.
