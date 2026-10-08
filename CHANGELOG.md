# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `slot(..., content=element)` -- a slot can hold an svg.py element instead of
  waiting for a consumer to fill it: the element, `w` by `h` in its own
  coordinates from its top-left, is placed with its box aligned on the anchor,
  and follows every camera.  VecView places it as given, never parsing, copying,
  or changing it.  The examples with mathematical labels now typeset them with
  VecTeX and hold them in slots, in place of `svg.TSpan` subscripts.
- `Animation(scene, *, duration, view_box, fps=30, repeat=1, background=None)`
  -- a scene that moves, sampled at evenly spaced times, rendered by the
  static renderer, and written as one standalone SVG that plays itself with
  native SVG timing.  Any argument of any drawing call may be a `Track`, inside
  lists and style keywords too; `place` takes a track of parts, for what
  changes in number, and the active camera may be a track.  A call holding a
  track is checked at `t = 0` and kept as made; `Scene.at(t)` and `Part.at(t)`
  give the drawing as it stands at `t`, and a scene with tracks renders as it
  stands at `t = 0`.  `Animation` also takes a pure function from seconds to a
  whole `Scene`.  Finite playback holds the exact
  frame at `duration`; `repeat=None` loops.  What every frame draws alike --
  identical definitions, and the elements every frame draws in the same paint
  order, wherever they fall -- is written once, in place, with its own ids.
  Each stretch that changes is one `<use>` whose `href` steps through its
  distinct contents with discrete timing, each written once in `<defs>` with
  its ids renamed `frame{i}-{id}` (the original kept as `data-vecview-id`);
  a stretch is cut at layer boundaries and per element wherever that is
  estimated to be smaller, a lone element is its own target, and an animation
  that never changes is a static document.  `animation.breakdown()` says where
  the file's bytes go, stretch by stretch.
- `vecview.animation`: `Track` with `keyframes` and `map`, the interpolators
  `linear` (numbers and 3D points) and `hold`, the easing `smoothstep`, and
  `rotate` and `scale`, which copy a part turned or scaled about a pivot and
  take tracks too.
- `examples/pendulum.py` -- a pendulum swinging through one seamless period,
  with its equations typeset by VecTeX.  See `docs/animation.md`.
- `examples/rotating_dipole.py` -- a rotating electric dipole and the exact
  retarded field it radiates: `E` as arrows and `B_z` as spiral shading over
  the equatorial plane, through one seamless period.

- `class_=` on every drawing call that takes style keywords: one string,
  space-separated, or a sequence of names.  An id names one object, a class a
  kind of object -- every gate, every oxygen -- for a stylesheet, a selector in
  the composing tool, or Inkscape to reach together.  Each top-level element a
  call emits carries the classes (every face, both strokes of an edge set, every
  slice of a sliced solid, both parts of a line a depth-sorted layer splits) and the
  elements inside a solid's group do not, so a selector matches each object
  once.  Names that are not strings raise `TypeError` at the call.
- `Part` and `place` -- objects drawn once, in their own coordinates, and
  placed any number of times: `place(layer, part, at=, rotate=(axis, deg),
  mirror=, scale=, id=, class_=)`.  Only rigid motions and a uniform scale, so
  every solid keeps its exact outline; faces stay wound to their normals under a
  mirror.  The placement's layer is added to the part's layers, `id` prefixes
  every id in the part, `class_` is added to its classes, and parts nest.  A
  part holds only world-space calls: no camera, `<defs>`, screen-space calls, or
  depth sorting, which stay the scene's.
- `extrude(section, along)` -- a planar cross-section in any plane swept along
  any vector, as correctly wound `Face`s named `"start"`, `"end"`, and
  `"side-{i}"`.  Wall `i` is built on section edge `i` as given, in either
  winding, so walls can be picked out by edge.  `prism_faces` is now the
  special case along `z`, with unchanged output.
- `vecview.outlines` -- footprints and cross-sections built in 2D: `rect`,
  `regular`, `offset`, `film` (a layer of given thickness deposited on a run of
  edges, numbered as `extrude` numbers walls), `union`, `difference`,
  `intersection`, and `to_plane` to lift an outline into 3D.  Results are
  counter-clockwise and ordered deterministically; a result with a hole
  raises `ValueError` rather than dropping it.
- `cut(faces, origin, normal)` -- a closed solid cut open by a plane, for a
  cutaway: faces are clipped and keep their names, and the cut is capped by
  faces named `"cut"` (or `"cut-0"`, ...) lying in the plane, so a section can
  be styled apart.  Non-convex solids work; a cap with a hole raises.
- `examples/kitaev_chain.py` -- a minimal Kitaev chain after Dvir et
  al., Nature 614, 445 (2023): a gated InSb nanowire running past draped Cr/Au
  contacts, a grounded Al shell of finite thickness that continues onto the
  dielectric, and a Majorana mode on each dot.
- A documentation page for `vecview.outlines`.

### Changed

- Depth-sorted layers render faster, with byte-identical output: exact
  visibility makes its shapely calls in batches -- one tree query, and one
  vectorised intersection for every overlapping pair -- and orders the painting
  of clipped surfaces with a heap.  The Dirac cone renders in 0.61 of the time,
  the solenoid in 0.79.
- The examples are self-contained scripts side by side in `examples/`, each
  writing its own SVG to `examples/out`; `examples/gallery/` and the shared
  `_common.py` are gone.  `python examples` builds them all, with timings and
  PNG previews, and `--docs` still refreshes `docs/gallery/`.
- `sort_by_depth(layer)` always decides visibility exactly, point by point:
  surfaces are clipped where an opaque surface hides them, lines are split, and
  a surface hidden entirely is dropped.  The painter's algorithm keyed on each
  element's mean depth is gone, and with it the need to cut geometry so it
  sorts: a coil wraps an unsliced core.  A sorted layer renders in up to a few
  seconds and can double in size, from its clip paths.
- `shapely` and `contourpy` are required dependencies, alongside `numpy` and
  `svg.py`, and the `occlusion` extra is gone: exact visibility always works,
  and robust polygon clipping is needed beyond it, for booleans on outlines.
  Both still load only when first needed.
- An unsliced cylinder or cone names its end disks `{id}-end0` and `{id}-end1`
  and its highlight gradient `{id}-shade`, as a sliced one already did, instead
  of `{id}-body-end0` and `{id}-body-shade`.  Slicing no longer renames what a
  consumer selects.
- Rendering raises `ValueError` when two elements share an id, naming every
  id used more than once.  Duplicate ids are invalid SVG, and a consumer
  selecting by id -- or a `url(#...)` fill -- would silently reach the wrong
  element.  Ids a call derives (`slab-pz`, `{id}-body`, `{id}-profile`) are
  checked too, so a hand-written id that collides with one is caught.
- Every example is rewritten with the new API -- `Part` and `place` for
  repeated and mirrored objects, `extrude` and `outlines` for cross-sections,
  `class_` for every kind of object -- and reads as the obvious way to draw its
  figure.  The pictures are unchanged apart from the depth-sorted ones.
- The documentation is checked against the code, page by page: wrong
  signatures, ids, and claims fixed, every runnable snippet run, and the
  missing public API documented.

### Removed

- `sort_by_depth(..., exact=True)`: every sorted layer is exact now, so the
  keyword is gone and passing it raises `TypeError`.

### Fixed

- Text is escaped: a label holding `<`, `>`, or `&` -- `"B < 0"`, `"Smith &
  Jones"` -- made the document unreadable, because svg.py writes text as it is
  given.  `text`, `text2d`, and their `svg.TSpan` runs are escaped now, the
  runs copied rather than changed.
- `double_arrow_shape` clamps a `head_len` longer than half the arrow, as
  `arrow_shape` already did, instead of returning a self-crossing polygon.
- `Scene.is_empty` is true exactly when rendering would find nothing to fit a
  viewBox to.  A scene holding only `add`, or `rect2d` and `text2d` with
  `grow=False`, was reported non-empty and then failed to render.
- The background `<rect>` takes the viewBox's rounded numbers, so it covers
  the viewBox exactly and no longer carries full-precision floats.

## [0.2.0] - 2026-10-06

### Added

- PyPI release workflow: validate the tag, run CI, build and check distributions,
  then publish with Trusted Publishing.

- `Scene.sort_by_depth(layer)` -- opt one layer in to the painter's algorithm.
  Its world-space elements are drawn back to front, keyed by the mean depth of
  the points that made them; ties keep insertion order and screen-space
  elements go on top. Other layers are untouched, and `with_camera` re-sorts.
- `Scene.sphere` -- the exact outline of a sphere as one `<circle>`, or a rotated
  `<ellipse>` under an oblique camera.
- `Scene.cylinder` and `Scene.cone` -- exact outlines of a cylinder, frustum, or
  cone: two straight sides and two elliptical arcs in one `<path>`, plus the end
  disk that faces the camera. `slices=n` cuts a long cylinder into separately
  sorted lengths that still draw as one seamless solid.
- `Scene.arrow3d` -- a solid arrow, cylindrical shaft and conical head in one group.
- `Scene.tube` -- a tube along a world-space curve, as overlapping stroked pieces
  that sort over and under each other without seams.
- `highlight=` on the solids -- a gradient fill lightest toward the upper left;
  spheres of one colour pair share one gradient.
- `Scene.edges` -- the edges of a convex solid, visible ones solid and hidden ones
  dropped or restyled (dashed), optionally one path per edge and trimmed at the
  vertices.
- `Scene.sphere_curve` -- a curve on a sphere, split exactly where it passes behind.
- `sort_by_depth(layer, exact=True)` -- exact visibility, with the new
  `occlusion` extra (`shapely`, `contourpy`).  Every surface keeps its native
  element, clipped to what no opaque surface hides: exactly between planes, and
  to a sub-pixel contour of the closed-form depths where a sphere, cylinder,
  cone, arrow, or tube is involved.  Lines are split where they pass behind a
  surface, and the hidden part dropped or drawn in a `back` style.  Opaque
  surfaces are painted so that what hides another comes after it.
- `Scene.polyline(..., back=...)` -- the style of the parts an exact layer hides.
- `Scene._repr_svg_`, so a scene displays inline in Jupyter.
- `text` and `text2d` accept a list of `svg.TSpan` runs, for subscripts.
- `arc_shape`, `helix`, `surface_faces`, `convex_polyhedron`, and `trim_corners`.
- `examples/gallery/` -- a perovskite cell, the fcc Brillouin zone, C60, the Bloch
  sphere, a gapped Dirac cone, a Néel skyrmion, a solenoid, and crossing mirror
  planes in an exact layer, each from one
  script; `python examples/gallery --docs` builds them all into the new Gallery
  page.

### Changed

- Use ty for type checking in development and CI instead of mypy.
- **A scene holds objects and cameras; a camera renders them.** Drawing calls
  only record, and nothing is projected until `render`, `save`, `bbox`, or
  `to_svg_document`. As in a 3D application, `scene.cameras` holds named
  cameras and `scene.camera` is the active one (formerly `scene.cam`), set by
  name or directly; the zero-argument `to_svg_document` uses it.
  `Scene(camera=None, *, cameras=None, pad, background)`, and `render(camera=None,
  *, pad, background)`, `save(path, camera=None)`, and `bbox(camera=None)` take a
  camera or a name. `with_camera` returns a copy with a different active camera.
  Output is byte-identical to before.
- Errors that need no camera -- a self-crossing footprint, a one-point tube, a
  highlight without a fill -- still raise at the call; those that need one -- a
  plane seen edge-on, an arrow along the projection ray -- raise on render.
- `Scene.items` and `Scene.defs` are gone; they belonged to one projection, and
  now live on the private canvas a render builds.
- `prism_walls` builds its path through the shared helper; output is unchanged.

## [0.1.2] - 2026-10-05

### Added

- `examples/altermagnetic_dot.py`: a device sketch of an altermagnetic quantum
  dot -- arc-shaped non-convex gates and tapered leads through `prism_walls`,
  Gaussian spin densities, camera-facing spins, a bias circuit, and labels at
  world points -- built once and replayed under four projections with
  `with_camera`.

## [0.1.1] - 2026-10-05

### Changed

- Prose spells the project VecView; the distribution, import, and command names
  stay `vecview`. The sibling projects are now FigWorks (formerly FigForge) and
  VecWire (formerly cirquit).

## [0.1.0] - 2026-10-05

### Added

- `Scene.slot(layer, pt3, w, h, *, id, align, dx, dy)` — reserve an empty,
  screen-aligned group pinned to a projected world point, for upright content
  such as TeX labels. The group is translated to the anchor and records `align`
  as `data-align`; a `w` × `h` box grows the fitted viewBox. The anchor is
  reprojected by `with_camera`, which a hand-placed `rect2d` was not.
- `Scene.silhouette(layer, solid)` — fill the projected convex hull of a solid as
  one polygon. Drawn in the wall colour under the cap, it replaces per-wall
  polygons and the hairline seams between them.
- `Scene.arrow(..., normal="camera")` — an arrow turned about its axis to show
  its widest face, resolved at draw time so a reprojection turns it too. Also
  accepts an explicit normal.
- `Scene.gaussian(layer, center, u, v, a, b, *, id, color)` — a soft Gaussian spot
  lying in a world plane, as one polygon filled by a radial gradient mapped
  through the plane's affine transform.
- `prism_faces(footprint, z0, z1)` — any simple footprint, convex or not, extruded
  along `z`, as correctly wound `Face`s that cull like a box. Self-intersecting
  or self-touching outlines raise `ValueError`.
- `Scene.prism_walls(layer, footprint, z0, z1)` — the camera-facing walls of an
  extrusion as one seamless `<path>`, one strip per run of consecutive facing
  walls. With the cap drawn over it, exact at any height for unstroked walls,
  including non-convex footprints.
- `annulus_sector(center, r_in, r_out, theta0_deg, theta1_deg, n)` — the
  counter-clockwise footprint of an annular sector or pie wedge.
- `ellipse_shape(center, u, v, a, b, n)` — the generalization of `circle_shape`.
- `Camera` is now an abstract base holding only the projection contract
  (`project`, `at`, `depth`, `visible`), with `ParallelCamera` beneath it for the
  affine machinery (`direction`, `screen_basis`, `foreshortening`,
  `plane_matrix`). A future perspective camera would subclass `Camera` directly
  and inherit none of the affine methods, which is the point of the split.
- `OrthographicCamera` — the axonometric camera, aimed by azimuth and elevation,
  with `isometric()` and `dimetric(ratio=...)` constructors, plus `axonometry()`
  to classify a projection from its actual ratios.
- `ObliqueCamera` — cavalier and cabinet projections, which no azimuth and
  elevation can reproduce.
- `Camera.foreshortening()` — the screen length of each unit world axis.
- `Scene.with_camera(cam)` — re-render a finished scene under another projection.
  Recorded world-space calls are replayed, giving output byte-identical to
  building from scratch with that camera.
- `Scene.faces(..., cull=True)` — cull back faces at draw time rather than in the
  caller, so a reprojection selects the right walls instead of keeping the
  original camera's.
- `Camera.plane_matrix()` and `Scene.plane()` — embed flat SVG content in a world
  plane. Exact rather than approximate: a parallel projection restricted to a
  plane is affine, which is what an SVG `matrix` expresses.
- Initial package, extracted from a single-module `vecview.py` used to draw an
  altermagnetic-slab polarizer schematic.
- `Camera.screen_basis(normal)` — the in-plane world directions that move a point
  purely right and purely down on screen, for any plane. Replaces a hand-rolled
  2×2 solve in the caller.
- `Camera.visible(faces)` and `Camera.faces_camera(normal)` — back-face culling
  by outward normal, replacing hand-listed wall corner pairs and a comment about
  which octant the camera occupies.
- `Camera.direction(vec)` — projects a displacement, ignoring `origin`.
- `Camera.depth(pts)` — signed distance along the view axis, for callers that want
  to order something by depth themselves.
- `box_faces(center, size)` and the `Face` record (`name`, `points`, `normal`) —
  an axis-aligned box as six named, correctly wound faces.
- `in_plane_dir(angle_deg, u, v)` — unit direction at an angle within a plane.
- `Scene.faces(layer, faces, **style)` — draw a group of faces with one style. An
  `id` is suffixed per face (`slab` → `slab-pz`, `slab-px`, …) rather than
  repeated, since duplicate ids are invalid SVG.
- `Scene.to_svg_document()` — a complete SVG document string, the whole surface a
  consumer needs to place a scene without importing this package.
- `Scene.save(path)`, `Scene.is_empty`, and `repr` for `Camera` and `Scene`.
- `Scene(cam, pad=..., background=...)` constructor defaults, so
  `to_svg_document()` needs no arguments.
- Public `unit(v)`.
- Type hints throughout, checked with `mypy --strict`, and a `py.typed` marker.
- Documentation, and a test suite covering projection, geometry, layering, and
  the document contract.

### Changed

- Coordinates that round to `-0.0` are written as `0.0`.
- **Python 3.12 is now the floor.** This drops the split where ruff targeted 3.11
  while mypy targeted 3.12 to parse numpy 2's PEP 695 stubs, and lets the private
  type aliases use `type` statements.
- Rename the distribution and import package from `svg3d` to `vecview`.
- Polygon and polyline coordinates are emitted as `svg.Point` pairs
  (`points="1.5,2.25 3,-4"`) rather than a flat coordinate run.
- Rendering an empty scene now raises `ValueError` instead of producing a
  document with a degenerate viewBox.
- `rect_shape` normalizes its axes, so an unnormalized direction no longer
  silently stretches the rectangle.
- `circle_shape` derives its in-plane basis from the shared helper, making the
  seam position consistent with other in-plane geometry.
