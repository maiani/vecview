# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `class_=` on every drawing call that takes style keywords: one string,
  space-separated, or a sequence of names.  An id names one object, a class a
  kind of object -- every gate, every oxygen -- for a stylesheet, a selector in
  the composing tool, or Inkscape to reach together.  Each top-level element a
  call emits carries the classes (every face, both strokes of an edge set, every
  slice of a sliced solid, both parts of a line an exact layer splits) and the
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

### Changed

- An unsliced cylinder or cone names its end disks `{id}-end0` and `{id}-end1`
  and its highlight gradient `{id}-shade`, as a sliced one already did, instead
  of `{id}-body-end0` and `{id}-body-shade`.  Slicing no longer renames what a
  consumer selects.
- Rendering raises `ValueError` when two elements share an id, naming every
  id used more than once.  Duplicate ids are invalid SVG, and a consumer
  selecting by id -- or a `url(#...)` fill -- would silently reach the wrong
  element.  Ids a call derives (`slab-pz`, `{id}-body`, `{id}-profile`) are
  checked too, so a hand-written id that collides with one is caught.

### Fixed

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
