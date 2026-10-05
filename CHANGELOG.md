# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
