# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

### Changed

- **Python 3.12 is now the floor.** This drops the split where ruff targeted 3.11
  while mypy targeted 3.12 to parse numpy 2's PEP 695 stubs, and lets the private
  type aliases use `type` statements.

- Rename the distribution and import package from `vecview` to `vecview`.

### Added

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

- Polygon and polyline coordinates are emitted as `svg.Point` pairs
  (`points="1.5,2.25 3,-4"`) rather than a flat coordinate run.
- Rendering an empty scene now raises `ValueError` instead of producing a
  document with a degenerate viewBox.
- `rect_shape` normalizes its axes, so an unnormalized direction no longer
  silently stretches the rectangle.
- `circle_shape` derives its in-plane basis from the shared helper, making the
  seam position consistent with other in-plane geometry.
