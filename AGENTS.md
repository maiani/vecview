# VecView contributor guide

## Scope

`vecview` projects world-space geometry into one SVG document, with an explicit
layer stack. It does not plot data, compile TeX, rasterize, export PDF, compose
multi-panel figures, or edit existing SVG.

The package is **independent and has no downstream dependencies**. Consumers
integrate through `Scene.to_svg_document()` alone; that direction is one-way.
Do not add a consumer-specific adapter, import, dependency, example, or test.

Naming the sibling projects is fine where it is only provenance: `vecview` is
developed alongside FigWorks and VecTeX as a suite, and a "Related projects"
link says so. What must stay out is *coupling* -- consumer-specific API
documentation, worked integration examples, or tests that import a consumer.
A tool that composes figures from `vecview` output documents that integration
on its own side.

## Architecture

- Keep the three-way split, and do not let it blur:
  - `shapes.py` knows numbers only — world-space arrays and `Face` records. No
    camera, no style, no SVG import.
  - `outlines.py` is the same in 2D: `(n, 2)` footprints and cross-sections,
    combined through `shapely`, which is the only module besides `_occlusion.py`
    that imports it. A result an outline cannot hold -- a hole -- raises
    rather than being dropped.
  - `camera.py` defines what a camera *is*: `Camera` (the abstract contract) and
    `ParallelCamera` (the affine machinery). Back-face culling lives here
    because it is a camera question, not a scene one.
  - `projections.py` holds the concrete projections: `OrthographicCamera` and
    `ObliqueCamera`.
  - Only the rendering modules import `svg`: `scene.py` and the private
    `_elements.py`, `_canvas.py`, and `_solids.py`. `shapes.py`, `camera.py`,
    `projections.py`, `_occlusion.py`, `_place.py`, and `_drawing.py` never do.
- `_occlusion.py` is private: exact visibility for `sort_by_depth(exact=True)`.
  The canvas describes each element of an exact layer as a `Surface` (outline
  plus a depth that goes on smoothly past its edge) or a `Line`, lazily, and
  `resolve` clips and splits them. Keep closed-form depths closed-form.
- `_vec.py` and `_types.py` are private. Re-export from `__init__.py` what should
  be public; `unit` is the only helper promoted so far.
- Geometry that needs a camera to be computed does not belong in `shapes.py`.
- Angles are degrees at every public boundary, radians nowhere.
- Keep the camera hierarchy honest. `Camera` promises only `project`, `at`,
  `depth`, and `visible`. `direction`, `screen_basis`, `foreshortening`, and
  `plane_matrix` belong on `ParallelCamera` because each assumes a screen offset
  independent of position. Moving any of them up would make them a silent wrong
  answer for a perspective camera, which would subclass `Camera` directly.
- `Scene.plane` reserves a group and nothing more. This package must not parse,
  normalize, or embed foreign SVG; a consumer fills the group by id.
- `Scene` records objects and nothing else; it never projects. Every public
  drawing method validates what it can without a camera and appends one record.
  The canvas holds a camera and implements each method of the same name;
  rendering replays the record onto a fresh canvas. Camera-dependent work and
  errors belong in the canvas, and a canvas never records, so its methods may
  call each other freely. `_Canvas` (`_canvas.py`) has the layer stack, exact
  visibility, and the flat calls; `_SolidCanvas` (`_solids.py`) adds the curved
  solids and is the canvas a render uses. `_elements.py` holds the SVG emission
  helpers both share.
- The world-space drawing calls live on the private `_Drawing` base
  (`_drawing.py`, with `Part`), shared by
  `Scene` and `Part`. `Scene` adds what only makes sense for a whole document:
  cameras, rendering, `sort_by_depth` (a property of the layer stack), `<defs>`,
  raw elements, and screen-space calls. Keep a call off `_Drawing` unless it has
  a world position, because a part must be able to move everything it holds.
- `place` copies a part's recorded calls with their geometry moved by a `_Frame`
  (`_place.py`).
  `_MOVES` says, per call, which arguments are points, directions, edges, or
  lengths. A new world-space call needs an entry there, or placing it fails with
  a `KeyError`; `tests/test_parts.py` places one of every call.

## Module size

- No Python file in the repository exceeds **1000 lines**, pylint's default
  `max-module-lines` (C0302). `tests/test_package_metadata.py` enforces it.
- Aim for a few hundred lines. When a module nears the limit, split it before
  adding to it, along a real seam of responsibility -- emission, visibility,
  solids -- not into arbitrary halves, and keep the output byte-identical: a
  split moves code, it does not change it.

## Deliberate non-features

Do not add these without the user changing the design first:

- **Automatic depth sorting across layers** (a z-buffer, or sorting a layer that
  did not ask). Layers are the model. A beam crossing a translucent slab has
  three parts that no automatic depth rule orders correctly. Depth only ever
  acts inside a layer passed to `Scene.sort_by_depth`: as the painter's
  algorithm, whose failures are fixed by cutting geometry back (`edges(trim=)`,
  `trim_corners`) or into pieces (`cylinder(slices=)`, `tube` chunks), or, with
  `exact=True`, as exact visibility by clipping.
- **Rasterizing or PDF export.** Runtime dependencies stay `numpy`, `svg.py`,
  `shapely`, and `contourpy`; a test in `tests/test_package_metadata.py`
  enforces it. The last two are required because robust polygon clipping and
  sub-pixel contours -- exact visibility, booleans on outlines -- are not worth
  reimplementing. There are no runtime extras.
- **Shading, materials, or lighting models.** This draws schematics, not renders.
  `highlight=` gradients are a fill style fixed on screen, with no light
  direction; keep them that way.

## Development

- Supported Python: 3.12 and newer.
- `uv sync --all-extras` for a full environment.
- Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check`,
  and `uv run pytest` before reporting a change complete.
- Run `uv run python examples/slab_polarizer.py --projection all`,
  `uv run python examples/altermagnetic_dot.py --projection all`, and
  `uv run python examples/gallery` after touching geometry or projection, and
  look at `examples/out/gallery/*.png` after touching a solid, an outline, or
  the depth sort. Refresh `docs/gallery/` with `--docs` when a figure changes. It is the realistic end-to-end check, and it renders
  pictures whose correctness is visible. Mirrored or upside-down content, or a
  beam that misses the slab, is the usual symptom of a projection bug.
- Keep output deterministic: layer ties resolve by insertion order, coordinates
  are rounded on emission. Byte-identical output is what keeps a figure diffable.

## Examples

The examples -- `examples/*.py`, `examples/gallery/`, and `docs/readme_figure.py`
-- are the showcase. People read them to learn the library, so they must read as
the obvious way to draw the figure, not as a record of how it was first got to
work.

- After adding a public function or parameter, go through every example and use
  it wherever it makes the code shorter or clearer, in the same piece of work.
  A helper an example wrote for itself that the library now provides is
  deleted, not kept beside the new call.
- Prefer the library's vocabulary to hand-rolled geometry: `extrude` over axis
  permutations, `place` over copy-pasted blocks, `cull=True` over
  `cam.visible`, `class_` for kinds of objects.
- Name things for what they are in the figure (`gate`, `contact`, `beam`), keep
  each function to one part of the picture, and comment the physics or the
  design decision, not the Python.
- A simplification must not change the picture: check the rendered PNGs, and
  say so if an id or a pixel changed on purpose.

## Testing

- Test geometry as numbers, not as rendered strings: winding against declared
  normal, a wave against an exact sine, a circle's radius. Reserve
  rendered-output assertions for document structure.
- Add a regression test for every projection or winding bug — a wrong normal is
  invisible until it culls the wrong wall.
- Test a projection through `foreshortening()`, not through its construction
  angles: the ratios are what the axonometric classification is defined by.
- Rendering under a camera must stay byte-identical to building the scene with
  that camera from the start. The parametrized test in `tests/test_scene.py`
  covers all five projections.
- `tests/test_document.py` pins what a consumer relies on: a parseable
  standalone document, a viewBox agreeing with `width`/`height`, geometry inside
  it, and deterministic bytes. It must not import a consumer.

## Documentation

Documentation is part of every change, not a follow-up. A change is not complete,
and must not be committed, until the documentation agrees with it.

- Every user-visible change -- a new call or parameter, a changed default or
  behaviour, a new error, a renamed id -- updates, in the same commit: the
  docstrings, the relevant page in `docs/`, `README.md` wherever it summarizes
  the feature, and `CHANGELOG.md` (Keep a Changelog 1.1.0).
- Documentation states what the code does, checked against the code: every
  signature, default, generated id, and error matches the source, and every
  runnable snippet runs. Do not document intentions or planned features.
- When a change makes a passage elsewhere wrong, fix that passage in the same
  commit, even on a page the change did not otherwise touch.
- When a figure's output changes, refresh `docs/gallery/` with
  `uv run python examples/gallery --docs`, and `docs/images/` with
  `uv run python docs/readme_figure.py`.
- Run `uv run zensical build` and confirm it reports no issues before reporting
  the change complete.

## Naming and packaging

- **`vecview` is the distribution and import name**, published on PyPI since
  0.2.0. See `docs/development.md` for the rename procedure, should it ever be
  needed.
- `pyproject.toml` is canonical for Python metadata.
- Do not commit, tag, upload, or publish unless the user explicitly asks.
- Before 1.0, make API changes directly: update consumers, tests, and docs in the
  same change. No compatibility aliases or deprecated wrappers.
