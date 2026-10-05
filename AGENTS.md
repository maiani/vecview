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
  - `camera.py` defines what a camera *is*: `Camera` (the abstract contract) and
    `ParallelCamera` (the affine machinery). Back-face culling lives here
    because it is a camera question, not a scene one.
  - `projections.py` holds the concrete projections: `OrthographicCamera` and
    `ObliqueCamera`.
  - `scene.py` is the **only** module that imports `svg`.
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
  `_Canvas` holds a camera and implements each method of the same name; rendering
  replays the record onto a fresh canvas. Camera-dependent work and errors
  belong in `_Canvas`, and a canvas never records, so its methods may call each
  other freely.

## Deliberate non-features

Do not add these without the user changing the design first:

- **Automatic depth sorting across layers** (a z-buffer, or sorting a layer that
  did not ask). Layers are the model. A beam crossing a translucent slab has
  three parts that no automatic depth rule orders correctly. Depth only ever
  acts inside a layer passed to `Scene.sort_by_depth`: as the painter's
  algorithm, whose failures are fixed by cutting geometry back (`edges(trim=)`,
  `trim_corners`) or into pieces (`cylinder(slices=)`, `tube` chunks), or, with
  `exact=True`, as exact visibility by clipping.
- **Rasterizing or PDF export.** Runtime dependencies stay `numpy` and `svg.py`;
  a test in `tests/test_package_metadata.py` enforces it. `shapely` and
  `contourpy` are the optional `occlusion` extra: only `_occlusion.py` imports
  them, inside `resolve`, so nothing but an exact layer needs them, and its
  tests use `pytest.importorskip`.
- **Shading, materials, or lighting models.** This draws schematics, not renders.
  `highlight=` gradients are a fill style fixed on screen, with no light
  direction; keep them that way.

## Development

- Supported Python: 3.12 and newer.
- `uv sync --all-extras` for a full environment.
- Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src`,
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
- Keep `docs/` in sync with user-visible changes in the same change, and run
  `uv run zensical build` before reporting documentation work complete.
- Keep `CHANGELOG.md` current in Keep a Changelog 1.1.0 format.

## Naming and packaging

- **`vecview` is the intended distribution and import name.** Confirm PyPI
  availability immediately before a first publication. See `docs/development.md`
  for the rename procedure.
- `pyproject.toml` is canonical for Python metadata.
- Do not commit, tag, upload, or publish unless the user explicitly asks.
- Before 1.0, make API changes directly: update consumers, tests, and docs in the
  same change. No compatibility aliases or deprecated wrappers.
