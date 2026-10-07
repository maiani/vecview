# Development

## Setup

```bash
uv sync --all-extras
```

## Validation

Run all four before reporting a change complete:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

The examples are also part of the surface being maintained. Run them after
touching geometry or projection, and look at `examples/out/*.png` after
touching a solid, an outline, or the depth sort:

```bash
uv run python examples/slab_polarizer.py --projection all
uv run python examples/altermagnetic_dot.py --projection all
uv run python examples            # --docs also refreshes docs/gallery/
```

They are the realistic end-to-end check, and their correctness is visible:
mirrored or upside-down content, or a beam that misses the slab, is the usual
symptom of a projection bug.

## The name

The distribution and import name is **`vecview`**, published on
[PyPI](https://pypi.org/project/vecview/). In prose the project is VecView.

If it needs to change in the future, the rename is mechanical — the import name
appears only in this repository's own code, tests, examples, metadata, and docs:

```bash
git grep -l vecview | xargs sed -i 's/vecview/newname/g'
git mv src/vecview src/newname
```

Avoid `axo-` names unless the package is intentionally committed to parallel
projection forever: axonometric excludes perspective and names the isometric,
dimetric, and trimetric cases.

## Design constraints

- **Runtime dependencies stay `numpy`, `svg.py`, `shapely`, and `contourpy`.**
  Rasterizing, PDF, and TeX belong to a consumer, not here. A test enforces
  this. `shapely` clips polygons and `contourpy` traces depth contours; both
  load only when they are first needed.
- **Keep the three-way split**: `shapes` knows only numbers, `camera` and
  `projections` know projection, and `scene` and `animation`, with their
  private rendering modules (`_elements`, `_canvas`, `_solids`,
  `_animation_svg`, `_animation_refs`), are the only code that touches
  `svg.py`. Geometry that needs a camera to be computed does not belong in
  `shapes`.
- **A scene records; a canvas projects.** Each public drawing method on `Scene`
  validates what it can without a camera and appends one record. Rendering
  replays the record onto a private canvas holding the camera, which is where
  camera-dependent work and errors belong.
- **Layers first.** A layer is depth-sorted only when it asks to be; see
  [Scenes](scenes.md#sorting-by-depth). Back-face culling of convex solids lives
  on `Camera`, not `Scene`.
- **Highlights are fill styles.** A highlight gradient sits in a fixed place on
  screen; there is no light direction, material, or shading model.
- **Degrees at the API surface**, radians nowhere.
- **Deterministic output.** Same scene, same bytes: layer ties resolve by
  insertion order and coordinates are rounded on emission, so a figure stays
  diffable in version control.
- **Before 1.0, make API changes directly** — update consumers, tests, and docs
  in the same change. No compatibility aliases or deprecated wrappers.

## Animation

`vecview.animation` holds the public names; the work is split by what it
touches:

| Module | Responsibility |
| --- | --- |
| `animation.py` | `Animation`: options, a scene of tracks or a time-to-scene callback, export. |
| `_tracks.py` | `Track`, interpolation, and easing: values only, no scenes or SVG. |
| `_transforms.py` | `rotate` and `scale` about a pivot, through `Part.place`. |
| `_animation_svg.py` | Sampling, sharing what frames draw alike, and SVG timing. |
| `_animation_refs.py` | Finding ids and references, and renaming a frame's. |
| `_numeric.py` | The finite-number checks the rest of the animation API shares. |

A sample is the static renderer's output: `_Canvas._elements` assembles a
scene's projected elements without fitting a viewBox. Keep it so -- an animation
must draw every frame exactly as a still would. Frames are compared by their
markup, since an element can hold content the dataclass comparison does not
see. Timing is checked in real browsers, not only in tests: a change to it
needs Chromium and Firefox to show the right frame at seeked times. Seek from
the `load` event: Firefox seeks an animated `href` wrongly past the first cycle
before the document has loaded, though it plays correctly.

## Testing

- Geometry is tested as numbers, not as rendered output: check that a face's
  winding matches its declared normal, that a wave is an exact sine, that a
  circle's points sit at the radius. Rendered-string assertions are for document
  structure only.
- Add a regression test for every projection or winding bug. A wrong normal is
  invisible until it culls the wrong wall.
- Test a projection through `foreshortening()`, not through its construction
  angles: the ratios are what the axonometric classification is defined by.
- Rendering under a camera must stay byte-identical to building the scene with
  that camera from the start; a parametrized test in `tests/test_scene.py` covers
  all five projections.
- `tests/test_document.py` pins the properties a consumer relies on: a parseable
  standalone document, a viewBox that agrees with `width`/`height`, geometry
  inside it, and byte-identical output across calls. It must not import any
  consumer — this package has no downstream dependencies, and adding one to a
  test would invert that.

## Documentation

Keep `docs/` in sync with user-visible changes in the same commit, and build
before reporting documentation work complete:

```bash
uv run zensical build
```

## Releases

`.github/workflows/publish.yml` publishes tags named `v<version>`. It checks
that the tag matches `pyproject.toml`, runs the CI matrix against that tag,
builds both distributions (the wheel from the sdist), and runs
`twine check --strict` before publishing the same artifacts. Python 3.15 is
experimental; its job may fail while the supported 3.12–3.14 jobs must pass.

Publishing goes through a
[Trusted Publisher on PyPI](https://docs.pypi.org/trusted-publishers/), configured
with these exact values:

| Field | Value |
| --- | --- |
| Project name | `vecview` |
| GitHub owner | `maiani` |
| Repository | `vecview` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

The GitHub repository has the matching `pypi` environment. Publishing
uses GitHub OIDC, with `id-token: write` granted only to the publish job;
no PyPI API token is needed. See
[PyPI's publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

For a release:

1. Update the version in `pyproject.toml` and `src/vecview/__init__.py` together.
2. Move the relevant `Unreleased` changelog entries into a dated release section.
   Update versioned README image and documentation URLs to the new tag.
3. Run validation and build the wheel and sdist. Commit the release changes.
4. Create and push the matching tag, such as `v0.2.0`. Its push triggers publishing.

The workflow can also be dispatched manually with an existing tag to retry
an upload. It always checks out `refs/tags/<tag>`, including for CI and builds.
Versions already uploaded to PyPI cannot be overwritten; fix release content
under a new version.
