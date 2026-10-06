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

The examples are also part of the surface being maintained:

```bash
uv run python examples/slab_polarizer.py --projection all
uv run python examples/altermagnetic_dot.py --projection all
```

## The name

The distribution and import name is **`vecview`**. It was confirmed available on
PyPI on 31 August 2026. Check availability again immediately before publishing.

If it needs to change in the future, the rename is mechanical — the import name
appears nowhere outside `src/vecview/`, its own tests, and the docs:

```bash
git grep -l vecview | xargs sed -i 's/vecview/newname/g'
git mv src/vecview src/newname
```

Avoid `axo-` names unless the package is intentionally committed to parallel
projection forever: axonometric excludes perspective and names the isometric,
dimetric, and trimetric cases.

## Design constraints

- **Runtime dependencies stay `numpy` and `svg.py`.** Rasterizing, PDF, and TeX
  belong to a consumer, not here. A test enforces this. Exact visibility uses
  `shapely` and `contourpy` from the optional `occlusion` extra, imported only
  when an exact layer renders.
- **Keep the three-way split**: `shapes` knows only numbers, `camera` knows
  projection, `scene` is the only module that touches `svg.py`. Geometry that
  needs a camera to be computed does not belong in `shapes`.
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

## Testing

- Geometry is tested as numbers, not as rendered output: check that a face's
  winding matches its declared normal, that a wave is an exact sine, that a
  circle's points sit at the radius. Rendered-string assertions are for document
  structure only.
- Add a regression test for every projection or winding bug. A wrong normal is
  invisible until it culls the wrong wall.
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
