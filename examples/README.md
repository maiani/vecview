# Examples

Every example is one self-contained script: it builds its figure in a `build()`
function and, run on its own, writes the SVG to `examples/out/`. Read any one of
them without the others.

```bash
uv run python examples/perovskite.py                       # one figure
uv run python examples/slab_polarizer.py --projection all  # under all five projections
uv run python examples/altermagnetic_dot.py --projection all
uv run python examples/pendulum.py                         # an animated SVG
```

From the repository root, `uv sync --all-extras` installs what they need, and

```bash
uv run python examples           # all of them, with timings and PNG previews
uv run python examples --docs    # and refresh the SVGs in docs/gallery/
```

builds them all. The PNG previews and the two device sketches use CairoSVG,
which is example-only: rasterizing is not VecView's job. The figures with
mathematical labels -- the Bloch sphere, the Brillouin zone, the Dirac cone, the
solenoid, the Kitaev chain, and the pendulum -- typeset them with
[VecTeX](https://github.com/maiani/vectex) 0.3 or newer
(`pip install 'vectex>=0.3'`), which needs a TeX engine (`pdflatex`, `xelatex`,
or `lualatex`) and `dvisvgm` on `PATH`; TeX Live and MiKTeX ship both. dvisvgm
reads PDF through MuPDF's `mutool` when Ghostscript is 10.01 or newer
(`mupdf-tools` on Debian and Ubuntu). Each renders its labels in one TeX run, with
`vectex.render_many`, and holds them upright at their points with
`Scene.slot(..., content=...)`.

## Figures

- `perovskite.py` — perovskite crystal cell
- `brillouin_zone.py` — fcc Brillouin zone and high-symmetry path
- `fullerene.py` — C60 ball-and-stick model
- `bloch_sphere.py` — Bloch sphere
- `dirac_cone.py` — gapped Dirac cone
- `skyrmion.py` — Néel skyrmion
- `solenoid.py` — solenoid on a core
- `kitaev_chain.py` — Kitaev chain in a nanowire
- `mirror_planes.py` — mirror planes in a cubic cell
- `altermagnetic_dot.py` — layered quantum-dot device
- `slab_polarizer.py` — spin texture and polarizing slab

## Animation

- `pendulum.py` — a pendulum swinging through one seamless period, with its
  equations typeset by TeX; tracks for the bob's position and the readouts
