# Gallery

The figures a physics or chemistry paper keeps redrawing, each built from code by
one script in `examples/gallery/` and shown here as the SVG it writes. Every one
is a few dozen lines of geometry: no layer is assigned per object, no occlusion
is worked out by hand, and every element keeps the id it was given, so the file
opens in Inkscape as named objects.

Build them all, with timings, from a checkout:

```bash
uv run python examples/gallery          # SVG and PNG to examples/out/gallery
uv run python examples/gallery --docs   # and refresh the SVGs on this page
```

Each script also runs on its own, `uv run python examples/gallery/perovskite.py`.
The mirror planes need the `occlusion` extra, and are skipped without it.

## Crystal structure

![A cubic perovskite unit cell](gallery/perovskite.svg)

A cubic perovskite ABO₃ cell: A cations on the corners, the B cation inside a
translucent BO₆ octahedron, and the cell edges dashed where the cell hides them.
Atoms, edges, and octahedron faces share one
[depth-sorted layer](scenes.md#sorting-by-depth). The octahedron comes from its
six corners alone through [`convex_polyhedron`](shapes.md#polyhedra), and both
its faces and the cell edges are cut back to the atom surfaces with
[`trim_corners`](shapes.md#polyhedra) and `edges(trim=...)`, which is what makes
the depth order exact where they meet.

`examples/gallery/perovskite.py`

## Brillouin zone

![The first Brillouin zone of the fcc lattice](gallery/brillouin_zone.svg)

The truncated octahedron of the fcc lattice, from its 24 corners: coplanar
triangles merge into eight hexagons and six squares. Hidden edges are dashed and
sit under the translucent faces, and the high-symmetry path Γ–X–W–K–Γ–L–U–W–L–K
runs inside. The axis labels are subscripted with `svg.TSpan` runs.

`examples/gallery/brillouin_zone.py`

## Molecule

![C60 in ball-and-stick](gallery/fullerene.svg)

Buckminsterfullerene: sixty atoms and ninety bonds, sorted together by depth.
Each bond is a [`cylinder`](scenes.md#curved-solids) cut back to the two atom
surfaces, and the thirty double bonds are drawn darker. Spheres and cylinders
are exact outlines — a `<circle>` per atom, and two lines and two elliptical
arcs per bond — so the file is small and each atom is one circle in an editor.

`examples/gallery/fullerene.py`

## Bloch sphere

![The Bloch sphere](gallery/bloch_sphere.svg)

A qubit state at polar angle θ and azimuth φ. The equator and a meridian are
[split exactly where they pass behind the sphere](scenes.md#hidden-lines) and
dashed there; the state is a solid [`arrow3d`](scenes.md#curved-solids), and the
angles are [`arc_shape`](shapes.md#arcs-and-helices) arcs in their own planes.

`examples/gallery/bloch_sphere.py`

## Band structure

![A gapped Dirac cone](gallery/dirac_cone.svg)

A gapped Dirac cone, E = ±√(k² + Δ²), with a chemical potential cutting the upper
band. Both bands are meshes from [`surface_faces`](shapes.md#surfaces), coloured
by energy, sorted together with the axes and the Fermi circle, which are thin
[tubes](scenes.md#curved-solids). The energy axis passes behind the front wall of
the upper band and shows again inside it, with no layer assigned by hand.

`examples/gallery/dirac_cone.py`

## Spin texture

![A Néel skyrmion](gallery/skyrmion.svg)

A Néel skyrmion: 289 solid spins on a square lattice, coloured by their
out-of-plane component, on a thin film. Every spin is one `<g>` named after its
lattice site, and the depth sort handles every overlap at any viewing angle.

`examples/gallery/skyrmion.py`

## Coil

![A solenoid](gallery/solenoid.svg)

A copper coil wound on a core. The coil is one tube along a
[`helix`](shapes.md#arcs-and-helices), and the core is a cylinder cut into
[slices](scenes.md#long-objects) along its length, so each turn passes behind the
core and comes round in front of it again.

`examples/gallery/solenoid.py`

## Mirror planes

![The three mirror planes of a cubic cell](gallery/mirror_planes.svg)

The three mirror planes of a cubic cell, crossing at its centre. Each plane is
partly in front of and partly behind each of the others — a cycle that no order of
drawing whole planes gets right. With
[exact visibility](scenes.md#exact-visibility) each plane is clipped to what shows
of it, the body diagonal is dashed exactly where a plane hides it, and the cell's
dashed back edges disappear where the planes cover them.

`examples/gallery/mirror_planes.py`

## Devices

![A quantum-dot device sketch](gallery/altermagnetic_dot.svg)

![A polarizing slab](gallery/slab_polarizer.svg)

The two worked examples one level up, `examples/altermagnetic_dot.py` and
`examples/slab_polarizer.py`, at their default projections. They predate the
curved solids and the depth sort, and are layered by hand: the right tool for a
handful of large, flat parts with a beam through them.
