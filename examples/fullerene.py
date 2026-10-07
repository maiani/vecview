"""Buckminsterfullerene, C60, in ball-and-stick.

Sixty atoms at the corners of a truncated icosahedron and the ninety bonds
between nearest neighbours, sorted together by depth.  Each bond is a cylinder
cut back to the surfaces of the two atoms it joins: a bond running into an
atom's centre would be partly inside the ball, which no depth order draws
correctly, while one that stops at the surface sorts exactly.  The thirty
double bonds, shared by two hexagons, are drawn darker.

Uses: ``sphere(highlight=...)``, ``cylinder(ends=False, highlight=...)``,
``sort_by_depth``, and ``convex_polyhedron`` to find the pentagons.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import vecview
from vecview import OrthographicCamera, Scene

GOLDEN = (1.0 + 5.0**0.5) / 2.0
BOND_LENGTH = 2.0  # the nearest-neighbour distance at these coordinates
ATOM_R = 0.42
BOND_R = 0.13


def atom_positions() -> np.ndarray:
    """Even permutations of (0, +-1, +-3g), (+-1, +-(2+g), +-2g), (+-g, +-2, +-g^3)."""
    seeds = [(0.0, 1.0, 3 * GOLDEN), (1.0, 2 + GOLDEN, 2 * GOLDEN), (GOLDEN, 2.0, GOLDEN**3)]
    points: set[tuple[float, ...]] = set()
    for seed in seeds:
        for signs in itertools.product((1.0, -1.0), repeat=3):
            p = [s * v for s, v in zip(signs, seed, strict=True)]
            for shift in range(3):  # cyclic permutations are the even ones
                points.add(tuple(round(p[(k + shift) % 3], 9) + 0.0 for k in range(3)))
    return np.array(sorted(points))


def pentagon_bonds(atoms: np.ndarray) -> set[frozenset[int]]:
    """The bonds round the twelve pentagons, read off the faces of the cage."""
    found: set[frozenset[int]] = set()
    for face in vecview.convex_polyhedron(atoms):
        if len(face.points) == 5:
            ring = [int(np.argmin(np.linalg.norm(atoms - p, axis=1))) for p in face.points]
            found |= {frozenset(pair) for pair in zip(ring, ring[1:] + ring[:1], strict=True)}
    return found


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=20.0, elev_deg=24.0, scale=48.0)
    scene = Scene(cam, pad=12.0, background="#ffffff")
    scene.sort_by_depth(10)
    atoms = atom_positions()
    assert len(atoms) == 60

    dist = np.linalg.norm(atoms[:, None] - atoms[None, :], axis=2)
    bonds = [
        (i, j) for i, j in zip(*np.nonzero(np.isclose(dist, BOND_LENGTH)), strict=True) if i < j
    ]
    assert len(bonds) == 90
    # Every bond borders two faces of the cage.  The double bonds are the
    # thirty shared by two hexagons: the ones on no pentagon.
    single = pentagon_bonds(atoms)
    assert len(single) == 60

    # Each bond stops just inside the two atom surfaces, so its ends are covered.
    inset = np.sqrt(ATOM_R**2 - BOND_R**2) - 0.02
    for k, (i, j) in enumerate(bonds):
        double = frozenset((i, j)) not in single
        a, b = atoms[i], atoms[j]
        d = (b - a) / np.linalg.norm(b - a)
        scene.cylinder(
            10,
            a + inset * d,
            b - inset * d,
            BOND_R,
            ends=False,
            fill="#4f5560" if double else "#c3c8cf",
            highlight="#b9bfc8" if double else "#f7f8f9",
            stroke="#3c4048",
            stroke_width=0.6,
            id=f"bond-{k}",
            class_=["bond", "double" if double else "single"],
        )
    for k, p in enumerate(atoms):
        scene.sphere(
            10,
            p,
            ATOM_R,
            fill="#3a3f47",
            highlight="#aab1bc",
            stroke="#16181c",
            stroke_width=0.7,
            id=f"C-{k}",
            class_="atom",
        )
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "fullerene.svg"))
