"""Buckminsterfullerene, C60, in ball-and-stick.

Sixty atoms at the corners of a truncated icosahedron and the ninety bonds
between nearest neighbours, sorted together by depth.  Each bond is a cylinder
cut back to the surfaces of the two atoms it joins: a bond running into an
atom's centre would be partly inside the ball, which no depth order draws
correctly, while one that stops at the surface sorts exactly.  The thirty
double bonds, shared by two hexagons, are drawn darker.

Uses: ``sphere(highlight=...)``, ``cylinder(ends=False, highlight=...)``,
``sort_by_depth``.
"""

from __future__ import annotations

import itertools

import numpy as np
from _common import export

from vecview import OrthographicCamera, Scene

NAME = "fullerene"

GOLDEN = (1.0 + 5.0**0.5) / 2.0
ATOM_R = 0.42
BOND_R = 0.13


def atoms() -> np.ndarray:
    """Even permutations of (0, +-1, +-3g), (+-1, +-(2+g), +-2g), (+-g, +-2, +-g^3)."""
    seeds = [(0.0, 1.0, 3 * GOLDEN), (1.0, 2 + GOLDEN, 2 * GOLDEN), (GOLDEN, 2.0, GOLDEN**3)]
    points: set[tuple[float, ...]] = set()
    for seed in seeds:
        for signs in itertools.product((1.0, -1.0), repeat=3):
            p = [s * v for s, v in zip(signs, seed, strict=True)]
            for shift in range(3):  # cyclic permutations are the even ones
                points.add(tuple(round(p[(k + shift) % 3], 9) + 0.0 for k in range(3)))
    return np.array(sorted(points))


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=20.0, elev_deg=24.0, scale=48.0)
    scene = Scene(cam, pad=12.0, background="#ffffff")
    scene.sort_by_depth(10)
    pts = atoms()
    assert len(pts) == 60

    # Bonds join nearest neighbours, at the edge length 2.  Every atom is in
    # exactly one pentagon, so a bond is double -- shared by two hexagons --
    # exactly when its two atoms are in different pentagons.
    dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    bonds = [(i, j) for i, j in zip(*np.nonzero(np.isclose(dist, 2.0)), strict=True) if i < j]
    assert len(bonds) == 90
    neighbours = {
        i: {j for a, b in bonds for i2, j in ((a, b), (b, a)) if i2 == i} for i in range(60)
    }
    pentagons = {frozenset(cycle) for a in range(60) for cycle in _five_cycles(a, neighbours)}

    for k, (i, j) in enumerate(bonds):
        double = not any({i, j} <= p for p in pentagons)
        a, b = pts[i], pts[j]
        d = (b - a) / np.linalg.norm(b - a)
        # Cut back to just inside each atom's surface, so the end is covered.
        inset = np.sqrt(ATOM_R**2 - BOND_R**2) - 0.02
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
        )
    for k, p in enumerate(pts):
        scene.sphere(
            10,
            p,
            ATOM_R,
            fill="#3a3f47",
            highlight="#aab1bc",
            stroke="#16181c",
            stroke_width=0.7,
            id=f"C-{k}",
        )
    return scene


def _five_cycles(start: int, neighbours: dict[int, set[int]]) -> list[tuple[int, ...]]:
    """Simple 5-cycles through ``start``: the pentagons that atom belongs to."""
    found = []
    for path in _walks(start, neighbours, 5):
        if start in neighbours[path[-1]]:
            found.append(path)
    return found


def _walks(start: int, neighbours: dict[int, set[int]], length: int) -> list[tuple[int, ...]]:
    paths = [(start,)]
    for _ in range(length - 1):
        paths = [(*p, n) for p in paths for n in sorted(neighbours[p[-1]]) if n not in p]
    return paths


if __name__ == "__main__":
    export(build(), NAME)
