"""A Néel skyrmion: 289 solid spins on a square lattice, on a thin film.

Each spin is an ``arrow3d`` -- a shaded shaft and head in one group -- coloured
by its out-of-plane component.  They share one depth-sorted layer, so spins in
front cover spins behind them at every viewing angle with no layers assigned by
hand, and the whole scene could be replayed under another camera.

Uses: ``arrow3d(highlight=...)``, ``sort_by_depth``, ``faces(cull=True)``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import vecview
from vecview import OrthographicCamera, Scene

N = 17  # spins per side
RADIUS = 3.2  # where m_z changes sign, in lattice constants
WALL = 1.3  # domain-wall width
UP, DOWN, MID = "#c0392b", "#1f5fa8", "#ece9e4"


def mix(a: str, b: str, t: float) -> str:
    """The colour a fraction ``t`` of the way from hex ``a`` to hex ``b``."""
    t = min(max(t, 0.0), 1.0)
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


def spin(x: float, y: float) -> np.ndarray:
    """The unit magnetization at (x, y): down at the core, up outside, radial in the wall.

    The polar angle follows the standard 360-degree domain-wall profile.
    """
    r = float(np.hypot(x, y))
    theta = 2.0 * np.arctan2(np.sinh(RADIUS / WALL), np.sinh(r / WALL))
    radial = np.array([x, y]) / r if r > 0 else np.zeros(2)
    return np.array([np.sin(theta) * radial[0], np.sin(theta) * radial[1], np.cos(theta)])


def color(mz: float) -> str:
    """Red for up, blue for down, through a pale grey in the plane."""
    return mix(MID, UP, mz) if mz >= 0 else mix(MID, DOWN, -mz)


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=-60.0, elev_deg=32.0, scale=34.0)
    scene = Scene(cam, pad=12.0, background="#ffffff")
    half = (N - 1) / 2.0

    film = vecview.box_faces((0, 0, -0.75), (N + 0.4, N + 0.4, 0.3))
    scene.faces(5, film, cull=True, fill="#e6e8ec", stroke="#9aa1ab", stroke_width=0.8, id="film")
    scene.sort_by_depth(10)
    for i, j in itertools.product(range(N), repeat=2):
        x, y = i - half, j - half
        m = spin(x, y)
        fill = color(float(m[2]))
        scene.arrow3d(
            10,
            (x, y, 0.0),
            m,
            0.9,
            pivot="mid",
            shaft_r=0.07,
            head_r=0.18,
            head_len=0.36,
            fill=fill,
            highlight=mix(fill, "#ffffff", 0.6),
            stroke=mix(fill, "#000000", 0.55),
            stroke_width=0.5,
            id=f"spin-{i}-{j}",
            class_="spin",
        )
    return scene


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(build().save(out / "skyrmion.svg"))
