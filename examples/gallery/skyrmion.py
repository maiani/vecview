"""A Néel skyrmion: 289 solid spins on a square lattice, on a thin film.

Each spin is an ``arrow3d`` -- a shaded shaft and head in one group -- coloured
by its out-of-plane component.  They share one depth-sorted layer, so spins in
front cover spins behind them at every viewing angle with no layers assigned by
hand, and the whole scene could be replayed under another camera.

Uses: ``arrow3d(highlight=...)``, ``sort_by_depth``, ``faces(cull=True)``.
"""

from __future__ import annotations

import numpy as np
from _common import export, mix

import vecview
from vecview import OrthographicCamera, Scene

NAME = "skyrmion"

N = 17  # spins per side
RADIUS = 3.2  # where m_z changes sign, in lattice constants
WALL = 1.3  # domain-wall width
UP, DOWN, MID = "#c0392b", "#1f5fa8", "#ece9e4"


def polar_angle(r: np.ndarray) -> np.ndarray:
    """Theta(r) for the standard 360-degree domain-wall profile: pi at the core, 0 outside."""
    return 2.0 * np.arctan2(np.sinh(RADIUS / WALL), np.sinh(r / WALL))


def color(mz: float) -> str:
    return mix(MID, UP, mz) if mz >= 0 else mix(MID, DOWN, -mz)


def build() -> Scene:
    cam = OrthographicCamera(azim_deg=-60.0, elev_deg=32.0, scale=34.0)
    scene = Scene(cam, pad=12.0, background="#ffffff")
    half = (N - 1) / 2.0

    film = vecview.box_faces((0, 0, -0.75), (N + 0.4, N + 0.4, 0.3))
    scene.faces(5, film, cull=True, fill="#e6e8ec", stroke="#9aa1ab", stroke_width=0.8, id="film")
    scene.sort_by_depth(10)
    for i in range(N):
        for j in range(N):
            x, y = i - half, j - half
            r = float(np.hypot(x, y))
            theta = float(polar_angle(np.array(r)))
            radial = np.array([x, y]) / r if r > 0 else np.zeros(2)
            m = np.array([np.sin(theta) * radial[0], np.sin(theta) * radial[1], np.cos(theta)])
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
            )
    return scene


if __name__ == "__main__":
    export(build(), NAME)
