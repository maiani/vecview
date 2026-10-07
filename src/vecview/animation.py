"""Animations: a scene for every time, exported as one self-playing SVG.

``Animation`` is the foundation, and any motion a frame callback can draw it can
play.  ``Track`` and the interpolation helpers turn keyframes into values, and
``rotate`` and ``scale`` turn or grow a part about a pivot.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from vecview._animation_svg import render_animation
from vecview._numeric import finite_number as _number
from vecview._tracks import Track, hold, linear, smoothstep
from vecview._transforms import rotate, scale
from vecview.scene import Scene

if TYPE_CHECKING:
    import svg


class Animation:
    """A pure seconds-to-Scene function exported as sampled SVG frames.

    Each callback result is a complete scene, so geometry, layers, styles and
    the active camera may change over time. The callback must return the same
    scene for a given time, independently of call order.

    Args:
        frame: Pure callback from seconds in ``[0, duration]`` to a ``Scene``.
        duration: Finite positive cycle length in seconds.
        view_box: Fixed ``(min_x, min_y, width, height)`` in projected SVG units.
        fps: Positive requested sample rate; defaults to 30.
        repeat: Positive integer total cycle count, or ``None`` to loop forever.
        background: Constant background fill; ``None`` is transparent.

    Scene padding and backgrounds do not affect animation output. Frame scenes
    need an active camera when rendered. Invalid option types raise
    ``TypeError``; invalid values raise ``ValueError``.
    """

    def __init__(
        self,
        frame: Callable[[float], Scene],
        *,
        duration: float,
        view_box: tuple[float, float, float, float],
        fps: float = 30.0,
        repeat: int | None = 1,
        background: str | None = None,
    ) -> None:
        """Store a callback and export settings without evaluating ``frame``."""
        if not callable(frame):
            raise TypeError("frame must be callable")
        self.duration = _number(duration, "duration")
        self.fps = _number(fps, "fps")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if isinstance(view_box, (str, bytes)):
            raise TypeError("view_box must contain four numbers")
        try:
            box = tuple(view_box)
        except TypeError as exc:
            raise TypeError("view_box must contain four numbers") from exc
        if len(box) != 4:
            raise ValueError("view_box must contain four numbers")
        x, y, width, height = (_number(v, "view_box coordinate") for v in box)
        if width <= 0 or height <= 0:
            raise ValueError("view_box width and height must be positive")
        if repeat is not None and (isinstance(repeat, bool) or not isinstance(repeat, int)):
            raise TypeError("repeat must be a positive integer or None")
        if repeat is not None and repeat <= 0:
            raise ValueError("repeat must be positive or None")
        if background is not None and not isinstance(background, str):
            raise TypeError("background must be a string or None")
        self._frame = frame
        self.view_box = (x, y, width, height)
        self.repeat = repeat
        self.background = background

    def frame(self, t: float) -> Scene:
        """Evaluate one cycle at exact seconds ``t`` in the closed interval.

        No wrapping, quantization, caching or mutation occurs. A non-finite or
        out-of-range time raises ``ValueError``; a non-Scene callback result
        raises ``TypeError``. Callback exceptions propagate.
        """
        seconds = _number(t, "time")
        if seconds < 0 or seconds > self.duration:
            raise ValueError(f"time must be in [0, {self.duration}]")
        result = self._frame(seconds)
        if not isinstance(result, Scene):
            raise TypeError(f"frame callback must return Scene, got {type(result).__name__}")
        return result

    def render(self) -> svg.SVG:
        """The animated document, as an inspectable SVG tree.

        Samples ``N = max(1, ceil(duration * fps))`` evenly spaced times per
        cycle, plus the exact frame at ``duration`` for finite playback, which
        is held once playback ends.  What every sample draws alike is written
        once, in place, and each distinct content of what changes once.
        """
        return render_animation(
            self.frame,
            duration=self.duration,
            view_box=self.view_box,
            fps=self.fps,
            repeat=self.repeat,
            background=self.background,
        )

    def to_svg_document(self) -> str:
        """Return the deterministic standalone SVG from :meth:`render`."""
        return str(self.render())

    def save(self, path: str | Path) -> Path:
        """Write animated SVG as UTF-8 and return the output path."""
        out = Path(path)
        out.write_text(self.to_svg_document(), encoding="utf-8")
        return out


__all__ = [
    "Animation",
    "Track",
    "hold",
    "linear",
    "rotate",
    "scale",
    "smoothstep",
]
