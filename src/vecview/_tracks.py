"""Time-to-value tracks, interpolation and easing; no scenes or SVG."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from typing import overload

from vecview._numeric import finite_number as _number
from vecview._numeric import finite_vector3 as _vector3
from vecview._types import Point3


class Track[T]:
    """A reusable pure function from finite seconds to a value.

    Tracks do not know an animation's duration or repeat settings. Ordinary
    callables work just as well in a frame callback.
    """

    def __init__(self, sample: Callable[[float], T]) -> None:
        """Wrap a callable taking seconds and returning any value."""
        if not callable(sample):
            raise TypeError("sample must be callable")
        self._sample = sample

    def __call__(self, t: float) -> T:
        """Evaluate the sample callback at finite seconds ``t``."""
        return self._sample(_number(t, "time"))

    @classmethod
    def keyframes(
        cls,
        keys: Sequence[tuple[float, T]],
        *,
        interpolate: Callable[[T, T, float], T],
        ease: Callable[[float], float] | None = None,
    ) -> Track[T]:
        """Build piecewise interpolated keys, holding the endpoint values.

        Times must be finite and strictly increasing in the supplied order.
        At exact key times, values are returned without calling ``ease`` or
        ``interpolate``. Between keys, ``ease`` remaps local progress before
        interpolation; its finite result may overshoot. Keys are snapshotted,
        but callers should treat their values as immutable.
        """
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
            raise TypeError("keys must be a sequence of (time, value) pairs")
        if not keys:
            raise ValueError("keys must not be empty")
        if not callable(interpolate):
            raise TypeError("interpolate must be callable")
        if ease is not None and not callable(ease):
            raise TypeError("ease must be callable or None")
        snapshot: list[tuple[float, T]] = []
        for index, entry in enumerate(keys):
            if not isinstance(entry, Sequence) or len(entry) != 2:
                raise TypeError(f"key {index} must be a (time, value) pair")
            at = _number(entry[0], f"key {index} time")
            if snapshot and at <= snapshot[-1][0]:
                raise ValueError("key times must be strictly increasing")
            snapshot.append((at, entry[1]))

        times = [at for at, _ in snapshot]

        def sample(t: float) -> T:
            i = bisect_right(times, t)
            if i == 0:
                return snapshot[0][1]
            if i == len(snapshot):
                return snapshot[-1][1]
            (a, left), (b, right) = snapshot[i - 1], snapshot[i]
            if t == a:
                return left
            progress = (t - a) / (b - a)
            eased = progress if ease is None else _number(ease(progress), "ease result")
            return interpolate(left, right, eased)

        return cls(sample)

    def map[U](self, function: Callable[[T], U]) -> Track[U]:
        """Return a track applying ``function`` to each sampled value."""
        if not callable(function):
            raise TypeError("function must be callable")
        return Track(lambda t: function(self(t)))


@overload
def linear(start: float, end: float, u: float) -> float: ...
@overload
def linear(start: Point3, end: Point3, u: float) -> tuple[float, float, float]: ...
def linear(
    start: float | Point3, end: float | Point3, u: float
) -> float | tuple[float, float, float]:
    """``(1-u)*start + u*end`` without clamping, for numbers or for 3D points."""
    progress = _number(u, "u")
    if isinstance(start, int | float) and isinstance(end, int | float):
        return (1.0 - progress) * _number(start, "start") + progress * _number(end, "end")
    a, b = _vector3(start, "start"), _vector3(end, "end")
    x, y, z = ((1.0 - progress) * p + progress * q for p, q in zip(a, b, strict=True))
    return x, y, z


def hold[T](start: T, end: T, u: float) -> T:
    """Return ``start`` while ``u < 1``, then ``end`` (useful for discrete values)."""
    return start if _number(u, "u") < 1 else end


def smoothstep(u: float) -> float:
    """Ease progress with ``3*u**2 - 2*u**3`` without clamping."""
    progress = _number(u, "u")
    return 3.0 * progress**2 - 2.0 * progress**3
