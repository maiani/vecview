"""Finite scalar and three-component value validation for animation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def finite_vector3(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a three-component vector")
    if not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a three-component vector")
    coords = tuple(value)
    if len(coords) != 3:
        raise ValueError(f"{name} must have three components")
    return cast(
        tuple[float, float, float],
        tuple(finite_number(coord, f"{name}[{i}]") for i, coord in enumerate(coords)),
    )
