"""Packaging invariants that are easy to break silently."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pytest

import vecview

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_matches_the_installed_distribution() -> None:
    assert vecview.__version__ == importlib.metadata.version("vecview")


def test_version_matches_pyproject(pyproject: dict) -> None:
    assert vecview.__version__ == pyproject["project"]["version"]


def test_all_names_are_importable() -> None:
    for name in vecview.__all__:
        assert hasattr(vecview, name), name


def test_all_has_no_duplicates() -> None:
    """Ordering is ruff's RUF022 to own (constants, then classes, then functions);
    duplicates are what a test can usefully catch."""
    assert len(vecview.__all__) == len(set(vecview.__all__))


def test_public_api_is_exactly_all() -> None:
    """A name reachable without an underscore but absent from __all__ is an accident."""
    # Submodules and the `from __future__` import are reachable but not API.
    not_api = {"annotations", "camera", "projections", "scene", "shapes"}
    public = {n for n in vars(vecview) if not n.startswith("_") and n not in not_api}
    assert public == set(vecview.__all__) - {"__version__"}


def test_package_ships_type_information() -> None:
    assert (ROOT / "src" / "vecview" / "py.typed").is_file()


def test_runtime_dependencies_stay_minimal(pyproject: dict) -> None:
    """Rasterizing and TeX belong to the composition layer, not here.

    shapely and contourpy are the geometry dependencies: robust polygon
    clipping and sub-pixel contours, for exact visibility and booleans on
    outlines, are not worth reimplementing.
    """
    names = {d.split(">")[0].split("=")[0].strip() for d in pyproject["project"]["dependencies"]}
    assert names == {"contourpy", "numpy", "shapely", "svg.py"}


MAX_MODULE_LINES = 1000  # pylint's default max-module-lines (C0302)


@pytest.mark.parametrize(
    "path",
    sorted(
        p
        for folder in ("src", "tests", "examples", "docs")
        for p in (ROOT / folder).rglob("*.py")
        if "__pycache__" not in p.parts
    ),
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_no_module_exceeds_the_length_limit(path: Path) -> None:
    """A module past the limit is split along a seam, as AGENTS.md says."""
    n = len(path.read_text(encoding="utf-8").splitlines())
    assert n <= MAX_MODULE_LINES, f"{path.relative_to(ROOT)} has {n} lines"
