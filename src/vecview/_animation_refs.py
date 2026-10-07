"""Ids and local references in animation frames: finding them, and namespacing a frame's."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from typing import Any

import svg

_URL = re.compile(r"url\(\s*(['\"]?)#([^)'\"\s]+)\1\s*\)")
_SELECTOR_ID = re.compile(r"(?<![\w-])#([\w-]+)")

type _Slot = tuple[svg.Element | dict[str, Any], str]


def walk(elements: Iterable[svg.Element]) -> Iterator[svg.Element]:
    """Every element, depth first."""
    for element in elements:
        yield element
        if element.elements:
            yield from walk(element.elements)


def _attributes(element: svg.Element) -> Iterator[tuple[str, str, _Slot]]:
    """Each set attribute other than ``id``: its local name, its value, and where it lives."""
    for name, value in vars(element).items():
        # Private state, like a wrapped fragment's markup, is not an attribute.
        if name in {"id", "elements", "text", "data", "extra"} or name[0] == "_" or value is None:
            continue
        yield name.rstrip("_").rsplit("__", 1)[-1], str(value), (element, name)
    for mapping in (element.extra, element.data):
        if mapping is not None:
            for name, value in mapping.items():
                yield name.rsplit(":", 1)[-1], str(value), (mapping, name)


def _set(slot: _Slot, value: str) -> None:
    owner, name = slot
    if isinstance(owner, dict):
        owner[name] = value
    else:
        setattr(owner, name, value)


def ids(elements: Iterable[svg.Element]) -> set[str]:
    """The ids the elements define."""
    return {element.id for element in walk(elements) if element.id}


def references(elements: Iterable[svg.Element]) -> set[str]:
    """The ids the elements point at: ``url(#...)``, ``href="#..."``, and stylesheet ids."""
    found: set[str] = set()
    for element in walk(elements):
        for name, value, _ in _attributes(element):
            if name == "href" and value.startswith("#"):
                found.add(value[1:])
            found.update(match[2] for match in _URL.finditer(value))
        if isinstance(element, svg.Style) and element.text:
            found.update(match[2] for match in _URL.finditer(element.text))
            found.update(_SELECTOR_ID.findall(_selectors(element.text)))
    return found


def _selectors(css: str) -> str:
    return " ".join(rule.partition("{")[0] for rule in css.split("}"))


def _scope_css(css: str, frame: str, target: Callable[[str], str]) -> str:
    """Confine a frame's stylesheet to its own group and its own ids."""
    if "@" in css or "/*" in css:
        raise ValueError(
            "a stylesheet that changes between animation frames cannot hold at-rules or comments"
        )
    rules = []
    for rule in css.split("}"):
        if not rule.strip():
            continue
        selectors, brace, body = rule.partition("{")
        if not brace:
            raise ValueError(f"cannot parse the CSS rule {rule.strip()!r} in an animation frame")
        scoped = ", ".join(
            f'[data-vecview-frame="{frame}"] '
            + _SELECTOR_ID.sub(lambda m: "#" + target(m[1]), selector.strip())
            for selector in selectors.split(",")
        )
        body = _URL.sub(lambda m: f"url(#{target(m[2])})", body.strip())
        rules.append(f"{scoped} {{{body}}}")
    return "".join(rules)


def namespace(elements: list[svg.Element], frame: str, shared: set[str]) -> list[svg.Element]:
    """A copy of one frame's elements, with its ids prefixed and its references following.

    Ids defined in the frame become ``frame{frame}-{id}``, and the element keeps
    its own as ``data-vecview-id``.  References to ``shared`` ids, written once
    for all frames, and to ids defined nowhere are left alone, as in a static
    scene.
    """
    elements = deepcopy(elements)
    own = ids(elements)
    prefix = f"frame{frame}-"

    def target(name: str) -> str:
        return prefix + name if name in own and name not in shared else name

    for element in walk(elements):
        for name, value, slot in _attributes(element):
            if name == "href" and value.startswith("#"):
                _set(slot, "#" + target(value[1:]))
            elif "url(" in value:
                _set(slot, _URL.sub(lambda m: f"url(#{target(m[2])})", value))
        if isinstance(element, svg.Style) and element.text:
            element.text = _scope_css(element.text, frame, target)
        if element.id:
            element.data = {**(element.data or {}), "vecview-id": element.id}
            element.id = target(element.id)
    return elements
