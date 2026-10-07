"""Sample an animation's scenes and play them back with native SVG timing.

Each sample is rendered by the static renderer, and what it draws is split by
how it changes.  What every sample draws alike is written once, in place: the
definitions with the same content, and the anchors -- elements every sample
draws in the same order.  Between two anchors, what is left of each sample is
a slot, split further into one slot per element when every sample has the same
number there and that is estimated to be smaller.  Each distinct content of a
slot is written once in ``<defs>``, and the slot is drawn by one ``<use>``
whose ``href`` steps through them with discrete timing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import svg

from vecview._animation_refs import ids, namespace, references, walk

if TYPE_CHECKING:
    from vecview.scene import Scene

# Rough markup sizes, in bytes, for choosing how finely to split a slot.
_TIMING = 160  # a <use> and its <animate>, less the values and key times
_RUN = 30  # one value and its key time
_WRAPPER = 45  # a <g> around content that is not a single element


class _Sample(NamedTuple):
    """One sample's definitions and items, each with the markup it is compared by.

    Markup rather than ``==``: an element may keep content where the dataclass
    comparison does not look, as a VecTeX fragment does.
    """

    defs: list[svg.Element]
    items: list[svg.Element]
    def_text: list[str]
    item_text: list[str]


class _Slot(NamedTuple):
    """A stretch of the paint order as every sample draws it.

    A fixed slot is an anchor: one element alike in every sample, written once.
    """

    items: list[list[svg.Element]]
    texts: list[tuple[str, ...]]
    fixed: bool


def _fmt(value: float) -> str:
    return f"{value:.12g}"


def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _sample(scene: Scene) -> _Sample:
    if scene.camera is None:
        raise ValueError("every animation frame needs an active camera")
    items = scene._project(None)._elements()
    defs: list[svg.Element] = []
    if items and isinstance(items[0], svg.Defs):
        defs, items = list(items[0].elements or []), items[1:]
    # A stylesheet applies wherever it is, so one in <defs> counts as drawn.
    items = [d for d in defs if isinstance(d, svg.Style)] + items
    defs = [d for d in defs if not isinstance(d, svg.Style)]
    return _Sample(defs, items, [str(d) for d in defs], [str(e) for e in items])


def _anchors(samples: list[_Sample]) -> list[list[int]]:
    """Each anchor's position in every sample, in paint order.

    The k-th copy of a markup in one sample is matched with the k-th in the
    others, and taken greedily in the first sample's order while it comes
    after the previous anchor everywhere.
    """

    def keyed(texts: list[str]) -> list[tuple[str, int]]:
        seen: Counter[str] = Counter()
        keys = []
        for text in texts:
            keys.append((text, seen[text]))
            seen[text] += 1
        return keys

    where = [{key: i for i, key in enumerate(keyed(s.item_text))} for s in samples]
    last = [-1] * len(samples)
    chain = []
    for key in keyed(samples[0].item_text):
        at = [w.get(key, -1) for w in where]
        if all(a > b for a, b in zip(at, last, strict=True)):
            chain.append(at)
            last = at
    return chain


def _cost(texts: list[tuple[str, ...]], *, wrapped: bool) -> int:
    """Roughly how many bytes a slot drawing these samples takes."""
    distinct = dict.fromkeys(texts)
    content = sum(len(text) for key in distinct for text in key)
    if len(distinct) == 1:
        return content
    runs = 1 + sum(a != b for a, b in pairwise(texts))
    return content + _TIMING + _RUN * runs + (_WRAPPER * len(distinct) if wrapped else 0)


def _split(items: list[list[svg.Element]], texts: list[tuple[str, ...]]) -> list[_Slot]:
    """One slot for a stretch, or one per element where that is smaller."""
    whole = [_Slot(items, texts, fixed=False)]
    sizes = {len(row) for row in texts}
    if len(sizes) != 1 or sizes == {1}:
        return whole
    parts = [
        _Slot([[row[j]] for row in items], [(row[j],) for row in texts], fixed=False)
        for j in range(sizes.pop())
    ]
    smaller = sum(_cost(p.texts, wrapped=False) for p in parts) < _cost(texts, wrapped=True)
    return parts if smaller else whole


def _layout(samples: list[_Sample]) -> list[_Slot]:
    """The paint order as anchors and the slots between them."""
    layout: list[_Slot] = []
    before = [-1] * len(samples)
    for at in [*_anchors(samples), None]:
        stop = [len(s.items) for s in samples] if at is None else at
        spans = list(zip(samples, before, stop, strict=True))
        items = [s.items[b + 1 : e] for s, b, e in spans]
        if any(items):
            layout += _split(items, [tuple(s.item_text[b + 1 : e]) for s, b, e in spans])
        if at is not None:
            pairs = list(zip(samples, at, strict=True))
            texts: list[tuple[str, ...]] = [(s.item_text[a],) for s, a in pairs]
            layout.append(_Slot([[s.items[a]] for s, a in pairs], texts, fixed=True))
            before = at
    return layout


def _private(sample: _Sample, shared: set[str]) -> dict[str, svg.Element]:
    """A sample's definitions that are not shared, by id."""
    pairs = zip(sample.defs, sample.def_text, strict=True)
    return {d.id: d for d, text in pairs if d.id and text not in shared}


def _reach(
    items: list[svg.Element], private: dict[str, svg.Element]
) -> tuple[list[svg.Element], set[str]]:
    """The private definitions ``items`` need, in order, and every id they all point at."""
    names = references(items)
    needed: set[str] = set()
    queue = list(names)
    while queue:
        name = queue.pop()
        if name in private and name not in needed:
            needed.add(name)
            more = references([private[name]])
            names |= more
            queue += more
    return [d for name, d in private.items() if name in needed], names


def _conflict(
    samples: list[_Sample], defs: list[svg.Element], layout: list[_Slot]
) -> tuple[int, int] | None:
    """The first and last of a run of slots that must be one, or ``None``.

    A slot's content is renamed per content, so it may point at itself, at what
    is shared, and at private definitions, but not into another slot.  A
    stylesheet that changes is scoped to the content written with it, so it
    takes every moving slot with it.
    """
    moving = [j for j, slot in enumerate(layout) if not slot.fixed]
    shared = {str(d) for d in defs}
    for i, sample in enumerate(samples):
        drawn = [e for j in moving for e in layout[j].items[i]]
        if len(moving) > 1 and any(isinstance(e, svg.Style) for e in walk(drawn)):
            return moving[0], moving[-1]
        owner = {name: j for j in moving for name in ids(layout[j].items[i])}
        private = _private(sample, shared)
        for j in moving:
            for name in _reach(layout[j].items[i], private)[1]:
                if owner.get(name, j) != j:
                    return min(j, owner[name]), max(j, owner[name])
    return None


def _settle(samples: list[_Sample], layout: list[_Slot]) -> tuple[list[svg.Element], list[_Slot]]:
    """The shared definitions and the layout, once nothing shared points at what is not."""
    first = samples[0]
    everywhere = set(first.def_text).intersection(*(s.def_text for s in samples[1:]))
    defs = [d for d, text in zip(first.defs, first.def_text, strict=True) if text in everywhere]
    everything = set().union(*(ids(s.defs) | ids(s.items) for s in samples))
    while True:
        shared = defs + [slot.items[0][0] for slot in layout if slot.fixed]
        private = everything - ids(shared)

        def closed(element: svg.Element, private: set[str] = private) -> bool:
            return not references([element]) & private

        kept = [d for d in defs if closed(d)]
        demoted = [
            s._replace(fixed=False) if s.fixed and not closed(s.items[0][0]) else s for s in layout
        ]
        span = _conflict(samples, kept, demoted)
        fixed = [s.fixed for s in layout]
        if span is None and len(kept) == len(defs) and [s.fixed for s in demoted] == fixed:
            return defs, layout
        defs, layout = kept, demoted
        if span is not None:
            a, b = span
            run = layout[a : b + 1]
            merged = _Slot(
                [[e for s in run for e in s.items[i]] for i in range(len(samples))],
                [tuple(t for s in run for t in s.texts[i]) for i in range(len(samples))],
                fixed=False,
            )
            layout = [*layout[:a], merged, *layout[b + 1 :]]


def _keys(slot: _Slot, private: list[dict[str, svg.Element]]) -> list[tuple[str, ...]]:
    """What a slot draws in each sample, with the private definitions that needs."""
    return [
        tuple(str(d) for d in _reach(items, defs)[0]) + texts
        for items, texts, defs in zip(slot.items, slot.texts, private, strict=True)
    ]


def _timing(shown: list[str], intervals: int, duration: float, repeat: int | None) -> svg.Element:
    """Step ``href`` through one target per sample, held once finite playback ends.

    Discrete keyTimes end at 1 so every viewer accepts them; the value there is
    the terminal sample's for finite playback, and never shows in a loop.
    """
    values: list[str] = []
    times: list[float] = []
    for i, target in enumerate(shown[:intervals]):
        if not values or target != values[-1]:
            values.append(target)
            times.append(i / intervals)
    values.append(shown[intervals] if repeat is not None else values[-1])
    times.append(1.0)
    return svg.Animate(
        attributeName="href",
        values=";".join(f"#{v}" for v in values),
        calcMode="discrete",
        extra={
            "keyTimes": ";".join(_fmt(t) for t in times),
            "dur": f"{_fmt(duration)}s",
            "repeatCount": "indefinite" if repeat is None else str(repeat),
            "fill": "freeze",
        },
    )


def render_animation(
    frame: Callable[[float], Scene],
    *,
    duration: float,
    view_box: tuple[float, float, float, float],
    fps: float,
    repeat: int | None,
    background: str | None,
) -> svg.SVG:
    """Render validated animation settings through the ordinary scene projector."""
    intervals = max(1, math.ceil(duration * fps))
    times = [i * duration / intervals for i in range(intervals)]
    if repeat is not None:
        times.append(duration)  # the frame held once playback ends
    samples = [_sample(frame(t)) for t in times]
    shared_defs, layout = _settle(samples, _layout(samples))
    shared_ids = ids(shared_defs) | ids(s.items[0][0] for s in layout if s.fixed)
    shared_text = {str(d) for d in shared_defs}
    private = [_private(s, shared_text) for s in samples]
    keys = [[] if s.fixed else _keys(s, private) for s in layout]
    moving = sum(len(set(k)) > 1 for k in keys)

    defs: list[svg.Element] = list(shared_defs)
    written: set[str] = set()
    body: list[svg.Element] = []

    def copy(slot: _Slot, i: int) -> list[svg.Element]:
        """Sample ``i`` of a slot renamed as frame ``i``, its private definitions written."""
        needed = _reach(slot.items[i], private[i])[0]
        copied = namespace(needed + slot.items[i], str(i), shared_ids)
        for d in copied[: len(needed)]:
            if d.id and d.id not in written:  # every needed definition has an id
                written.add(d.id)
                defs.append(d)
        return copied[len(needed) :]

    def group(name: str | None, i: int, own: list[svg.Element]) -> svg.Element:
        return svg.G(id=name, data={"vecview-frame": str(i)}, elements=own)

    stretch = 0
    for slot, slot_keys in zip(layout, keys, strict=True):
        if slot.fixed:
            body.append(slot.items[0][0])
            continue
        styled = any(isinstance(e, svg.Style) for row in slot.items for e in walk(row))
        first = {key: slot_keys.index(key) for key in dict.fromkeys(slot_keys)}
        if len(first) == 1:  # alike in every sample, but pointing at private definitions
            own = copy(slot, 0)
            body += [group(None, 0, own)] if styled else own
            continue
        targets: dict[tuple[str, ...], str] = {}
        for key, i in first.items():
            own = copy(slot, i)
            if len(own) == 1 and own[0].id and not styled:
                targets[key] = own[0].id
                defs += own
            else:
                targets[key] = f"frame{i}" if moving < 2 else f"frame{i}.{stretch}"
                defs.append(group(targets[key], i, own))
        shown = [targets[key] for key in slot_keys]
        timing = _timing(shown, intervals, duration, repeat)
        body.append(svg.Use(href=f"#{shown[0]}", elements=[timing]))
        stretch += 1

    x, y, width, height = view_box
    elements: list[svg.Element] = [svg.Defs(elements=defs)] if defs else []
    if background is not None:
        elements.append(
            svg.Rect(
                x=_number(x),
                y=_number(y),
                width=_number(width),
                height=_number(height),
                fill=background,
            )
        )
    elements += body
    return svg.SVG(
        width=_number(width),
        height=_number(height),
        viewBox=svg.ViewBoxSpec(*map(_number, view_box)),
        overflow="hidden",
        elements=elements,
    )
