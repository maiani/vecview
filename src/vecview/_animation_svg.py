"""Sample an animation's scenes and play them back with native SVG timing.

Each sample is rendered by the static renderer.  What every sample draws alike
is written once: definitions with the same id and content, and the elements at
the bottom and top of the paint order.  What is left of a sample goes in a
group shown only during its interval, and consecutive samples that are left
with the same content share one group.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import takewhile
from typing import TYPE_CHECKING, NamedTuple

import svg

from vecview._animation_refs import ids, namespace, references

if TYPE_CHECKING:
    from vecview.scene import Scene


class _Sample(NamedTuple):
    """One sample's definitions and items, each with the markup it is compared by.

    Markup rather than ``==``: an element may keep content where the dataclass
    comparison does not look, as a VecTeX fragment does.
    """

    defs: list[svg.Element]
    items: list[svg.Element]
    def_text: list[str]
    item_text: list[str]


def _fmt(value: float) -> str:
    return format(value, ".12g")


def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _sample(scene: Scene) -> _Sample:
    if scene.camera is None:
        raise ValueError("every animation frame needs an active camera")
    items = scene._project(None)._elements()
    defs: list[svg.Element] = []
    if items and isinstance(items[0], svg.Defs):
        defs, items = list(items[0].elements or []), items[1:]
    return _Sample(defs, items, [str(d) for d in defs], [str(e) for e in items])


def _alike(rows: Sequence[Sequence[str]]) -> int:
    """How many leading entries every row has in common."""
    columns = zip(*rows, strict=False)
    return sum(1 for _ in takewhile(lambda column: len(set(column)) == 1, columns))


def _shared(samples: list[_Sample]) -> tuple[list[svg.Element], int, int]:
    """The definitions every sample has alike, and how many items to share at each end.

    Nothing shared may point at something that is not, since that is renamed
    per frame; so shrink the shared part until it is closed under references.
    """
    first = samples[0]
    everywhere = set(first.def_text).intersection(*(s.def_text for s in samples[1:]))
    defs = [
        d for d, text in zip(first.defs, first.def_text, strict=True) if d.id and text in everywhere
    ]
    head = _alike([s.item_text for s in samples])
    shortest = min(len(s.items) for s in samples)
    tail = min(_alike([s.item_text[::-1] for s in samples]), shortest - head)
    everything = set().union(*(ids(s.defs) | ids(s.items) for s in samples))
    while True:
        shared = defs + first.items[:head] + first.items[len(first.items) - tail :]
        private = everything - ids(shared)

        def closed(element: svg.Element, private: set[str] = private) -> bool:
            return not references([element]) & private

        new_defs = [d for d in defs if closed(d)]
        new_head = sum(1 for _ in takewhile(closed, first.items[:head]))
        top = first.items[len(first.items) - tail :][::-1]
        new_tail = sum(1 for _ in takewhile(closed, top))
        if (len(new_defs), new_head, new_tail) == (len(defs), head, tail):
            return defs, head, tail
        defs, head, tail = new_defs, new_head, new_tail


def _timing(
    start: int, stop: int, intervals: int, duration: float, repeat: int | None, *, hold: bool
) -> svg.Element:
    """Show a group from sample ``start`` up to ``stop`` in every cycle.

    Discrete keyTimes end at 1 so every viewer accepts them; with ``hold`` the
    group is the terminal frame too and stays shown once playback ends.
    """
    values, times = [], []
    if start:
        values.append("none")
        times.append(0.0)
    values.append("inline")
    times.append(start / intervals)
    if stop < intervals:
        values.append("none")
        times.append(stop / intervals)
    values.append("inline" if hold else "none")
    times.append(1.0)
    return svg.Animate(
        attributeName="display",
        values=";".join(values),
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
    defs, head, tail = _shared(samples)
    shared_ids = ids(defs) | ids(samples[0].items[:head])
    shared_ids |= ids(samples[0].items[len(samples[0].items) - tail :])

    shared_text = {str(d) for d in defs}

    def rest(sample: _Sample) -> tuple[list[svg.Element], tuple[str, ...]]:
        """What is left of a sample to draw in its own group, and its markup."""
        private = [
            (d, text)
            for d, text in zip(sample.defs, sample.def_text, strict=True)
            if text not in shared_text
        ]
        stop = len(sample.items) - tail
        items = sample.items[head:stop]
        text = tuple(t for _, t in private) + tuple(sample.item_text[head:stop])
        return ([svg.Defs(elements=[d for d, _ in private])] if private else []) + items, text

    rests = [rest(s) for s in samples]
    terminal = rests.pop() if repeat is not None else None
    runs: list[tuple[int, int]] = []
    for i, (_, text) in enumerate(rests):
        if runs and text == rests[runs[-1][0]][1]:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    hold = terminal is not None and terminal[1] == rests[runs[-1][0]][1]

    groups: list[svg.Element] = []
    for start, stop in runs:
        content = rests[start][0]
        if not content:
            continue
        held = hold and stop == intervals
        children = namespace(content, str(start), shared_ids)
        children.append(_timing(start, stop, intervals, duration, repeat, hold=held))
        groups.append(
            svg.G(
                elements=children,
                display="inline" if start == 0 else "none",
                data={"vecview-frame": str(start)},
            )
        )
    if terminal is not None and terminal[0] and not hold:
        children = namespace(terminal[0], str(intervals), shared_ids)
        children.append(
            svg.Set(
                attributeName="display",
                to="inline",
                extra={"begin": f"{_fmt(duration * (repeat or 1))}s", "fill": "freeze"},
            )
        )
        groups.append(
            svg.G(elements=children, display="none", data={"vecview-frame": str(intervals)})
        )

    first = samples[0].items
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
    elements += first[:head] + groups + first[len(first) - tail :]
    return svg.SVG(
        width=_number(width),
        height=_number(height),
        viewBox=svg.ViewBoxSpec(*map(_number, view_box)),
        overflow="hidden",
        elements=elements,
    )
