"""Curved solids on the canvas: spheres, cylinders, cones, solid arrows, and tubes.

Each has an exact outline under any parallel projection, as one native element
or group, and describes its exact depth for an exact layer.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
import svg

from vecview._canvas import _ball_axes, _Canvas, _coverage, _parallel
from vecview._elements import _ellipse_axes, _ellipse_element, _named, _num, _path, _points, _slug
from vecview._occlusion import (
    Piece,
    Surface,
    capsule_depth,
)
from vecview._types import Array, Point3, Points3, Style
from vecview._vec import as_points, basis_for, convex_hull, unit
from vecview.shapes import Face, Pivot


def _frustum_outline(
    c0: Array,
    c1: Array,
    axes: Array,
    r0: float,
    r1: float,
    *,
    closed: bool = True,
    arcs: tuple[bool, bool] = (True, True),
) -> list[svg.PathData]:
    """Exact outline of a projected frustum: two tangent lines and two elliptical arcs.

    Under a parallel projection the two end circles become the homothetic
    ellipses ``c_k + r_k * axes @ (cos t, sin t)``, and the solid's outline is
    their convex hull.  Mapping the screen back through ``axes`` turns both
    ellipses into circles, where the outer common tangents have a closed form;
    mapping the tangent points forward again gives the outline, with each
    curved side as one SVG arc rather than a polyline.  ``r1 = 0`` is a cone.

    With ``closed=False`` the result is for stroking only: the two sides, and
    the end arcs that ``arcs`` asks for.  That is how a sliced cylinder strokes
    its silhouette without a line across every joint.  The degenerate views
    fall back to the closed outline.
    """
    rx, ry, angle, major = _ellipse_axes(axes)
    det = float(np.linalg.det(axes))
    if abs(det) <= 1e-9 * rx * rx:
        # The end circles are seen edge-on and project to segments.
        ends = np.array(
            [c0 + r0 * rx * major, c0 - r0 * rx * major, c1 + r1 * rx * major, c1 - r1 * rx * major]
        )
        return _path([ends[convex_hull(ends)]], closed=True)

    def at(c: Array, r: float, phi: float) -> svg.Point:
        x, y = c + r * (axes @ np.array([math.cos(phi), math.sin(phi)]))
        return svg.Point(_num(x), _num(y))

    def arc(r: float, large: bool, sweep: bool, end: svg.Point) -> svg.Arc:
        return svg.Arc(_num(rx * r, 3), _num(ry * r, 3), _num(angle, 3), large, sweep, end.x, end.y)

    d = np.linalg.solve(axes, c1 - c0)
    dist = float(np.linalg.norm(d))
    if dist <= abs(r0 - r1) * (1.0 + 1e-12):
        # Seen down the axis: the larger end covers everything else.
        c, r = (c0, r0) if r0 >= r1 else (c1, r1)
        p, q = at(c, r, 0.0), at(c, r, math.pi)
        return [svg.M(p.x, p.y), arc(r, False, True, q), arc(r, False, True, p), svg.Z()]
    theta = math.atan2(float(d[1]), float(d[0]))
    alpha = math.acos(min(max((r0 - r1) / dist, -1.0), 1.0))
    # Both arcs run toward decreasing parameter angle, which is SVG's positive
    # sweep exactly when `axes` reverses orientation.
    sweep = det < 0
    start, back = at(c0, r0, theta + alpha), at(c0, r0, theta - alpha)
    if r1 > 0:
        near, far = at(c1, r1, theta + alpha), at(c1, r1, theta - alpha)
    else:
        near = far = svg.Point(_num(c1[0]), _num(c1[1]))
    far_arc = arc(r1, 2 * alpha > math.pi, sweep, far)
    near_arc = arc(r0, 2 * alpha < math.pi, sweep, start)
    commands: list[svg.PathData] = [svg.M(start.x, start.y), svg.L(near.x, near.y)]
    if closed:
        if r1 > 0:
            commands.append(far_arc)
        return [*commands, svg.L(back.x, back.y), near_arc, svg.Z()]
    commands.append(far_arc if arcs[1] and r1 > 0 else svg.M(far.x, far.y))
    commands.append(svg.L(back.x, back.y))
    if arcs[0]:
        commands.append(near_arc)
    return commands


def _frustum_faces(a: Array, b: Array, r0: float, r1: float, n: int = 32) -> list[Face]:
    """A frustum as a convex polyhedron circumscribing it, for occlusion only.

    Circumscribed rather than inscribed, so the facets cover the exact outline
    that is drawn rather than falling a sliver short of it.
    """
    axis = b - a
    along = unit(axis)
    e1, e2 = basis_for(axis)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    grow = 1.0 / math.cos(math.pi / n)
    ring = grow * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
    bottom, top = a + r0 * ring, b + r1 * ring
    length = float(np.linalg.norm(axis))
    faces: list[Face] = []
    for i in range(n):
        j = (i + 1) % n
        radial = unit(ring[i] + ring[j])
        normal = unit(radial * length + along * (r0 - r1))
        faces.append(Face(f"side-{i}", np.array([bottom[i], bottom[j], top[j], top[i]]), normal))
    faces.append(Face("-axis", bottom[::-1].copy(), -along))
    if r1 > 0:
        faces.append(Face("+axis", top.copy(), along))
    return faces


class _SolidCanvas(_Canvas):
    """The complete canvas: :class:`_Canvas` plus the curved solids."""

    def sphere(
        self,
        layer: int,
        center: Point3,
        radius: float,
        *,
        highlight: str | None = None,
        **style: Style,
    ) -> None:
        """A sphere, drawn as its exact outline: one ``<circle>`` or ``<ellipse>``."""
        cam = _parallel(self.cam, "outline a sphere")
        c = np.asarray(self.cam.at(center))
        axes = _ball_axes(cam, radius)
        if highlight is not None:
            style["fill"] = self._ball_gradient(style, highlight)
        reach = np.linalg.norm(axes, axis=1)
        self._grow(np.array([c - reach, c + reach]))
        self._emit(layer, _ellipse_element(c, axes, **style), self._depth([center]))
        opaque, pad = _coverage(style)
        self._shape(lambda: Surface([self._sphere_piece(center, radius, pad)], opaque, pad))

    def _ball_gradient(self, style: Mapping[str, Style], highlight: str) -> str:
        """Add (once) the radial gradient shared by spheres of one colour pair."""
        fill = style.get("fill")
        if not isinstance(fill, str):
            raise ValueError("highlight needs a fill colour to shade toward")
        gid = f"ball-{_slug(fill)}-{_slug(highlight)}"
        if gid not in self._shared:
            self._shared.add(gid)
            self.defs.append(
                svg.RadialGradient(
                    id=gid,
                    cx=0.5,
                    cy=0.5,
                    r=0.5,
                    fx=0.36,
                    fy=0.32,
                    elements=[
                        svg.Stop(offset=0, stop_color=highlight),
                        svg.Stop(offset=1, stop_color=fill),
                    ],
                )
            )
        return f"url(#{gid})"

    def _frustum(
        self,
        p0: Point3,
        p1: Point3,
        r0: float,
        r1: float,
        *,
        ends: tuple[bool, bool],
        end_style: Mapping[str, Style] | None,
        highlight: str | None,
        name: str | None,
        style: Mapping[str, Style],
    ) -> list[svg.Element]:
        """Body outline plus whichever end disks face the camera, as elements.

        ``ends`` says which of the two end disks may be drawn at all; each is
        then drawn only if it faces the camera, after the body, which is exact
        for a convex solid.
        """
        cam = _parallel(self.cam, "outline a solid of revolution")
        if r0 <= 0 or r1 < 0:
            raise ValueError(f"need r0 > 0 and r1 >= 0, got r0={r0}, r1={r1}")
        a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        axis = b - a
        if float(np.linalg.norm(axis)) == 0.0:
            raise ValueError("the two ends coincide, so the solid has no axis")
        axes = cam.scale * cam.matrix @ np.column_stack(basis_for(axis))
        c0, c1 = cam.project([a, b])
        reach = np.linalg.norm(axes, axis=1)
        self._grow(np.array([c0 - r0 * reach, c0 + r0 * reach, c1 - r1 * reach, c1 + r1 * reach]))

        body: dict[str, Style] = dict(style)
        if highlight is not None:
            fill = body.get("fill")
            if not isinstance(fill, str) or name is None:
                raise ValueError("highlight needs a fill colour and an id to name the gradient")
            body["fill"] = f"url(#{name}-shade)"
            self.defs.append(self._shade_across(f"{name}-shade", c0, c1, axes, r0, fill, highlight))
        label: dict[str, Style] = {} if name is None else {"id": name}
        elements: list[svg.Element] = [
            svg.Path(d=_frustum_outline(c0, c1, axes, r0, r1), **label, **body)
        ]
        disk: dict[str, Style] = {**style, **(end_style or {})}
        toward = float(np.dot(unit(axis), cam.view))
        for k, (c, r, faces) in enumerate(((c0, r0, toward < 0), (c1, r1, toward > 0))):
            if ends[k] and faces and r > 0:
                end: dict[str, Style] = {} if name is None else {"id": f"{name}-end{k}"}
                elements.append(_ellipse_element(c, r * axes, **end, **disk))
        return elements

    def _sliced_cylinder(
        self,
        layer: int,
        p0: Point3,
        p1: Point3,
        r0: float,
        r1: float,
        *,
        slices: int,
        ends: bool,
        end_style: Mapping[str, Style] | None,
        highlight: str | None,
        base: str | None,
        style: Mapping[str, Style],
    ) -> None:
        """:meth:`cylinder` cut into ``slices`` separately sorted lengths."""
        cam = _parallel(self.cam, "outline a solid of revolution")
        if r0 <= 0 or r1 < 0:
            raise ValueError(f"need r0 > 0 and r1 >= 0, got r0={r0}, r1={r1}")
        a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        axis = b - a
        if float(np.linalg.norm(axis)) == 0.0:
            raise ValueError("the two ends coincide, so the solid has no axis")
        axes = cam.scale * cam.matrix @ np.column_stack(basis_for(axis))
        reach = np.linalg.norm(axes, axis=1)
        c0, c1 = cam.project([a, b])
        self._grow(np.array([c0 - r0 * reach, c0 + r0 * reach, c1 - r1 * reach, c1 + r1 * reach]))

        fill: dict[str, Style] = {k: v for k, v in style.items() if not k.startswith("stroke")}
        edge: dict[str, Style] = {k: v for k, v in style.items() if k.startswith("stroke")}
        if highlight is not None:
            colour = fill.get("fill")
            if not isinstance(colour, str) or base is None:
                raise ValueError("highlight needs a fill colour and an id to name the gradient")
            fill["fill"] = f"url(#{base}-shade)"
            self.defs.append(
                self._shade_across(f"{base}-shade", c0, c1, axes, max(r0, r1), colour, highlight)
            )
        disk: dict[str, Style] = {**style, **(end_style or {})}
        toward = float(np.dot(unit(axis), cam.view))
        # Each slice's body overruns into its neighbours, so no seam shows at a
        # joint whichever of the two is drawn on top.
        overrun = 0.15 / slices
        cuts = np.linspace(0.0, 1.0, slices + 1)

        def at(t: float) -> tuple[Array, float]:
            return a + t * axis, r0 + t * (r1 - r0)

        for k in range(slices):
            lo, hi = max(cuts[k] - overrun, 0.0), min(cuts[k + 1] + overrun, 1.0)
            (pa, ra), (pb, rb) = at(lo), at(hi)
            qa, qb = cam.project([pa, pb])
            first, last = k == 0, k == slices - 1
            parts: list[svg.Element] = [
                svg.Path(
                    d=_frustum_outline(qa, qb, axes, ra, rb),
                    **_named(base, f"{k}-body"),
                    **fill,
                )
            ]
            if edge.get("stroke"):
                parts.append(
                    svg.Path(
                        d=_frustum_outline(qa, qb, axes, ra, rb, closed=False, arcs=(first, last)),
                        fill="none",
                        **_named(base, f"{k}-edge"),
                        **edge,
                    )
                )
            for cap, centre, r, faces in ((first, qa, ra, toward < 0), (last, qb, rb, toward > 0)):
                if ends and cap and faces and r > 0:
                    end = "0" if centre is qa else "1"
                    label = _named(base, f"end{end}")
                    parts.append(_ellipse_element(centre, r * axes, **label, **disk))
            group = svg.G(elements=parts, **_named(base, str(k)))
            self._emit(layer, group, self._depth([at(cuts[k])[0], at(cuts[k + 1])[0]]))
            self._solid_shape([(pa, pb, ra, rb)], style)

    @staticmethod
    def _shade_across(
        gid: str, c0: Array, c1: Array, axes: Array, r: float, fill: str, highlight: str
    ) -> svg.Element:
        """A linear gradient across a projected solid, lightest toward the upper left."""
        along = c1 - c0
        length = float(np.linalg.norm(along))
        side = np.array([-along[1], along[0]]) / length if length > 1e-9 else np.array([-1.0, -1.0])
        side = side / float(np.linalg.norm(side))
        if side[0] + side[1] > 0:
            side = -side
        half = r * float(np.linalg.norm(axes.T @ side))
        mid = (c0 + c1) / 2.0
        (x1, y1), (x2, y2) = mid + half * side, mid - half * side
        return svg.LinearGradient(
            id=gid,
            gradientUnits="userSpaceOnUse",
            x1=_num(x1),
            y1=_num(y1),
            x2=_num(x2),
            y2=_num(y2),
            elements=[
                svg.Stop(offset=0, stop_color=fill),
                svg.Stop(offset=0.3, stop_color=highlight),
                svg.Stop(offset=1, stop_color=fill),
            ],
        )

    def cylinder(
        self,
        layer: int,
        p0: Point3,
        p1: Point3,
        radius: float,
        *,
        r1: float | None = None,
        ends: bool = True,
        end_style: Mapping[str, Style] | None = None,
        highlight: str | None = None,
        slices: int = 1,
        **style: Style,
    ) -> None:
        """A cylinder from ``p0`` to ``p1``, or a frustum when ``r1`` differs."""
        base = style.pop("id", None)
        if slices < 1:
            raise ValueError(f"slices must be at least 1, got {slices}")
        if slices > 1:
            self._sliced_cylinder(
                layer,
                p0,
                p1,
                radius,
                radius if r1 is None else r1,
                slices=slices,
                ends=ends,
                end_style=end_style,
                highlight=highlight,
                base=base,
                style=style,
            )
            return
        elements = self._frustum(
            p0,
            p1,
            radius,
            radius if r1 is None else r1,
            ends=(ends, ends),
            end_style=end_style,
            highlight=highlight,
            name=None if base is None else f"{base}-body",
            style=style,
        )
        group = svg.G(elements=elements, **({} if base is None else {"id": base}))
        self._emit(layer, group, self._depth([p0, p1]))
        a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
        self._solid_shape([(a, b, radius, radius if r1 is None else r1)], style)

    def _solid_shape(
        self, parts: Sequence[tuple[Array, Array, float, float]], style: Mapping[str, Style]
    ) -> None:
        """Describe the solids of revolution just emitted as one surface."""
        opaque, pad = _coverage(style)

        def surface() -> Surface | None:
            grow = self._grown(pad)
            pieces = []
            for a, b, r0, r1 in parts:
                along = unit(b - a) * grow
                faces = _frustum_faces(a - along, b + along, r0 + grow, r1 + grow)
                if piece := self._convex_piece(faces):
                    pieces.append(piece)
            return Surface(pieces, opaque, pad) if pieces else None

        self._shape(surface)

    def cone(
        self,
        layer: int,
        base: Point3,
        apex: Point3,
        radius: float,
        *,
        end: bool = True,
        end_style: Mapping[str, Style] | None = None,
        highlight: str | None = None,
        **style: Style,
    ) -> None:
        """A cone from a disk of ``radius`` at ``base`` to a point at ``apex``."""
        self.cylinder(
            layer,
            base,
            apex,
            radius,
            r1=0.0,
            ends=end,
            end_style=end_style,
            highlight=highlight,
            **style,
        )

    def arrow3d(
        self,
        layer: int,
        origin: Point3,
        direction: Point3,
        length: float,
        *,
        shaft_r: float,
        head_r: float,
        head_len: float,
        pivot: Pivot = "tail",
        end_style: Mapping[str, Style] | None = None,
        highlight: str | None = None,
        **style: Style,
    ) -> None:
        """A solid arrow: a cylindrical shaft and a conical head, in one ``<g>``."""
        d = unit(direction)
        if not d.any():
            raise ValueError("an arrow needs a non-zero direction")
        base = style.pop("id", None)
        tail = np.asarray(origin, dtype=np.float64) - (d * length / 2.0 if pivot == "mid" else 0.0)
        tip = tail + d * length
        neck = tail + d * max(length - head_len, 0.0)

        def piece(a: Array, b: Array, r0: float, r1: float, part: str) -> list[svg.Element]:
            return self._frustum(
                a,
                b,
                r0,
                r1,
                ends=(True, False),
                end_style=end_style,
                highlight=highlight,
                name=None if base is None else f"{base}-{part}",
                style=style,
            )

        head = piece(neck, tip, head_r, 0.0, "head")
        shaft = piece(tail, neck, shaft_r, shaft_r, "shaft") if length > head_len else []
        toward = float(np.dot(d, _parallel(self.cam, "draw a solid arrow").view)) >= 0
        elements = shaft + head if toward else head + shaft
        group = svg.G(elements=elements, **({} if base is None else {"id": base}))
        self._emit(layer, group, self._depth([tail, tip]))
        parts = [(neck, tip, head_r, 0.0)]
        if length > head_len:
            parts.append((tail, neck, shaft_r, shaft_r))
        self._solid_shape(parts, style)

    def tube(
        self,
        layer: int,
        pts3: Points3,
        radius: float,
        *,
        chunk: int = 4,
        **style: Style,
    ) -> None:
        """A tube of ``radius`` along a world-space curve -- a coil, a field line, a bond path."""
        cam = _parallel(self.cam, "draw a tube")
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        pts = as_points(pts3)
        if len(pts) < 2:
            raise ValueError("a tube needs at least two points")
        body = style.pop("fill", "#000000")
        edge = style.pop("stroke", None)
        edge_w = float(style.pop("stroke_width", 1.0))
        base = style.pop("id", None)
        s = np.linalg.svd(cam.matrix, compute_uv=False) * cam.scale
        width = 2.0 * radius * math.sqrt(float(s[0] * s[1]))
        p = cam.project(pts)
        margin = width / 2.0 + (edge_w if edge else 0.0)
        self._grow(np.vstack([p - margin, p + margin]))
        last = len(pts) - 1
        line: dict[str, Style] = dict(fill="none", stroke_linecap="butt", stroke_linejoin="round")
        for k, start in enumerate(range(0, last, chunk)):
            # Outlines of neighbouring pieces overlap by a segment, so their
            # square ends leave no notch on the outside of a bend, and each body
            # overruns its own outline by a segment at either inner end, so an
            # outline's square end lies under body colour rather than leaving an
            # anti-aliased seam across the tube.
            stop = min(start + chunk + 3, last)
            strokes: list[svg.Element] = []
            if edge:
                inner = p[(start + 1 if start else 0) : min(start + chunk + 2, last) + 1]
                strokes.append(
                    svg.Polyline(
                        points=_points(inner),
                        stroke=edge,
                        stroke_width=_num(width + 2 * edge_w, 3),
                        **line,
                    )
                )
            strokes.append(
                svg.Polyline(
                    points=_points(p[start : stop + 1]),
                    stroke=body,
                    stroke_width=_num(width, 3),
                    **line,
                )
            )
            label: dict[str, Style] = {} if base is None else {"id": f"{base}-{k}"}
            self._emit(
                layer,
                svg.G(elements=strokes, **label, **style),
                self._depth(pts[start : stop + 1]),
            )
            self._tube_shape(
                pts[start : stop + 1],
                width,
                radius,
                {**style, "fill": body},
                edge_w if edge else 0.0,
                (id(pts), k),
            )

    def _tube_shape(
        self,
        run: Array,
        width: float,
        radius: float,
        style: Mapping[str, Style],
        edge: float,
        chain: tuple[int, int],
    ) -> None:
        """A tube piece: the chain of capsules along its centre line, exactly."""
        opaque, _ = _coverage(style)

        def surface() -> Surface | None:
            cam = _parallel(self.cam, "resolve occlusion")
            segments = np.array([(a, b) for a, b in itertools.pairwise(run) if np.any(a != b)])
            if not len(segments):
                return None
            back = np.linalg.pinv(cam.matrix * cam.scale)
            grown = radius + self._grown(edge)
            depth, covers = capsule_depth(segments, grown, back, cam.origin, cam.view)
            piece = Piece(cam.project(run), depth, covers=covers, width=width / 2.0)
            return Surface([piece], opaque, edge, chain)

        self._shape(surface)
