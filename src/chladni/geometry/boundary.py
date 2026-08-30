"""Boundaries and their corners.

A `Corner` is a frozen object, not an entry in four parallel arrays. lappy carries
corner-indexed `corners`/`corner_idx`/`corner_angle0`/`corner_angle1` beside segment-indexed
`int_angles`, and its own `_find_corners` docstring records the consequence: on a half-disk
`corners` had 2 entries while the angle arrays had 3, "so no `orders` length could be masked
consistently". Objects cannot desync.

A corner is also not purely geometric. The singularity exponent depends on the boundary
conditions either side, so a Dirichlet/Neumann change across a straight edge is singular at
interior angle exactly pi -- invisible to lappy's `abs(turning_angle) > 1e-10`, which only
notices three layers up, in `eigfun_integrals.corner_specs`, as a reason to bail out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ._adaptive import allocate, refine
from ._core import PREC, Frozen, _set
from .quadrature import Panel, Quadrature, nodes_from_plan, segment_nodes
from .segments import LineSegment, Segment, chord_dist_matrix, polyline_is_simple

_MAX_BLOCKING_CHORDS = 4000
#: Wall-following steps tried in each direction when hunting a polyline branch cut.
_MAX_WALK_STEPS = 400


@dataclass(frozen=True)
class Corner:
    """A junction the solver must treat as singular.

    `clearance` is the distance to the nearest NON-adjacent piece of boundary, a proxy for
    the radius of the largest disk about the corner inside the domain, which bounds where a
    local corner expansion stays valid. `branch_angle` is a direction in which a ray leaves
    into the exterior, for the branch cut of a fractional-order expansion. Both live in
    lappy but in different modules -- `corner_clearance` in eigfun_integrals, and
    `corner_branch_cut_rays` in geometry -- hand-rolling the same polyline/owner-mask
    pattern. Both are purely geometric, so both are computed once, here.

    Geometry stops at the angle and the BC pair. It never computes nu = pi/omega; that is
    what keeps lappy's `pts_per_seg(domain, fb_basis)` from having anywhere to reappear.
    """

    z: complex
    seg_out: int
    seg_in: int
    omega: float
    t_in: complex
    t_out: complex
    bc_in: float
    bc_out: float
    clearance: float
    branch_angle: float
    angle_atol: float = 1e-12
    bdry_idx: int = 0
    #: True when this corner sits on a hole rather than on the outer boundary. Set by
    #: `Domain.corners`; a Boundary on its own cannot know.
    on_hole: bool = False
    #: The adjacent segments themselves. Indices are the JOIN KEY -- `Quadrature.seg_idx`
    #: and `bdry_idx` are integer arrays, so selecting a corner's nodes is
    #: `q.seg_idx == c.seg_out` -- while these are the accessor, because the first thing a
    #: caller does with `clearance` is divide it by the adjacent segment's length.
    #: Safe to hold only because segments are immutable and can never drift. They point at
    #: leaves, never at the owning Boundary or Domain: a back-reference would close the
    #: cycle that lappy's cache.py exists to undo. Excluded from equality and repr so a
    #: Corner still compares by value and prints as numbers.
    segment_out: object = field(default=None, compare=False, repr=False)
    segment_in: object = field(default=None, compare=False, repr=False)
    #: Intermediate vertices of a polyline branch cut, empty when a straight ray suffices.
    #: `branch_angle` is then the direction of the FINAL ray, leaving the last vertex.
    branch_vertices: tuple = ()

    def branch_path(self, length):
        """The cut as a polyline: the corner, any intermediate vertices, then a point
        `length` along the final ray. What a plotter draws and a test walks."""
        if not np.isfinite(self.branch_angle):
            raise ValueError("this corner has no branch cut")
        tail = self.branch_vertices[-1] if self.branch_vertices else self.z
        return np.array([self.z, *self.branch_vertices,
                         tail + length * np.exp(1j * self.branch_angle)])

    @property
    def is_geometric(self):
        return abs(self.omega - np.pi) > self.angle_atol

    @property
    def is_reentrant(self):
        return self.omega > np.pi + self.angle_atol

    @property
    def is_bc_junction(self):
        return self.bc_in != self.bc_out

    @property
    def is_singular(self):
        return self.is_geometric or self.is_bc_junction

    @property
    def admits_exterior_branch_cut(self):
        """Whether a fractional-order expansion can be branch-cut without cutting Omega.

        A corner-adapted Fourier-Bessel function carries r^(k*nu) with nu = pi/omega, which
        for fractional k*nu needs a branch cut from the corner. The cut must kill every loop
        in Omega that winds around the corner point.

        On the outer boundary the adjacent exterior component reaches infinity, so the ray
        at `branch_angle` does it. On a HOLE it cannot: a loop in Omega encircling the hole
        winds around every point of that hole's boundary, this corner included, so omega
        gains 2*pi on it. Killing that loop needs a cut joining the hole to the outer
        boundary, which necessarily crosses Omega, and a cut confined to the bounded hole
        blocks nothing at all. The obstruction is topological, not a matter of searching
        harder for a direction.

        A hole corner is therefore usable only by expansions that are single-valued without
        a cut -- nu integral, i.e. omega = pi/m -- which is the caller's test to make, since
        geometry does not compute nu.

        `branch_angle` stays finite at a hole corner: it is still a genuine direction into
        the adjacent exterior component, which is what you want for placing a fundamental
        solution's source inside the hole. It is this flag, not that angle, that says
        whether the direction extends to a legitimate cut.
        """
        return (not self.on_hole) and bool(np.isfinite(self.branch_angle))


def _blocked_arcs(z, b0, b1):
    """Angular intervals, as (start, width), subtended at `z` by the chords b0 -> b1."""
    a = np.angle(b0 - z)
    d = np.angle((b1 - z) / (b0 - z))  # wrapped into (-pi, pi]
    start = np.where(d >= 0, a, a + d)
    return start % (2 * np.pi), np.abs(d)


def _is_blocked(theta, start, width):
    return np.any(((np.asarray(theta)[..., None] - start) % (2 * np.pi)) <= width, axis=-1)


def _path_crosses(path, b0, b1, origin, atol=1e-9):
    """Does the open polyline `path` cross any boundary chord, ignoring the start point?

    The cut begins ON the boundary, at the corner, so intersections within `atol` of
    `origin` are the cut leaving its own corner and are not crossings.
    """
    p0, p1 = path[:-1], path[1:]
    r = (p1 - p0)[:, None]
    a = b0[None, :]
    sv = (b1 - b0)[None, :]
    denom = r.real * sv.imag - r.imag * sv.real
    ok = np.abs(denom) > 1e-14
    d = a - p0[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(ok, (d.real * sv.imag - d.imag * sv.real) / np.where(ok, denom, 1.0), -1.0)
        u = np.where(ok, (d.real * r.imag - d.imag * r.real) / np.where(ok, denom, 1.0), -1.0)
    hit = ok & (t > atol) & (t < 1 - atol) & (u > atol) & (u < 1 - atol)
    if not hit.any():
        return False
    where = p0[:, None] + t * r
    return bool((hit & (np.abs(where - origin) > atol)).any())


class Boundary(Frozen):
    """A contiguous chain of segments; closed when the last meets the first."""

    def __init__(self, segments, validate=True):
        segments = tuple(segments)
        if not segments:
            raise ValueError("a boundary needs at least one segment")
        for s in segments:
            if not isinstance(s, Segment):
                raise TypeError(f"expected Segment, got {type(s).__name__}")
        prec = min(s.prec for s in segments)
        scale = max(max(s.length for s in segments), 1.0)
        tol = max(prec, 1e-12) * scale
        for i in range(len(segments) - 1):
            if abs(segments[i].p1 - segments[i + 1].p0) > tol:
                raise ValueError(
                    f"segments are not contiguous: segment {i} ends at {segments[i].p1:.6g} "
                    f"but segment {i + 1} starts at {segments[i + 1].p0:.6g}"
                )
        _set(self, segments=segments, prec=prec, _tol=tol, _validated=bool(validate))
        if validate and not self.is_simple:
            raise ValueError("boundary must be a simple curve")

    # --- basic geometry -----------------------------------------------------------------

    @cached_property
    def seg_lengths(self):
        return np.array([s.length for s in self.segments])

    @cached_property
    def length(self):
        return float(self.seg_lengths.sum())

    @property
    def angle_atol(self):
        return max(self.prec, 1e-12)

    @property
    def is_contiguous(self):
        return True  # enforced at construction

    @cached_property
    def is_closed(self):
        return abs(self.segments[-1].p1 - self.segments[0].p0) <= self._tol

    @cached_property
    def is_simple(self):
        if len(self.segments) == 1:
            # Nothing to cross but itself, and a segment can answer that analytically: an
            # arc is simple iff its sweep is at most 2*pi. Going through the polyline
            # instead cost a disk 4.6 ms of construction and an annulus 12.8 ms, building
            # 16k points to rediscover a closed form.
            return self.segments[0].is_simple
        return polyline_is_simple(self.polyline_pts, self.prec * self.length)

    @cached_property
    def polyline_pts(self):
        parts = [self.segments[0].polyline_pts]
        parts += [s.polyline_pts[1:] for s in self.segments[1:]]
        return np.concatenate(parts)

    @cached_property
    def bbox(self):
        boxes = np.array([[s.bbox[0].real, s.bbox[0].imag, s.bbox[1].real, s.bbox[1].imag]
                          for s in self.segments])
        return (complex(boxes[:, 0].min(), boxes[:, 1].min()),
                complex(boxes[:, 2].max(), boxes[:, 3].max()))

    @cached_property
    def _polyline_cache(self):
        return self.polyline_uncached()

    def polyline(self):
        """Sub-segments b0[k] -> b1[k] with the index of the segment that owns each.

        Cached: `_branch_angle` calls this once per corner, so a twelve-corner curved
        boundary rebuilt and reconcatenated its whole polyline twelve times.
        """
        return self._polyline_cache

    def polyline_uncached(self):
        b0, b1, owner = [], [], []
        for j, seg in enumerate(self.segments):
            pts = seg.polyline_pts
            b0.append(pts[:-1])
            b1.append(pts[1:])
            owner.append(np.full(len(pts) - 1, j, dtype=int))
        return np.concatenate(b0), np.concatenate(b1), np.concatenate(owner)

    @cached_property
    def signed_area(self):
        """(1/2) oint (x . n) ds -- positive for CCW.

        Taken from the quadrature rather than the polyline: chords make an inscribed
        polygon, which for a circle at prec=1e-8 is short of the true area by more than
        prec. The quadrature is exact on a polygon at any order and spectral on a curve.
        """
        def half_flux(n):
            q = self.quad(n)
            return 0.5 * float((q.radial(0j) * q.wts).sum())

        n0 = 8 * len(self.segments)
        return float(refine(half_flux, n0, self.prec, scale=self.length**2 / (4 * np.pi)))

    @cached_property
    def orientation(self):
        return 1 if self.signed_area >= 0 else -1

    @cached_property
    def _line_endpoints(self):
        """Endpoint arrays when every segment is straight, else None.

        Lets `dist` answer for all segments in one vectorized call instead of one Python
        call per segment. inradius's compass search evaluates ~48 times, so a 400-gon was
        paying 19k small numpy calls per inradius.
        """
        if all(isinstance(s, LineSegment) for s in self.segments):
            return (np.array([s.p0 for s in self.segments]),
                    np.array([s.p1 for s in self.segments]))
        return None

    def dist(self, pt):
        ends = self._line_endpoints
        if ends is not None:
            return chord_dist_matrix(*ends, pt).min(axis=1).reshape(np.shape(np.asarray(pt)))
        return np.min([s.dist(pt) for s in self.segments], axis=0)

    def _dist_matrix(self, pts):
        """(n_points, n_segments) distances -- needed unreduced, to mask adjacency."""
        ends = self._line_endpoints
        if ends is not None:
            return chord_dist_matrix(*ends, pts)
        return np.stack([np.atleast_1d(s.dist(pts)) for s in self.segments], axis=1)

    def crossings(self, z):
        return sum(s.crossings(z) for s in self.segments)

    # --- corners ------------------------------------------------------------------------

    @cached_property
    def corners(self):
        """All singular junctions, built in a batch.

        Every step is vectorized over corners. Done one corner at a time -- a distance
        against every non-adjacent segment, then a blocked-arc scan over the whole polyline
        -- this was O(n^2) with a Python call at each cell: 136 ms for a 200-gon and 534 ms
        for a 400-gon, which is the size range a shape optimizer generates.
        """
        n = len(self.segments)
        idx = np.arange(0 if self.is_closed else 1, n)
        if not len(idx):
            return ()
        prev = (idx - 1) % n
        t_out = np.array([self.segments[i].T0 for i in idx])
        t_in = np.array([self.segments[j].T1 for j in prev])
        omega = np.angle(-t_in / t_out) % (2 * np.pi)
        bc_out = np.array([self.segments[i].bc for i in idx])
        bc_in = np.array([self.segments[j].bc for j in prev])

        keep = (np.abs(omega - np.pi) > self.angle_atol) | (bc_in != bc_out)
        if not keep.any():
            return ()
        idx, prev = idx[keep], prev[keep]
        omega, t_in, t_out = omega[keep], t_in[keep], t_out[keep]
        bc_in, bc_out = bc_in[keep], bc_out[keep]
        zs = np.array([self.segments[i].p0 for i in idx])

        clear = self._clearances(zs, idx, prev)
        branch, branch_polys = self._branch_angles(zs, idx, prev, t_out, omega)
        return tuple(
            Corner(z=zs[k], seg_out=int(idx[k]), seg_in=int(prev[k]), omega=float(omega[k]),
                   t_in=t_in[k], t_out=t_out[k], bc_in=float(bc_in[k]), bc_out=float(bc_out[k]),
                   clearance=float(clear[k]), branch_angle=float(branch[k]),
                   angle_atol=self.angle_atol,
                   segment_out=self.segments[int(idx[k])],
                   segment_in=self.segments[int(prev[k])],
                   branch_vertices=branch_polys[k])
            for k in range(len(idx))
        )

    def _clearances(self, zs, seg_out, seg_in):
        """Distance from each corner to the nearest NON-adjacent segment OF THIS BOUNDARY.

        `Domain.corners` tightens this against the other boundaries. Read from a Boundary
        in isolation it is an upper bound, which is the honest answer a lone boundary can
        give: it cannot see a hole it knows nothing about.
        """
        d = self._dist_matrix(zs)
        rows = np.arange(len(zs))
        d[rows, seg_out] = np.inf
        d[rows, seg_in] = np.inf
        return d.min(axis=1)

    def _branch_angles(self, zs, seg_out, seg_in, t_out, omega):
        """A direction from each corner into the exterior.

        The exterior sector runs from the end of the interior wedge round to its start, so
        its bisector is angle(t_out) + omega/2 + pi -- canonical and exact whenever it is
        unobstructed, which is the common case. Only a corner whose bisector is blocked
        pays for the fallback scan.
        """
        bisector = (np.angle(t_out) + omega / 2 + np.pi) % (2 * np.pi)
        if self.orientation < 0:  # noqa: SIM108
            # A clockwise loop is a hole: the exterior component next to these corners is
            # the hole's own bounded interior. "Does the ray escape without re-entering
            # Omega" has no answer there -- every ray crosses the hole and leaves -- so the
            # blocking test fires on the correct direction and the fallback then hands back
            # one pointing INTO Omega. Measured on a unit hole in a 3x3 square: the true
            # exterior bisector at (1,1) is 0.25*pi, and the fallback returned 1.25*pi.
            # The bisector is right; it is the escape test that does not apply. Whether the
            # direction extends to a legitimate branch cut is `admits_exterior_branch_cut`,
            # and for a hole corner it does not, for topological reasons.
            return bisector, [() for _ in range(len(zs))]
        b0, b1, owner = self.polyline()
        if len(b0) > _MAX_BLOCKING_CHORDS:
            pick = np.linspace(0, len(b0) - 1, _MAX_BLOCKING_CHORDS).astype(int)
            b0, b1, owner = b0[pick], b1[pick], owner[pick]

        free = np.ones(len(zs), dtype=bool)
        step = max(1, (1 << 22) // max(len(b0), 1))
        for lo in range(0, len(zs), step):
            hi = min(lo + step, len(zs))
            v0 = b0[None, :] - zs[lo:hi, None]
            v1 = b1[None, :] - zs[lo:hi, None]
            # A corner IS an endpoint of its own two chords, so v0 or v1 vanishes there and
            # the direction is undefined. Those columns are masked out below anyway, but
            # dividing first and masking after leaks a nan through a comparison; guard it.
            touching = (v0 == 0) | (v1 == 0)
            v0s = np.where(touching, 1.0, v0)
            d = np.angle(np.where(touching, 1.0, v1) / v0s)
            a = np.angle(v0s)
            start = np.where(d >= 0, a, a + d) % (2 * np.pi)
            hit = ((bisector[lo:hi, None] - start) % (2 * np.pi)) <= np.abs(d)
            hit &= ~touching
            hit &= (owner[None, :] != seg_out[lo:hi, None]) & (owner[None, :] != seg_in[lo:hi, None])
            free[lo:hi] = ~hit.any(axis=1)

        out = bisector.copy()
        polylines = [() for _ in range(len(zs))]
        for k in np.flatnonzero(~free):
            angle = self._branch_fallback(zs[k], int(seg_out[k]), int(seg_in[k]))
            if not np.isfinite(angle):
                # surrounded: no straight ray escapes, so wall-follow one out
                verts, angle = self._polyline_branch_cut(zs[k], int(seg_out[k]), int(seg_in[k]))
                polylines[k] = verts
            out[k] = angle
        return out, polylines

    def free_ray_from(self, z, exclude=()):
        """Midpoint of the widest run of directions from `z` that no chord blocks, or NaN.

        A direction that meets nothing escapes to infinity, so this is the sightline test.
        `exclude` drops segments by index -- used only at a corner, whose own two segments
        touch it and would block everything.
        """
        b0, b1, owner = self.polyline()
        keep = np.ones(len(b0), dtype=bool)
        for o in exclude:
            keep &= owner != o
        if not keep.any():
            return float("nan")
        b0, b1 = b0[keep], b1[keep]
        if len(b0) > _MAX_BLOCKING_CHORDS:
            pick = np.linspace(0, len(b0) - 1, _MAX_BLOCKING_CHORDS).astype(int)
            b0, b1 = b0[pick], b1[pick]
        start, width = _blocked_arcs(z, b0, b1)
        grid = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
        ok = ~_is_blocked(grid, start, width)
        if not ok.any():
            return float("nan")
        run = np.flatnonzero(ok)
        runs = np.split(run, np.flatnonzero(np.diff(run) > 1) + 1)
        if len(runs) > 1 and ok[0] and ok[-1]:
            runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
        best = max(runs, key=len)
        return float(grid[best[len(best) // 2]])

    def _branch_fallback(self, z, seg_out, seg_in):
        return self.free_ray_from(z, exclude=(seg_out, seg_in))

    def _polyline_branch_cut(self, z, seg_out, seg_in, max_steps=_MAX_WALK_STEPS):
        """A polyline cut for a corner with no straight sightline out.

        Ported from lappy's `corner_branch_cut_polyline`: offset the boundary outward by a
        small eps, wall-follow it from the corner until reaching a vertex that CAN see
        infinity, then attach that ray. Both directions round the boundary are tried and
        the shorter valid cut wins.

        A spiral is the shape that needs it -- its inner coils surround corners whose
        exterior is a channel that turns, so no ray escapes. On lappy's 24-vertex spiral
        13 of 24 corners are in that position, and without this they carry no branch cut at
        all, which quietly bars a fractional-order expansion from every one of them.

        Heuristic, and honestly so: it follows the wall rather than searching for a
        shortest cut, and it can fail, in which case the corner reports no cut rather than
        a wrong one.
        """
        b0, b1, owner = self.polyline()
        pts, edge = b0, b1 - b0
        n = len(pts)
        seg_n = -1j * edge / np.abs(edge)
        vert_n = seg_n + np.roll(seg_n, 1)
        mag = np.abs(vert_n)
        vert_n = np.where(mag > 1e-12, vert_n / np.where(mag > 1e-12, mag, 1.0), seg_n)
        eps = 0.05 * float(np.min(np.abs(edge)))
        offset = pts + eps * vert_n
        k0 = int(np.abs(pts - z).argmin())
        stride = max(1, n // max_steps)

        best = None
        for direction in (+1, -1):
            verts = []
            for j in range(1, max_steps + 1):
                q = offset[(k0 + direction * j * stride) % n]
                verts.append(q)
                beta = self.free_ray_from(q)
                if not np.isfinite(beta):
                    continue
                path = np.array([z, *verts])
                if _path_crosses(path, b0, b1, z):
                    break
                length = float(np.abs(np.diff(path)).sum())
                cand = (len(verts), length, tuple(verts), float(beta))
                if best is None or cand[:2] < best[:2]:
                    best = cand
                break
        if best is None:
            return (), float("nan")
        return best[2], best[3]

    # --- sampling and quadrature --------------------------------------------------------

    def _grade_flags(self, grading):
        n = len(self.segments)
        if grading is None:
            return [False] * n, [False] * n
        if grading != "corner":
            raise ValueError(f"unknown grading {grading!r}")
        singular = {c.seg_out for c in self.corners if c.is_singular}
        starts = [i in singular for i in range(n)]
        ends = [((i + 1) % n) in singular for i in range(n)]
        return starts, ends

    def quad(self, n=None, *, n_per_seg=None, rule="legendre", grading=None, bdry_idx=0,
             panels=None, weight_family="arclength"):
        """Boundary quadrature.

        `n` is a TOTAL, split across segments by arclength with a floor. lappy offers only
        per-segment counts, which is what pushed an allocation decision onto callers until
        someone wrote the basis-aware `pts_per_seg` to make it for them.
        """
        segs = self.segments
        if panels is not None:
            return self._quad_from_plan(panels, bdry_idx, weight_family)
        if n_per_seg is not None:
            counts = np.asarray(n_per_seg, dtype=int)
            if len(counts) != len(segs):
                raise ValueError(f"n_per_seg has {len(counts)} entries for {len(segs)} segments")
        elif n is not None:
            counts = allocate(int(n), self.seg_lengths)
        else:
            raise ValueError("give either n or n_per_seg")
        starts, ends = self._grade_flags(grading)
        panel0 = 0
        blocks, panels = [], []
        for i, seg in enumerate(segs):
            if counts[i] <= 0:
                continue
            d = segment_nodes(seg, int(counts[i]), rule=rule,
                              grade_start=starts[i], grade_end=ends[i], panel0=panel0)
            panel0 += d["n_panels"]
            d["seg_idx"] = np.full(len(d["tau"]), i, dtype=int)
            d["bc"] = np.full(len(d["tau"]), seg.bc)
            blocks.append(d)
            panels.extend((i, lo, hi) for lo, hi in d["panels"])
        cat = lambda key: np.concatenate([b[key] for b in blocks])
        m = sum(len(b["tau"]) for b in blocks)
        return Quadrature(
            pts=cat("pts"), normals=cat("normals"), tangents=cat("tangents"),
            wts=cat("wts"), tau=cat("tau"), seg_idx=cat("seg_idx"),
            bdry_idx=np.full(m, bdry_idx, dtype=int), bc=cat("bc"),
            panel_id=cat("panel_id"), panels=tuple(panels),
        )

    def _quad_from_plan(self, panels, bdry_idx, weight_family):
        """Assemble a caller's panel plan. Geometry supplies frames, weights, provenance
        and BC tagging; the plan supplies only where the nodes go and what they weigh."""
        by_seg = {}
        for pan in panels:
            by_seg.setdefault(int(pan.seg_idx), []).append(pan)
        panel0 = 0
        blocks, spans = [], []
        for i in sorted(by_seg):
            plan = sorted(by_seg[i], key=lambda q: q.tau0)
            d = nodes_from_plan(self.segments[i], plan, panel0=panel0)
            panel0 += d["n_panels"]
            d["seg_idx"] = np.full(len(d["tau"]), i, dtype=int)
            d["bc"] = np.full(len(d["tau"]), self.segments[i].bc)
            blocks.append(d)
            spans.extend((i, lo, hi) for lo, hi in d["panels"])
        cat = lambda key: np.concatenate([b[key] for b in blocks])
        m = sum(len(b["tau"]) for b in blocks)
        return Quadrature(
            pts=cat("pts"), normals=cat("normals"), tangents=cat("tangents"),
            wts=cat("wts"), tau=cat("tau"), seg_idx=cat("seg_idx"),
            bdry_idx=np.full(m, bdry_idx, dtype=int), bc=cat("bc"),
            panel_id=cat("panel_id"), panels=tuple(spans), weight_family=weight_family,
        )

    def sample(self, n=None, *, n_per_seg=None, grading=None):
        return self.quad(n, n_per_seg=n_per_seg, grading=grading).pts

    # --- construction and transforms ----------------------------------------------------

    @classmethod
    def from_vertices(cls, vertices, bc="dir", close=True, validate=True, prec=PREC):
        v = np.asarray(vertices, dtype=complex)
        if len(v) < 2:
            raise ValueError("need at least two vertices")
        pairs = list(zip(v, np.roll(v, -1))) if close else list(zip(v[:-1], v[1:]))
        return cls([LineSegment(a, b, bc=bc, prec=prec) for a, b in pairs], validate=validate)

    def _rebuilt(self, segments):
        return Boundary(segments, validate=False)

    def reversed(self):
        return self._rebuilt([s.reversed() for s in reversed(self.segments)])

    def translated(self, dz):
        return self._rebuilt([s.translated(dz) for s in self.segments])

    def scaled(self, s, about=0j):
        return self._rebuilt([seg.scaled(s, about=about) for seg in self.segments])

    def rotated(self, theta, about=0j):
        return self.scaled(np.exp(1j * float(theta)), about=about)

    def with_bc(self, bc):
        return self._rebuilt([s.with_bc(bc) for s in self.segments])

    def with_prec(self, prec):
        return self._rebuilt([s.with_prec(prec) for s in self.segments])

    def __add__(self, dz):
        return self.translated(dz)

    __radd__ = __add__

    def __mul__(self, s):
        return self.scaled(s)

    __rmul__ = __mul__

    def __len__(self):
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def plot(self, **kwargs):
        """Draw this boundary. See `chladni.plotting.plot_boundary`."""
        from ..plotting import plot_boundary

        return plot_boundary(self, **kwargs)

    def __repr__(self):
        return f"Boundary({len(self.segments)} segments, length={self.length:.6g})"
