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

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ._adaptive import allocate, refine
from ._core import PREC, Frozen, _set
from .quadrature import Quadrature, segment_nodes
from .segments import LineSegment, Segment, polyline_is_simple

_MAX_BLOCKING_CHORDS = 4000


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

    Geometry stops at the angle and the BC pair. It never computes nu = pi/theta; that is
    what keeps lappy's `pts_per_seg(domain, fb_basis)` from having anywhere to reappear.
    """

    z: complex
    seg_out: int
    seg_in: int
    theta: float
    t_in: complex
    t_out: complex
    bc_in: float
    bc_out: float
    clearance: float
    branch_angle: float
    angle_atol: float = 1e-12

    @property
    def is_geometric(self):
        return abs(self.theta - np.pi) > self.angle_atol

    @property
    def is_reentrant(self):
        return self.theta > np.pi + self.angle_atol

    @property
    def is_bc_junction(self):
        return self.bc_in != self.bc_out

    @property
    def is_singular(self):
        return self.is_geometric or self.is_bc_junction


def _blocked_arcs(z, b0, b1):
    """Angular intervals, as (start, width), subtended at `z` by the chords b0 -> b1."""
    a = np.angle(b0 - z)
    d = np.angle((b1 - z) / (b0 - z))  # wrapped into (-pi, pi]
    start = np.where(d >= 0, a, a + d)
    return start % (2 * np.pi), np.abs(d)


def _is_blocked(phi, start, width):
    return np.any(((np.asarray(phi)[..., None] - start) % (2 * np.pi)) <= width, axis=-1)


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

    def dist(self, pt):
        return np.min([s.dist(pt) for s in self.segments], axis=0)

    def crossings(self, z):
        return sum(s.crossings(z) for s in self.segments)

    # --- corners ------------------------------------------------------------------------

    @cached_property
    def corners(self):
        n = len(self.segments)
        first = 0 if self.is_closed else 1
        out = []
        for i in range(first, n):
            prev = self.segments[(i - 1) % n]
            cur = self.segments[i]
            t_in, t_out = prev.T1, cur.T0
            theta = float(np.angle(-t_in / t_out) % (2 * np.pi))
            geometric = abs(theta - np.pi) > self.angle_atol
            junction = prev.bc != cur.bc
            if not (geometric or junction):
                continue
            z = cur.p0
            out.append(
                Corner(
                    z=z,
                    seg_out=i,
                    seg_in=(i - 1) % n,
                    theta=theta,
                    t_in=t_in,
                    t_out=t_out,
                    bc_in=prev.bc,
                    bc_out=cur.bc,
                    clearance=self._clearance(z, i, (i - 1) % n),
                    branch_angle=self._branch_angle(z, i, (i - 1) % n, t_out, theta),
                    angle_atol=self.angle_atol,
                )
            )
        return tuple(out)

    def _clearance(self, z, seg_out, seg_in):
        others = [s for k, s in enumerate(self.segments) if k not in (seg_out, seg_in)]
        if not others:
            return float(np.inf)
        return float(min(float(s.dist(z)) for s in others))

    def _branch_angle(self, z, seg_out, seg_in, t_out, theta):
        """A direction from the corner into the exterior.

        The exterior sector runs from the end of the interior wedge round to its start, so
        its bisector is angle(t_out) + theta/2 + pi. That is canonical and exact whenever it
        is unobstructed, which is the common case; otherwise take the midpoint of the widest
        unobstructed run of directions.
        """
        b0, b1, owner = self.polyline()
        keep = (owner != seg_out) & (owner != seg_in)
        if not keep.any():
            return float((np.angle(t_out) + theta / 2 + np.pi) % (2 * np.pi))
        b0, b1 = b0[keep], b1[keep]
        if len(b0) > _MAX_BLOCKING_CHORDS:
            idx = np.linspace(0, len(b0) - 1, _MAX_BLOCKING_CHORDS).astype(int)
            b0, b1 = b0[idx], b1[idx]
        start, width = _blocked_arcs(z, b0, b1)
        bisector = float((np.angle(t_out) + theta / 2 + np.pi) % (2 * np.pi))
        if not _is_blocked(bisector, start, width):
            return bisector
        grid = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
        free = ~_is_blocked(grid, start, width)
        if not free.any():
            return float("nan")
        # widest circular run of free directions
        idx = np.flatnonzero(free)
        splits = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splits + 1)
        if len(runs) > 1 and free[0] and free[-1]:
            runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
        best = max(runs, key=len)
        return float(grid[best[len(best) // 2]])

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

    def quad(self, n=None, *, n_per_seg=None, rule="legendre", grading=None, bdry_idx=0):
        """Boundary quadrature.

        `n` is a TOTAL, split across segments by arclength with a floor. lappy offers only
        per-segment counts, which is what pushed an allocation decision onto callers until
        someone wrote the basis-aware `pts_per_seg` to make it for them.
        """
        segs = self.segments
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

    def __repr__(self):
        return f"Boundary({len(self.segments)} segments, length={self.length:.6g})"
