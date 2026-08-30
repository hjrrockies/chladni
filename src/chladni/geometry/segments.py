"""Boundary segments.

Every segment carries a native parameter ``t`` in ``[t0, t1]`` supplied by whoever built it,
and exposes a normalized arclength parameter ``tau`` in ``[0, 1]`` as the public one, so
``|dp(tau)| == length`` identically. That property is what all the boundary quadrature rests
on, and it holds here by construction rather than by refinement: ``t'(s) = 1/|p'(t)|``
analytically, so ``dp(tau) = p'(t) * L / |p'(t)|``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from ._adaptive import (
    _NEWTON_ITERS,
    _ANCHOR_ORDER,
    _REMAINDER_ORDER,
    adaptive_arclength_table,
    adaptive_polyline,
    gauss_legendre,
)
from ._core import PREC, Frozen, _set, check_prec, normalize_bc
from .quadrature import Quadrature, nodes_from_plan, segment_nodes


#: Above this many (point, chord) pairs, sweep instead of forming the outer product.
_DENSE_CROSSING_BUDGET = 1 << 22


def _cross(a, b):
    """2D cross product of complex-encoded vectors."""
    return a.real * b.imag - a.imag * b.real


def chord_dist_matrix(b0, b1, pts):
    """(n_points, n_chords) distances to the chords b0[k] -> b1[k], in one vectorized pass."""
    flat = np.atleast_1d(np.asarray(pts, dtype=complex)).ravel()
    d = b1 - b0
    denom = np.where(np.abs(d) ** 2 == 0.0, 1.0, np.abs(d) ** 2)
    t = np.clip(((flat[:, None] - b0[None, :]) * d.conjugate()[None, :]).real / denom[None, :], 0.0, 1.0)
    return np.abs(flat[:, None] - (b0[None, :] + t * d[None, :]))


def _polyline_dist(b0, b1, pt):
    """Distance from each point to the nearest of the chords b0[k] -> b1[k]."""
    pt = np.asarray(pt, dtype=complex)
    return chord_dist_matrix(b0, b1, pt).min(axis=1).reshape(np.shape(pt))


def _chord_crossings(b0, b1, z):
    """Horizontal +x ray crossing count against a set of chords, half-open in y.

    Two algorithms, chosen by problem size. Sort-and-sweep rather than an (N, M) outer
    product once the chord count makes the product expensive. The straddle test
    ``(ya <= y) != (yb <= y)`` is exactly ``min(ya,yb) <= y < max(ya,yb)``, so sorting the
    query heights once turns each chord's affected points into a contiguous slice found by
    two binary searches. Work is then proportional to the straddles that actually occur --
    about 2N for a polyline of a bounded curve -- instead of N times the chord count.

    The outer-product form was O(N*M) in memory as well as time: a spline boundary carries
    ~16k chords at prec=1e-8, so one temporary at N=1e5 wanted ~13 GB, and containment on a
    spline domain took 605 ms at N=1e4 against 2 ms for a polygon.
    """
    z = np.asarray(z, dtype=complex)
    flat = np.atleast_1d(z).ravel()
    n = len(flat)
    if n == 0:
        return np.zeros(np.shape(z), dtype=int)
    y, x = flat.imag, flat.real
    ya, yb = b0.imag, b1.imag

    if n * len(b0) <= _DENSE_CROSSING_BUDGET:
        # Few chords: the outer product beats the sweep, which would pay an O(N log N)
        # sort to save work that is not there. A LineSegment carries ONE chord, so every
        # polygon lands here -- sorting per segment made an L-shape 19x slower.
        straddles = (ya[None, :] <= y[:, None]) != (yb[None, :] <= y[:, None])
        dy = np.where(yb == ya, 1.0, yb - ya)
        t = (y[:, None] - ya[None, :]) / dy[None, :]
        xi = b0.real[None, :] + t * (b1.real - b0.real)[None, :]
        return (straddles & (xi > x[:, None])).sum(axis=1).reshape(np.shape(z))

    order = np.argsort(y, kind="stable")
    ys = y[order]

    ylo, yhi = np.minimum(ya, yb), np.maximum(ya, yb)
    lo = np.searchsorted(ys, ylo, side="left")
    hi = np.searchsorted(ys, yhi, side="left")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return np.zeros(np.shape(z), dtype=int)

    chord = np.repeat(np.arange(len(b0)), counts)
    starts = np.repeat(lo, counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    pt = order[starts + offsets]

    dy = yb[chord] - ya[chord]
    t = (y[pt] - ya[chord]) / dy
    xi = b0.real[chord] + t * (b1.real - b0.real)[chord]
    hit = pt[xi > x[pt]]
    return np.bincount(hit, minlength=n).reshape(np.shape(z))


def _vectorized(f, sample):
    """Return a callable that maps an array of parameters to a complex array."""

    def direct(t):
        return np.asarray(f(t), dtype=complex)

    try:
        out = direct(sample)
        if out.shape == sample.shape:
            return direct
    except Exception:  # noqa: BLE001 -- any failure just means it is not vectorized
        pass
    vec = np.vectorize(f, otypes=[complex])
    return lambda t: np.asarray(vec(t), dtype=complex)


def polyline_is_simple(pts, tol):
    """Simplicity of a polyline at working precision: no exact self-crossing, and no two
    non-adjacent vertices closer than `tol`.

    The tolerance is not belt-and-braces. Exact predicates miss a curve that *touches*
    itself rather than crossing a region: the figure-eight sin(t) + i sin(t) cos(t) returns
    to the origin at three parameters whose floating-point images differ by ~1e-16, so no
    two chords intersect exactly and JTS reports it simple. A boundary that approaches
    itself that closely is degenerate for the eigensolver whether or not it touches, so
    proximity is the right question to ask at this layer.
    """
    from scipy.spatial import cKDTree
    from shapely import LineString

    from ._core import as_xy

    xy = as_xy(pts)
    if not LineString(xy).is_simple:
        return False
    n = len(xy)
    closed = np.abs(pts[0] - pts[-1]) <= tol
    pairs = cKDTree(xy).query_pairs(r=max(tol, 0.0), output_type="ndarray")
    if not len(pairs):
        return True
    far = np.abs(pairs[:, 1] - pairs[:, 0]) > 1
    if closed:
        far &= ~((pairs[:, 0] == 0) & (pairs[:, 1] == n - 1))
    return not bool(far.any())


class Segment(Frozen, ABC):
    """Abstract boundary segment in normalized arclength parameterization."""

    def __init__(self, bc="dir", prec=PREC):
        _set(self, bc=normalize_bc(bc), prec=check_prec(prec))

    # --- parameterization ---------------------------------------------------------------

    @abstractmethod
    def p(self, tau):
        """Position at normalized arclength tau in [0, 1]."""

    @abstractmethod
    def dp(self, tau):
        """dp/dtau. Its modulus is `length` everywhere, by construction."""

    @property
    @abstractmethod
    def length(self):
        """Arclength of the segment."""

    def T(self, tau):
        d = self.dp(tau)
        return d / np.abs(d)

    def N(self, tau):
        """Outward unit normal, under the CCW-outer / CW-hole orientation convention."""
        return -1j * self.T(tau)

    @property
    def p0(self):
        return complex(self.p(0.0))

    @property
    def p1(self):
        return complex(self.p(1.0))

    @property
    def T0(self):
        return complex(self.T(0.0))

    @property
    def T1(self):
        return complex(self.T(1.0))

    @property
    def break_ts(self):
        """Derivative breaks in the segment's OWN parameter, strictly inside (t0, t1)."""
        return np.empty(0)

    @property
    def break_taus(self):
        """Derivative breaks in tau. Quadrature panels must break here."""
        return np.empty(0)

    @property
    def is_closed(self):
        return abs(self.p1 - self.p0) <= self.prec * self.length

    # --- polyline -----------------------------------------------------------------------

    @property
    def polyline_pts(self):
        """Points whose chords track the segment to within prec * length."""
        raise NotImplementedError

    @cached_property
    def bbox(self):
        """(lower-left, upper-right) as complex. Exact for lines and arcs."""
        pts = self.polyline_pts
        return (complex(pts.real.min(), pts.imag.min()), complex(pts.real.max(), pts.imag.max()))

    def _chords(self):
        pts = self.polyline_pts
        return pts[:-1], pts[1:]

    def dist(self, pt):
        b0, b1 = self._chords()
        return _polyline_dist(b0, b1, pt)

    def crossings(self, z):
        """Count of +x ray crossings, for the point-in-domain parity test."""
        b0, b1 = self._chords()
        return _chord_crossings(b0, b1, z)

    # --- sampling and quadrature --------------------------------------------------------

    def sample(self, n, *, grading=None):
        return self.quad(n, grading=grading).pts

    def quad(self, n=None, *, rule="legendre", grading=None, grade_start=False,
             grade_end=False, panels=None, weight_family="arclength"):
        """Quadrature on this segment.

        `panels` lets a higher layer supply its own plan -- the seam a corner-adapted
        Gauss-Jacobi rule needs. Geometry still owns the arclength map, the frames, the
        provenance and the BC tagging; only the choice of rule, which needs nu, sits above.
        Without it that layer would have to redo the affine panel map by hand, which is what
        lappy's PointSet docstring records a downstream package having to do.
        """
        if panels is not None:
            d = nodes_from_plan(self, panels)
        else:
            if grading == "corner":
                grade_start = grade_end = True
            elif grading is not None:
                raise ValueError(f"unknown grading {grading!r}")
            if n is None:
                raise ValueError("give either n or panels")
            d = segment_nodes(self, n, rule=rule, grade_start=grade_start, grade_end=grade_end)
        m = len(d["tau"])
        return Quadrature(
            pts=d["pts"],
            normals=d["normals"],
            tangents=d["tangents"],
            wts=d["wts"],
            tau=d["tau"],
            seg_idx=np.zeros(m, dtype=int),
            bdry_idx=np.zeros(m, dtype=int),
            bc=np.full(m, self.bc),
            panel_id=d["panel_id"],
            panels=d["panels"],
            weight_family=weight_family,
        )

    # --- predicates ---------------------------------------------------------------------

    @cached_property
    def is_simple(self):
        return polyline_is_simple(self.polyline_pts, self.prec * self.length)

    def intersection(self, other):
        if isinstance(self, LineSegment) and isinstance(other, LineSegment):
            return _line_line_intersection(self, other)
        from shapely import LineString

        from ._core import as_xy, from_xy

        hit = LineString(as_xy(self.polyline_pts)).intersection(
            LineString(as_xy(other.polyline_pts))
        )
        if hit.is_empty:
            return np.empty(0, dtype=complex)
        geoms = getattr(hit, "geoms", [hit])
        out = []
        for g in geoms:
            coords = np.asarray(g.coords) if hasattr(g, "coords") else np.empty((0, 2))
            if len(coords):
                out.append(from_xy(coords))
        return np.unique(np.concatenate(out)) if out else np.empty(0, dtype=complex)

    def intersects(self, other):
        return len(self.intersection(other)) > 0

    # --- transforms ---------------------------------------------------------------------

    @abstractmethod
    def translated(self, dz):
        ...

    @abstractmethod
    def scaled(self, s, about=0j):
        ...

    @abstractmethod
    def reversed(self):
        ...

    def rotated(self, theta, about=0j):
        return self.scaled(np.exp(1j * float(theta)), about=about)

    @abstractmethod
    def with_bc(self, bc):
        ...

    @abstractmethod
    def with_prec(self, prec):
        ...

    def __add__(self, dz):
        return self.translated(dz)

    __radd__ = __add__

    def __mul__(self, s):
        return self.scaled(s)

    __rmul__ = __mul__

    def __neg__(self):
        return self.reversed()

    def plot(self, **kwargs):
        """Draw this segment. See `chladni.plotting.plot_segment`.

        matplotlib is imported here, not at module scope: the geometry layer must stay
        usable without it. lappy's geometry.py carries a top-level
        `import matplotlib.pyplot as plt`, which makes layer 1 depend on a plotting library.
        """
        from ..plotting import plot_segment

        return plot_segment(self, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}(p0={self.p0:.4g}, p1={self.p1:.4g}, length={self.length:.6g})"


def _line_line_intersection(a, b):
    p, r = a.p0, a.p1 - a.p0
    q, s = b.p0, b.p1 - b.p0
    denom = _cross(r, s)
    scale = abs(r) * abs(s)
    if abs(denom) <= 1e-14 * max(scale, 1.0):
        if abs(_cross(q - p, r)) > 1e-12 * max(abs(r), 1.0):
            return np.empty(0, dtype=complex)  # parallel, not collinear
        u = np.array([0.0, 1.0, ((q - p) * r.conjugate()).real / abs(r) ** 2,
                      ((q + s - p) * r.conjugate()).real / abs(r) ** 2])
        lo = max(0.0, min(u[2], u[3]))
        hi = min(1.0, max(u[2], u[3]))
        if lo > hi:
            return np.empty(0, dtype=complex)
        return np.unique(np.array([p + lo * r, p + hi * r]))
    t = _cross(q - p, s) / denom
    w = _cross(q - p, r) / denom
    if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= w <= 1 + 1e-12:
        return np.array([p + np.clip(t, 0.0, 1.0) * r])
    return np.empty(0, dtype=complex)


class LineSegment(Segment):
    """A straight segment. Everything about it is closed form."""

    def __init__(self, p0, p1, bc="dir", prec=PREC):
        super().__init__(bc, prec)
        a, b = complex(p0), complex(p1)
        if a == b:
            raise ValueError(f"degenerate zero-length segment at {a!r}")
        _set(self, _a=a, _b=b)

    @property
    def length(self):
        return abs(self._b - self._a)

    def p(self, tau):
        t = np.asarray(tau, dtype=float)
        return self._a + t * (self._b - self._a)

    def dp(self, tau):
        t = np.asarray(tau, dtype=float)
        return np.full(t.shape, self._b - self._a, dtype=complex)

    @property
    def p0(self):
        return self._a

    @property
    def p1(self):
        return self._b

    @property
    def polyline_pts(self):
        return np.array([self._a, self._b])

    @cached_property
    def is_simple(self):
        return True

    @cached_property
    def bbox(self):
        return (complex(min(self._a.real, self._b.real), min(self._a.imag, self._b.imag)),
                complex(max(self._a.real, self._b.real), max(self._a.imag, self._b.imag)))

    def dist(self, pt):
        pt = np.asarray(pt, dtype=complex)
        d = self._b - self._a
        t = np.clip(((pt - self._a) * d.conjugate()).real / abs(d) ** 2, 0.0, 1.0)
        return np.abs(pt - (self._a + t * d))

    def translated(self, dz):
        return LineSegment(self._a + dz, self._b + dz, self.bc, self.prec)

    def scaled(self, s, about=0j):
        s, about = complex(s), complex(about)
        return LineSegment(about + s * (self._a - about), about + s * (self._b - about), self.bc, self.prec)

    def reversed(self):
        return LineSegment(self._b, self._a, self.bc, self.prec)

    def with_bc(self, bc):
        return LineSegment(self._a, self._b, bc, self.prec)

    def with_prec(self, prec):
        return LineSegment(self._a, self._b, self.bc, prec)


class ArcSegment(Segment):
    """A circular arc. lappy builds every arc as a general ParametricSegment and pays the
    adaptive machinery for something with a closed form; this does not."""

    def __init__(self, center, radius, theta0, theta1, bc="dir", prec=PREC):
        super().__init__(bc, prec)
        r = float(radius)
        if r <= 0:
            raise ValueError(f"radius must be positive, got {r!r}")
        if theta1 == theta0:
            raise ValueError("degenerate zero-length arc")
        _set(self, center=complex(center), radius=r, theta0=float(theta0), theta1=float(theta1))

    @property
    def sweep(self):
        return self.theta1 - self.theta0

    @property
    def length(self):
        return self.radius * abs(self.sweep)

    def _theta(self, tau):
        return self.theta0 + np.asarray(tau, dtype=float) * self.sweep

    def p(self, tau):
        return self.center + self.radius * np.exp(1j * self._theta(tau))

    def dp(self, tau):
        return 1j * self.sweep * self.radius * np.exp(1j * self._theta(tau))

    @cached_property
    def polyline_pts(self):
        taus = adaptive_polyline(self.p, self.prec * self.length)
        return self.p(taus)

    @cached_property
    def is_simple(self):
        return abs(self.sweep) <= 2 * np.pi + 1e-12

    @cached_property
    def bbox(self):
        """Endpoints plus whichever axis extremes the sweep actually reaches -- exact, and
        it does not force the 16k-point polyline into existence just to find a box."""
        lo, hi = min(self.theta0, self.theta1), max(self.theta0, self.theta1)
        k0, k1 = int(np.ceil(lo / (np.pi / 2))), int(np.floor(hi / (np.pi / 2)))
        angles = np.concatenate([[lo, hi], (np.pi / 2) * np.arange(k0, k1 + 1)])
        pts = self.center + self.radius * np.exp(1j * angles)
        return (complex(pts.real.min(), pts.imag.min()), complex(pts.real.max(), pts.imag.max()))

    def dist(self, pt):
        pt = np.asarray(pt, dtype=complex)
        v = pt - self.center
        lo, hi = min(self.theta0, self.theta1), max(self.theta0, self.theta1)
        inside = ((np.angle(v) - lo) % (2 * np.pi)) <= (hi - lo)
        radial = np.abs(np.abs(v) - self.radius)
        ends = np.minimum(np.abs(pt - self.p0), np.abs(pt - self.p1))
        return np.where(inside, radial, ends)

    def crossings(self, z):
        """Exact circle/ray parity, split into sub-arcs on which y is monotone.

        y = cy + r sin(theta) turns at cos(theta) == 0, so cutting there leaves each piece
        with a fixed branch of x = cx +- r sqrt(1 - s^2). This keeps `contains` exact for
        every arc-bounded domain -- notably the annulus, where a polyline hole would report
        points in its chord gaps as interior.
        """
        z = np.asarray(z, dtype=complex)
        cx, cy, r = self.center.real, self.center.imag, self.radius
        lo, hi = min(self.theta0, self.theta1), max(self.theta0, self.theta1)
        k0 = int(np.ceil((lo - np.pi / 2) / np.pi))
        k1 = int(np.floor((hi - np.pi / 2) / np.pi))
        cuts = np.unique(np.concatenate([[lo, hi], np.pi / 2 + np.pi * np.arange(k0, k1 + 1)]))
        cuts = cuts[(cuts >= lo) & (cuts <= hi)]
        ys = cy + r * np.sin(cuts)
        if abs(abs(self.sweep) - 2 * np.pi) <= 1e-12:
            # A closed arc has no real seam, but sin(theta0) and sin(theta0 + 2*pi) differ
            # by ~1e-16. That leaves a sliver of y uncovered by the monotone pieces, and a
            # query point inside it is misclassified -- which is exactly where a hole's own
            # polyline closes, so an annulus rejected its hole as "not contained".
            ys[-1] = ys[0]
        total = np.zeros(np.shape(z), dtype=int)
        y, xq = z.imag, z.real
        for k, (a_, b_) in enumerate(zip(cuts[:-1], cuts[1:])):
            if b_ <= a_:
                continue
            sigma = np.sign(np.cos(0.5 * (a_ + b_)))
            ya, yb = ys[k], ys[k + 1]
            straddles = (ya <= y) != (yb <= y)
            s = np.clip((y - cy) / r, -1.0, 1.0)
            xi = cx + r * sigma * np.sqrt(np.maximum(0.0, 1.0 - s * s))
            total = total + (straddles & (xi > xq)).astype(int)
        return total

    def translated(self, dz):
        return ArcSegment(self.center + dz, self.radius, self.theta0, self.theta1, self.bc, self.prec)

    def scaled(self, s, about=0j):
        s, about = complex(s), complex(about)
        if s == 0:
            raise ValueError("scale factor must be nonzero")
        phi = np.angle(s)
        return ArcSegment(
            about + s * (self.center - about), abs(s) * self.radius,
            self.theta0 + phi, self.theta1 + phi, self.bc, self.prec,
        )

    def reversed(self):
        return ArcSegment(self.center, self.radius, self.theta1, self.theta0, self.bc, self.prec)

    def with_bc(self, bc):
        return ArcSegment(self.center, self.radius, self.theta0, self.theta1, bc, self.prec)

    def with_prec(self, prec):
        return ArcSegment(self.center, self.radius, self.theta0, self.theta1, self.bc, prec)


class ParametricSegment(Segment):
    """A segment given by differentiable ``p(t)`` and ``dp(t)`` on ``[t0, t1]``."""

    def __init__(self, p, dp, t0, t1, bc="dir", prec=PREC, break_ts=()):
        super().__init__(bc, prec)
        t0, t1 = float(t0), float(t1)
        if t1 <= t0:
            raise ValueError(f"t1 ({t1}) must exceed t0 ({t0})")
        sample = np.linspace(t0, t1, 5)
        _set(
            self,
            _pf=_vectorized(p, sample),
            _dpf=_vectorized(dp, sample),
            _p_raw=p,
            _dp_raw=dp,
            t0=t0,
            t1=t1,
            _breaks=np.sort(np.asarray(break_ts, dtype=float)),
        )

    # --- the arclength map --------------------------------------------------------------

    def _speed(self, t):
        return np.abs(self._dpf(np.asarray(t, dtype=float)))

    @cached_property
    def _reparam(self):
        chord = np.abs(np.diff(self._pf(np.linspace(self.t0, self.t1, 65)))).sum()
        eps_abs = self.prec * max(float(chord), 1e-300)
        nodes = adaptive_arclength_table(self._speed, self.t0, self.t1, self.prec, eps_abs)
        if len(self._breaks):
            nodes = np.unique(np.concatenate([nodes, self._breaks]))
        x, w = gauss_legendre(_ANCHOR_ORDER)
        lo, hi = nodes[:-1], nodes[1:]
        span = hi - lo
        vals = self._speed((lo[:, None] + span[:, None] * x[None, :]).ravel()).reshape(len(lo), -1)
        s_nodes = np.concatenate([[0.0], np.cumsum(span * (vals @ w))])
        if s_nodes[-1] <= 0:
            raise ValueError("degenerate zero-length segment")
        guess = PchipInterpolator(s_nodes, nodes)
        return nodes, s_nodes, guess

    @property
    def length(self):
        return float(self._reparam[1][-1])

    def _s_of_t(self, t):
        """Exact arclength from t0: anchor at the bracketing node, integrate the remainder."""
        nodes, s_nodes, _ = self._reparam
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i = np.clip(np.searchsorted(nodes, t, side="right") - 1, 0, len(nodes) - 2)
        lo = nodes[i]
        span = t - lo
        x, w = gauss_legendre(_REMAINDER_ORDER)
        vals = self._speed((lo[:, None] + span[:, None] * x[None, :]).ravel()).reshape(len(lo), -1)
        return s_nodes[i] + span * (vals @ w)

    def _t_of_s(self, s):
        """Safeguarded Newton on s(t) - s, with ds/dt = |p'(t)|."""
        nodes, s_nodes, guess = self._reparam
        s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, s_nodes[-1])
        i = np.clip(np.searchsorted(s_nodes, s, side="right") - 1, 0, len(nodes) - 2)
        lo, hi = nodes[i].copy(), nodes[i + 1].copy()
        t = np.clip(np.asarray(guess(s), dtype=float), lo, hi)
        for _ in range(_NEWTON_ITERS):
            f = self._s_of_t(t) - s
            lo = np.where(f < 0, t, lo)
            hi = np.where(f < 0, hi, t)
            d = self._speed(t)
            step = np.where(d > 0, f / np.where(d > 0, d, 1.0), 0.0)
            nxt = t - step
            bad = ~np.isfinite(nxt) | (nxt < lo) | (nxt > hi)
            t = np.where(bad, 0.5 * (lo + hi), nxt)
        return t

    def _t(self, tau):
        tau = np.asarray(tau, dtype=float)
        return self._t_of_s(tau.ravel() * self.length).reshape(tau.shape)

    def p(self, tau):
        return self._pf(self._t(tau))

    def dp(self, tau):
        d = self._dpf(self._t(tau))
        return d / np.abs(d) * self.length

    @property
    def p0(self):
        return complex(self._pf(np.asarray(self.t0)))

    @property
    def p1(self):
        return complex(self._pf(np.asarray(self.t1)))

    @property
    def break_ts(self):
        return self._breaks

    @cached_property
    def break_taus(self):
        if not len(self._breaks):
            return np.empty(0)
        taus = np.asarray(self._s_of_t(self._breaks)) / self.length
        return np.unique(taus[(taus > 1e-12) & (taus < 1 - 1e-12)])

    @cached_property
    def polyline_pts(self):
        """Built in the NATIVE parameter, so `is_simple` never needs the arclength map."""
        chord = np.abs(np.diff(self._pf(np.linspace(self.t0, self.t1, 65)))).sum()
        ts = adaptive_polyline(self._pf, self.prec * float(chord), self.t0, self.t1)
        return self._pf(ts)

    # --- transforms ---------------------------------------------------------------------

    def _rebuilt(self, p, dp, breaks=None, bc=None, prec=None):
        return ParametricSegment(
            p, dp, self.t0, self.t1,
            self.bc if bc is None else bc,
            self.prec if prec is None else prec,
            self._breaks if breaks is None else breaks,
        )

    def translated(self, dz):
        dz = complex(dz)
        pf, dpf = self._p_raw, self._dp_raw
        return self._rebuilt(lambda t: np.asarray(pf(t)) + dz, dpf)

    def scaled(self, s, about=0j):
        s, about = complex(s), complex(about)
        if s == 0:
            raise ValueError("scale factor must be nonzero")
        pf, dpf = self._p_raw, self._dp_raw
        return self._rebuilt(
            lambda t: about + s * (np.asarray(pf(t)) - about),
            lambda t: s * np.asarray(dpf(t)),
        )

    def reversed(self):
        pf, dpf, a, b = self._p_raw, self._dp_raw, self.t0, self.t1
        breaks = np.sort(a + b - self._breaks) if len(self._breaks) else self._breaks
        return self._rebuilt(
            lambda t: np.asarray(pf(a + b - np.asarray(t))),
            lambda t: -np.asarray(dpf(a + b - np.asarray(t))),
            breaks=breaks,
        )

    def with_bc(self, bc):
        return self._rebuilt(self._p_raw, self._dp_raw, bc=bc)

    def with_prec(self, prec):
        return self._rebuilt(self._p_raw, self._dp_raw, prec=prec)


class SplineSegment(ParametricSegment):
    """A cubic-spline segment. Its knots are derivative breaks, and quadrature panels break
    there: a degree-k B-spline is only C^(k-1) at its knots."""

    def __init__(self, spline, bc="dir", prec=PREC):
        knots = np.asarray(spline.x, dtype=float)
        deriv = spline.derivative()
        super().__init__(spline, deriv, knots[0], knots[-1], bc, prec, break_ts=knots[1:-1])
        _set(self, spline=spline)

    @classmethod
    def interp_from_pts(cls, pts, bc="dir", prec=PREC, spline_bc_type="natural"):
        pts = np.asarray(pts, dtype=complex)
        if len(pts) < 3:
            raise ValueError("need at least three points to interpolate a spline")
        closed = abs(pts[-1] - pts[0]) <= 1e-14 * max(1.0, np.abs(pts).max())
        d = np.abs(np.diff(pts))
        if np.any(d == 0):
            keep = np.concatenate([[True], d > 0])
            pts, d = pts[keep], np.abs(np.diff(pts[keep]))
        t = np.concatenate([[0.0], np.cumsum(d)])
        t /= t[-1]
        kind = "periodic" if closed else spline_bc_type
        return cls(CubicSpline(t, pts, bc_type=kind), bc=bc, prec=prec)

    def _rebuilt(self, p, dp, breaks=None, bc=None, prec=None):
        # A transformed spline is no longer a CubicSpline object, but it keeps its knots.
        return ParametricSegment(
            p, dp, self.t0, self.t1,
            self.bc if bc is None else bc,
            self.prec if prec is None else prec,
            self._breaks if breaks is None else breaks,
        )
