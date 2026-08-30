"""Adaptive arclength and polyline machinery.

Ported in approach from lappy, whose docstrings record the measurements that settled the
design; they are not re-derived here.

The arclength map is built as an adaptive TABLE used for bracketing plus a monotone cubic
initial guess, after which `t_of_s` *solves* s(t) = s by safeguarded Newton with
t'(s) = 1/|p'(t)|. lappy's earlier approach interpolated the inverse, which made p(tau) only
C^1 in tau: ellipse boundary integrals stalled at ~7e-5 regardless of quadrature order, and
|dp/dtau| - len sat at 8.6e-3. With the solve, constant speed holds by construction.

Subdivision here is level-synchronous and vectorized -- every interval at a given depth is
tested in one call -- rather than a per-interval Python recursion.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_TABLE_ORDER = 7        # cheap rule for the subdivision error estimate
_ANCHOR_ORDER = 24      # machine-precision arclength at the table nodes
_REMAINDER_ORDER = 12   # short span inside one table panel; insensitive to order
_NEWTON_ITERS = 4       # two reach 1.8e-16 on lappy's measurements; this is margin


@lru_cache(maxsize=64)
def gauss_legendre(order):
    """Gauss-Legendre nodes on [0, 1] with weights summing to 1."""
    x, w = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (x + 1.0), 0.5 * w


def panel_integrals(f, lo, hi, order):
    """Integrate a scalar f over each [lo[k], hi[k]] with one Gauss rule per panel."""
    x, w = gauss_legendre(order)
    t = lo[:, None] + (hi - lo)[:, None] * x[None, :]
    vals = np.asarray(f(t.ravel()), dtype=float).reshape(t.shape)
    return (hi - lo) * (vals @ w)


def _subdivide(test, t0, t1, max_depth, min_panels=16):
    """Bisect [t0, t1] until `test(lo, mid, hi)` accepts every panel. Returns sorted edges.

    Refinement starts from a uniform partition of `min_panels`, never from the whole
    interval. A midpoint-based error estimate can read exactly zero on a symmetric curve --
    the figure-eight p(t) = sin t + i sin t cos t returns to the origin at its own midpoint,
    so a single-panel start accepted the entire curve as one chord and reported it simple.
    A uniform floor costs a handful of evaluations and removes that whole class of aliasing.
    """
    grid = np.linspace(float(t0), float(t1), int(min_panels) + 1)
    lo = grid[:-1].copy()
    hi = grid[1:].copy()
    edges = grid.tolist()
    for _ in range(max_depth):
        if lo.size == 0:
            break
        mid = 0.5 * (lo + hi)
        bad = ~test(lo, mid, hi)
        if not bad.any():
            break
        edges.extend(mid[bad].tolist())
        lo = np.concatenate([lo[bad], mid[bad]])
        hi = np.concatenate([mid[bad], hi[bad]])
    return np.unique(np.asarray(edges, dtype=float))


def adaptive_arclength_table(speed, t0, t1, eps_rel, eps_abs, max_depth=40):
    """Table nodes whose panels each integrate |p'| to tolerance.

    The table supplies brackets and an initial guess only; its own arclengths are recomputed
    at higher order by the caller, so `eps_rel` sets the quality of a guess and nothing else.
    """

    def test(lo, mid, hi):
        whole = panel_integrals(speed, lo, hi, _TABLE_ORDER)
        halves = panel_integrals(speed, lo, mid, _TABLE_ORDER) + panel_integrals(
            speed, mid, hi, _TABLE_ORDER
        )
        err = np.abs(whole - halves)
        return err <= np.maximum(eps_abs, eps_rel * np.abs(halves))

    return _subdivide(test, t0, t1, max_depth)


def adaptive_polyline(p, eps_abs, t0=0.0, t1=1.0, max_depth=26):
    """Parameter values whose chords track the curve to within `eps_abs`.

    The error measured is the sagitta |p(mid) - (p(lo) + p(hi))/2|, which is exactly the
    deviation the chord makes from the curve at its worst point for a smooth arc.
    """

    def test(lo, mid, hi):
        dev = np.abs(p(mid) - 0.5 * (p(lo) + p(hi)))
        return dev <= eps_abs

    return _subdivide(test, t0, t1, max_depth)


def refine(fn, n0, prec, scale=None, max_iter=6):
    """Evaluate fn(n) at doubling n until the change falls under prec.

    Used for quantities defined by a boundary integral -- area, centroid -- so a polygon
    converges at the first step while a curved boundary pays only what it needs.

    `scale` is a floor for the convergence test, and it is not optional in practice: a
    purely relative criterion cannot converge on a quantity whose true value is zero. The
    disk's centroid integral is ~1e-13 of pure rounding noise, so successive values never
    agree to a relative 1e-8, and n doubled to 8192 -- where `leggauss` is O(n^2) and costs
    seconds per call. Pass the natural magnitude of the quantity instead.
    """
    n = int(n0)
    prev = fn(n)
    floor = 0.0 if scale is None else abs(scale)
    for _ in range(max_iter):
        n *= 2
        cur = fn(n)
        ref = max(abs(cur), abs(prev), floor)
        if ref == 0.0 or abs(cur - prev) <= prec * ref:
            return cur
        prev = cur
    return prev


def allocate(total, weights, min_each=2):
    """Split `total` nodes across panels proportionally to `weights`, by largest remainder.

    Every panel gets at least `min_each`: a one-point Gauss rule is not a useful default,
    and a segment with no nodes would silently vanish from the quadrature.
    """
    w = np.asarray(weights, dtype=float)
    k = len(w)
    if total is None:
        raise ValueError("a node count is required")
    total = int(total)
    if total < k * min_each:
        return np.full(k, max(1, total // k if total >= k else 1), dtype=int)
    base = np.full(k, min_each, dtype=int)
    remaining = total - base.sum()
    share = w / w.sum() * remaining
    counts = np.floor(share).astype(int)
    left = remaining - counts.sum()
    if left > 0:
        order = np.argsort(-(share - counts))
        counts[order[:left]] += 1
    return base + counts
