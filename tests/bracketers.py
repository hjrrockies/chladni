"""The BRACKETING phase of each candidate search, exposed on its own.

TEST SUPPORT ONLY.

A bracketer is

    brackets(tensions, a, b, *, prec, modes=None) -> [(lo, hi), ...]

and it must COMMIT: the returned intervals are the ones it believes hold an eigenvalue, not a
subdivision of `[a, b]`. That distinction is what makes a partitioning method comparable to a
window-producing one. Measured, on `B:d=5.9e-03` over `[10, 88]` with 4 true eigenvalues: swing
cells tile the interval (total width 78.0 of 78.0) and so score coverage 4/4 for free, while
uniform windows cover 5.9 of 78.0 and have to earn it.

`prec` is the only thing that varies across a sweep. The pencil is FIXED at best resolution
(`rect_suite.pencil_for`, plane waves, `rtol = 1e-16`), so any difference in bracketing
behaviour across `prec` is attributable to the algorithm rather than to the objective.

`prec` tells a bracketer when to STOP splitting: two eigenvalues closer than `prec * lam` may
share a bracket, and nothing is gained by resolving below that. The stopping scale is known a
priori and needs no estimate of the pencil -- the pencil's own limit is `eps/c`, and the
measured headroom at `prec = 1e-14` on these cases is 2.8x to 7.0x.

These are extracted from `searches.py` rather than refactored into it: the five searches remain
exactly as scored elsewhere, and the helpers (`_local_minima`, `_spacing`, `_descend`, `_scan`,
`_cells`, `_direct`) are reused rather than reimplemented. Verified beforehand that none of the
five is entangled -- in every one the bracketing pass terminates before the first
`minimize_scalar` and no refined value re-enters subdivision.
"""
from __future__ import annotations

import numpy as np

import searches as _s
from harness import _local_minima, _spacing

__all__ = ["uniform_grid_brackets", "lappy_brackets", "lappy_repaired_brackets",
           "lipschitz_brackets", "swing_brackets", "single_bracket", "fine_grid",
           "BRACKETERS", "CONTROLS"]


def _cached(tensions):
    """A memoizing wrapper plus the sorted sample cache, which is where several of the
    methods keep their brackets rather than returning them."""
    cache = {}

    def T(x):
        k = float(x)
        if k not in cache:
            cache[k] = tensions(k)
        return cache[k]

    return T, cache


def _from_cache(cache, tau=None):
    xs = np.array(sorted(cache))
    v = np.array([cache[x][0] for x in xs])
    idx = _local_minima(v)
    if tau is not None:
        idx = idx[v[idx] < tau]
    return [(float(xs[i - 1]), float(xs[i + 1])) for i in idx]


def _plateau(T, a, b, n=48):
    xs = np.linspace(a, b, n)
    return float(np.median([T(x)[0] for x in xs]))


def _direct(f, lo, hi, tau, budget):
    """Lipschitz-FREE subdivision: keep the lower-right hull of (width, value), plus anything
    already below `tau`. Reimplemented here because `searches.py`'s copy is a closure inside
    `swing_direct_search`. No constant to calibrate, which matters because section 22 measured
    the Lipschitz calibration unsound."""
    ivs = [(lo, hi, f(0.5 * (lo + hi)))]
    used = 1
    while used < budget:
        w = np.array([v[1] - v[0] for v in ivs])
        val = np.array([v[2] for v in ivs])
        keep = set()
        for wid in np.unique(w):
            m = np.nonzero(w == wid)[0]
            keep.add(int(m[np.argmin(val[m])]))
        keep |= {int(i) for i in np.nonzero(val < tau)[0]}
        new_iv = []
        for i in sorted(keep, reverse=True)[:6]:
            q0, q1, _ = ivs.pop(i)
            m1, m2 = q0 + (q1 - q0) / 3, q1 - (q1 - q0) / 3
            new_iv += [(q0, m1, f(0.5 * (q0 + m1))), (m1, m2, f(0.5 * (m1 + m2))),
                       (m2, q1, f(0.5 * (m2 + q1)))]
            used += 3
        if not new_iv:
            break
        ivs += new_iv
    pts = np.array(sorted(0.5 * (v[0] + v[1]) for v in ivs))
    return pts, np.array([f(x) for x in pts])


# --- 1. uniform grid ----------------------------------------------------------------------

def uniform_grid_brackets(n_per_well=8, tau_frac=0.5, n_max=20000):
    """Sample once, take discrete minima, bracket with the neighbouring samples.

    Deliberately NOT prec-aware beyond a cap: a uniform grid cannot reach a spacing of
    `prec * lam` at any usable cost, so its brackets are grid-spacing wide whatever `prec`
    asks for. That is the honest behaviour of the method and the results should show it.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        T, cache = _cached(tensions)
        n = max(int(n_per_well * (b - a) / _spacing(T, a, b)), 16)
        n = min(n, n_max)
        xs = np.linspace(a, b, n)
        v = np.array([T(x)[0] for x in xs])
        tau = tau_frac * float(np.median(v))
        idx = _local_minima(v)
        idx = idx[v[idx] < tau]
        return [(float(xs[i - 1]), float(xs[i + 1])) for i in idx]

    return brackets


# --- 2. lappy, original -------------------------------------------------------------------

def lappy_brackets(ppl=20, ttol=1e-3, max_recurse=8, noise_factor=8.0):
    """`_scan` already fills an explicit `brackets` list; this is a thin wrapper.

    Its recursion is gated on `sigma_2`'s VALUE against `ttol`, which section 11 measured to
    be a property of the grid rather than of the pair -- a grid of spacing `h` sees about
    `c*h/2`, so the gate engages when `h <= 2*ttol/c` whatever the separation.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        T, _ = _cached(tensions)
        n = max(int(ppl * (b - a) / _spacing(T, a, b)), 32)
        out = []
        _s._scan(T, a, b, n, out, max_recurse, noise_factor, ttol)
        return [(float(lo), float(hi)) for lo, hi in out]

    return brackets


# --- 3. lappy, repaired -------------------------------------------------------------------

def lappy_repaired_brackets(ppl=20, n_sub=9, max_depth=6, tau_frac=0.5):
    """Recurse into EVERY discrete minimum, unconditionally, to a floor of `prec * lam`.

    The repair of section 13: the original's `sigma_2` gate is replaced by unconditional
    recursion. The floor here is `prec * b` rather than `floor_frac * (b - a)` -- section 26
    found `floor_frac` load-bearing and unexplained, losing a different eigenvalue at each
    value, and `prec * lam` is the scale the precision contract actually names.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        T, cache = _cached(tensions)
        n = max(int(ppl * (b - a) / _spacing(T, a, b)), 32)
        _s._descend(T, a, b, n, n_sub, max_depth, max(prec * abs(b), 1e-13))
        return _from_cache(cache, tau_frac * _plateau(T, a, b))

    return brackets


# --- 4. Lipschitz branch and bound --------------------------------------------------------

def lipschitz_brackets(gamma=2.0, tau_frac=0.5, alpha=0.25, n_pilot=48, max_evals=4000):
    """Piyavskii-Shubert envelope, then brackets read off the SAMPLE CACHE.

    Not off the candidate intervals: `searches.py:249` records that Phase B was rewritten to
    "refine from the SAMPLES, not the candidate brackets", leaving `candidates` dead at
    searches.py:201/220/225. Returning that list would be returning the abandoned design.

    Section 22 measured the `L` calibration UNSOUND -- on GWW1 a 20-point pilot with
    `gamma = 2` underestimated the true slope, ratio 0.672 -- so the envelope's completeness
    claim can be false. It is included to measure that, not because it is expected to win.
    """
    import heapq

    def brackets(tensions, a, b, *, prec, modes=None):
        T, cache = _cached(tensions)
        xp = np.linspace(a, b, n_pilot)
        vp = np.array([T(x)[0] for x in xp])
        S = float(np.median(vp))
        L = gamma * float(np.max(np.abs(np.diff(vp) / np.diff(xp))))
        tau = tau_frac * S
        floor = max(prec * abs(b), 1e-13)
        heap = [(-(b - a), a, b)]
        heapq.heapify(heap)
        while heap and len(cache) < max_evals:
            _, lo, hi = heapq.heappop(heap)
            h = hi - lo
            v0, v1 = T(lo)[0], T(hi)[0]
            s_hat = 0.5 * (v0 + v1 - L * h)
            if s_hat > tau:
                continue                      # certified free, given L
            if h < max(alpha * S / L, floor):
                continue
            m = 0.5 * (lo + hi)
            T(m)
            heapq.heappush(heap, (-(m - lo), lo, m))
            heapq.heappush(heap, (-(hi - m), m, hi))
        return _from_cache(cache, tau)

    return brackets


# --- 5. swing cells, optionally narrowed by DIRECT ----------------------------------------

def swing_brackets(per_gap=6.0, thresh=5.0, tau_frac=0.5, narrow=True,
                   direct_budget=14, commit_probe=9):
    """Cells from the eigenvector swing, each crossing bisected, then committed against `tau`.

    Two levels, and both are reported: `narrow=False` returns the cells themselves,
    `narrow=True` runs DIRECT inside each and returns the sample-neighbour bracket. The
    `vals >= tau` drop is the COMMITMENT step -- without it this is a subdivision of `[a, b]`
    and coverage is free.

    The crossing is BISECTED, not taken at the peak sample: section 25 measured a peak
    reported at 46.83 for a crossing truly at 48.86, which put both members of a pair in one
    cell. Detection brackets the crossing to a sampling interval; bisection separates it, at
    `log2(h/d)` mode evaluations and independent of how coarse the stride was.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        T, cache = _cached(tensions)
        if modes is None:
            return uniform_grid_brackets()(tensions, a, b, prec=prec)
        tol = max(prec * abs(b), 1e-13)
        xp = np.linspace(a, b, 48)
        S = float(np.median([T(x)[0] for x in xp]))
        tau = tau_frac * S
        gap = (b - a) / max(len(_local_minima(np.array([T(x)[0] for x in xp]))), 1)
        n = int(np.clip(np.ceil((b - a) / (gap / per_gap)), 16, 4000)) + 1
        xs = np.linspace(a, b, n)
        U = np.array([modes(x) for x in xs])
        ang = 1.0 - np.abs(np.sum(U[:-1] * U[1:], axis=1))
        med = float(np.median(ang))
        cuts = []
        if np.isfinite(med) and med > 0:
            j = np.nonzero((ang[1:-1] > ang[:-2]) & (ang[1:-1] >= ang[2:]))[0] + 1
            for i in j[ang[j] > thresh * med]:
                lo, hi, ul, uh = xs[i], xs[i + 1], U[i], U[i + 1]
                for _ in range(60):
                    if hi - lo <= tol:
                        break
                    m = 0.5 * (lo + hi)
                    um = modes(m)
                    if 1.0 - abs(float(ul @ um)) >= 1.0 - abs(float(um @ uh)):
                        hi, uh = m, um
                    else:
                        lo, ul = m, um
                cuts.append(0.5 * (lo + hi))
        edges = np.concatenate([[a], np.array(sorted(cuts)), [b]])
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            if not narrow:
                # COMMITMENT: a cell is claimed only if it looks like it holds a root. Without
                # this the cells tile [a, b] and coverage is free -- see the module docstring.
                probe = np.linspace(lo, hi, max(commit_probe, 3))
                if min(T(x)[0] for x in probe) < tau:
                    out.append((float(lo), float(hi)))
                continue
            pts, vals = _direct(lambda x: T(x)[0], lo, hi, tau, direct_budget)
            i = int(np.argmin(vals))
            if vals[i] >= tau:
                continue
            bl = pts[i - 1] if i > 0 else lo
            br = pts[i + 1] if i + 1 < pts.size else hi
            out.append((float(bl), float(br)))
        return out

    return brackets


# --- adversarial controls -----------------------------------------------------------------
#
# Without these the metrics are unvalidated. `single_bracket` scores coverage 1.0 and spurious
# 0 while being useless, so purity must be what rejects it; `fine_grid` has perfect purity and
# absurd cost, so spurious and cost must be what reject it. If either scores well the metric is
# broken rather than the control.

def single_bracket():
    """`[(a, b)]`. Must fail PURITY on every case with more than one level."""
    def brackets(tensions, a, b, *, prec, modes=None):
        tensions(0.5 * (a + b))
        return [(float(a), float(b))]

    return brackets


def fine_grid(n=4000):
    """`n` equal slivers, committed only where the tension is below `tau`. Must fail on cost,
    and on spurious unless the commitment is doing real work."""
    def brackets(tensions, a, b, *, prec, modes=None):
        xs = np.linspace(a, b, n + 1)
        v = np.array([tensions(0.5 * (xs[i] + xs[i + 1]))[0] for i in range(n)])
        tau = 0.5 * float(np.median(v))
        return [(float(xs[i]), float(xs[i + 1])) for i in np.nonzero(v < tau)[0]]

    return brackets


BRACKETERS = {
    "uniform_grid":   uniform_grid_brackets,
    "lappy":          lappy_brackets,
    "lappy_repaired": lappy_repaired_brackets,
    "lipschitz":      lipschitz_brackets,
    "swing_cells":    lambda **kw: swing_brackets(narrow=False, **kw),
    "swing_direct":   lambda **kw: swing_brackets(narrow=True, **kw),
}

CONTROLS = {"single_bracket": single_bracket, "fine_grid": fine_grid}
