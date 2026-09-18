"""Candidate eigenvalue searches, behind the harness interface.

TEST SUPPORT ONLY. When one of these wins it moves into `chladni/mps/`, rewritten against
its own spec; until then they live here so they can be scored against each other without
committing the library to any of them.

Each is `search(tensions, a, b) -> [(x, mult), ...]`, and none of them sees the domain, the
basis or the pencil.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

#: One list, at the top. This file carried THREE `__all__` rebinds, one per section,
#: each silently shadowing the last -- so the name a reader saw depended on where they
#: stopped reading. `lipschitz_search` was in two of them and had no caller at all.
__all__ = ["lappy_search", "lappy_repaired_search", "swing_direct_search"]


def _local_minima(v):
    """Discrete local minima, ties broken left. See `harness._local_minima`."""
    return np.nonzero((v[1:-1] <= v[:-2]) & (v[1:-1] < v[2:]))[0] + 1


def _spacing(tensions, a, b):
    """The same crude spacing estimate the uniform-grid baseline uses, so that both searches
    start from the same information about the interval."""
    xs = np.linspace(a, b, 64)
    s1 = np.array([tensions(x)[0] for x in xs])
    lvl = 0.5 * np.median(s1)
    below = s1 < lvl
    n = max(int(np.sum(below[1:] & ~below[:-1])), 1)
    return (b - a) / n


def lappy_search(ppl=20, ttol=1e-3, ltol=1e-8, polish_frac=1e-2, max_recurse=8,
                 noise_factor=4.0, n_pts=None):
    """lappy's `solve_interval`, reimplemented as the baseline to beat.

    Faithful to the shape of the original rather than to its code. The parts that matter for
    how it succeeds and fails:

    * **A uniform grid and discrete local minima of sigma_1.** Anything narrower than the
      grid is invisible, which is the mechanism behind its known missed eigenvalues.
    * **sigma_2 guides the recursion.** Where sigma_2 has a discrete minimum that no sigma_1
      bracket covers, the region is re-gridded and re-searched -- lappy's way of splitting a
      pair too close for sigma_1 to separate. Only `tensions(x)[:2]` is ever used, though the
      backend computes the whole vector.
    * **Parabolic refinement of sigma_1 squared**, which is right: squaring removes the
      Lipschitz kink at the bottom of the cone.
    * **Multiplicity by a PREFIX rule against an absolute threshold** -- the longest run
      `1..kappa` for which every branch is both a local minimum and below `ttol`. lappy's own
      `docs/todo.md` calls this "still structurally fragile"; the measured reason it is
      fragile is that an absolute `ttol` cannot serve a floor `eps` that varies by fifty-fold
      within one scan.
    * **A noise-limited stop.** A subinterval whose peak-to-peak is within `noise_factor` of
      its own floor is declared noise and not refined further.
    """

    def search(tensions, a, b, *, modes=None):
        n = n_pts or max(int(ppl * (b - a) / _spacing(tensions, a, b)), 32)
        cache = {}

        def T(x):
            k = float(x)
            if k not in cache:
                cache[k] = tensions(k)
            return cache[k]

        brackets = []
        _scan(T, a, b, n, brackets, max_recurse, noise_factor, ttol)

        out = []
        for lo, hi in brackets:
            r = minimize_scalar(lambda x: T(x)[0] ** 2, bounds=(lo, hi), method="bounded",
                                options={"xatol": max(ltol * polish_frac, 1e-15)})
            x, f = r.x, np.sqrt(max(r.fun, 0.0))
            if f > ttol:
                continue                       # rejected: the dip is not deep enough
            out.append((x, lo, hi, f))

        out.sort()
        merged = []
        for x, lo, hi, f in out:
            if merged and abs(x - merged[-1][0]) <= ltol * max(abs(x), 1.0):
                if f < merged[-1][3]:
                    merged[-1] = (x, lo, hi, f)
                continue
            merged.append((x, lo, hi, f))

        return [(x, _prefix_multiplicity(T, x, lo, hi, ttol)) for x, lo, hi, _ in merged]

    return search


def _scan(T, a, b, n, brackets, depth, noise_factor, ttol):
    """One grid pass; recurse where sigma_2 says a bracket is hiding a second root."""
    xs = np.linspace(a, b, n)
    S = np.array([T(x)[:2] for x in xs])
    s1, s2 = S[:, 0], S[:, 1]

    if np.ptp(s1) <= noise_factor * s1.min():
        return                                  # noise-limited: nothing to resolve here

    i1 = _local_minima(s1)
    for i in i1:
        brackets.append((xs[i - 1], xs[i + 1]))

    if depth <= 0:
        return
    # sigma_2's minima mark crossings: a pair too close for sigma_1 to have split yet.
    for j in _local_minima(s2):
        if s2[j] > ttol:
            continue
        if any(xs[i - 1] <= xs[j] <= xs[i + 1] for i in i1) and len(i1) >= 2:
            continue
        lo, hi = xs[max(j - 1, 0)], xs[min(j + 1, len(xs) - 1)]
        if hi > lo:
            _scan(T, lo, hi, n, brackets, depth - 1, noise_factor, ttol)


def _prefix_multiplicity(T, x, lo, hi, ttol):
    """lappy's rule: the longest prefix of branches that are both small and locally minimal.

    Note what it does NOT do: it applies an ABSOLUTE threshold `ttol` to every branch, with
    no gap test. That is the part measured to fail once the separation drops below `eps/c`,
    where the second branch at one member of a pair is genuinely small.
    """
    t, tl, tr = T(x), T(lo), T(hi)
    n = min(len(t), len(tl), len(tr))
    t, tl, tr = t[:n], tl[:n], tr[:n]
    is_locmin = (t <= tl) & (t <= tr) & ((t != tl) | (t != tr))
    is_small = t <= ttol
    m = 0
    while m < n and is_locmin[m] and is_small[m]:
        m += 1
    return max(m, 1)


def _gap_multiplicity(T, x, lo, hi, tau, G):
    """(C1) largest ratio gap in the low spectrum, and (C2) each counted branch is a local
    minimum here. Both are needed; see the module docstring."""
    t, tl, tr = T(x), T(lo), T(hi)
    n = min(len(t), len(tl), len(tr), 8)
    t, tl, tr = t[:n], tl[:n], tr[:n]
    if n < 2:
        return 1
    ratios = t[1:] / np.maximum(t[:-1], 1e-300)
    kappa = int(np.argmax(ratios)) + 1
    if ratios[kappa - 1] < G or t[0] > tau:
        kappa = 1
    while kappa > 1 and not (t[kappa - 1] < tl[kappa - 1] and t[kappa - 1] < tr[kappa - 1]):
        kappa -= 1
    return max(kappa, 1)


def lappy_repaired_search(ppl=20, ttol=1e-3, ltol=1e-8, polish_frac=1e-2, n_sub=9,
                          max_depth=60, G=10.0, gap_multiplicity=True, n_pts=None,
                          floor_frac=1e-3):
    """lappy with its two measured defects repaired, and nothing else changed.

    The point of this search is to find out how much of the branch-and-bound's advantage is
    available without the branch and bound. Both repairs are small.

    **Defect 1 -- the recursion trigger.** lappy refines a bracket only where the sampled
    `sigma_2` is below `ttol`. That quantity is about `c*h/2`, a property of the GRID, so the
    rescue engages when the grid is already fine and not when the pair is tight. The repair
    is to drop the condition entirely: recurse into EVERY discrete minimum of `sigma_1`,
    unconditionally, until the sub-interval is narrower than `floor_frac` of its parent's
    parent or no new minimum appears. That is adaptive bisection, and it is where the
    `O(log(1/d))` scaling comes from -- effort concentrates near dips and nowhere else.

    There is no cleverer trigger available. Measured: the offset between `sigma_2`'s minimum
    and `sigma_1`'s becomes visible at exactly the coarseness where `sigma_1` first shows two
    minima, so nothing detects a pair earlier than `sigma_1` does. The only question is how
    cheaply you can get to that resolution, and the answer is "only refine where it is dark".

    **Defect 2 -- multiplicity.** The prefix rule tests each branch against an ABSOLUTE
    `ttol`. The floor varies by five orders within a single L-shape scan, so no absolute
    threshold works. Replaced by (C1) the largest ratio gap plus (C2) the local-minimum
    condition, both measured necessary. `gap_multiplicity=False` keeps lappy's original rule,
    so the two defects can be scored separately.
    """

    def search(tensions, a, b, *, modes=None):
        cache = {}

        def T(x):
            k = float(x)
            if k not in cache:
                cache[k] = tensions(k)
            return cache[k]

        n = n_pts or max(int(ppl * (b - a) / _spacing(T, a, b)), 32)
        _descend(T, a, b, n, n_sub, max_depth, floor_frac * (b - a))

        xs = np.array(sorted(cache))
        v = np.array([cache[x][0] for x in xs])
        out = []
        for i in _local_minima(v):
            lo, hi = xs[i - 1], xs[i + 1]
            r = minimize_scalar(lambda x: T(x)[0] ** 2, bounds=(lo, hi), method="bounded",
                                options={"xatol": max(ltol * polish_frac, 1e-15)})
            f = np.sqrt(max(r.fun, 0.0))
            if f > ttol:
                continue
            h = max(1e-9 * max(abs(r.x), 1.0), 1e-12)
            if not (T(r.x - h)[0] > f and T(r.x + h)[0] > f):
                continue                      # a wall point next to a deeper root
            out.append((r.x, f, lo, hi))

        out.sort()
        merged = []
        for x, f, lo, hi in out:
            if merged and abs(x - merged[-1][0]) <= ltol * max(abs(x), 1.0):
                if f < merged[-1][1]:
                    merged[-1] = (x, f, lo, hi)
                continue
            merged.append((x, f, lo, hi))

        res = []
        for x, f, lo, hi in merged:
            w = max(hi - lo, 1e-12)
            if gap_multiplicity:
                res.append((x, _gap_multiplicity(T, x, x - w, x + w, ttol, G)))
            else:
                res.append((x, _prefix_multiplicity(T, x, x - w, x + w, ttol)))
        return res

    return search


def _descend(T, a, b, n, n_sub, depth, floor):
    """Grid, then recurse into every sigma_1 minimum. No gate, no condition."""
    xs = np.linspace(a, b, n)
    v = np.array([T(x)[0] for x in xs])
    if depth <= 0 or (b - a) < floor:
        return
    for i in _local_minima(v):
        lo, hi = xs[i - 1], xs[i + 1]
        if hi - lo < floor:
            continue
        _descend(T, lo, hi, n_sub, n_sub, depth - 1, floor)


def swing_direct_search(per_gap=6.0, tau_frac=0.5, thresh=5.0, G=10.0, P=2.0,
                        prec=1e-8, direct_budget=14, max_cells=200):
    """The composed candidate: swing cells, DIRECT inside each, refine in a one-root bracket.

    Three measured results, in order (`notes/landscape_findings.md` sections 22 and 23):

    * the minimising eigenvector's IDENTITY changes at branch crossings, which lie BETWEEN
      eigenvalues, so peaks in the swing partition `[a, b]` into cells holding one eigenvalue
      each -- at 4 samples per gap on six of seven real domains, for 5-7% more cost per sample
      than a tension evaluation;
    * the swing splits a pair about 290x tighter than `sigma_1` does at the same stride, and
      reads ZERO peaks at an exact degeneracy, so a peak inside a dip means two DISTINCT
      eigenvalues;
    * DIRECT needs no Lipschitz constant, which matters because a pilot-calibrated `L` was
      measured UNSOUND -- it underestimated the true slope on GWW1.

    Falls back to a plain adaptive pass when the case supplies no `modes`, so it can still be
    scored against cases built for the older contract.
    """
    from scipy.optimize import minimize_scalar

    def _locate(modes, lo, hi, ulo, uhi, tol, depth=24):
        """Bisect toward the crossing inside `[lo, hi]`, one mode evaluation per level.

        A detected peak only localizes the crossing to +/- h/2, and when the pair separation
        is below `h` the cell boundary can land OUTSIDE the pair -- measured on
        rect(1.01, 1), where the crossing at 48.86 was reported at 46.83 and both members
        fell in one cell. Detection is not separation.

        The repair is that a peak is a feature to BISECT TOWARD, not a sample to accept. The
        crossing lies in whichever half carries the larger angle, halves are equal width so
        the comparison is fair, and the cost is `log2(h/d)` -- the same logarithmic behaviour
        adaptive subdivision has on the tension, obtained here from the eigenvector.
        """
        for _ in range(depth):
            if hi - lo <= tol:
                break
            m = 0.5 * (lo + hi)
            um = modes(m)
            left = 1.0 - abs(float(ulo @ um))
            right = 1.0 - abs(float(um @ uhi))
            if left >= right:
                hi, uhi = m, um
            else:
                lo, ulo = m, um
        return 0.5 * (lo + hi)

    def _cells(modes, a, b, n, tol):
        xs = np.linspace(a, b, n)
        U = np.array([modes(x) for x in xs])
        ang = 1.0 - np.abs(np.sum(U[:-1] * U[1:], axis=1))
        med = np.median(ang)
        if not np.isfinite(med) or med <= 0:
            return np.array([a, b])
        j = np.nonzero((ang[1:-1] > ang[:-2]) & (ang[1:-1] >= ang[2:]))[0] + 1
        j = j[ang[j] > thresh * med]
        cuts = [_locate(modes, xs[i], xs[i + 1], U[i], U[i + 1], tol) for i in j]
        return np.concatenate([[a], np.array(sorted(cuts)), [b]])

    def _direct(f, lo, hi, tau, budget):
        """Lipschitz-free subdivision: keep the lower-right hull of (width, value)."""
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
            new = []
            for i in sorted(keep, reverse=True)[:6]:
                p, q, _ = ivs.pop(i)
                m1, m2 = p + (q - p) / 3, q - (q - p) / 3
                new += [(p, m1, f(0.5 * (p + m1))), (m1, m2, f(0.5 * (m1 + m2))),
                        (m2, q, f(0.5 * (m2 + q)))]
                used += 3
            if not new:
                break
            ivs += new
        pts = np.array(sorted(0.5 * (v[0] + v[1]) for v in ivs))
        return pts, np.array([f(x) for x in pts])

    def search(tensions, a, b, *, modes=None):
        # One pilot pass serves BOTH the spacing estimate and the plateau. The plateau is a
        # median over an O(1)-varying curve and does not need cell-stride resolution, so
        # re-scanning the cell grid for it was pure waste -- it doubled the tension cost.
        xp = np.linspace(a, b, 48)
        sp = np.array([tensions(x)[0] for x in xp])
        tau = tau_frac * np.median(sp)
        lvl = 0.5 * np.median(sp)
        below = sp < lvl
        nwell = max(int(np.sum(below[1:] & ~below[:-1])), 1)
        gap = (b - a) / nwell
        n = int(np.clip(np.ceil((b - a) / (gap / per_gap)), 16, max_cells * 4)) + 1
        # Localize each crossing to the PRECISION CONTRACT, not to a fraction of the
        # estimated mean gap. `1e-3 * gap` was the floor here and it is the same defect as
        # section 25 at a smaller scale: with gap ~ 26 that floor is 2.6e-02, WIDER than a
        # pair at d = 5.9e-03, so the boundary can again land outside the pair. The tolerance
        # must not be tied to a quantity the pair is smaller than -- and `gap` is estimated
        # from a pilot that has already merged the pair, so it is exactly the wrong scale.
        tol = max(prec * abs(b), 1e-12)
        edges = (_cells(modes, a, b, n, tol) if modes is not None
                 else np.array([a, b]))

        f = lambda x: tensions(x)[0]
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi - lo <= 0:
                continue
            pts, vals = _direct(f, lo, hi, tau, direct_budget)
            i = int(np.argmin(vals))
            if vals[i] >= tau:
                continue
            bl = pts[i - 1] if i > 0 else lo
            br = pts[i + 1] if i + 1 < pts.size else hi
            r = minimize_scalar(lambda x: f(x)**2, bounds=(bl, br), method="bounded",
                                options={"xatol": max(prec * abs(pts[i]), 1e-13)})
            x, val = r.x, np.sqrt(max(r.fun, 0.0))
            if val >= tau or not (lo <= x <= hi):
                continue
            h = 0.25 * (hi - lo)
            out.append((x, _gap_multiplicity(tensions, x, x - h, x + h, tau, G)))
        out.sort()
        return out

    return search
