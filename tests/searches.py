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

__all__ = ["lappy_search"]


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

    def search(tensions, a, b):
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


# =========================================================================================
# The Lipschitz cone method, per docs/design/lipschitz-cone-mps.md
# =========================================================================================

import heapq


def lipschitz_search(gamma=2.0, tau_accept=1e-3, tau_scan=None, alpha=0.25, G=10.0,
                     n_pilot=30, ltol=1e-8, tol_merge=1e-6, max_evals=200000):
    """Piyavskii-Shubert branch and bound, then Brent, then multiplicity from the cache.

    The four phases of `lipschitz-cone-mps.md`, and the three places this project has
    already measured something that bears on them:

    * **Phase A** prunes an interval when the lower envelope
      `s_hat = (s_i + s_j - L*h)/2` exceeds `tau_scan`, and keeps every pruned interval in
      an ATTIC so that raising `L` can revive it. Pruning is the only irreversible act.
    * **The isolation criterion uses section 3.4's OFFSET detector**, not sigma_2's value.
      That distinction is the whole reason this method might beat the baseline: scoring
      lappy showed its sigma_2 gate tests the sampled MINIMUM VALUE, which on a grid of
      spacing h is about `c*h/2` -- a property of the grid, not of the pair. A minimum's
      LOCATION survives coarse sampling where its depth does not, and section 9.2 of the
      lab notes confirms the offset tracks `d/2` on real pencils to about 1%.
    * **Phase C** uses both conditions measured to be necessary: (C1) the largest ratio gap,
      and (C2) that each counted branch is a local minimum at the point. (C2) is
      load-bearing rather than belt-and-braces -- once `d <~ eps/c` the gap rule alone
      reports multiplicity 2 at a SIMPLE eigenvalue and double-counts.

    Deliberately NOT implemented: the two-level basis of section 8, which needs the pencil
    rather than just the tension function, and the Weyl cross-check of the same section,
    which belongs to the harness because it audits any search rather than this one.
    """
    tau_scan = tau_scan if tau_scan is not None else 10.0 * tau_accept

    def search(tensions, a, b):
        cache = {}

        def T(x):
            k = float(x)
            if k not in cache:
                if len(cache) >= max_evals:
                    raise RuntimeError(f"evaluation cap {max_evals} reached")
                cache[k] = tensions(k)
            return cache[k]

        def s1(x):
            return float(T(x)[0])

        # --- pilot: calibrate L and the plateau -----------------------------------------
        xs = list(np.linspace(a, b, n_pilot))
        vals = [s1(x) for x in xs]
        S = float(np.median(vals))
        slopes = [abs(vals[i + 1] - vals[i]) / (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
        L = gamma * max(max(slopes), 1e-12)

        def envelope(x0, x1, v0, v1):
            h = x1 - x0
            s_hat = 0.5 * (v0 + v1 - L * h)
            x_hat = 0.5 * (x0 + x1) + (v0 - v1) / (2.0 * L)
            return s_hat, min(max(x_hat, x0 + 1e-15 * h), x1 - 1e-15 * h)

        diag = {"L_pilot": L, "max_observed_slope": max(slopes), "attic_revivals": 0,
                "plateau_S": S}
        heap, attic, candidates = [], [], []
        for i in range(len(xs) - 1):
            sh, _ = envelope(xs[i], xs[i + 1], vals[i], vals[i + 1])
            heapq.heappush(heap, (sh, xs[i], xs[i + 1], vals[i], vals[i + 1]))

        while heap:
            sh, x0, x1, v0, v1 = heapq.heappop(heap)
            if sh > tau_scan:
                attic.append((x0, x1, v0, v1))
                continue
            h = x1 - x0
            if h < alpha * S / L:
                # Section 3.4 in full: the offset between sigma_2's minimum and sigma_1's
                # ESTIMATES d/2, and that estimate sets how much further to subdivide. An
                # earlier version used the signal only as a boolean veto, which is half the
                # idea and does not work: the isolation width alpha*S/L is 3.75e-02 on these
                # cases, a fixed scale with no relation to the separation, so a 4e-03 pair
                # was declared isolated and returned as one root.
                d_est = _pair_separation(T, x0, x1, tau_scan)
                if d_est is None or h < 0.25 * d_est:
                    candidates.append((x0, x1))
                    continue
            if h < 1e-13 * max(abs(x0), 1.0):
                candidates.append((x0, x1))
                continue
            _, xm = envelope(x0, x1, v0, v1)
            vm = s1(xm)
            grew = False
            for (p, q, vp, vq) in ((x0, xm, v0, vm), (xm, x1, vm, v1)):
                sl = abs(vq - vp) / max(q - p, 1e-300)
                diag["max_observed_slope"] = max(diag["max_observed_slope"], sl)
                if sl > L / gamma:
                    L = gamma * sl
                    grew = True
                sh2, _ = envelope(p, q, vp, vq)
                heapq.heappush(heap, (sh2, p, q, vp, vq))
            if grew and attic:
                revived, keep = [], []
                for (p, q, vp, vq) in attic:
                    sh2, _ = envelope(p, q, vp, vq)
                    (revived if sh2 <= tau_scan else keep).append((p, q, vp, vq))
                attic = keep
                diag["attic_revivals"] += len(revived)
                for (p, q, vp, vq) in revived:
                    sh2, _ = envelope(p, q, vp, vq)
                    heapq.heappush(heap, (sh2, p, q, vp, vq))

        # --- Phase B: refine ------------------------------------------------------------
        # Refine from the SAMPLES, not from the candidate brackets.
        #
        # Piyavskii-Shubert evaluates at the envelope's minimiser, which for a cone is its
        # APEX -- so the branch and bound lands on a root and then splits there, leaving it
        # on the shared endpoint of two brackets and in the interior of neither. Three
        # attempts at repairing that after the fact each fixed some cases and broke others:
        # unioning every touching pair swallowed resolved pairs (a 4e-3 pair went 1.000 ->
        # 0.500), and unioning only where a refinement "landed on an edge" missed, because
        # Brent's bounded method halts about 1e-7 from the boundary rather than at it.
        #
        # The structure was wrong. The sample set already contains the answer: the apex is a
        # local minimum AMONG THE SAMPLES, with neighbours on both sides. So sort the cache,
        # take discrete local minima of it, and bracket each with its neighbours. That uses
        # every evaluation the scan paid for and needs no bracket bookkeeping at all.
        xs_all = np.array(sorted(cache))
        v_all = np.array([cache[x][0] for x in xs_all])
        idx = _local_minima(v_all)
        xtol = max(ltol * 1e-2, 1e-15)

        found = []
        for i in idx:
            lo, hi = xs_all[i - 1], xs_all[i + 1]
            r = minimize_scalar(lambda x: s1(x) ** 2, bounds=(lo, hi), method="bounded",
                                options={"xatol": xtol})
            f = np.sqrt(max(r.fun, 0.0))
            if f > tau_accept:
                continue
            if not _is_interior_minimum(s1, r.x, lo, hi, f):
                continue
            found.append((r.x, f, lo, hi))

        # --- Phase D (collapse) then C (multiplicity) -----------------------------------
        found.sort()
        collapsed = []
        for x, f, x0, x1 in found:
            if collapsed and abs(x - collapsed[-1][0]) <= tol_merge * max(abs(x), 1.0):
                if f < collapsed[-1][1]:
                    collapsed[-1] = (x, f, x0, x1)
                continue
            collapsed.append((x, f, x0, x1))

        out = []
        for x, f, x0, x1 in collapsed:
            w = max(abs(x1 - x0), 1e-12)
            out.append((x, _gap_multiplicity(T, x, x - w, x + w, tau_accept, G)))

        # Section 9: the completeness claim is CONDITIONAL on L, so report what it rested
        # on. A large margin between L and the largest observed slope means the scan was
        # conservative; a small one means the constant was being pushed.
        diag["L_final"] = L
        diag["margin"] = L / max(diag["max_observed_slope"], 1e-300)
        diag["evaluations"] = len(cache)
        search.diagnostics = diag
        return out

    return search


def _is_interior_minimum(s1, x, x0, x1, f):
    """Reject a refinement that stopped on the WALL of a neighbouring root.

    `minimize_scalar` with `bounds` returns the best point IN the interval, which is an
    endpoint whenever the true minimum lies outside it. Next to a deep root that endpoint is
    still far below `tau_accept`, so the threshold test passes and a phantom root is
    reported a few times 1e-4 away from a real one. Measured before this check existed: 8
    phantoms against 20 true roots, every one of them 2e-5 to 4e-4 from a genuine root with
    sigma_1 between 1.6e-04 and 9.8e-04.

    The fix is condition (C2) applied to the root itself: a minimum has to be lower than its
    own neighbourhood, not merely small.
    """
    h = max(1e-9 * max(abs(x), 1.0), 1e-12)
    if not (x0 + 1e-12 < x < x1 - 1e-12):
        # landed on a bracket edge: only keep it if it still beats both sides
        pass
    return s1(x - h) > f and s1(x + h) > f


def _pair_separation(T, x0, x1, tau):
    """Section 3.4: estimate `d` for a pair inside this interval, or None if there is one.

    sigma_1 has two minima with a local maximum between them; sigma_2 has ONE, at the
    crossing. So `2 * |argmin sigma_2 - argmin sigma_1|` estimates `d`.

    The reason to use the offset rather than sigma_2's value -- which is what lappy's gate
    tests -- is that a minimum's LOCATION survives coarse sampling while its DEPTH does not.
    On a grid of spacing h, sigma_2's sampled minimum is about `c*h/2` however tight the
    pair is, so a value test measures the grid; the location is still good to about h.
    Section 9.2 of the lab notes confirms the offset tracks d/2 on real pencils to ~1%.
    """
    xs = np.linspace(x0, x1, 9)
    S = np.array([T(x)[:2] for x in xs])
    if S[:, 1].min() > tau:
        return None
    off = abs(xs[int(np.argmin(S[:, 0]))] - xs[int(np.argmin(S[:, 1]))])
    return 2.0 * off if off > 0 else None


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


__all__ = ["lappy_search", "lipschitz_search"]


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

    def search(tensions, a, b):
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


__all__ = ["lappy_search", "lipschitz_search", "lappy_repaired_search"]
