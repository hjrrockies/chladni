"""Approach test: a Weyl-anchored scan that corrects its OWN stride, locally.

Section 20.2 measured the defect this is aimed at: six samples per gap loses eigenvalues on
four of seven real domains, full recall needs about 18 per MEDIAN gap, and the governing
quantity is the MINIMUM LOCAL gap, which runs 0.13 to 0.79 of Weyl's mean. So no a-priori
stride is safe, and the correction has to happen during the solve.

The idea under test: the Weyl deficit is currently a scalar reported at the end
("worst over the gaps"). Make it a PROFILE over cuts, so it says not only that a level is
missing but WHERE, and re-scan only that interval. Cost then goes as the number of tight spots
rather than as the whole range.

The decisive question is not whether it converges. It is whether it can tell
"refine harder" from "the basis is inadequate" -- because those two look identical from the
deficit alone, and only one of them is fixable by more evaluations. `H_shape` (a genuine miss,
deficit 1.18) and `rect81` (floor 4.9e-02 at some levels, deficit 2.55) are the two cases.

Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "notes")
sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import minimize_scalar

import curve_lab as cl
from chladni.asymp import weyl_count


def weyl_profile(levels, mults, dom, a, b):
    """Deficit AT each cut, not just the worst. Cuts sit midway between distinct levels.

    Returns `(cuts, deficit)`. A jump of about 1 between consecutive cuts localizes a missing
    eigenvalue to the interval between the levels those cuts straddle.
    """
    lv = np.asarray(levels, dtype=float)
    if lv.size < 2:
        return np.array([]), np.array([])
    order = np.argsort(lv)
    lv, mu = lv[order], np.asarray(mults)[order]
    cuts = 0.5 * (lv[:-1] + lv[1:])
    below = np.cumsum(mu)[:-1]
    expect = np.array([weyl_count(c, dom) for c in cuts])
    # NO anchor at `a`. The three-term count is an ASYMPTOTIC and is meaningless at the low
    # end -- weyl_count(a) measured -0.69 on H_shape, -0.75 on rect81, -1.04 on chevron.
    # Subtracting it adds that much to every deficit, which is a phantom the loop then chases:
    # H_shape burned 958 evaluations converging to a deficit that was really -0.17.
    # The count is absolute, so this requires the found set to be a COMPLETE PREFIX from the
    # bottom of the spectrum, i.e. `a` below lam_1. That is the same precondition
    # `asymp.weyl_deficit` carries, and it should be asserted rather than papered over.
    return cuts, expect - below


def local_minima(y):
    return np.nonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1


class Budget:
    def __init__(self):
        self.n = 0

    def __call__(self, p, x):
        self.n += 1
        return p.tensions(x)


def multiplicity(p, lam, plateau, G=10.0, P=2.0, h_frac=0.02, nb=6, budget=None):
    ev = budget if budget is not None else (lambda p, x: p.tensions(x))
    s = ev(p, lam)[:nb]
    s = s[np.isfinite(s)]
    if s.size < 2:
        return 1
    r = s[1:] / np.maximum(s[:-1], 1e-300)
    j = int(np.argmax(r))
    k1 = j + 1 if r[j] >= G else s.size
    h = h_frac * abs(lam)
    lo, hi = ev(p, lam - h), ev(p, lam + h)
    k = 0
    for i in range(min(k1, len(lo), len(hi))):
        if lo[i] > P * s[i] and hi[i] > P * s[i]:
            k += 1
        else:
            break
    return max(k, 1)


def sweep_interval(p, lo, hi, n, budget):
    xs = np.linspace(lo, hi, n)
    return xs, np.array([budget(p, x)[0] for x in xs])


def adaptive(dom, p, a, b, *, prec=1e-8, s0=8, tau_frac=0.5, max_rounds=6, verbose=True):
    """Weyl-anchored scan, then locally re-scan wherever the deficit profile says a level is
    missing. Stops on a clean audit, on the round budget, or when a re-scan buys nothing --
    which is the basis-inadequate signal."""
    budget = Budget()
    gap = 4 * np.pi / dom.area
    h = gap / s0
    n = int(np.ceil((b - a) / h)) + 1
    xs, s1 = sweep_interval(p, a, b, n, budget)
    plateau = np.median(s1)
    tau = tau_frac * plateau
    intervals = [(a, b)]
    cacheX, cacheY = list(xs), list(s1)
    history = []

    for rnd in range(max_rounds):
        X = np.array(cacheX)
        o = np.argsort(X)
        X, Y = X[o], np.array(cacheY)[o]
        idx = local_minima(Y)
        idx = idx[Y[idx] < tau]
        levels, mults = [], []
        for i in idx:
            r = minimize_scalar(lambda z: budget(p, z)[0]**2,
                                bounds=(X[i - 1], X[i + 1]), method="bounded",
                                options={"xatol": max(prec * X[i], 1e-13)})
            levels.append(r.x)
            mults.append(multiplicity(p, r.x, plateau, budget=budget))
        cuts, prof = weyl_profile(levels, mults, dom, a, b)
        worst = float(np.max(prof)) if prof.size else np.nan
        history.append((rnd, len(levels), int(np.sum(mults)), worst, budget.n))
        if verbose:
            print(f"      round {rnd}: {len(levels):3d} levels, mult sum "
                  f"{int(np.sum(mults)):3d}, worst deficit {worst:6.2f}, "
                  f"{budget.n:5d} evals")
        if not np.isfinite(worst) or worst < 0.5:
            break
        # localize: every cut whose deficit exceeds the previous one by >= 0.5
        prev = np.concatenate([[0.0], prof[:-1]])
        hot = np.nonzero(prof - prev >= 0.5)[0]
        if hot.size == 0:
            hot = np.array([int(np.argmax(prof))])
        lv = np.sort(np.asarray(levels))
        added = 0
        for c in hot:
            lo = lv[c] if c < lv.size else lv[-1]
            hi = lv[c + 1] if c + 1 < lv.size else b
            m = max(int(np.ceil((hi - lo) / (h / 2**(rnd + 1)))) + 1, 5)
            nx, ny = sweep_interval(p, lo, hi, m, budget)
            cacheX.extend(nx)
            cacheY.extend(ny)
            added += m
        if added == 0:
            break
    return dict(levels=np.array(levels), mults=np.array(mults), evals=budget.n,
                deficit=worst, history=history, plateau=plateau)
