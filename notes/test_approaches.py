"""Tests for the shortlist. Each one falsifies something load-bearing; see `approaches.md`.

A2  Weyl-stratified sampling   -- does concentrating by density beat uniform?
A6  Lipschitz calibration      -- does a pilot's max secant slope bound the true slope?
A7  DIRECT, Lipschitz-free     -- does it match a calibrated envelope with no constant?

All three run off the cached fine scans, so they cost nothing.
"""
import sys
import warnings

sys.path.insert(0, "notes")
sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

import curve_lab as cl
import curve_study as cs
from chladni.asymp import weyl_count


def _truth(lbl):
    lams, S, area, gap = cs.load(lbl)
    s1 = S[:, 0]
    idx, plateau = cs.find_dips(lams, s1)
    return lams, s1, lams[idx], plateau, gap


def _recall(found, truth, tol):
    if truth.size == 0:
        return 1.0
    return sum(1 for t in truth
               if found.size and np.min(np.abs(found - t)) <= tol) / truth.size


def _extract(x, y, tau):
    i = np.nonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1
    return x[i[y[i] < tau]]


# --- A2: Weyl-stratified against uniform, at EQUAL point count ---------------------------

def a2_stratified():
    print("A2  Weyl-stratified vs uniform sampling, same number of points\n")
    print(f"{'domain':>9} {'min/mean gap':>13} {'n':>5} "
          f"{'uniform':>8} {'stratified':>11} {'delta':>7}")
    for lbl, expr, win, _ in cl.ROSTER:
        lams, s1, truth, plateau, gap = _truth(lbl)
        dom = cl.domain(expr)
        obs = np.diff(truth)
        ratio = obs.min() / gap if obs.size else np.nan
        a, b = lams[0], lams[-1]
        # stratify: equispaced in the Weyl count, mapped back to lam by interpolation
        grid = np.linspace(a, b, 4001)
        N = np.array([weyl_count(g, dom) for g in grid])
        N = np.maximum.accumulate(N)
        for n in (40, 60, 90, 130):
            xu = np.linspace(a, b, n)
            targets = np.linspace(N[0], N[-1], n)
            xs = np.interp(targets, N, grid)
            yu = np.interp(xu, lams, s1)
            ys = np.interp(xs, lams, s1)
            tol = 0.5 * (b - a) / n
            ru = _recall(_extract(xu, yu, 0.5 * plateau), truth, tol)
            rs = _recall(_extract(xs, ys, 0.5 * plateau), truth, tol)
            tag = f"{ratio:13.3f}" if n == 40 else " " * 13
            print(f"{lbl if n == 40 else '':>9} {tag} {n:5d} "
                  f"{ru:8.3f} {rs:11.3f} {rs - ru:+7.3f}")
        print()


# --- A6: is the pilot's max secant slope an upper bound on the true slope? ----------------

def a6_lipschitz():
    print("\nA6  does a pilot bound the true slope?  L_pilot = gamma * max|ds/dlam| "
          "over n_pilot points, gamma = 2\n")
    print(f"{'domain':>9} {'true max':>10} " + "".join(f"{f'n={n}':>12}"
                                                       for n in (20, 40, 80, 160)))
    for lbl, expr, win, _ in cl.ROSTER:
        lams, s1, truth, plateau, gap = _truth(lbl)
        true_max = np.max(np.abs(np.diff(s1) / np.diff(lams)))
        row = []
        for n in (20, 40, 80, 160):
            xp = np.linspace(lams[0], lams[-1], n)
            yp = np.interp(xp, lams, s1)
            Lp = 2.0 * np.max(np.abs(np.diff(yp) / np.diff(xp)))
            row.append(f"{Lp/true_max:12.3f}")
        print(f"{lbl:>9} {true_max:10.4f} " + "".join(row))
    print("\n   ratio < 1 means the pilot UNDERESTIMATES the true slope and the "
          "certificate is unsound.")


# --- A7: DIRECT-style, Lipschitz-free ----------------------------------------------------

def a7_direct(budget=200):
    """Keep every interval that is potentially-optimal for SOME Lipschitz constant, and
    additionally every interval whose midpoint is below tau. No constant to calibrate."""
    print(f"\n\nA7  DIRECT-style Lipschitz-free subdivision, budget {budget} evaluations\n")
    print(f"{'domain':>9} {'recall':>8} {'n_found':>8} {'evals':>7} "
          f"{'uniform recall @ same n':>24}")
    for lbl, expr, win, _ in cl.ROSTER:
        lams, s1, truth, plateau, gap = _truth(lbl)
        f = lambda x: np.interp(x, lams, s1)
        a, b = lams[0], lams[-1]
        tau = 0.5 * plateau
        ivs = [(a, b, f(0.5 * (a + b)))]
        used = 1
        while used < budget:
            # potentially optimal: on the lower-right convex hull of (width, value)
            w = np.array([iv[1] - iv[0] for iv in ivs])
            v = np.array([iv[2] for iv in ivs])
            keep = set()
            for wid in np.unique(w):
                m = np.nonzero(w == wid)[0]
                keep.add(int(m[np.argmin(v[m])]))
            # plus anything already below tau, so dips keep splitting
            keep |= {int(i) for i in np.nonzero(v < tau)[0]}
            order = sorted(keep, key=lambda i: (-w[i], v[i]))[:12]
            new = []
            for i in sorted(order, reverse=True):
                lo, hi, _ = ivs.pop(i)
                m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                new += [(lo, m1, f(0.5 * (lo + m1))),
                        (m1, m2, f(0.5 * (m1 + m2))),
                        (m2, hi, f(0.5 * (m2 + hi)))]
                used += 3
            ivs += new
            if not new:
                break
        pts = np.sort(np.array([0.5 * (iv[0] + iv[1]) for iv in ivs]))
        vals = f(pts)
        found = _extract(pts, vals, tau)
        tol = 0.5 * (b - a) / used
        r = _recall(found, truth, max(tol, 0.02 * gap))
        xu = np.linspace(a, b, used)
        ru = _recall(_extract(xu, f(xu), tau), truth, max(0.5 * (b - a) / used, 0.02 * gap))
        print(f"{lbl:>9} {r:8.3f} {found.size:8d} {used:7d} {ru:24.3f}")


if __name__ == "__main__":
    a2_stratified()
    a6_lipschitz()
    a7_direct()


# --- A14: is d(psi)/dk = (x . grad psi) / k, exactly? ------------------------------------
#
# Every particular solution here is psi(x) = Psi(k*x) about its own centre: a plane wave is
# exp(i*k*d.(x-x0)) and a corner family is J_nu(k*r)*e^{i*nu*t} with r measured from the
# corner. Euler's identity for a homogeneous argument then gives the claim exactly. If it
# holds, `dsigma/dlam` is available for one gradient evaluation instead of two extra tension
# evaluations, and approaches A10, A15 and A16 all open up.

def a14_derivative():
    from chladni.geometry import L_shape
    from chladni.helmholtz import FourierBessel, PlaneWave
    print("\n\nA14  d(psi)/dk  ==  (x . grad psi) / k ?   "
          "max relative error over columns and points\n")
    dom = L_shape()
    pts = dom.interior_pts(60, method="grid")
    k = 3.1
    h = 1e-6
    corner = [c for c in dom.corners if c.is_reentrant][0]
    cases = [("PlaneWave(origin)", PlaneWave.uniform(12, radius=dom.diameter), 0j),
             ("FB @ reentrant", FourierBessel.at_corner(corner, n=10, domain=dom), corner.z)]
    print(f"{'family':>20} {'centre':>14} {'max rel err':>13} {'FD noise floor':>15}")
    for name, b, ctr in cases:
        fd = (b.values(k + h, pts) - b.values(k - h, pts)) / (2 * h)
        g = b.grad(k, pts)
        x = pts - ctr
        # x . grad, with grad returned as a complex-plane vector per point
        xdg = (x.real[:, None] * g.real + x.imag[:, None] * g.imag) / k
        scale = np.maximum(np.abs(fd).max(axis=0), 1e-300)
        err = np.abs(fd - xdg).max(axis=0) / scale
        # what a central difference can deliver at all
        floor = (h**2) + np.finfo(float).eps / h
        print(f"{name:>20} {str(ctr):>14} {err.max():13.2e} {floor:15.2e}")


# --- A19: does the eigenvector swing sample more coarsely than the dip? -------------------
#
# Section 15 measured that the interior images of the minimising coefficient vector are
# nearly parallel within a well and nearly orthogonal across distinct eigenfunctions. If the
# ANGLE between c(lam) and c(lam+h) peaks sharply at an eigenvalue, it is an indicator that
# may survive a coarser stride than sigma_1's dip -- which is exactly section 20.2's problem.

def a19_swing(lbl="L_shape", nper=(4, 6, 8, 12, 16)):
    import curve_lab as cl
    lams, s1, truth, plateau, gap = _truth(lbl)
    dom = cl.domain(dict((r[0], r[1]) for r in [(x[0], x[1]) for x in cl.ROSTER])[lbl])
    a, b = lams[0], lams[-1]
    p = cl.make(dom, a, b)

    def img(x):
        _, I = p.pencil(x)
        u = I @ p.coef(x, 1)[:, 0]
        return u / np.linalg.norm(u)

    print(f"\n\nA19  eigenvector swing vs sigma_1 dip, {lbl}: recall at matched stride\n")
    print(f"{'per gap':>8} {'n':>5} {'dip recall':>11} {'swing recall':>13} "
          f"{'swing peak/median':>18}")
    for s in nper:
        n = int(np.ceil((b - a) / (gap / s))) + 1
        xs = np.linspace(a, b, n)
        ys = np.interp(xs, lams, s1)
        U = np.array([img(x) for x in xs])
        ang = 1.0 - np.abs(np.sum(U[:-1] * U[1:], axis=1))     # swing between neighbours
        mid = 0.5 * (xs[:-1] + xs[1:])
        tol = 1.5 * (gap / s)
        rd = _recall(_extract(xs, ys, 0.5 * plateau), truth, tol)
        # a swing peak: local max of the angle
        j = np.nonzero((ang[1:-1] > ang[:-2]) & (ang[1:-1] >= ang[2:]))[0] + 1
        j = j[ang[j] > 5 * np.median(ang)]
        rs = _recall(mid[j], truth, tol)
        print(f"{s:8d} {n:5d} {rd:11.3f} {rs:13.3f} "
              f"{(ang.max()/max(np.median(ang),1e-300)):18.1e}")


# --- A19b: the swing as a CROSSING detector, i.e. a complete cell decomposition -----------
#
# A19's original hypothesis was falsified: the swing does NOT peak at eigenvalues (recall
# 0.000 at fine stride). It peaks at BRANCH CROSSINGS, which lie BETWEEN eigenvalues -- and
# that is far more useful. A peak in every gap partitions [a, b] into cells holding exactly
# one eigenvalue each, which is a completeness certificate of a kind no sampling schedule
# gives, and section 20.2's stride problem restated in a form that may be much cheaper.

def a19b_cells(strides=(3, 4, 6, 8), thresh=5.0):
    import time
    import curve_lab as cl
    print("\n\nA19b  swing peaks as a CELL DECOMPOSITION: one peak per eigenvalue gap\n")
    print(f"{'domain':>9} {'per gap':>8} {'n':>5} {'gaps':>5} {'gaps hit':>9} "
          f"{'frac':>6} {'peaks':>6} {'extra':>6} {'ms/eval':>8}")
    for lbl, expr, win, _ in cl.ROSTER:
        lams, s1, truth, plateau, gap = _truth(lbl)
        dom = cl.domain(expr)
        a, b = lams[0], lams[-1]
        p = cl.make(dom, a, b)

        def img(x):
            _, I = p.pencil(x)
            u = I @ p.coef(x, 1)[:, 0]
            return u / np.linalg.norm(u)

        gaps = [(truth[i], truth[i + 1]) for i in range(len(truth) - 1)]
        for s in strides:
            n = int(np.ceil((b - a) / (gap / s))) + 1
            xs = np.linspace(a, b, n)
            t0 = time.time()
            U = np.array([img(x) for x in xs])
            ms = (time.time() - t0) / n * 1000
            ang = 1.0 - np.abs(np.sum(U[:-1] * U[1:], axis=1))
            mid = 0.5 * (xs[:-1] + xs[1:])
            j = np.nonzero((ang[1:-1] > ang[:-2]) & (ang[1:-1] >= ang[2:]))[0] + 1
            j = j[ang[j] > thresh * np.median(ang)]
            pk = mid[j]
            hit = sum(1 for lo, hi in gaps if np.any((pk > lo) & (pk < hi)))
            extra = len(pk) - hit
            print(f"{lbl if s == strides[0] else '':>9} {s:8d} {n:5d} {len(gaps):5d} "
                  f"{hit:9d} {hit/max(len(gaps),1):6.3f} {len(pk):6d} {extra:6d} {ms:8.1f}")
        print()
