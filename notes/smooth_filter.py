"""Smooth spectral filtering as an alternative to hard rank truncation. EXPERIMENT ONLY.

Nothing here touches src/. Everything is a free function over raw arrays, built on top of
the public `chladni.mps` seam (`assemble`, `equilibrate`) plus gsvd4py directly.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
from gsvd4py import gsvdvals

EPS = np.finfo(float).eps


def stack_svd(B, I):
    """U (stacked), s, Vt of the equilibrated stacked pencil."""
    A = np.vstack([B, I])
    U, s, Vt = la.svd(A, full_matrices=False)
    return U, s, Vt


def tension_from(Br, Ir):
    c, s = gsvdvals(Br, Ir)
    sigma = np.divide(c, s, out=np.full(c.shape, np.inf), where=(s != 0))
    return np.sort(sigma)


# --- the three policies ---------------------------------------------------------------------

def tension_trunc(B, I, rank=None, rtol=1.5e-15):
    """Hard truncation, the current `reg='svd'` path (threshold if rank is None)."""
    m = B.shape[0]
    U, s, Vt = stack_svd(B, I)
    r = int((s >= rtol * s[0]).sum()) if rank is None else int(rank)
    Ur = U[:, :r]
    return tension_from(Ur[:m], Ur[m:]), r


def tension_weighted(B, I, w):
    """A COMMON right-diagonal weighting of both blocks. Included to show it is a no-op."""
    m = B.shape[0]
    U, s, Vt = stack_svd(B, I)
    Uw = U * w
    return tension_from(Uw[:m], Uw[m:])


def tension_filtered(B, I, g, null_cut=EPS):
    """Smooth filter as a PENALTY: min_c sqrt(||Bc||^2 + ||Gamma c||^2)/||Ic||.

    In whitened coordinates y = diag(s) V^T c the pencil is ([U_B; diag(g)], U_I), where
    `g_j` is the penalty weight on whitened direction j. g=0 keeps a direction untouched,
    g=inf forbids it (= hard truncation), and anything in between is a smooth filter.

    `g` is an array of length len(s) or a callable g(s, s0).
    Directions below `null_cut * s[0]` are dropped outright -- they are exact noise and the
    penalty on them would be astronomically large anyway.
    """
    m = B.shape[0]
    U, s, Vt = stack_svd(B, I)
    keep = s >= null_cut * s[0]
    U, s = U[:, keep], s[keep]
    gv = np.asarray(g(s, s[0]), dtype=float)
    Baug = np.vstack([U[:m], np.diag(gv)])
    return tension_from(Baug, U[m:]), int(keep.sum())


def tension_tikhonov(B, I, mu):
    """The same thing with Gamma = mu*I, straight from the equilibrated pencil, no SVD.

    Identical to `tension_filtered` with g_j = mu / s_j.
    """
    n = B.shape[1]
    Baug = np.vstack([B, mu * np.eye(n)])
    return tension_from(Baug, I)


# --- filter shapes --------------------------------------------------------------------------

def g_tikhonov(mu):
    return lambda s, s0: mu / s


def g_relative(rtol, p=1.0):
    """Penalty that is 1 exactly at the cutoff and turns over with power `p` in log s."""
    return lambda s, s0: (rtol * s0 / s) ** p


# --- the measurements ------------------------------------------------------------------------
#
# Run from `tests/`:  PYTHONPATH=. ../.venv/bin/python ../notes/smooth_filter.py
# Produces the tables in `notes/smooth_filter.md`. EXPERIMENTAL; nothing in `src/` uses it.

def _invariance():
    """The GSVD is invariant under ANY invertible right factor -- so weighting is not a filter."""
    rng = np.random.default_rng(0)
    B, I = rng.standard_normal((40, 12)), rng.standard_normal((20, 12))
    t0 = tension_from(B, I)
    print("right-factor invariance (max relative change in the tension vector):")
    for label, M in (("diag, mild", np.diag(10.0**rng.uniform(-2, 0, 12))),
                     ("diag, 1e-8 span", np.diag(10.0**rng.uniform(-8, 0, 12))),
                     ("random invertible", rng.standard_normal((12, 12)))):
        t = tension_from(B @ M, I @ M)
        print(f"  {label:>20}: {np.max(np.abs(t - t0) / t0):.3e}")


def _rank_change_points(p, lo, hi, n=600, want=6):
    """Parameters at which the threshold rank steps, bisected to 1e-10."""
    grid = np.linspace(lo, hi, n)
    rk = np.array([p.rank_at(float(l)) for l in grid])
    out = []
    for j in np.nonzero(np.diff(rk))[0][:want]:
        a, b = grid[j], grid[j + 1]
        ra = p.rank_at(float(a))
        while b - a > 1e-10:
            mid = 0.5 * (a + b)
            if p.rank_at(float(mid)) == ra:
                a = mid
            else:
                b = mid
        out.append((a, b))
    return out


def _plateau_noise(f, c, width=1e-8, n=11):
    """Relative residual of a local linear fit -- how noisy one evaluation is."""
    xs = c * (1 + np.linspace(-width, width, n))
    ys = np.array([f(x) for x in xs])
    fit = np.polyval(np.polyfit(xs - c, ys, 1), xs - c)
    return np.abs(ys - fit).max() / np.abs(ys).mean()


def _trade_curve():
    """floor(mu) against noise(mu), with the two incumbents on the same axes."""
    import rect_suite as rs
    from chladni.mps import MPSPencil

    case = rs.family_A()[0]
    p = case.pencil
    truth = np.unique(rs.complete_truth(case))
    frozen = MPSPencil.from_basis(p.basis, p.bdry, p.interior, reg=p.reg, rtol=p.rtol,
                                  lam_ref=(case.a, case.b))
    pts = _rank_change_points(p, 20.0, 300.0)
    pencils = [(p.pencil(a), p.pencil(b)) for a, b in pts]
    at_truth = [p.pencil(float(l)) for l in truth]
    centres = [40.0, 90.0, 160.0, 230.0, 290.0]

    print(f"\nrect(1, 1/phi) over [{case.a:g}, {case.b:g}], {p.n_cols} columns, "
          f"frozen rank {frozen.rank}, threshold rank "
          f"{min(p.rank_at(float(l)) for l in truth)}.."
          f"{max(p.rank_at(float(l)) for l in truth)}")
    print(f"\n{'policy':>10} {'worst floor':>12} {'worst step':>11} {'worst noise':>12}")
    for mu in np.logspace(-15, -7, 9):
        fl = max(tension_tikhonov(B, I, mu)[0] for (B, I) in at_truth)
        st = max(abs(tension_tikhonov(Bh, Ih, mu)[0] - tension_tikhonov(Bl, Il, mu)[0])
                 / min(tension_tikhonov(Bh, Ih, mu)[0], tension_tikhonov(Bl, Il, mu)[0])
                 for (Bl, Il), (Bh, Ih) in pencils)
        ns = max(_plateau_noise(
            lambda x, mu=mu: tension_tikhonov(*p.pencil(float(x)), mu)[0], c)
            for c in centres)
        print(f"{'mu=%.0e' % mu:>10} {fl:12.3e} {st:11.2e} {ns:12.2e}")
    for name, obj in (("THRESHOLD", p), ("FROZEN", frozen)):
        f = lambda x: float(obj.tension(float(x)))                          # noqa: E731
        fl = max(f(float(l)) for l in truth)
        st = max(abs(f(b) - f(a)) / min(f(a), f(b)) for a, b in pts)
        ns = max(_plateau_noise(f, c) for c in centres)
        print(f"{name:>10} {fl:12.3e} {st:11.2e} {ns:12.2e}")


def _slope_divergence():
    """The apparent slope as the difference step refines onto a rank change."""
    import rect_suite as rs
    from chladni.mps import MPSPencil

    case = rs.family_A()[0]
    p = case.pencil
    frozen = MPSPencil.from_basis(p.basis, p.bdry, p.interior, reg=p.reg, rtol=p.rtol,
                                  lam_ref=(case.a, case.b))
    (a, b), = _rank_change_points(p, 38.0, 38.6, n=60, want=1)
    c = 0.5 * (a + b)
    pol = [("THRESH", lambda x: float(p.tension(float(x)))),
           ("FROZEN", lambda x: float(frozen.tension(float(x))))]
    pol += [(f"tik {mu:.0e}",
             lambda x, mu=mu: tension_tikhonov(*p.pencil(float(x)), mu)[0])
            for mu in (1e-13, 1e-11, 1e-9)]
    print(f"\ncentred difference at the rank change lam = {c:.9f}\n")
    print(f"{'h':>10} " + " ".join(f"{n:>11}" for n, _ in pol))
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        print(f"{h:10.0e} " + " ".join(
            f"{abs(f(c + h) - f(c - h)) / (2 * h):11.3e}" for _, f in pol))


# --- the hybrid ------------------------------------------------------------------------------

def tension_hybrid(B, I, c, power, null_cut=None, g_max=1e8, return_diag=False):
    """Hard-cut the null zone, smooth-penalise the band: `g_j = (c*s_1/s_j)**power`.

    Three zones in `s_j/s_1`, and the two parameters place them:

    * above the ramp, `g_j -> 0` and the direction is untouched;
    * across the ramp -- centred at `c`, one `1/power` decade wide -- `g_j` climbs smoothly;
    * below, `g_j` exceeds `g_max` and the column is DROPPED rather than carried with an
      astronomical penalty. `f_j = 1/(1+g_j**2) < 1e-16` there, so dropping changes the
      objective by less than the arithmetic can see, and it bounds the dynamic range of
      `diag(g)`, which is what an uncapped `power >= 4` destroys.

    `null_cut` additionally drops everything below `null_cut * s_1` whatever `g` says;
    it defaults to `eps`, the one cut that is not a judgement call.
    """
    m = B.shape[0]
    U, s, _ = stack_svd(B, I)
    if s.size == 0 or s[0] <= 0.0:
        raise ValueError("empty or zero pencil")
    null_cut = EPS if null_cut is None else null_cut
    r = s / s[0]
    with np.errstate(over="ignore", divide="ignore"):
        g = (c / r) ** power
    keep = (r >= null_cut) & (g <= g_max)
    U, g = U[:, keep], g[keep]
    Baug = np.vstack([U[:m], np.diag(g)])
    out = tension_from(Baug, U[m:])
    return (out, int(keep.sum())) if return_diag else out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    _invariance()
    _trade_curve()
    _slope_divergence()

