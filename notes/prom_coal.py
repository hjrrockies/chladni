"""The two undefined constants of the operational definition, measured.

P (PROMINENCE) makes "sigma_1 has a local minimum here" decidable. Inside the flat bottom of
a well -- half-width eps*sqrt(2*eps_mach)/c -- the curve is roughness, not signal, so a strict
detector finds either nothing or a crowd. P is the factor by which sigma_1 must rise on both
sides for a minimum to count. The question is whether true wells and roughness minima are
separated by a wide margin, as G and tau_gram were, or not.

KAPPA (COALESCENCE) makes "sigma_j has a local minimum AT lam_i" decidable, since at a real
degeneracy the branches are numerically coincident but not bit-identical. Measured as
|argmin sigma_1 - argmin sigma_2| on rect(1+delta, 1): at delta = 0 that is the coalescence
residual, and at delta > 0 section 9.2 says it is d/2. kappa has to sit between them.

Run log for section 16 of `landscape_findings.md`. Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, peak_prominences

from chladni.geometry import rect
from chladni.helmholtz import FourierBessel, PlaneWave
from chladni.mps import MPSPencil

PI2 = np.pi**2
U = np.finfo(float).eps


def lams(delta):
    a = 1.0 + delta
    return PI2 * (4 / a**2 + 1), PI2 * (1 / a**2 + 4), PI2 * (1 / a**2 + 1)


def build(delta, nw=40, nb=800, ni=400, seed=0, fb=None):
    """`fb=n` uses a Fourier-Bessel family at ONE corner instead of plane waves.

    A structurally deficient basis rather than a merely small one, and a far better ladder:
    n = 4, 6, 8, 10, 14 gives eps = 7.5e-03, 1.7e-05, 1.2e-08, 3.3e-12, 1.8e-15 -- five clean
    decades, against the plane waves' abrupt 1e-03 -> 1e-06 -> 1e-10 -> 1e-14.
    """
    dom = rect(1.0 + delta, 1.0)
    _, lam_b, lam_c = lams(delta)
    b = (FourierBessel.at_corner(dom.corners[0], n=fb, domain=dom) if fb
         else PlaneWave.uniform(nw, radius=dom.diameter))
    return MPSPencil.from_basis(b, dom.bdry_quad(nb),
                                dom.interior_pts(ni, rng=np.random.default_rng(seed)),
                                lam_ref=(lam_c * 0.9, lam_b * 1.1))


def wall_slope(p, lam_star, h=0.6):
    """c in LAM units: the far-field slope of the cone, one gap-tenth out."""
    return 0.5 * (p.tension(lam_star + h) + p.tension(lam_star - h)) / h


def prominences(s):
    """Ratio prominence of every discrete local minimum, via log-space peak prominence."""
    y = -np.log(s)
    idx, _ = find_peaks(y)
    if idx.size == 0:
        return idx, np.array([])
    return idx, np.exp(peak_prominences(y, idx)[0])


def sample(p, lam_c, W, n=401):
    xs = np.linspace(lam_c - W, lam_c + W, n)
    return prominences(np.array([p.tension(x) for x in xs]))


def measure_P():
    """A flat bottom only exists when `eps` is LARGE, so starve the basis.

    40 plane waves on a rectangle give eps ~ 7e-15 and a flat half-width of 1.8e-21 -- below
    float spacing, so the well is conical all the way down and there is no roughness regime
    to measure at all. Section 10 said exactly this. 6-16 waves puts the floor back up.

    And prominence measured over a window of half-width W is W/(eps/c) by construction -- it
    measures the WINDOW. The intrinsic number is the whole well: plateau over floor.
    """
    _, _, lam_c = lams(1e-2)
    ulp = np.spacing(lam_c)
    bases = ([("pw", nw, dict(nw=nw)) for nw in (6, 8, 10, 12, 16, 40)]
             + [("fb", n, dict(fb=n)) for n in (4, 5, 6, 7, 8, 10, 14)])
    print("A. PROMINENCE\n")
    print("   true wells, window = 0.4 * the mean eigenvalue gap (4*pi/area = 12.4)\n")
    print(f"{'basis':>8} {'eps':>10} {'plateau':>9} {'eps/c':>9} "
          f"{'n_min':>6} {'P (deepest)':>12} {'plateau/eps':>12}")
    rows = []
    for kind, size, kw in bases:
        p = build(1e-2, **kw)
        nw = f"{kind}{size}"
        eps, c = p.tension(lam_c), wall_slope(p, lam_c)
        S = p.tension(lam_c + 3.0)
        w = eps * np.sqrt(2 * U) / c
        rows.append((nw, p, eps, c, S, w))
        idx, pr = sample(p, lam_c, 0.4 * 12.44)
        best = pr.max() if pr.size else np.nan
        print(f"{nw:>8} {eps:10.2e} {S:9.2e} {eps/c:9.2e} "
              f"{idx.size:6d} {best:12.2e} {S/eps:12.2e}")

    print("\n   roughness minima, window = 100 * the flat half-width\n")
    print(f"{'basis':>8} {'flat w':>10} {'h/ulp':>9} {'n_min':>6} "
          f"{'max P - 1':>11} {'med P - 1':>11}")
    for nw, p, eps, c, S, w in rows:
        W = 100 * w
        h = 2 * W / 400
        if h < 4 * ulp:
            print(f"{nw:>8} {w:10.2e} {h/ulp:9.2f} {'--':>6} "
                  f"{'below float spacing':>11}")
            continue
        idx, pr = sample(p, lam_c, W)
        if idx.size == 0:
            print(f"{nw:>8} {w:10.2e} {h/ulp:9.1f} {0:6d} "
                  f"{'no minima at all':>11}")
            continue
        print(f"{nw:>8} {w:10.2e} {h/ulp:9.1f} {idx.size:6d} "
              f"{pr.max() - 1:11.2e} {np.median(pr) - 1:11.2e}")


def argmin_branch(p, j, lo, hi, xatol=1e-14):
    f = lambda x: p.tensions(x)[j]**2
    return minimize_scalar(f, bounds=(lo, hi), method="bounded",
                           options={"xatol": xatol}).x


def measure_kappa():
    print("\n\nB. COALESCENCE: |argmin sigma_1 - argmin sigma_2| against the separation\n")
    print(f"{'delta':>9} {'d(lam)':>10} {'d/2':>10} {'|Delta|':>10} "
          f"{'|Delta|/(d/2)':>14} {'eps':>9} {'eps/c':>9} {'|D|/(eps/c)':>12}")
    for delta in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0]:
        p = build(delta)
        lam_a, lam_b, _ = lams(delta)
        d = lam_b - lam_a
        mid = 0.5 * (lam_a + lam_b)
        W = max(2 * d, 0.5)
        x1 = argmin_branch(p, 0, mid - W, mid + W)
        x2 = argmin_branch(p, 1, mid - W, mid + W)
        eps, c = p.tension(lam_a), wall_slope(p, lam_a)
        D = abs(x1 - x2)
        r = D / (0.5 * d) if d > 0 else np.nan
        print(f"{delta:9.1e} {d:10.3e} {0.5*d:10.3e} {D:10.3e} {r:14.3f} "
              f"{eps:9.2e} {eps/c:9.2e} {D/(eps/c):12.2f}")

    print("\n   the delta = 0 residual, against the optimizer tolerance and the seed\n")
    print(f"{'xatol':>10} {'seed':>5} {'|Delta|':>11} {'eps/c':>10} {'|D|/(eps/c)':>12}")
    lam_a, _, _ = lams(0.0)
    for xatol in (1e-12, 1e-14, 1e-16):
        for seed in (0, 1, 2):
            p = build(0.0, seed=seed)
            x1 = argmin_branch(p, 0, lam_a - 0.5, lam_a + 0.5, xatol)
            x2 = argmin_branch(p, 1, lam_a - 0.5, lam_a + 0.5, xatol)
            c = wall_slope(p, lam_a)
            knee = p.tension(lam_a) / c
            print(f"{xatol:10.0e} {seed:5d} {abs(x1-x2):11.3e} {knee:10.2e} "
                  f"{abs(x1-x2)/knee:12.2f}")


if __name__ == "__main__":
    measure_P()
    measure_kappa()
