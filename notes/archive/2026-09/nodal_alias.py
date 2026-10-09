"""Does a structured interior fill interpolate the nodal lines of some eigenfunction?

This is the interior block's PRIMARY job, and it is not what section 17 measured. The tension
is `||A_B c|| / ||A_I c||`, so the interior points exist to keep the denominator away from
zero. If they happen to lie on the nodal set of the target eigenfunction, `||A_I u||` collapses
for a function that is NOT small, the tension is inflated, and the eigenvalue is missed
outright. Random points make that probabilistically impossible. A grid could be commensurate.

Section 17 measured the ACCURACY of the discrete inner product on two well-behaved functions.
That is a different quantity from its WORST-CASE degeneracy over all modes, and a fill can be
excellent at the first while failing catastrophically at the second.

The rectangle makes it exact and needs no MPS at all: `u_mn = sin(m*pi*x/a)*sin(n*pi*y/b)` has
`mean(u**2) = 1/4` over the domain, so

    R = mean over the interior points of u**2, divided by 1/4

is the normalization the interior block actually applies. R = 1 is a fair sample; R -> 0 is the
failure. The quantity that matters is the WORST R over the modes in range.

Run log for section 18 of `landscape_findings.md`. Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

from chladni.geometry import rect

sys.path.insert(0, "notes")
from interior_pts import points  # noqa: E402


def R(pts, m, n, a=1.0, b=1.0):
    u = np.sin(m * np.pi * pts.real / a) * np.sin(n * np.pi * pts.imag / b)
    return np.mean(u**2) / 0.25


def worst(pts, mmax, a=1.0, b=1.0):
    """The most-annihilated mode, and its indices."""
    best, at = np.inf, None
    for m in range(1, mmax + 1):
        for n in range(1, mmax + 1):
            r = R(pts, m, n, a, b)
            if r < best:
                best, at = r, (m, n)
    return best, at


def sweep(a=1.0, mmax=50):
    dom = rect(a, 1.0)
    print(f"rect({a}, 1), worst R over modes m, n = 1 .. {mmax}   (R = 1 is a fair sample)\n")
    print(f"{'n_int':>7} | {'grid: worst R':>14} {'at':>9} | "
          f"{'random: worst R (12 seeds)':>28} | {'sobol':>10}")
    for ni in (200, 400, 800, 1600):
        g = points(dom, "grid", ni)
        gw, gat = worst(g, mmax, a)
        rw = []
        for s in range(12):
            rw.append(worst(points(dom, "random", ni, s), mmax, a)[0])
        rw = np.array(rw)
        sw = worst(points(dom, "sobol", ni), mmax, a)[0]
        print(f"{ni:7d} | {gw:14.2e} {str(gat):>9} | "
              f"med {np.median(rw):8.3f}  min {rw.min():8.3f} | {sw:10.3f}")


def arithmetic(mmax=60):
    """The mechanism: `interior_pts` grid puts x at k/(side+1), so mode `side+1` is annihilated
    exactly, and every DIVISOR of `side+1` is partially annihilated. Whether that is harmless
    depends on the factorization of an integer nobody chose deliberately."""
    dom = rect(1.0, 1.0)
    print(f"\n\nthe mechanism: grid x-values are k/(side+1), side = ceil(sqrt(4*n_int))\n")
    print(f"{'n_int':>7} {'side':>5} {'side+1':>7} {'factors':>18} "
          f"{'worst R':>10} {'at':>10}")
    for ni in (100, 200, 300, 400, 500, 600, 800, 1000, 1225, 1600):
        side = int(np.ceil(np.sqrt(ni * 4.0)))
        s1 = side + 1
        f = [d for d in range(2, s1 + 1) if s1 % d == 0]
        g = points(dom, "grid", ni)
        gw, gat = worst(g, mmax)
        print(f"{ni:7d} {side:5d} {s1:7d} {str(f[:5]):>18} {gw:10.2e} {str(gat):>10}")


if __name__ == "__main__":
    sweep()
    arithmetic()
