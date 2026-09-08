"""Tighten `m` in kappa = m * eps/c. Section 16 took it from three seeds.

A loose `m` is not a neutral error. kappa too large MERGES two genuinely distinct eigenvalues
and reports one level with multiplicity 2, throwing away resolution the solver actually had --
it makes us underestimate the solver. And the basis contract `gap > 2*m*eps/c` is derived from
it, so a loose `m` also declares adequate bases inadequate.

What is actually being measured: at an EXACT degeneracy the discretization splits the pair into
two nearby simple eigenvalues. `|argmin sigma_1 - argmin sigma_2|` is that splitting. It is NOT
minimizer noise -- the location uncertainty of a well with floor eps and relative roughness eta
is eps*sqrt(2*eta)/c, about 4e-14 here, four orders below what is measured.

Two domains with protected degeneracies, 30 seeds each.

Run log for section 19 of `landscape_findings.md`. Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import minimize_scalar

from chladni.geometry import disk, eq_tri, rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil

PI2 = np.pi**2


def pencil(dom, lam, ni, seed, nw=40, nb=800):
    b = PlaneWave.uniform(nw, radius=dom.diameter)
    return MPSPencil.from_basis(b, dom.bdry_quad(nb),
                                dom.interior_pts(ni, rng=np.random.default_rng(seed)),
                                lam_ref=(lam * 0.9, lam * 1.1))


def splitting(dom, lam, ni, seed, W=None):
    """|argmin s1 - argmin s2| at an exact degeneracy, in units of eps/c."""
    p = pencil(dom, lam, ni, seed)
    W = W if W is not None else 0.03 * lam
    c = 0.5 * (p.tension(lam + W) + p.tension(lam - W)) / W
    xs = [minimize_scalar(lambda x: p.tensions(x)[j]**2, bounds=(lam - W, lam + W),
                          method="bounded", options={"xatol": 1e-14}).x for j in (0, 1)]
    eps = p.tension(lam)
    return abs(xs[0] - xs[1]) / (eps / c), eps, c


CASES = [
    ("rect(1,1)  modes (2,1)/(1,2)", rect(1.0, 1.0), 5 * PI2),
    ("disk       j_1,1 double     ", disk(1.0), 3.831705970207512**2),
    ("eq_tri     arithmetic double", eq_tri(1.0), None),
]


def main(nseed=30):
    from chladni.reference import eq_tri_eigs
    e = eq_tri_eigs(12)
    dbl = [v for i, v in enumerate(e[:-1]) if abs(e[i + 1] - v) < 1e-10]
    cases = [(lbl, d, lam if lam is not None else dbl[0]) for lbl, d, lam in CASES]

    print(f"|argmin s1 - argmin s2| / (eps/c) at an exact degeneracy, {nseed} seeds\n")
    print(f"{'case':<30} {'lam':>10} {'eps':>9} | {'med':>8} {'p90':>8} "
          f"{'max':>8} | {'min':>8}")
    allv = []
    for lbl, dom, lam in cases:
        v, es = [], []
        for s in range(nseed):
            try:
                r, eps, _ = splitting(dom, lam, 400, s)
            except Exception:
                continue
            v.append(r)
            es.append(eps)
        v = np.array(v)
        allv.append(v)
        print(f"{lbl:<30} {lam:10.4f} {np.median(es):9.1e} | {np.median(v):8.2f} "
              f"{np.percentile(v, 90):8.2f} {v.max():8.2f} | {v.min():8.2f}")

    a = np.concatenate(allv)
    print(f"\n{'pooled':<30} {'':>10} {'':>9} | {np.median(a):8.2f} "
          f"{np.percentile(a, 90):8.2f} {a.max():8.2f} | {a.min():8.2f}")
    print(f"\n   n = {a.size}.  p95 = {np.percentile(a, 95):.1f}, "
          f"p99 = {np.percentile(a, 99):.1f}, max = {a.max():.1f}")
    print(f"   fraction above 50: {np.mean(a > 50):.3f}   "
          f"above 100: {np.mean(a > 100):.3f}   above 171: {np.mean(a > 171):.3f}")


if __name__ == "__main__":
    main()
