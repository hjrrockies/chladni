"""`Spectrum.bound` under-reports the error on a starved basis: `delta_res` is not an upper bound.

Found 2026-10-06 while measuring whether `FourierBessel.at_corner(..., domain=)` scaling matters
(it does not: equilibration undoes any fixed column scale). The case: the L-shape with 14
Fourier-Bessel terms at EACH of its six corners, asked for `prec=1e-12` -- a basis whose tension
floor is ~4e-9, so the request is out of its reach. The search correctly flags 6 of 7 levels as
short of `prec`, but on 4 of 7 the reported `bound = min(delta_res, mp_bound)` is SMALLER than the
true error against the 14-digit table. The Moler-Payne bound covers every level; `delta_res` does
not, by up to ~4e6x, and `min` passes it through. `attained_prec` is wrong with it.

Run from the repository root:

    .venv/bin/python notes/delta_res_undercover.py

Measured 2026-10-06, macOS/Accelerate, at `5f5398a`..`03c29db` (no `src/` change between):

    domain=True:  attained 3.4e-10, unmet 6 of 7
       29.5216669411 err 1.9e-04 delta_res 4.4e-11 mp 3.6e-03  covered=False
       31.9126359634 err 6.3e-09 delta_res 6.8e-11 mp 1.4e-06  covered=False
       41.4745098870 err 3.2e-09 delta_res 5.8e-11 mp 3.3e-06  covered=False
       44.9484877527 err 2.9e-08 delta_res 4.2e-10 mp 3.7e-06  covered=False
    domain=False: the same four levels uncovered (errors 6e-10 .. 4e-8)

Not established: whether `delta_res` under-covers on ADEQUATE bases (every adequate-basis row in
`examples/lshape_tour.ipynb` held), or on the penalty path (`prec >= 1e-8`; this basis's bars held
at 1e-8). One domain, one basis family.
"""
import warnings

import numpy as np

from chladni.geometry import L_shape
from chladni.helmholtz import FourierBessel, PrecisionWarning
from chladni.mps import MPSEigensolver
from chladni.reference import L_shape_eigs

warnings.simplefilter("ignore", PrecisionWarning)

dom = L_shape()
TRUTH = np.asarray(L_shape_eigs(7), dtype=float)          # 14 digits; all 7 lie in [5, 45]

n_uncovered = 0
for use_domain in (True, False):
    basis = None
    for c in dom.corners:
        f = FourierBessel.at_corner(c, n=14, domain=dom if use_domain else None)
        basis = f if basis is None else basis + f
    s = MPSEigensolver(dom, basis=basis, n_bdry=3 * len(basis), n_int=len(basis), prec=1e-12)
    res = s.search_interval(5.0, 45.0)
    d = res.diagnostics
    print(f"domain={use_domain}: {len(basis)} columns, tension at lam_1 {s.tension(TRUTH[0]):.1e}, "
          f"attained {d.attained_prec:.1e}, unmet {len(d.unmet)} of {len(res.levels)}")
    for L in res.levels:
        err = float(np.min(np.abs(TRUTH - L.lam)))
        ok = err <= L.bound
        n_uncovered += not ok
        print(f"   {L.lam:.10f} err {err:.1e} delta_res {L.delta_res:.1e} mp {L.mp_bound:.1e} "
              f"bound {L.bound:.1e} covered={ok}  mp_covers={err <= L.mp_bound}")

print(f"\n{n_uncovered} level(s) whose reported bound is below the true error")
