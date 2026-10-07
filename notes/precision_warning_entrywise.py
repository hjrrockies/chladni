"""`FourierBessel`'s `PrecisionWarning` fires on per-ENTRY relative error, which is not substantive.

Found 2026-10-06 reading `examples/lshape_tour.ipynb`. `Sampling._check_accuracy`
(`helmholtz/_core.py`) warns when `max(rel_err)` over every point and column exceeds the family's
`prec`. In the series branch `rel_err = eps * largest_term / |S|` (`helmholtz/_bessel.py`,
`_j_series`), which blows up wherever `J_mu(k r)` is near a zero: the relative error of a
near-zero entry is ill-conditioned. The solve is normwise -- equilibration and the GSVD see error
relative to a column's scale -- so those entries perturb nothing.

This script, on the L-shape with 80 reentrant-corner terms and 240 boundary points, scans `lam`
in [5, 45] for the worst `rel_err`, then checks the worst entries against mpmath at 40 digits.

Run from the repository root:

    .venv/bin/python notes/precision_warning_entrywise.py

Measured 2026-10-06 (macOS/Accelerate, `d0b4a9a`):

    worst rel_err over the window: 2.1e-08 at lam=42.1000
     mu      k*r     rel_err  |entry|/colmax  true rel err  true err/colmax
     1.333   7.4909  2.1e-08  1.9e-06         2.7e-09       5.1e-15
     0.667   6.5320  1.2e-11  1.5e-04         3.1e-12       4.8e-16
     3.333   6.7851  1.2e-11  1.0e-03         1.2e-12       1.2e-15
     5.333   9.1684  1.0e-11  3.9e-03         2.7e-12       1.1e-14
    max over entries of rel_err*|entry|/colmax: 2.6e-13

and end to end, the same basis gives eigenvalues to ~4e-14 whether the family's `prec` is 1e-8
(warns occasionally) or 1e-12 (warns 185 times in one `search_interval(5, 45)`).

Not established: the scipy branch's bound has no 1/|S| term and was not the trigger here; branch C
(Olver/Debye) was not examined; one domain, one basis.
"""
import warnings

import mpmath as mp
import numpy as np

from chladni.geometry import L_shape
from chladni.helmholtz import FourierBessel, PrecisionWarning
from chladni.helmholtz._bessel import log_j
from chladni.mps import lam_to_k

warnings.simplefilter("ignore", PrecisionWarning)
mp.mp.dps = 40

dom = L_shape()
corner = next(c for c in dom.corners if c.is_reentrant)
fb = FourierBessel.at_corner(corner, n=80, domain=dom)
smp = fb.at(dom.bdry_quad(240))
r, mu = smp.r, fb.orders


def finite(a):
    return np.where(np.isfinite(a), a, 0.0)


worst, lam = max((float(finite(log_j(mu[None, :], lam_to_k(x) * r[:, None], False).rel_err).max()), x)
                 for x in np.linspace(5, 45, 401))
print(f"worst rel_err over the window: {worst:.1e} at lam={lam:.4f}")

k = lam_to_k(lam)
s = log_j(mu[None, :], k * r[:, None], False)
rel = finite(s.rel_err)
V = smp.values(k)                              # the scaled columns the pencil assembles
colmax = np.abs(V).max(axis=0)

print(" mu      k*r     rel_err  |entry|/colmax  true rel err  true err/colmax")
seen = set()
for flat in np.argsort(rel.ravel())[::-1]:
    i, j = np.unravel_index(flat, rel.shape)
    key = (j, round(float(r[i]), 12))
    if key in seen:                            # symmetric points repeat the same (mu, k r)
        continue
    seen.add(key)
    x, nu = k * r[i], mu[j]
    true = mp.besselj(nu, x)
    sigma = nu * mp.log(x / 2) - mp.loggamma(nu + 1)
    got = mp.mpf(float(s.S[i, j])) * mp.exp(sigma)
    true_rel = float(abs((got - true) / true))
    frac = abs(V[i, j]) / colmax[j]
    print(f" {nu:6.3f}  {x:7.4f}  {rel[i, j]:7.1e}  {frac:14.1e}  {true_rel:12.1e}  {true_rel * frac:13.1e}")
    if len(seen) == 4:
        break

print(f"max over entries of rel_err*|entry|/colmax: {np.nanmax(rel * np.abs(V) / colmax[None, :]):.1e}")
