"""The L-shape's exact double `10 pi^2` comes back with multiplicity 0 to 4, unflagged.  (2026-10-07)

Found by `notes/bound_coverage.py`. The basis is `tests/hard_roster.recipe_basis` -- 80
Fourier-Bessel terms at the reentrant corner, the L-shape's reference recipe -- so this is an
ADEQUATE basis: the tension at `10 pi^2` is 1.5e-13, `sigma_3` there 0.09. `10 pi^2 = 98.696`
is a true double (two square modes on the three unit cells; `L_shape_eigs` lists it twice).

Mechanism at `prec = 1e-12`, window [5, 100] (traced 2026-10-07): the reach rule reads branch 2
at `d_est = sigma_2/c = 1.4e-11` against an emitted bracket reaching 1.2e-11 on its near side,
so it DECLINES (`ambiguous=True`, mu=1). The side search (`search.py`, "LOCATE it") then refines
either side of the root for a DISTINCT partner, finds none -- the partner is the root itself --
and resolves the level as certain. Nothing else locates the second copy. The design comment
assumes a declined branch is "one bracket holding two DISTINCT eigenvalues, the partner at the
bracket edge"; an exact degeneracy at the floor is the other case.

    .venv/bin/python notes/double_10pi2.py        # ~1 min

Measured 2026-10-07 at the `overnight/2026-10-07` branch point (before any search change):

    prec   window      count  flagged
    1e-12  [5, 100]      1     no
    1e-12  [5, 110]      1     no
    1e-12  [90, 105]     2     no   (as two simple levels 6e-12 apart)
    1e-12  [95, 100]     2     no   (one level, mult 2: correct)
    1e-13  [5, 100]      3     no
    1e-13  [5, 110]      4     no
    1e-13  [90, 105]     0     no
    1e-13  [95, 100]     0     no

`rtol` (default against `SUITE_RTOL`) changes nothing.
"""
import sys
import warnings

import numpy as np

sys.path.insert(0, "tests")

from chladni.geometry import L_shape  # noqa: E402
from chladni.helmholtz import PrecisionWarning  # noqa: E402
from chladni.mps import MPSEigensolver  # noqa: E402
from hard_roster import recipe_basis  # noqa: E402

warnings.simplefilter("ignore", PrecisionWarning)

dom = L_shape()
basis = recipe_basis(dom, 80)
n = len(basis)
t = 10 * np.pi ** 2
print(f"{n} columns")
print("prec   window      count  flagged  levels near 10 pi^2 (offset, mult, uncertain)")
for prec in (1e-12, 1e-13):
    s = MPSEigensolver(dom, basis=basis, n_bdry=3 * n, n_int=n, prec=prec)
    for lo, hi in ((5, 100), (5, 110), (90, 105), (95, 100)):
        res = s.search_interval(lo, hi)
        near = [L for L in res.levels if abs(L.lam - t) < 0.05]
        count = sum(L.mult for L in near)
        flagged = any(L.count_uncertain or L.grouping_ambiguous for L in near)
        print(f"{prec:.0e}  [{lo:3d}, {hi:3d}]   {count}     {'yes' if flagged else 'no '}     "
              f"{[(f'{L.lam - t:+.1e}', L.mult, L.count_uncertain) for L in near]}")
