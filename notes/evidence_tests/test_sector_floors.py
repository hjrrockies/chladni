"""What a corner Fourier-Bessel basis on a disk SECTOR can serve, measured before sectors are used.

Plan Phase 2. The sector is the only closed-form family with both a singular apex and a dial for
near-degeneracy: branches of different angular index cross exactly as the opening angle moves.
Before a crossing family is built on it, the floor at the modes that cross has to be known, or
every cell past it measures the basis.

Measured 2026-09-14 on a fixed `SUITE_RTOL`, at the four crossings `tests/sector_suite.py` will
dial, with the fewest columns that represent both modes:

    omega    crossing          lam       tension at modes     3*eps/c/lam
     63.74   (1,2) x (2,1)     90.78     7.1e-14 / 1.1e-14      5.2e-14
     72.40   (1,3) x (2,2)    151.38     2.9e-15 / 8.7e-14      5.1e-14
    139.70   (1,2) x (3,1)     55.17     4.1e-14 / 1.3e-14      2.8e-14
    216.91   (1,2) x (4,1)     45.82     1.4e-14 / 4.4e-15      9.8e-15   reentrant

So the crossings are good to about 5e-14 for accuracy and not to 1e-14, except the reentrant one.
Adding columns does not move the crossing modes, but it DEGRADES the worst mode in a window:
7e-14 with one column and 1e-11 with six, tracking `Sampling.accuracy`'s own bound from 5e-12 to
1.3e-9. That is the Bessel kernel at high order, and it is why a sector case must carry its floor
through `basis_sufficiency` rather than be assumed reference-grade.

The recorded angles are rounded: at 63.74 degrees the (1,2)/(2,1) pair is still 3e-5 apart. A
dial must solve for the crossing, not read it from this table.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from chladni.geometry import disk_sector
from chladni.helmholtz import FourierBessel
from chladni.mps import MPSPencil
from chladni.reference import sector_eig
from rect_suite import SUITE_RTOL

pytestmark = pytest.mark.slow

CROSSINGS = [(63.74, (1, 2), (2, 1), 1e-13), (72.40, (1, 3), (2, 2), 1e-13),
             (139.70, (1, 2), (3, 1), 1e-13), (216.91, (1, 2), (4, 1), 1e-14)]


def _pencil(omega_deg, n):
    om = np.deg2rad(omega_deg)
    dom = disk_sector(1.0, om)
    apex = min(dom.corners, key=lambda c: abs(c.z))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fb = FourierBessel.at_corner(apex, n=n, domain=dom)
    return om, MPSPencil.from_basis(fb, dom.bdry_quad(max(3 * n, 120)),
                                   dom.interior_pts(max(n, 40), rng=np.random.default_rng(0)),
                                   rtol=SUITE_RTOL)


def _bracketing(p, lams):
    worst = 0.0
    for x in lams:
        c = 0.5 * (p.tension(x * 1.005) + p.tension(x * 0.995)) / (0.005 * x)
        worst = max(worst, 3.0 * p.tension(x) / c / x)
    return worst


@pytest.mark.parametrize("omega, m1, m2, serves", CROSSINGS)
def test_the_crossing_modes_serve_the_precision_measured(omega, m1, m2, serves):
    """Pinned at the resolution the measurement supports, so a basis or kernel change that loses
    an order shows up here rather than as a crossing-family failure somewhere downstream."""
    om, p = _pencil(omega, max(m1[0], m2[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lams = [sector_eig(*m1, 1.0, om), sector_eig(*m2, 1.0, om)]
        assert _bracketing(p, lams) <= serves


@pytest.mark.parametrize("omega", [63.74, 139.70])
def test_more_columns_DEGRADE_the_worst_mode_and_the_accuracy_bound_says_so(omega):
    """The warning is not noise to be filtered: `Sampling.accuracy` rises with the order, and the
    worst tension rises with it. Asserted, not suppressed."""
    rows = []
    for n in (1, 6):
        om, p = _pencil(omega, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lams = [x for x in (sector_eig(l, r, 1.0, om) for l in range(1, n + 1)
                                for r in (1, 2, 3)) if x < 600]
            worst = max(p.tension(x) for x in lams)
            bound = float(np.max(p.bdry.accuracy(np.sqrt(max(lams)))))
        rows.append((worst, bound))
    (w1, b1), (w6, b6) = rows
    assert w6 > 10 * w1, f"worst tension {w1:.1e} -> {w6:.1e}"
    assert b6 > 10 * b1, f"accuracy bound {b1:.1e} -> {b6:.1e}"
    assert b6 > 1e-10, "and the bound is large enough to disqualify 1e-14 on its own"
