"""Evidence moved out of `tests/test_moler_payne.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 3: rungs below 1e-10 are parked. Both tests here
run at `prec = 1e-12` and both are `ulp_sensitive`, so neither is a contract douse needs, but
each records a measurement:

* the `tau 2e-14` row: the Moler-Payne bound is valid AND within 10x on the rectangle when the
  penalty sits near rounding (2 of 16 jitter seeds flipped it; Linux EPYC 9V45);
* the starved basis: evidence for TODO's "`delta_res` is admitted into `Level.bound` at the very
  edge of `prec`" item -- on a Linux EPYC 9V45 runner its level at 44.948 had `delta_res`
  4.445e-11 against `prec*lam` 4.495e-11, admitted by 1%, and the bound failed to cover.

Moved verbatim with their helpers; the `tau 2e-14` case is now its own function, with the
parametrization's other rows removed. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_moler_payne_evidence.py
"""
import numpy as np
import pytest

from chladni import reference as ref
from chladni.eigensolver import lam_to_k
from chladni.geometry import L_shape
from chladni.integrals import cauchy as C
from chladni.integrals import certify as M
from chladni.integrals import plan as P
from chladni.mps import MPSPencil
from chladni.mps.objective import Objective
from chladni.mps.refine import refine

pytestmark = pytest.mark.slow


def _bound(p, lam, x, domain, prec=1e-12):
    """The bound for the candidate `x` (equilibrated coordinates) of pencil `p` at `lam`."""
    fc = p.family_coef(lam, x)
    if fc.ndim == 1:
        fc = fc[:, None]
    iq = P.integral_quad(domain, lam, prec=prec)
    cd = C.cauchy_data(p.basis, fc, lam, iq.quad)
    k = lam_to_k(lam)

    def evaluate(z):
        V = p.basis.values(k, z)
        return V @ fc, np.abs(V) @ np.abs(fc)

    return M.moler_payne(cd, evaluate, domain, M.x0_probes(domain, iq.x0))


def _rect_case():
    import rect_suite as rs
    return rs.family_A()[0]


@pytest.mark.ulp_sensitive   # 2 of 16 seeds; Linux EPYC 9V45
def test_the_bound_covers_the_true_error_on_every_rectangle_level_tau_2e_14():
    """`notes/moler_payne_vs_tension.py bound`, now through `chladni.integrals` with a Rellich
    norm instead of an interior rule: five eigenvalues of `rect(1, 1/phi)`, each refined on the
    pencil, the bound held against the exact eigenvalue. Valid on every row; within 2-6x where
    the penalty sits near rounding, as the note measured."""
    case = _rect_case()
    p0 = case.pencil
    n = p0.n_cols
    p, prec = (MPSPencil(p0.bdry, p0.interior, reg="tikhonov_qr", prec=1e-12,
                         rtol=2e-14 / np.sqrt(n)), 1e-12)
    for lam0 in (float(t) for t in case.truth[[0, 1, 2, 4, 7]]):
        lo, hi = lam0 * (1 - 1e-6), lam0 * (1 + 1.3e-6)
        obj = Objective(lambda z: p.tensions(float(z)), lo, hi, prec=prec, variable="lam")
        r = refine(obj, lo, hi, prec=prec)
        mp = _bound(p, r.lam, p.coef(r.lam)[:, 0], case.domain)
        err = abs(r.lam - lam0)
        assert mp.reason == ""
        assert mp.abs[0] >= err, (lam0, mp.abs[0], err)
        assert mp.abs[0] <= 10 * max(err, 1e-15 * lam0), (lam0, mp.abs[0], err)


@pytest.mark.ulp_sensitive  # flips under tests/ulp_jitter.py and across CPUs; notes/jitter_sweep.py
@pytest.mark.filterwarnings("ignore::chladni.helmholtz.PrecisionWarning")
def test_a_starved_basis_reports_a_bound_that_covers_its_error():
    """`notes/delta_res_undercover.py`, `notes/bound_coverage.md`: 14 Fourier-Bessel terms at
    each L-shape corner asked for 1e-12, out of the basis's reach. `delta_res` read up to ~1e5x
    under the true error on such levels and `min(delta_res, mp_bound)` passed it through, so the
    reported bound and `attained_prec` were wrong. A `delta_res` that does not meet `prec` is
    the basis's floor, not a resolution limit, and is not admitted: the level reports its
    Moler-Payne bound, which covered every level of the grid."""
    from chladni.helmholtz import FourierBessel
    from chladni.mps import MPSEigensolver

    dom = L_shape()
    b = None
    for c in dom.corners:
        f = FourierBessel.at_corner(c, n=14, domain=dom)
        b = f if b is None else b + f
    s = MPSEigensolver(dom, basis=b, n_bdry=3 * len(b), n_int=len(b), prec=1e-12)
    res = s.search_interval(5.0, 45.0)
    truth = ref.L_shape_eigs(10)
    worst = 0.0
    starved = 0
    for L in res.levels:
        err = float(np.min(np.abs(truth - L.lam)))
        assert err <= L.bound, (L.lam, err, L.delta_res, L.mp_bound, L.bound)
        if L.delta_res > 1e-12 * L.lam:
            starved += 1
            assert L.bound == L.mp_bound
        worst = max(worst, err / L.lam)
    assert starved >= 4                       # the case still exercises the rule
    assert res.diagnostics.attained_prec >= worst

