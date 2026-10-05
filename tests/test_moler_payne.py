"""The Moler-Payne bound from boundary data: `chladni.integrals.certify` (M3).

The contract is an INEQUALITY: some true eigenvalue lies within the bound. So every test here
knows the true eigenvalue and checks two things -- the bound covers the error (validity), and it
does not cover it by a factor that would make it useless (tightness).
"""
import sys

import numpy as np
import pytest

from chladni import reference as ref
from chladni.eigensolver import lam_to_k
from chladni.geometry import L_shape, rect
from chladni.integrals import cauchy as C
from chladni.integrals import certify as M
from chladni.integrals import plan as P
from chladni.mps import MPSPencil
from chladni.mps.objective import Objective
from chladni.mps.refine import refine

sys.path.insert(0, "tests")

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


# --- the notes table, through the layer -----------------------------------------------------------

def _rect_case():
    import rect_suite as rs
    return rs.family_A()[0]


@pytest.mark.parametrize("which", ["default", "tau 2e-14", "threshold"])
def test_the_bound_covers_the_true_error_on_every_rectangle_level(which):
    """`notes/moler_payne_vs_tension.py bound`, now through `chladni.integrals` with a Rellich
    norm instead of an interior rule: five eigenvalues of `rect(1, 1/phi)`, each refined on the
    pencil, the bound held against the exact eigenvalue. Valid on every row; within 2-6x where
    the penalty sits near rounding, as the note measured."""
    case = _rect_case()
    p0 = case.pencil
    n = p0.n_cols
    p, prec = {
        "default": (MPSPencil(p0.bdry, p0.interior, prec=1e-8), 1e-8),
        "tau 2e-14": (MPSPencil(p0.bdry, p0.interior, reg="tikhonov_qr", prec=1e-12,
                                rtol=2e-14 / np.sqrt(n)), 1e-12),
        "threshold": (p0, 1e-12),
    }[which]
    for lam0 in (float(t) for t in case.truth[[0, 1, 2, 4, 7]]):
        lo, hi = lam0 * (1 - 1e-6), lam0 * (1 + 1.3e-6)
        obj = Objective(lambda z: p.tensions(float(z)), lo, hi, prec=prec, variable="lam")
        r = refine(obj, lo, hi, prec=prec)
        mp = _bound(p, r.lam, p.coef(r.lam)[:, 0], case.domain)
        err = abs(r.lam - lam0)
        assert mp.reason == ""
        assert mp.abs[0] >= err, (lam0, mp.abs[0], err)
        if which == "tau 2e-14":
            assert mp.abs[0] <= 10 * max(err, 1e-15 * lam0), (lam0, mp.abs[0], err)


# --- a singular corner, at known offsets ----------------------------------------------------------

@pytest.mark.parametrize("index", [1, 3])
@pytest.mark.parametrize("offset", [1e-11, 1e-8, 1e-5])
def test_the_bound_covers_a_KNOWN_offset_on_the_L_shape_and_tightly(index, offset):
    """A candidate taken at `lam = lam_k (1 + offset)` is off by exactly that much, which is a
    ground truth no solver output gives. The first L-shape eigenfunction is singular at the
    re-entrant corner, the third smooth there. The bound must cover the offset and not exceed it
    by more than a modest factor -- here 30, where lappy's sampled maximum and the inclusion's
    own slack both live."""
    from hard_roster import solver_for

    dom = L_shape()
    lam0 = float(ref.L_shape_eigs(index)[index - 1])
    p = solver_for(dom, lam0 * 1.3, label="L", prec=1e-10).pencil
    lam = lam0 * (1 + offset)
    mp = _bound(p, lam, p.coef(lam)[:, 0], dom)
    err = abs(lam - lam0)
    assert mp.abs[0] >= err, (mp.abs[0], err)
    assert mp.abs[0] <= 30 * err, (mp.abs[0], err)
    assert mp.cancellation[0] < 10


def test_at_the_eigenvalue_the_bound_reaches_the_rounding_floor_and_no_further():
    """At the tabulated eigenvalue (14 digits) the candidate's boundary values are rounding, and
    the bound must say so rather than claim zero: it sits at a few times `eps * cancellation`,
    and it still covers the table's own uncertainty."""
    from hard_roster import solver_for

    dom = L_shape()
    lam0 = float(ref.L_shape_eigs(1)[0])
    p = solver_for(dom, lam0 * 1.3, label="L", prec=1e-10).pencil
    mp = _bound(p, lam0, p.coef(lam0)[:, 0], dom)
    floor = np.finfo(float).eps * mp.cancellation[0]
    assert 1e-16 < mp.rel[0] < 1e-12
    assert mp.rel[0] >= floor


# --- what it refuses ------------------------------------------------------------------------------

def test_a_neumann_boundary_is_refused_with_a_reason():
    """The inclusion rests on `u - v` vanishing on the whole boundary. Not Dirichlet: no bound."""
    dom = rect(1.0, 1.0, bc="neu")
    from chladni.helmholtz import PlaneWave

    fam = PlaneWave.uniform(6, radius=dom.diameter)
    iq = P.integral_quad(dom, 20.0, prec=1e-8)
    c = np.zeros(len(fam))
    c[0] = 1.0
    cd = C.cauchy_data(fam, c, 20.0, iq.quad)
    mp = M.moler_payne(cd, lambda z: (fam.values(np.sqrt(20.0), z) @ c[:, None],) * 2, dom,
                       [iq.x0])
    assert np.all(np.isinf(mp.abs)) and "Dirichlet" in mp.reason


def test_a_candidate_far_from_any_eigenvalue_gets_no_finite_bound():
    """`eps >= 1` says nothing, and must be reported as nothing."""
    from hard_roster import solver_for

    dom = L_shape()
    lam = 12.0                                   # between lam_1 = 9.64 and lam_2 = 15.2
    p = solver_for(dom, 20.0, label="L", prec=1e-10).pencil
    mp = _bound(p, lam, p.coef(lam)[:, 0], dom)
    assert mp.eps[0] > 1e-3
    assert mp.abs[0] >= min(abs(lam - float(e)) for e in ref.L_shape_eigs(3))


# --- wired into the solver ------------------------------------------------------------------------

def _l_solver(certify=True):
    from hard_roster import recipe_basis
    from chladni.mps import MPSEigensolver

    dom = L_shape()
    b = recipe_basis(dom, 80)
    n = len(b)
    return MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=1e-10,
                          rng=np.random.default_rng(0), certify=certify)


def test_every_level_of_a_dirichlet_solve_carries_a_bound_that_covers_its_error():
    """The end-to-end contract: `solve_interval` on the L-shape, five levels against the 14-digit
    table. Each level's Moler-Payne bound is finite, covers the true error, and is what the
    Spectrum reports wherever it is tighter than the tension's bar."""
    s = _l_solver()
    res = s.search_interval(8.0, 33.0)
    truth = ref.L_shape_eigs(6)
    assert len(res.levels) == 5
    for L in res.levels:
        true = float(truth[np.argmin(np.abs(truth - L.lam))])
        assert L.bound_reason == "" and np.isfinite(L.mp_bound), L
        assert L.mp_bound >= abs(L.lam - true) - 1e-14 * true, (L.lam, L.mp_bound, true)
        assert L.bound == min(L.delta_res, L.mp_bound)
    spec = s.solve_interval(8.0, 33.0)
    assert np.allclose(spec.bound, [L.bound for L in sorted(res.levels, key=lambda x: x.lam)])


def test_attained_precision_reads_the_tighter_bound():
    """`Level.attained_prec` and the diagnostics' `unmet` follow the reported bound, so a level
    the tension's bar called short but the bound certifies is no longer reported short."""
    s = _l_solver()
    res = s.search_interval(8.0, 33.0)
    for L in res.levels:
        assert L.attained_prec == pytest.approx(L.bound / L.lam)
    assert res.diagnostics.attained_prec == pytest.approx(max(L.attained_prec for L in res.levels))


def test_certify_false_leaves_the_tension_bar_alone():
    s = _l_solver(certify=False)
    res = s.search_interval(8.0, 33.0)
    assert all(np.isnan(L.mp_bound) and L.bound == L.delta_res for L in res.levels)


def test_a_neumann_solve_says_why_it_has_no_bound():
    from chladni.helmholtz import PlaneWave
    from chladni.mps import MPSEigensolver

    dom = rect(1.0, 0.7, bc="neu")
    b = PlaneWave.uniform(14, radius=dom.diameter)
    s = MPSEigensolver(dom, basis=b, n_bdry=3 * len(b), n_int=len(b), prec=1e-8,
                       rng=np.random.default_rng(0))
    res = s.search_interval(5.0, 30.0)
    assert res.levels and all("Dirichlet" in L.bound_reason for L in res.levels)
    assert all(L.bound == L.delta_res for L in res.levels)
