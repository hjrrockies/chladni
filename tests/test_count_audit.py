"""Phase 4: the count audit -- a dropped eigenvalue, which no returned level can flag.  (2026-10-11)

`asymp.count_audit` compares the corner-corrected Weyl count against the returned eigenvalues at
every mid-gap cut, and takes the MEAN deficit. Calibrated and validated in
`notes/count_audit.md`: on 819 real solves labelled by finite-element counts it caught 100 of 100
drops, including a drop balanced by a duplicate, with no false alarm in 716 complete solves.
Douse's guard today takes the WORST cut instead, which gave 20 false alarms. Across all 18 shapes
with known spectra the thresholds clear every complete spectrum; the cost is that a single drop
on a shallow spectrum is caught ~94% of the time, and less deep in the spectrum. A backstop.

    short      mean deficit > 0.25
    excess     mean deficit < -0.6
    (complete spectra, 18 shapes: [-0.48, 0.23]; douse benchmark drops >= 0.47)
    unaudited  not a prefix window, a non-Dirichlet boundary, or nothing to cut

Prefix windows only: Weyl counts from 0, so the window must start below the Faber-Krahn bound on
lam_1. Without a mid-gap top, the window's top is a coarse check only (+-1.5): a top that is not
in a gap is one unaveraged reading of Weyl's remainder, [-1.57, 1.84] on complete spectra.
"""
import numpy as np
import pytest

from chladni.asymp import count_audit
from chladni.geometry import L_shape, rect
from chladni.reference import L_shape_eigs

N = 20
EIGS = L_shape_eigs(N + 1)            # with multiplicity; 49.348 and 98.696 are doubles
TOP = 0.5 * (EIGS[N - 1] + EIGS[N])   # a mid-gap cut above the first N (EIGS[N-1] != EIGS[N])
LO = 5.0                              # below the Faber-Krahn bound on the L-shape, ~6.06


@pytest.fixture(scope="module")
def dom():
    return L_shape()


def _levels(e):
    """Distinct values, so a test can drop or duplicate a whole LEVEL."""
    return np.unique(np.round(e, 9))


# --- the verdicts, on the L-shape's 14-digit table ------------------------------------------

def test_a_complete_spectrum_is_ok(dom):
    a = count_audit(EIGS[:N], dom, lo=LO, top=TOP, top_is_gap=True)
    assert a.verdict == "ok", a
    assert abs(a.statistic) < 0.1


def test_most_single_drops_with_headroom_are_short(dom):
    """The calibrated rate, not a per-drop guarantee: on spectra of <= 30 levels, ~94% of single
    drops with four levels above them are caught (`notes/count_audit.md`). On the L-shape's first
    20, every level with at least four above it."""
    lv = _levels(EIGS[:N])
    caught = []
    for x in lv[:-4]:
        e = EIGS[:N][np.abs(EIGS[:N] - x) > 1e-9]
        caught.append(count_audit(e, dom, lo=LO, top=TOP, top_is_gap=True).verdict == "short")
    assert np.mean(caught) >= 0.9, caught


@pytest.mark.parametrize("shape", ["rect_5", "ngon8", "L_shape", "eq_tri"])
def test_no_complete_prefix_alarms(shape):
    """The thresholds clear every complete spectrum measured, including the extremes that set
    them: a 5:1 rectangle reaches +0.23 at 10 levels, an 8-gon cut at 6 levels -0.48."""
    from chladni import geometry as G
    from chladni import reference as R

    dom, e = {"rect_5": (G.rect(1.0, 0.2), R.rect_eigs(40, 1.0, 0.2)),
              "ngon8": (G.reg_ngon(8), R.reg_ngon_eigs(10, 8)),
              "L_shape": (G.L_shape(), R.L_shape_eigs(25)),
              "eq_tri": (G.eq_tri(), R.eq_tri_eigs(40))}[shape]
    e = np.sort(np.asarray(e, float))
    lv = _levels(e)
    # Below Faber-Krahn, not merely below lam_1: on the 5:1 rectangle lam_1 = 257 but the bound is
    # 91, so a window from lam_1 / 2 cannot be certified a prefix and is (rightly) unaudited.
    from chladni.bounds import faber_krahn
    lo = 0.5 * faber_krahn(dom)
    bad = []
    for n in range(6, len(lv)):
        sub = e[e <= lv[n - 1] * (1 + 1e-9)]
        a = count_audit(sub, dom, lo=lo, top=0.5 * (lv[n - 1] + lv[n]), top_is_gap=True)
        if a.verdict != "ok":
            bad.append((n, a.verdict, round(a.statistic, 3)))
    assert not bad, f"{shape}: complete prefixes that alarm: {bad}"


def test_a_duplicated_eigenvalue_is_excess(dom):
    e = np.sort(np.append(EIGS[:N], EIGS[2] * (1 + 1e-5)))    # one eigenvalue returned twice
    a = count_audit(e, dom, lo=LO, top=TOP, top_is_gap=True)
    assert a.verdict == "excess", a


def test_a_drop_balanced_by_a_duplicate_is_still_short(dom):
    """The total is right, the spectrum is not: douse's T2.1, and T2.2 on the benchmark (dropped
    118.38, returned 284.12 twice). A total-count check cannot see it; the mean deficit can,
    because the drop moves every cut above it and the duplicate only those above IT."""
    e = EIGS[:N][np.abs(EIGS[:N] - EIGS[1]) > 1e-9]
    e = np.sort(np.append(e, EIGS[N - 2] * (1 + 1e-5)))
    assert e.size == N
    a = count_audit(e, dom, lo=LO, top=TOP, top_is_gap=True)
    assert a.verdict == "short", a


def test_the_step_is_located(dom):
    """Where the deficit first reaches half an eigenvalue: the place to enrich. A pointer, not a
    certificate -- Weyl's remainder moves it by a level or so."""
    e = EIGS[:N][np.abs(EIGS[:N] - EIGS[3]) > 1e-9]
    a = count_audit(e, dom, lo=LO, top=TOP, top_is_gap=True)
    assert EIGS[1] <= a.step_at <= EIGS[6], a


# --- what it will not audit, and the coarse top --------------------------------------------

def test_a_window_that_does_not_start_below_lam_1_is_unaudited(dom):
    a = count_audit(EIGS[3:N], dom, lo=20.0, top=TOP, top_is_gap=True)
    assert a.verdict == "unaudited" and "prefix" in a.reason


def test_a_non_dirichlet_boundary_is_unaudited():
    """The corner sum and the calibration are Dirichlet's."""
    a = count_audit([1.0, 2.0, 3.0], rect(1.0, 0.7, bc="neu"), lo=0.5)
    assert a.verdict == "unaudited" and "Dirichlet" in a.reason


def test_nothing_to_cut_is_unaudited(dom):
    a = count_audit([EIGS[0]], dom, lo=LO)
    assert a.verdict == "unaudited"


def test_an_empty_window_with_a_top_is_short(dom):
    """A basis that finds nothing at all -- the starved L-shape at n = 8 returns no level in
    [5, 40] -- is caught by the coarse top check, which needs no mid-gap."""
    a = count_audit([], dom, lo=LO, top=40.0)
    assert a.verdict == "short", a


def test_a_top_that_is_not_in_a_gap_does_not_raise_a_false_alarm(dom):
    """Just below an eigenvalue the deficit there reads half an eigenvalue high, just above it
    half low, on top of Weyl's remainder. The coarse check (+-2.5) tolerates both; the mean over
    the gaps decides."""
    for top in (EIGS[N - 1] * (1 - 1e-9), EIGS[N - 1] * (1 + 1e-9)):
        e = EIGS[EIGS <= top]
        a = count_audit(e, dom, lo=LO, top=top)
        assert a.verdict == "ok", (top, a)


# --- wired into the solver ------------------------------------------------------------------

def _starved_lshape(n):
    from chladni.helmholtz import FourierBessel
    from chladni.mps import MPSEigensolver

    dom = L_shape()
    corner = max(dom.corners, key=lambda c: c.omega)
    fb = FourierBessel.at_corner(corner, n=n, domain=dom)
    return MPSEigensolver(dom, basis=fb, n_bdry=3 * n, n_int=n, prec=1e-10, rng=0)


@pytest.mark.slow
def test_search_interval_carries_the_audit():
    res = _starved_lshape(16).search_interval(LO, 40.0)
    a = res.diagnostics.count_audit
    assert a is not None and a.verdict == "ok", a       # all five levels in [5, 40] found


@pytest.mark.slow
def test_search_interval_audits_a_window_the_basis_could_not_see():
    """n = 8 returns nothing in [5, 40] (`tests/test_level_resolution.py`): a whole-window drop."""
    res = _starved_lshape(8).search_interval(LO, 40.0)
    assert not res.levels
    assert res.diagnostics.count_audit.verdict == "short"


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
def test_a_benchmark_drop_is_caught():
    """ngon12 on the stand-in basis D at 1e-10 drops whole doubles (`notes/phase3_rescore.md`):
    no returned level is near them, so only the audit sees it."""
    from test_level_resolution import _bench_solve

    res, ref = _bench_solve("T2.9-ngon12")
    n_ret = sum(L.mult for L in res.levels)
    assert n_ret < ref["fem_count_below_hi"], "the case still drops: a precondition of the test"
    assert res.diagnostics.count_audit.verdict == "short", res.diagnostics.count_audit
