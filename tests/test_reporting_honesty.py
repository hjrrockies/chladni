"""What a solve REPORTS agrees with what it computed.  (2026-11-03)

Regression tests for the claim defects of `notes/accuracy_audit_2026-11-03.md`, section 1: a
refused Moler-Payne bound read as met, a margin that contradicted `met`, count doubts dropped
on the warm path, a refusal that raised `NameError`, tolerances with an absolute floor that made
them scale-dependent, and the kernel crashing at its own vertex.
"""
import warnings

import numpy as np
import pytest

from chladni import eigensolver
from chladni._core import PrecisionShortfall
from chladni.eigenproblem import Eigenproblem, _record
from chladni.geometry import L_shape, Polygon, rect
from chladni.helmholtz import FourierBessel
from chladni.mps import bracket, search
from chladni.mps.refine import DELTA_SAFETY
from chladni.mps.search import Level, SolveDiagnostics, SolveResult


# --- 1.1 / 1.4: the bar, `met` and `margin` tell one story -------------------------------------

def test_a_refused_moler_payne_bound_is_not_met():
    """`mp_bound = inf` is the worst Moler-Payne outcome; `delta_res` was not admitted, so it
    cannot stand in for the bound."""
    L = Level(lam=10.0, mult=1, delta_res=5e-10, prec=1e-10, mp_bound=np.inf,
              delta_res_admitted=False, bound_reason="eps >= 1")
    assert not np.isfinite(L.bound)
    assert not L.met(1e-10)
    assert L.count_shortfall


@pytest.mark.parametrize("delta_res, mp_bound, admitted", [
    (5e-10, 5e-8, False),       # failed the cross-check: MP is the bar
    (5e-10, np.inf, False),     # MP refused
    (5e-10, 4e-10, True),       # both, MP tighter
    (5e-10, 1e-8, True),        # both, delta_res tighter
    (5e-10, np.nan, True),      # no MP computed
    (3e-9, 5e-10, False),       # delta_res missed, MP met
])
def test_margin_and_met_agree_on_every_path(delta_res, mp_bound, admitted):
    """`met <=> margin >= DELTA_SAFETY`, whichever bar the level reports."""
    L = Level(lam=10.0, mult=1, delta_res=delta_res, prec=1e-10, mp_bound=mp_bound,
              delta_res_admitted=admitted)
    m = L.margin
    assert L.met(1e-10) == (np.isfinite(m) and m >= DELTA_SAFETY), (L.bound, m)


def _result(levels, **diag):
    return SolveResult(levels=tuple(levels), diagnostics=SolveDiagnostics(**diag))


def test_a_refused_bound_is_flagged_in_the_record():
    L = Level(lam=10.0, mult=1, delta_res=5e-10, prec=1e-10, mp_bound=np.inf,
              delta_res_admitted=False, bound_reason="the norm under-estimate is zero")
    spec = _record(_result([L]), 1, 1e-10, None, [])
    assert any("Moler-Payne" in f and "refused" in f for f in spec.flags), spec.flags


# --- 1.2: count doubts survive the warm path ----------------------------------------------------

def test_an_uncertain_level_is_flagged_even_when_the_diagnostics_did_not_count_it():
    L = Level(lam=10.0, mult=2, delta_res=1e-12, prec=1e-10, count_uncertain=True,
              count_alt=1, uncertain_reason="reach and gap disagree")
    spec = _record(_result([L]), 2, 1e-10, None, [])
    assert any(f.startswith("uncertain") for f in spec.flags), spec.flags


@pytest.mark.filterwarnings("ignore:FourierBessel can deliver")   # removed in Phase 3
def test_track_levels_keeps_the_count_doubts_of_a_merged_window(monkeypatch):
    s = Eigenproblem(rect(1.0, 1.1)).solve(3, prec=1e-8).solver
    doubt = (10.0, 2, 1, "reach and gap disagree")

    def fake_search(lo, hi, prec=None, **kw):
        lam = 0.5 * (lo + hi)
        return _result([Level(lam=lam, mult=2, delta_res=1e-12 * lam, prec=prec,
                              count_uncertain=True, count_alt=1)],
                       n_uncertain=1, uncertain=(doubt,), rejected_deep=((lam, 1e-3, "x"),),
                       n_ambiguous=1, evaluations=7)
    monkeypatch.setitem(vars(s), "search_interval", fake_search)      # the solver is frozen
    monkeypatch.setitem(vars(s), "certify", False)
    prev = np.array([20.0, 20.001])
    res, why = s.track_levels(prev, np.array([0.01, 0.01]), mults=[1, 1], prec=1e-8)
    assert why is None
    d = res.diagnostics
    assert d.n_uncertain == 1 and d.uncertain == (doubt,)
    assert len(d.rejected_deep) == 1 and d.n_ambiguous == 1


# --- 1.3: the documented exception, not NameError -----------------------------------------------

def test_a_declined_shape_raises_precision_shortfall(monkeypatch):
    ep = Eigenproblem(rect(1.0, 1.0))
    declined = type("D", (), {"reason": "the shape is refused"})()
    monkeypatch.setattr(ep, "_declined", lambda n, prec: declined)
    with pytest.raises(PrecisionShortfall, match="refused"):
        ep.solve(3, prec=1e-8, on_shortfall="raise")


# --- 1.6: relative tolerances stay relative at any scale ----------------------------------------

def test_bracket_purity_is_relative_below_k_one():
    prec_u = 1e-8
    hi = 0.5                                     # k < 1: a large domain
    assert bracket.width_is_pure(hi - 0.4 * prec_u * hi, hi, prec_u)
    assert not bracket.width_is_pure(hi - 0.6 * prec_u, hi, prec_u)    # 1.2 prec_u * hi wide


def test_the_merge_licence_is_relative_below_lam_one():
    """Two roots 3x the licence apart at lam = 0.1 are two levels, as they would be at lam = 10."""
    assert search._licence(0.1, 1e-8) == pytest.approx(1e-9)
    assert search._licence(10.0, 1e-8) == pytest.approx(1e-7)


def test_a_small_boundary_with_a_relative_gap_is_not_contiguous():
    from chladni.geometry.segments import LineSegment
    from chladni.geometry.boundary import Boundary
    s = 1e-6
    gap = 5e-3 * s                              # 0.5% of the side
    segs = [LineSegment(0, s), LineSegment(s, s + 1j * s),
            LineSegment(s + 1j * s + gap, 1j * s), LineSegment(1j * s, 0)]
    with pytest.raises(ValueError, match="contiguous"):
        Boundary(segs)


# --- 1.10: the kernel at its own vertex ---------------------------------------------------------

def test_fourier_bessel_at_its_own_vertex_is_exact_and_quiet():
    dom = L_shape()
    c = next(c for c in dom.corners if c.is_reentrant)
    fam = FourierBessel.at_corner(c, n=6, domain=dom)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        V = fam.values(5.0, np.array([c.z]))
    assert np.all(V == 0.0)


# --- 1.11: a trustworthy count is not reported as uncertain -------------------------------------

def test_an_ambiguous_grouping_is_not_an_uncertain_count():
    L = Level(lam=10.0, mult=3, delta_res=1e-12, prec=1e-10, grouping_ambiguous=True)
    S = eigensolver.Level.from_search(L, 1e-10)
    assert S.grouping_ambiguous and not S.count_uncertain


# --- small ones ---------------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:FourierBessel can deliver")   # removed in Phase 3
def test_findability_is_cached_per_probe_count():
    s = Eigenproblem(rect(1.0, 1.1)).solve(2, prec=1e-8).solver
    a = s.findability(10.0, 40.0, dk=0.05, n=3)
    b = s.findability(10.0, 40.0, dk=0.05, n=5)
    assert a is not b


def test_an_unmeasured_level_does_not_report_the_request_as_attained():
    L = Level(lam=10.0, mult=1, prec=1e-10)            # no bar at all
    unmet, attained = search._precision_summary([L], 1e-10)
    assert not attained <= 1e-10
