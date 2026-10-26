"""Phase 5: the black-box path, `Eigenproblem(dom).solve(n, prec)`.  (2026-10-12)

`docs/design/douse_requirements.md`: a Dirichlet polygon in, the first `n` eigenvalues out --
sorted, levels whole at `n`, each with its bound and flags -- with the basis planned by
`chladni.plan`. The planner climbs `LADDER` until every level up to `n + HEADROOM` meets `prec`
and the count audit is ok. Otherwise it returns its best rung FLAGGED, or raises
`PrecisionShortfall` under `on_shortfall="raise"`. Tested against ground truth: the L-shape's
14-digit table and closed forms.
"""
import numpy as np
import pytest

from chladni.eigenproblem import Eigenproblem
from chladni.eigensolver import Spectrum
from chladni.geometry import L_shape, Polygon, eq_tri, rect
from chladni.mps import PrecisionShortfall
from chladni.reference import L_shape_eigs, eq_tri_eigs, rect_eigs

PREC = 1e-10

# Fourier-Bessel's per-column error bound is very pessimistic (`notes/hard_cases.md`, part 3).
pytestmark = pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")


def _check_against(spec, truth, n, prec):
    """Every returned value within its bound of the truth, and the first n right with
    multiplicity, the level at n whole."""
    e = spec.unfolded
    assert e.size >= n
    for x, m, b in zip(spec.lam, spec.mult, spec.bound):
        j = np.argmin(np.abs(truth - x))
        assert abs(truth[j] - x) <= b + 1e-14 * x, (x, truth[j], b)
        assert m == int(np.sum(np.abs(truth - truth[j]) <= 1e-12 * truth[j])), (x, m)


@pytest.mark.slow
def test_the_L_shape_to_1e10_against_its_table():
    spec = Eigenproblem(L_shape()).solve(10, prec=PREC)
    assert isinstance(spec.spectrum, Spectrum) and spec.plan is not None
    _check_against(spec, L_shape_eigs(25), 10, PREC)
    assert spec.achieved and not spec.flags, spec.flags
    assert spec.count_audit is not None and spec.count_audit.verdict == "ok"
    assert len(spec.levels) == len(spec.lam)
    assert all(L.met for L in spec.levels)


@pytest.mark.slow
def test_a_level_straddling_n_comes_back_whole():
    """The square: 2, 5, 5, 8, 10, 10, ... x pi^2. `solve(5)` returns six values -- the double at
    10 pi^2 whole -- never a level split at n."""
    spec = Eigenproblem(rect(1.0, 1.0)).solve(5, prec=PREC)
    assert spec.count == 6 and spec.mult[-1] == 2
    _check_against(spec, np.asarray(rect_eigs(12, 1.0, 1.0), float), 5, PREC)


@pytest.mark.slow
def test_the_equilateral_triangle_counts_its_doubles():
    spec = Eigenproblem(eq_tri()).solve(8, prec=PREC)
    _check_against(spec, np.asarray(eq_tri_eigs(20), float), 8, PREC)


@pytest.mark.slow
def test_a_plan_is_reused_on_a_nearby_shape():
    """douse: plan once, solve every nearby iterate with it."""
    ep = Eigenproblem(L_shape())
    plan = ep.plan(8, prec=PREC)
    moved = Polygon([0, 1.002j, -1 + 1j, -1.001 - 1j, 1 - 0.999j, 1.0], bc="dir")
    spec = Eigenproblem(moved).solve(8, prec=PREC, plan=plan)
    assert spec.plan is plan and spec.count >= 8


@pytest.mark.slow
def test_a_tighter_request_than_the_plan_is_flagged_not_enlarged():
    """Changing prec is an explicit re-plan: the solve reports, it does not enrich silently."""
    ep = Eigenproblem(L_shape())
    plan = ep.plan(8, prec=1e-8)
    spec = ep.solve(8, prec=1e-12, plan=plan)
    assert spec.plan is plan
    assert any("plan" in f for f in spec.flags), spec.flags


@pytest.mark.robust          # three rungs, twice: minutes
def test_a_request_no_rung_accepts_is_flagged_or_raised():
    """1e-14 on the L-shape: the hybrid top rung meets it on the first six levels, but not on
    the HEADROOM above them that the count audit relies on, so no rung is accepted. Default: the
    best rung, FLAGGED -- the six values may be fine, and the flag says the audit could not be
    trusted to show they are complete. On request: `PrecisionShortfall`."""
    ep = Eigenproblem(L_shape())
    spec = ep.solve(6, prec=1e-14)
    assert any(f.startswith("planner:") for f in spec.flags), spec.flags
    with pytest.raises(PrecisionShortfall):
        ep.solve(6, prec=1e-14, on_shortfall="raise")


def test_the_planner_refuses_what_it_does_not_cover():
    from chladni.geometry import disk
    with pytest.raises(NotImplementedError):
        Eigenproblem(disk(1.0)).solve(3)


def test_the_old_spectrum_still_builds():
    s = Spectrum.from_unfolded([1.0, 2.0, 2.0])
    assert s.count == 3 and len(s.levels) == 2 and s.flags == () and s.count_audit is None
    assert all(np.isnan(L.margin) for L in s.levels)             # nothing measured per level


@pytest.mark.slow
def test_a_long_thin_rectangle_reaches_1e10():
    """4:1. No corner is singular, so the plan has to reach its plane-wave rung."""
    spec = Eigenproblem(rect(1.6, 0.4)).solve(3, prec=PREC)
    assert not spec.flags, spec.flags
    _check_against(spec, np.asarray(rect_eigs(12, 1.6, 0.4), float), 3, PREC)


@pytest.mark.slow
def test_a_spectrum_carries_the_searchs_evaluation_counts():
    """What a solve cost, by stage, for a caller (douse) logging it per iterate."""
    spec = Eigenproblem(rect(1.0, 1.0)).solve(2, prec=1e-8)
    d = spec.diagnostics
    assert d is not None and d.evaluations > 0
    assert d.search_evals + d.refine_evals + d.multiplicity_evals <= d.evaluations


@pytest.mark.slow
def test_plan_and_solve_hands_back_the_rung_it_accepted():
    """The planner solved the accepted rung to select it; `plan_and_solve` returns that answer,
    and it is the one `solve(plan=plan)` recomputes (same basis, same domain, deterministic)."""
    ep = Eigenproblem(L_shape())
    plan, rec = ep.plan_and_solve(3, prec=1e-8)
    again = ep.solve(3, prec=1e-8, plan=plan)
    np.testing.assert_array_equal(rec.unfolded, again.unfolded)
    assert rec.flags == again.flags and rec.plan is plan and rec.diagnostics is not None


# --- the best-rung fallback: which flagged answer comes back (2026-10-26) ----------------------
#
# `notes/triangle_apex_planner.md`: on the triangle 0, 1, 0.1+0.25i every rung missed 1e-10, all
# tied at "0 met", and the first rung -- the poorest basis, values off by 2e-5 -- was returned
# while rungs 1-2 held values right to ~1e-10. Two defects: an early-aborted rung always scored
# 0 (`located` arrives as floats, which the counter could not read), and a tie kept the first.

def _aborted(margin, located):
    from chladni.mps.search import Level
    stop = Level(lam=float(located[-1]), mult=1, delta_res=1.0, prec=PREC)
    return PrecisionShortfall("stopped early", level=stop, margin=margin, located=located)


def test_an_aborted_rung_counts_the_values_it_met_before_stopping():
    from chladni.eigenproblem import HEADROOM, _rank
    met, _ = _rank(_aborted(0.5, [10.0, 20.0, 30.0]), 15, PREC)
    assert met == pytest.approx(2 / (15 + HEADROOM))     # the last is the one that stopped it


def test_a_tie_on_levels_met_goes_to_the_larger_margin():
    from chladni.eigenproblem import _rank
    poor, rich = _rank(_aborted(0.046, [289.0]), 15, PREC), _rank(_aborted(0.13, [289.0]), 15, PREC)
    assert poor[0] == rich[0] and rich > poor


def test_more_levels_met_still_outranks_a_larger_margin():
    from chladni.eigenproblem import _rank
    assert _rank(_aborted(0.01, [1.0, 2.0, 3.0]), 15, PREC) > _rank(_aborted(1.9, [1.0]), 15, PREC)


def test_the_fallback_returns_the_best_ranked_rung_not_the_first(monkeypatch):
    """No rung accepted, every rung aborted at 0 met: the rung with the largest stopping
    margin is the one solved in full and returned."""
    from chladni.plan import LADDER
    margins = {0: 0.046, 1: 0.13, 2: 0.16}
    full = []

    def fake(self, plan, n, prec, on_shortfall, solver=None):
        from types import SimpleNamespace
        if on_shortfall == "raise" and plan.rung in margins:
            raise _aborted(margins[plan.rung], [289.0])
        if on_shortfall == "flag":
            full.append(plan.rung)
        declined = SimpleNamespace(declined=True, n_uncertain=0, count_audit=None)
        return SimpleNamespace(levels=(), diagnostics=declined), (0.0, 1.0)

    monkeypatch.setattr(Eigenproblem, "_solve_with", fake)
    dom = Polygon(np.array([0, 1, 0.1 + 0.25j]))
    plan, _, _, accepted = Eigenproblem(dom)._plan(15, PREC, False)
    assert not accepted and plan.rung == 2 and full == [2]
    assert len(LADDER) > 2
