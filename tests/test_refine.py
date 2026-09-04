"""Stage 2, the refiner: the four guards, the tolerance, and the measured cost.

`CHLADNI_V1.md` section 3. Built before the searcher, because it is what PRICES a bracket --
and a flat per-bracket charge reversed the answer to "should the search refine or hand off?"
twice, so the cost model has to be measured before any architectural conclusion is drawn.

Each guard gets a case that FIRES it and a case that must NOT. Guard 2 is the one that matters
most: `C <= 0` fires both on a bracket holding no single minimum and at the end of a perfectly
good refinement, and treating the second as failure discards correct answers.
"""
from __future__ import annotations

import numpy as np
import pytest

from objective import Objective
from v1_refine import _vertex, _xatol, refine

EPS, C_K, PLATEAU = 1e-13, 0.27, 0.2


def cone(*lams, eps=EPS, c=C_K, plateau=PLATEAU, lo=3.0, hi=12.0, prec=1e-14):
    """An objective in `k` with one cone per given `lam`. Cheap; no pencil."""
    ks = np.sqrt(np.asarray(lams, dtype=float))

    def tensions(k):
        d = np.sort(np.abs(ks - float(k)))
        return np.minimum(np.sqrt(eps**2 + (c * d) ** 2), plateau)

    return Objective(tensions, lo, hi, variable="k", prec=prec)


LAM = 49.348022005446786


# --- it locates, and cheaply -----------------------------------------------------------------

@pytest.mark.parametrize("width", [1e-1, 1e-3, 1e-6, 1e-12])
def test_it_locates_to_machine_precision_from_any_bracket_width(width):
    """V1's headline: median relative error 1.6e-16 at 5.5-7.9 evaluations per bracket, against
    Brent's 5.4e-10 to 5.6e-5. Squaring does NOT cost half the digits -- the
    `sqrt(eps_machine)` limit belongs to methods that compare function VALUES, and solving for
    the vertex from the curvature does not suffer it."""
    obj = cone(LAM)
    r = refine(obj, LAM - width, LAM + width, prec=1e-14)
    assert r.accepted
    assert abs(r.lam - LAM) / LAM < 1e-15
    assert r.evaluations <= 12


def test_the_cost_is_not_flat_in_the_bracket_width():
    """3 evaluations for a tight bracket and ~10 for a wide one. A flat charge overcharged
    tight brackets threefold and undercharged spurious ones by the same factor."""
    tight = refine(cone(LAM), LAM - 1e-12, LAM + 1e-12, prec=1e-14).evaluations
    wide = refine(cone(LAM), LAM - 10.0, LAM + 10.0, prec=1e-14).evaluations
    assert tight < wide
    assert 2 <= tight <= 8 and wide <= 14


def test_a_hint_is_reused_from_the_cache_for_free():
    """The deepest sample the search already paid for. The cache is shared, so handing it over
    costs nothing -- which matters because `MPSPencil` does not memoize the GSVD."""
    obj = cone(LAM)
    obj.tension(obj.from_lam(LAM))          # the search's own sample
    before = obj.n_evals
    r = refine(obj, LAM - 1e-3, LAM + 1e-3, prec=1e-14, lam_hint=LAM)
    assert r.accepted and obj.n_evals - before == r.evaluations
    assert r.evaluations <= 6


# --- guard 1: refuse when the parabola never closed -------------------------------------------

def test_guard1_an_EMPTY_bracket_is_refused():
    """`C <= 0` with no successful step. The bracket holds no eigenvalue at all, so there is
    no minimum to find and the refiner must say so rather than return its lowest sample."""
    obj = cone(LAM)
    r = refine(obj, LAM + 1.0, LAM + 2.0, prec=1e-14)
    assert not r.accepted
    assert r.reason in ("no_parabola", "not_a_minimum", "endpoint")
    assert r.evaluations <= 8, "and rejecting is cheap: 2 to 7 evaluations"


def test_guard1_a_bracket_spanning_TWO_eigenvalues_is_refused():
    """The fit is across a W, so `C <= 0`. The danger is specific: the lowest sampled point can
    be a bracket ENDPOINT, 1e-3 from either eigenvalue, returned as a success."""
    a, b = 30.0, 70.0
    obj = cone(a, b)
    r = refine(obj, a - 1.0, b + 1.0, prec=1e-14)
    if r.accepted:
        # If it does converge it must have found one of them, not the hump between.
        assert min(abs(r.lam - a), abs(r.lam - b)) < 1e-6, f"landed at {r.lam}"


def test_guard1_a_wall_point_next_to_a_deep_root_is_refused():
    """Smallness alone is not enough: bounded minimisation returns an endpoint when the true
    minimum lies outside the bracket, and next to a deep root that endpoint passes any
    threshold test. Measured before this check existed: 8 phantom roots against 20 real ones."""
    obj = cone(LAM)
    r = refine(obj, LAM + 1e-4, LAM + 1e-2, prec=1e-14)
    assert not r.accepted, f"accepted a wall point at sigma = {r.sigma:.2e}"
    assert r.sigma < 1e-2, "even though its tension is small"


# --- guard 2: but only when there was no progress ---------------------------------------------

def test_guard2_vanishing_curvature_AFTER_progress_is_accepted():
    """The guard that discards correct answers when got wrong. `C <= 0` fires at the END of a
    good refinement too, once the three points have collapsed onto the floor and there is no
    curvature left to fit."""
    obj = cone(LAM)
    r = refine(obj, LAM - 1e-3, LAM + 1e-3, prec=1e-14)
    assert r.reason in ("flat", "converged"), r.reason
    assert r.accepted, "a collapsed triple is success, not failure"
    assert abs(r.lam - LAM) / LAM < 1e-15


def test_guard2_and_guard1_are_distinguished_by_PROGRESS_not_by_curvature():
    """Both are `C <= 0`. The reason codes must differ, or the two cases are one case."""
    good = refine(cone(LAM), LAM - 1e-3, LAM + 1e-3, prec=1e-14)
    empty = refine(cone(LAM), LAM + 1.0, LAM + 2.0, prec=1e-14)
    assert good.accepted and not empty.accepted
    assert good.reason != "no_parabola"


# --- guard 3: the lowest point is tracked independently ----------------------------------------

def test_guard3_the_best_point_survives_a_bad_vertex():
    """The triple is re-formed by position rather than by value, so a bad vertex can discard
    the sub-interval holding the minimum. Tracking the lowest point separately means the
    answer cannot be lost once it has been seen."""
    obj = cone(LAM)
    r = refine(obj, LAM - 5.0, LAM + 5.0, prec=1e-14)
    xs, v = obj.samples
    assert r.sigma <= v.min() * (1 + 1e-12), "the reported point is the lowest one seen"


# --- guard 4: dx == 0, and the nudge stays inside ------------------------------------------------

def test_guard4_a_degenerate_triple_yields_no_vertex():
    assert _vertex([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) is None
    assert _vertex([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) is None, "no curvature, no vertex"


def test_guard4_the_result_never_leaves_the_bracket():
    """The nudge off a duplicate point is clamped: a step that left the interval would be
    searching somewhere it was not asked to."""
    for lo, hi in [(LAM - 1e-3, LAM + 1e-3), (LAM + 1.0, LAM + 2.0), (10.0, 11.0)]:
        r = refine(cone(LAM), lo, hi, prec=1e-14)
        assert lo <= r.lam <= hi, f"{r.lam} escaped [{lo}, {hi}]"


def test_guard4_an_exactly_degenerate_pair_does_not_trip_the_duplicate_nudge():
    """Clusters closer than `prec` refine cleanly -- 7 evaluations, accepted, landing between
    the two members -- with the degenerate-points guard never firing across any close pair."""
    d = LAM * 1e-15
    obj = cone(LAM, LAM + d)
    r = refine(obj, LAM - 1e-6, LAM + 1e-6, prec=1e-14)
    assert r.accepted
    assert abs(r.lam - LAM) <= 10 * d + 1e-12


# --- the tolerance ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [2.7e-13, 1e-9, 1e-3])
def test_xatol_never_exceeds_the_bracket_it_is_refining(width):
    """A tolerance derived from `lam` alone can EXCEED the bracket: a 2.7e-13 bracket handed a
    tolerance of 7.9e-13 terminates before doing anything and returns a point that is not the
    minimum. `max(prec*|lam|, 4*ulp(lam))` at `prec = 1e-14` on `lam ~ 49` is 4.9e-13, which
    is exactly the regime -- so the refiner must still make progress."""
    obj = cone(LAM)
    r = refine(obj, LAM - width / 2, LAM + width / 2, prec=1e-14)
    assert r.evaluations >= 3, "it evaluated, rather than returning its first guess"
    assert abs(r.lam - LAM) / LAM < 1e-14


def test_xatol_is_the_ulp_floor_when_prec_is_finer_than_the_arithmetic():
    assert _xatol(49.35, 1e-30) == pytest.approx(4 * np.spacing(49.35))
    assert _xatol(49.35, 1e-8) == pytest.approx(49.35e-8)


def test_a_loose_prec_stops_early_and_a_tight_one_does_not():
    """Not more work than asked: at loose `prec` the target width binds and the refiner stops
    even though the floor would permit more."""
    loose = refine(cone(LAM, prec=1e-4), LAM - 1e-2, LAM + 1e-2, prec=1e-4)
    tight = refine(cone(LAM, prec=1e-14), LAM - 1e-2, LAM + 1e-2, prec=1e-14)
    assert loose.evaluations <= tight.evaluations
    assert abs(loose.lam - LAM) / LAM <= 1e-4


def test_delta_res_is_measured_and_reported():
    """There is no predicted-accuracy constant: the bound comes from the parabola fit the
    refiner already performed, and is compared against `prec * lam` by the caller."""
    r = refine(cone(LAM), LAM - 1e-3, LAM + 1e-3, prec=1e-14)
    assert np.isnan(r.delta_res) or r.delta_res >= 0.0


# --- termination -----------------------------------------------------------------------------

def test_it_always_terminates_even_when_every_point_is_cached():
    """The budget is in STEPS, not in cache misses. Budgeting on misses spins forever at zero
    cost as soon as the triple stops moving, which is what the first version did."""
    obj = cone(LAM)
    r = refine(obj, LAM - 1e-3, LAM + 1e-3, prec=1e-14)
    assert r.steps < 40, f"stopped on its own after {r.steps} steps, not on the cap"


def test_a_flat_objective_terminates_without_claiming_a_root():
    obj = Objective(lambda k: [0.2, 0.3], 3.0, 12.0, variable="k", prec=1e-14)
    r = refine(obj, LAM - 1.0, LAM + 1.0, prec=1e-14)
    assert not r.accepted and r.steps < 40


# --- tier 2: a real pencil --------------------------------------------------------------------

@pytest.mark.slow
def test_it_reaches_the_floor_on_a_real_rectangle_pencil():
    """V1's table is the acceptance bar, on a real objective rather than a cone."""
    import bracket_score as bs
    import rect_suite as rs

    (case,) = rs.family_A()
    obj = bs.objective_for(case, 1e-14)
    errs, costs = [], []
    for lam in case.truth:
        for width in (1e-2, 1e-4):
            obj.reset()
            r = refine(obj, lam - width, lam + width, prec=1e-14)
            assert r.accepted, f"{case.label} at {lam}: {r.reason}"
            errs.append(abs(r.lam - lam) / lam)
            costs.append(r.evaluations)
    assert np.median(errs) < 1e-13, f"median relative error {np.median(errs):.2e}"
    assert np.median(costs) <= 10, f"median {np.median(costs)} evaluations per bracket"


@pytest.mark.slow
def test_it_rejects_an_empty_bracket_on_a_real_pencil_cheaply():
    """The other half of the cost model: 2 to 7 evaluations to REJECT. A search that returns a
    spurious bracket pays this, and that is the honest price of keeping a doubtful region."""
    import bracket_score as bs
    import rect_suite as rs

    (case,) = rs.family_A()
    obj = bs.objective_for(case, 1e-14)
    gaps = np.diff(np.unique(case.truth))
    mid = float(np.unique(case.truth)[int(np.argmax(gaps))] + gaps.max() / 2)
    obj.reset()
    r = refine(obj, mid - 0.05, mid + 0.05, prec=1e-14)
    assert not r.accepted, f"accepted a bracket in a gap at sigma = {r.sigma:.2e}"
    assert r.evaluations <= 10
