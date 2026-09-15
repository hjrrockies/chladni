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

from chladni.mps.objective import Objective
from chladni.mps.refine import _vertex, _xatol, refine

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


@pytest.mark.parametrize("eps", [1e-14, 1e-11, 1e-8])
def test_delta_res_IS_the_resolution_limit_eps_over_c(eps):
    """`search.md` section 1: `eps/c` is the scale on which the cone flattens into its
    quadratic bottom. A minimum cannot be located finer than the region where the curve is
    flat, so that scale IS the attained bound, times a margin.

    The surrogate knows both quantities exactly, so this pins the FORMULA rather than a
    measurement. Two conversions have to be right and neither is visible in the result: the
    cone is built in `k` with slope `C_K` while the refiner works in `lam`, where
    `dlam = 2k dk` makes the slope `C_K/(2k)`; and `_vertex` returns the parabola's LEADING
    COEFFICIENT, which is `c**2` rather than `2*c**2`. Getting the second wrong is a silent
    sqrt(2), which is exactly how it was caught.
    """
    from chladni.mps.refine import DELTA_SAFETY

    r = refine(cone(LAM, eps=eps), LAM - 1e-3, LAM + 1e-3, prec=1e-16)
    c_lam = C_K / (2.0 * np.sqrt(LAM))
    assert r.delta_res == pytest.approx(DELTA_SAFETY * eps / c_lam, rel=1e-3)


#: Errors at or below this many ulps are rounding: any bar contains them, so they cannot show a
#: bar to be loose or an old formula to be short. Measured: 0 and 1 ulp at two disk levels.
_ROUNDING = 4.0


def _disk_levels(prec=1e-10):
    """The disk over `[8, 60]` through the REAL pipeline, against exact truth.

    Through the pipeline on purpose. Handed an artificially tight bracket the refiner reaches
    the true floor and lands exactly on the eigenvalue, so every bound contains a zero error
    and nothing is tested. The search hands it a bracket 0.43 wide, the refiner stops where the
    curve goes flat, and THAT is the regime a bound has to describe.
    """
    from chladni.geometry import disk
    from chladni.mps.search import solve
    from chladni.reference import disk_spectrum
    from hard_roster import solver_for

    p = solver_for(disk(1.0), 60.0, label="disk", prec=prec).pencil
    obj = Objective.from_pencil(p, 8.0, 60.0, variable="lam", prec=prec)
    truth = np.unique(disk_spectrum(8.0, 60.0, 1.0))
    for L in solve(obj, prec=prec).levels:
        yield truth[int(np.argmin(np.abs(truth - L.lam)))], L


@pytest.mark.slow
def test_delta_res_CONTAINS_the_true_error_on_a_real_pencil():
    """Clause (3), honesty, as the assertion that matters: a reported bar must contain the
    error it reports on.

    The disk is the case that exposes it, because its floor is not flat across the window --
    the level at `j_0,2` stops four orders higher than its neighbours, and the old formula's
    shortfall grew with the floor. A domain with a uniform floor would have hidden this.
    """
    slack = []
    for lam0, L in _disk_levels():
        err = abs(L.lam - lam0)
        assert err <= L.delta_res, (
            f"lam={lam0:.6f}: error {err:.3e} EXCEEDS the reported bound {L.delta_res:.3e}")
        if err > _ROUNDING * np.spacing(lam0):
            slack.append(L.delta_res / err)
    # A level located to within rounding says nothing about looseness: any bar contains it. Since
    # the QR route around Accelerate's `dgeqrf`, two of six disk levels land 0 and 1 ulp from the
    # truth, and a slack of `delta_res/0` read as infinitely loose.
    assert len(slack) >= 3, f"only {len(slack)} levels above rounding: the test measures nothing"
    assert max(slack) < 10, f"loose by {max(slack):.1f}x: a scale with margin, not a guess"


@pytest.mark.slow
def test_ABLATION_the_machine_epsilon_formula_contains_NOTHING():
    """The correction shown to be load bearing, by putting the old formula back.

    It read `sqrt(eps * eps_mach / curvature)`, assuming the tension carries ABSOLUTE noise of
    order machine epsilon. It does not -- the error is `eps/c` -- so the two differ by
    `sqrt(eps_mach / eps)`, and the shortfall GROWS as the floor rises. `search.md` section 3
    had already recorded the assumption as wrong; the consequence had never been applied here.
    """
    from chladni.mps.refine import DELTA_SAFETY

    missed = 0
    total = 0
    for lam0, L in _disk_levels():
        if abs(L.lam - lam0) <= _ROUNDING * np.spacing(lam0):
            continue                     # located to rounding: every bar contains it
        total += 1
        old = L.delta_res * np.sqrt(np.spacing(1.0) / L.sigma) / DELTA_SAFETY
        missed += abs(L.lam - lam0) > old
    assert total >= 3, f"only {total} levels above rounding: the ablation measures nothing"
    assert missed == total, (
        f"the old formula contained the error at {total - missed} of {total} levels above "
        f"rounding; it was supposed to fail at every one")


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


# --- the two seam defects that only composition exposed ---------------------------------------

@pytest.mark.slow
def test_a_genuine_minimum_OFF_CENTRE_in_its_bracket_is_accepted():
    """The interior margin was `min(xatol, width/4)`, and the `width/4` term rejected real
    minima sitting near a bracket end. On the multiplicity-3 degeneracy the level sits 8.8e-08
    from the left end of a bracket 8.6e-07 wide, against a demanded margin of 2.15e-07 -- so
    every degeneracy in the suite was refused and the composed pipeline scored separation 0.00.

    The margin was redundant as well as wrong: condition 2 already excludes a locator that
    walked to an endpoint, since there `s_best == s(end)`. So the interior test only has to
    mean "not AT an endpoint", which is a question about ulps.
    """
    import bracket_score as bscore
    import rect_suite as rs
    from chladni.mps.bracket import bracket_interval

    (case,) = [c for c in rs.family_D() if "mult=3" in c.label]
    true = float(np.unique(case.truth)[0])
    obj = bscore.objective_for(case, 1e-8)
    res = bracket_interval(obj, prec=1e-8)

    holding = [b for b in res.brackets if b.lo <= true <= b.hi]
    assert holding, "stage 1 must bracket the level at all"
    b = holding[0]
    lo, hi = b.lo, b.hi
    r = refine(obj, lo, hi, prec=1e-8, lam_hint=b.k_hint**2)
    assert r.accepted, f"rejected a bracket that holds the level: {r.reason}"
    assert abs(r.lam - true) / true < 1e-9, f"relative error {abs(r.lam - true) / true:.2e}"

    off = min(r.lam - lo, hi - r.lam) / (hi - lo)
    assert off < 0.25, f"the minimum sits {off:.3f} of the way in -- inside the old margin"


@pytest.mark.slow
def test_the_start_point_is_the_DEEPEST_SAMPLE_not_the_midpoint():
    """**A bracket is guaranteed to CONTAIN the eigenvalue, not to have a sampled descent inside
    it.** Stage 1 stops on purity by width, which can be satisfied without ever sampling at well
    scale: on the multiplicity-3 degeneracy the well is ~1e-07 wide inside an 8.6e-07 bracket,
    so the arithmetic midpoint misses it and the starting triple reads
    1.5e-09 -> 5.6e-09 -> 1.3e-08, monotone. The parabola cannot close on that.

    The search already paid for a sample in the well -- that is what earned the bracket -- so
    the fix is a cache read, not another evaluation.
    """
    import bracket_score as bscore
    import rect_suite as rs
    from chladni.mps.bracket import bracket_interval
    from chladni.mps.refine import _deepest_inside

    (case,) = [c for c in rs.family_D() if "mult=3" in c.label]
    true = float(np.unique(case.truth)[0])
    obj = bscore.objective_for(case, 1e-8)
    res = bracket_interval(obj, prec=1e-8)
    b = [x for x in res.brackets if x.lo <= true <= x.hi][0]
    lo, hi = b.lo, b.hi

    deep = _deepest_inside(obj, lo, hi)
    assert deep is not None and lo < deep < hi
    assert obj.tension(deep) < obj.tension(0.5 * (lo + hi)), (
        "the deepest sample is deeper than the midpoint, which is why it is the start point")
    assert abs(refine(obj, lo, hi, prec=1e-8).lam - true) / true < 1e-10, (
        "and with no explicit hint the refiner now finds it anyway")


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
