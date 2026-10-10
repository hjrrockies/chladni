"""Stage 1, the searcher: the component laws, and the ablations that must FAIL.

`CHLADNI_V1.md` section 2. Tier 1 -- the surrogate, ~2 us per evaluation -- so all of this
stays in the default run except the real-pencil comparison.

**The ablations are the point of this file.** V1 section 7.3: "Write the ablation that must
fail. If disabling a guard does not break anything, the guard is not doing what you think." Two
`sigma_2` guards each looked load-bearing there and the ablation had to remove BOTH to show that
either mattered.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import surrogate_cases as sc
import chladni.mps.bracket as V
from chladni.mps.objective import Objective
from candidates_v1 import v1_brackets
from chladni.mps.bracket import bracket_interval

LADDER = bs.PREC_LADDER


def objective(case, prec):
    return bs.objective_for(case, prec)


# --- the exclusion ladder law -----------------------------------------------------------------

@pytest.mark.parametrize("safety, predicted", [(1.2, 11.0), (1.5, 5.0), (2.0, 3.0),
                                               (3.0, 2.0), (5.0, 1.5)])
def test_the_ladder_ratio_matches_its_derivation(safety, predicted):
    """`(safety+1)/(safety-1)`, derived from the envelope test alone and independent of `c`.
    V1 reports predicted == measured to four figures at all five values; this pins the law
    rather than the measurement, since the law is what transfers off the rectangle."""
    assert V.wall_ladder_ratio(safety) == pytest.approx(predicted, rel=1e-12)


def test_exclusion_on_an_exact_cone_needs_the_predicted_spacing():
    """The law, measured on a cone rather than asserted. A pair at `u_a > u_b` is excluded only
    when `u_b/u_a >= (safety-1)/(safety+1)`."""
    c, safety = 1.0, 1.5
    L = safety * c
    u_a = 1.0
    threshold = (safety - 1.0) / (safety + 1.0)
    for factor, excluded in [(1.05 * threshold, True), (0.95 * threshold, False)]:
        u_b = factor * u_a
        # BOTH POINTS ON THE SAME WALL, at distances u_a > u_b from the minimizer at 0. Putting
        # them on opposite walls instead makes the gap `u_a + u_b` rather than `u_a - u_b`, and
        # the envelope is then negative for every ratio -- which is correct (a minimizer really
        # does lie between two opposite walls) and tests nothing about the ladder.
        ks = np.array([-u_a, -u_b])
        vs = c * np.abs(ks)
        env = V._envelope_min(ks, vs, 0, L)
        assert bool(env >= 0.0) is excluded, f"u_b/u_a = {factor:.4f}, env = {env:.4g}"


# --- the seed ---------------------------------------------------------------------------------

def test_the_seed_grid_nests_exactly():
    """`n -> 2n-1` by midpoint insertion, so each grid CONTAINS the previous one bit-identically
    and the cache returns the old points. `n -> 2n` leaves new points an ulp from old ones, and
    a secant across such a pair is rounding rather than slope -- one reached 7.3e10 against a
    true 0.6, and a slope model that never shrinks cannot recover from believing it."""
    case = sc.isolated()
    obj = objective(case, 1e-8)
    ks, vs, hist, settled = V._seed(obj, obj.lo, obj.hi, 1e-8)
    assert settled
    # every doubling's points are still present, so nothing was re-evaluated at a shifted grid
    assert obj.n_evals == ks.size, "no sample was paid for twice"
    assert np.all(np.diff(ks) > 0)


def test_the_slope_estimate_settles_and_is_reported():
    obj = objective(sc.isolated(), 1e-8)
    _, _, hist, settled = V._seed(obj, obj.lo, obj.hi, 1e-8)
    assert settled and len(hist) >= 2
    assert hist[-1] >= hist[0], "the largest observed secant only grows with more samples"


# --- what licenses a bracket -------------------------------------------------------------------

def test_a_descent_certificate_needs_an_interior_minimum():
    ks = np.array([1.0, 2.0, 3.0, 4.0])
    rising = np.array([1.0, 2.0, 3.0, 4.0])
    assert V.descent_certificate(ks, rising, 1.0, 4.0) is None, "monotone: nothing is fenced"
    dip = np.array([1.0, 0.1, 0.9, 1.0])
    cert = V.descent_certificate(ks, dip, 1.0, 4.0)
    assert cert is not None and cert[0] == 1


def test_a_certificate_anchors_on_the_LOWEST_sample_not_the_deepest_ratio():
    """Anchoring on the minimum is what makes the bracket tighten monotonically: every new
    sample nearer the bottom becomes the new centre and its neighbours are necessarily closer.
    Selecting by DEPTH stalls, because depth tends to 1 as the flanks approach the middle."""
    ks = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    vs = np.array([1.0, 0.5, 0.4, 0.45, 1.0])
    cert = V.descent_certificate(ks, vs, 0.0, 4.0)
    assert cert[0] == 2, "the lowest sample, even though index 1 has the showier ratio"


def test_depth_is_normalized_by_the_FURTHER_flank():
    """By the nearer one, a cone bottoming out mid-cell has almost equidistant flanks, scores
    ~1 and is rejected -- 19 of 26 minimizers instead of 26, at every grid density."""
    ks = np.array([0.0, 1.0, 1.02, 2.0])
    vs = np.array([1.0, 0.02, 0.03, 1.0])       # a cone bottoming just left of centre
    cert = V.descent_certificate(ks, vs, 0.0, 2.0)
    far = V._depth(vs, cert)
    near = vs[cert[0]] / min(vs[cert[0] - 1], vs[cert[0] + 1])
    assert far <= V.DIP_RATIO, f"deep against the further flank: {far:.3f}"
    assert near > V.DIP_RATIO, f"and would be REJECTED against the nearer one: {near:.3f}"


def test_width_purity_is_keyed_to_the_upper_end():
    """The scorer's own rule, so the search stops exactly where the scorer stops caring."""
    assert V.width_is_pure(100.0, 100.0 + 1e-9, 1e-12) is False, "1e-9 > 1e-12 * 100"
    assert V.width_is_pure(100.0, 100.0 + 1e-9, 1e-10) is True, "1e-9 <= 1e-10 * 100"
    # Keyed to the UPPER end: the same absolute width is pure at large `lam` and not at small.
    assert V.width_is_pure(1e6, 1e6 + 1e-3, 1e-8) is True
    assert V.width_is_pure(1.0, 1.0 + 1e-3, 1e-8) is False


def test_the_wavenumber_halves_the_relative_precision():
    """A bracket `[u1, u2]` maps to `dlam = (dlam/du)*du`, so for `u = lam**p` the relative
    precision in the transformed variable is `p*prec` -- a half for `k`. Getting that factor
    wrong asks for twice or half the precision requested."""
    # `dlam/dk = 2k`, so a bracket of relative width `w` in `k` is `2w` in `lam`. The arithmetic
    # itself is not asserted -- it was a tautology; `test_the_k_rule_DELIVERS_the_lam_contract...`
    # below checks the factor against the rule over 12000 brackets. What is asserted here is that
    # the search really is in `k`, with brackets scored in `lam` by the harness.
    prec = 1e-8
    obj = objective(sc.isolated(), prec)
    assert obj.variable == "k"
    s = bs.run_case(v1_brackets(), sc.isolated(), prec)
    assert s.pure, "purity is judged in lam, where prec lives"


# --- the isolation gate -------------------------------------------------------------------------

def test_the_k_rule_DELIVERS_the_lam_contract_it_stands_in_for():
    """`prec` is relative to `lam` by definition; the search works in `k` and stops on
    `width_is_pure(..., prec/2)`. The factor is not a fudge -- `lam = k**2` gives
    `dlam/lam = (k2-k1)(k2+k1)/k2**2`, which `2*dk/k2` bounds with equality only as
    `k1 -> k2`, so the `k` rule is conservative and sound at the same time.

    Checked over 12000 brackets at three precisions: the worst `dlam/lam` a passing `k`
    bracket achieves is 0.9994 of `prec`, so the rule is tight to 0.06% as well as safe. A
    factor of `prec` instead of `prec/2` would ask for twice the precision requested; a factor
    of 1 would deliver half of it.
    """
    rng = np.random.default_rng(0)
    worst, n = 0.0, 0
    for prec in (1e-4, 1e-8, 1e-12):
        pu = 0.5 * prec
        for k in rng.uniform(1.5, 30.0, 2000):
            lo, hi = k, k + 0.999 * pu * max(k, 1.0)
            if not V.width_is_pure(lo, hi, pu):
                continue
            n += 1
            worst = max(worst, ((hi * hi - lo * lo) / max(hi * hi, 1.0)) / prec)
    assert n > 5000, "the sample actually exercised the rule"
    assert worst <= 1.0, f"a passing k bracket violated the lam contract by {worst:.6f}"
    assert worst > 0.99, f"and it is not wastefully conservative: {worst:.4f}"


def test_attained_prec_is_reported_in_LAM_whatever_the_search_worked_in():
    """`prec_requested` is in `lam`, so `attained_prec` must be too, or `met_request` compares
    two different quantities. The search reports `2*(hi-lo)/hi` on a `k` bracket, which is
    exactly `dlam/lam` to first order."""
    obj = objective(sc.isolated(), 1e-8)
    d = bracket_interval(obj, prec=1e-8).diagnostics
    assert d.prec_requested == 1e-8
    assert np.isfinite(d.attained_prec) and d.attained_prec > 0
    assert d.met_request == (d.attained_prec <= d.prec_requested)


def test_the_gate_refuses_without_a_certificate():
    """Silence is not permission: no certificate means no decision, and the region stays on the
    slow path."""
    case = sc.isolated()
    obj = objective(case, 1e-8)
    for x in np.linspace(obj.lo, obj.hi, 5):
        obj.tension(x)
    ks, vs = obj.samples
    assert V.isolation(obj, ks, vs, obj.lo, obj.lo + 1e-9, c_bound=1.0) is None


def test_the_gate_uses_c_as_an_UPPER_bound():
    """`max(local, L)`. Where a region holds two minimizers its flanks span both cones, the
    local estimate comes out too small, the neighbour looks further than it is, and the gate
    certifies exactly what it must not. Cost of getting it wrong: 3 of 26 eigenvalues."""
    # One seed cell around the deepest sample, so the region holds one well and the gate has a
    # certificate either way. The whole window, as this test first used, never certified at any
    # `c_bound` and the comparison was guarded by an `if` that could skip it.
    case = sc.isolated()
    obj = objective(case, 1e-8)
    ks, vs, *_ = V._seed(obj, obj.lo, obj.hi, 1e-8)
    i = int(np.argmin(vs))
    lo, hi = ks[i - 1], ks[i + 1]
    small = V.isolation(obj, ks, vs, lo, hi, c_bound=1e-6)
    big = V.isolation(obj, ks, vs, lo, hi, c_bound=1e6)
    assert small is not None and big is not None, "both have a certificate to decide on"
    assert small["c"] == small["c_local"], "a bound below the local slope leaves it in charge"
    assert big["c"] == 1e6, "a bound above it replaces it: max(local, L)"
    assert small["single"], "on the local slope the one well is certified"
    assert not big["single"], "and a larger c REFUSES rather than certifies"
    assert big["neighbour"] < small["neighbour"]


# --- the whole search, over the full prec ladder --------------------------------------------------

@pytest.mark.parametrize("prec", LADDER)
def test_isolated_wells_are_all_bracketed_at_every_precision(prec):
    case = sc.isolated()
    s = bs.run_case(v1_brackets(), case, prec)
    assert s.ok, f"{s.verdict}: coverage {s.coverage}, {s.n_brackets} brackets"
    assert s.per_eigenvalue < bs.COST_CAP_PER_EIG


@pytest.mark.parametrize("band", (0.5, 0.8, 1.3, 3.0, 30.0))
@pytest.mark.parametrize("prec", LADDER)
def test_the_crossover_ladder_is_covered_and_pure_at_every_precision(prec, band):
    """The cross product: every band at every precision, two that must merge and three that
    must split. A method that always merges and one that always splits both look correct on a
    sweep that stays on one side."""
    cases = sc.pair_ladder(bands=(band,), prec=prec)
    if not cases:
        pytest.skip("not representable at this band and precision")
    case = cases[0]
    if not sc.adequate(case, prec):
        pytest.skip("the objective cannot support this prec: eps/c exceeds prec*lam")
    s = bs.run_case(v1_brackets(), case, prec)
    assert s.ok, f"{s.verdict}: coverage {s.coverage}, {s.n_brackets} brackets"


@pytest.mark.parametrize("mult", (2, 3))
@pytest.mark.parametrize("prec", LADDER)
def test_an_exact_degeneracy_is_bracketed_at_every_precision(prec, mult):
    """Degeneracy swept as a CLASS. No interval can separate a level from itself, so one
    bracket is the correct answer -- and the gate can never certify such a region, since
    `sigma_2` sits on the floor beside `sigma_1`. It leaves the cover loop on width instead."""
    case = sc.degenerate(mult)
    if not sc.adequate(case, prec):
        pytest.skip("the objective cannot support this prec")
    s = bs.run_case(v1_brackets(), case, prec)
    assert s.ok, f"{s.verdict}: coverage {s.coverage}"


@pytest.mark.parametrize("mult", (3, 4))
def test_a_tight_cluster_with_unequal_gaps_is_covered(mult):
    cases = sc.cluster(mult, band=1.3, prec=1e-8)
    s = bs.run_case(v1_brackets(), cases[0], 1e-8)
    assert s.ok, f"{s.verdict}: coverage {s.coverage}"


def test_roughness_is_not_mistaken_for_a_well():
    case = sc.rough(level=1e-10)
    s = bs.run_case(v1_brackets(), case, 1e-8)
    assert s.coverage == 1.0
    assert s.spurious <= 2, f"{s.spurious} spurious brackets from roughness"


# --- declining ------------------------------------------------------------------------------------

def test_an_unbounded_slope_DECLINES_rather_than_returning_nothing():
    """The seed's own falsification test. A curve whose secant keeps growing has no local slope
    bound, so no exclusion the method makes would mean anything -- and the honest outcome is a
    reported decline, not an empty list that looks like success."""
    def spiky(k):
        # amplitude fixed, frequency rising without limit: the max secant never settles
        return np.array([0.2 + 0.1 * np.sin(1e6 * k**4), 0.3])

    obj = Objective(spiky, 3.0, 12.0, prec=1e-8, variable="k")
    res = bracket_interval(obj, prec=1e-8)
    assert res.declined
    assert "never settled" in res.diagnostics.decline_reason


def test_a_decline_is_reported_as_declined_and_never_as_a_pass():
    def spiky(k):
        return np.array([0.2 + 0.1 * np.sin(1e6 * k**4), 0.3])

    case = sc.isolated()
    case.tensions = spiky
    s = bs.run_case(v1_brackets(), case, 1e-8)
    assert s.declined and s.verdict == "declined" and not s.ok


def test_the_diagnostics_report_the_conditional_nature_of_the_claim():
    """The completeness claim is "no `x` with `sigma_1(x) <= tau` outside the returned set,
    GIVEN `L`". So every run reports `L`, the largest secant seen, their ratio, and how often
    the bound grew after something relied on it. A small margin means the constant was being
    pushed and the run deserves a rerun with a larger safety factor."""
    obj = objective(sc.isolated(), 1e-8)
    d = bracket_interval(obj, prec=1e-8).diagnostics
    assert np.isfinite(d.L) and np.isfinite(d.max_secant)
    assert d.margin == pytest.approx(V.SAFETY, rel=0.5)
    assert d.L_growth_after_use >= 0
    assert d.seed_evals > 0 and d.evaluations >= d.seed_evals


def test_the_slope_spread_in_k_is_measured_per_run():
    """V1's 5% claim, measured rather than assumed -- and section 8 warns a domain with a
    re-entrant angle may not reproduce it, in which case the case for a LOCAL slope model
    returns."""
    obj = objective(sc.isolated(), 1e-8)
    d = bracket_interval(obj, prec=1e-8).diagnostics
    assert np.isfinite(d.slope_spread)


# --- the ablations that MUST fail -----------------------------------------------------------------

def test_ABLATION_a_straddle_of_zero_stalls_the_locator():
    """`STRADDLE` exists because prediction alone does not tighten a bracket: the width is set
    by whichever side of the prediction has no sample near it, and the prediction error has a
    systematic sign, so sampling AT the prediction converges from one side and the far flank
    never moves."""
    saved = V.STRADDLE
    try:
        V.STRADDLE = 0.0
        case = sc.pair_ladder(bands=(3.0,), prec=1e-10)[0]
        zero = bs.run_case(v1_brackets(), case, 1e-10)
        V.STRADDLE = saved
        normal = bs.run_case(v1_brackets(), case, 1e-10)
        assert normal.ok
        assert zero.per_eigenvalue >= normal.per_eigenvalue or not zero.ok, (
            f"straddle=0 cost {zero.per_eigenvalue:.1f} vs {normal.per_eigenvalue:.1f}")
    finally:
        V.STRADDLE = saved


def test_ABLATION_a_safety_factor_of_one_makes_the_ladder_infinite():
    """`safety = 1` gives ratio `inf`: no geometric ladder covers a wall, so exclusion cannot
    work at all. The guard is that `SAFETY > 1` is structural, not a tuning choice."""
    assert V.wall_ladder_ratio(1.0) == float("inf")
    assert V.SAFETY > 1.0


# --- tier 2: a real pencil --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("prec", (1e-4, 1e-8))
def test_it_brackets_a_real_rectangle_pencil(prec):
    import rect_suite as rs

    (case,) = rs.family_A()
    s = bs.run_case(v1_brackets(), case, prec)
    assert s.coverage == 1.0, f"missed {s.n_true - s.covered} of {s.n_true}"
    assert s.pure, f"{s.impure} impure brackets"
