"""Stage 4: the composed pipeline, and what composition costs that the stages did not.

`docs/design/candidates.md` section 1 -- `solve` is the only mandatory seam, so it is where v1
and a method that does not decompose are compared on equal terms.

Every stage passed its own tests before this file existed, and composing them failed on every
exact degeneracy. That is the point of the file: seam defects are invisible from either side.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import level_score as ls
import rect_suite as rs
import chladni.mps.multiplicity as M
from chladni.mps.objective import Declined, Objective
from candidates_v1 import v1_solve
from chladni.mps.search import solve



def _cone(lams, eps=1e-13, c_k=0.27, plateau=0.2, lo=15.0, hi=110.0, prec=1e-8):
    ks = np.sqrt(np.asarray(lams, dtype=float))

    def tensions(lam):
        d = np.sort(np.abs(ks - np.sqrt(float(lam))))
        return np.minimum(np.sqrt(eps**2 + (c_k * d) ** 2), plateau)

    return Objective(tensions, lo, hi, prec=prec, variable="lam")


# --- the seam ---------------------------------------------------------------------------------

def test_the_seam_returns_lam_mult_pairs():
    """`[(lam, mult), ...]`, which is what a monolithic candidate must also return."""
    lams = [2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]
    out = v1_solve()(_cone(lams).tensions, 15.0, 110.0, prec=1e-8, objective=_cone(lams))
    assert isinstance(out, list) and out
    for item in out:
        lam, mult = item
        assert lam > 0 and isinstance(mult, int) and mult >= 1


def test_a_decline_propagates_as_a_VALUE_not_an_empty_list():
    """An empty list looks like success. `Declined` is falsey and carries the reason."""
    def spiky(lam):
        return np.array([0.2 + 0.1 * np.sin(1e6 * lam**2), 0.3])

    obj = Objective(spiky, 15.0, 110.0, prec=1e-8, variable="lam")
    out = v1_solve()(obj.tensions, obj.lo, obj.hi, prec=1e-8, objective=obj)
    assert isinstance(out, Declined) and not out
    assert out.reason


def test_the_result_is_sorted_and_unfoldable():
    lams = [2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]
    res = solve(_cone(lams), prec=1e-8)
    got = [x.lam for x in res.levels]
    assert got == sorted(got)
    assert res.unfolded.size == sum(x.mult for x in res.levels)


# --- clause 2: accuracy is reported, not assumed -----------------------------------------------

def test_each_level_reports_the_accuracy_it_ACHIEVED():
    """`search.md` clause (2): within `prec`, OR report `delta_res` as the attained bound.
    There is no predicted-accuracy constant -- it is measured from the parabola fit."""
    res = solve(_cone([2 * np.pi**2, 5 * np.pi**2]), prec=1e-8)
    assert res.levels
    for x in res.levels:
        assert np.isnan(x.attained_prec) or x.attained_prec >= 0.0
    d = res.diagnostics
    assert d.prec_requested == 1e-8
    assert d.met_request == (not d.unmet)


def test_clause_3_reports_what_a_caller_cannot_otherwise_see():
    """`S` is unknown on every domain that matters, so the honesty clause is what stands in for
    truth on a real run: where the evaluations went, what the search declined, what the refiner
    rejected, and how many roots were folded together."""
    res = solve(_cone([2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]), prec=1e-8)
    d = res.diagnostics
    assert d.evaluations == (d.search_evals + d.refine_evals + d.multiplicity_evals
                            + d.probe_evals + d.side_evals)
    assert d.search is not None and np.isfinite(d.search.L)
    assert d.n_brackets >= len(res.levels)


def test_stage_3_SPENDS_NOTHING_inside_the_pipeline():
    """The contract that makes the cost model honest. It was NOT free at first: estimating the
    wall slope with an explicit `lam +/- h` probe cost two evaluations per level, and charging
    them to stage 3 contradicted its own docstring. The slope now comes from the two cached
    samples either side of the root."""
    res = solve(_cone([2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]), prec=1e-8)
    assert res.diagnostics.multiplicity_evals == 0


# --- Phase D's distance prefilter, and what it replaces ----------------------------------------

def test_roots_within_prec_are_ONE_level_and_read_once():
    """Two located roots inside `prec*lam` are one level of `S(prec)`. Reading multiplicity at
    each separately double-counts, because the wide bracket spans the whole group -- correctly,
    since the group IS the level -- so each fragment declares the group's total."""
    lam = 5 * np.pi**2
    res = solve(_cone([2 * np.pi**2, lam, lam * (1 + 1e-10), 8 * np.pi**2]), prec=1e-8)
    near = [x for x in res.levels if abs(x.lam - lam) / lam < 1e-6]
    assert len(near) == 1, f"a sub-licence pair must come back as one level, got {len(near)}"


def test_reading_once_per_group_gets_the_COLLAPSE_branch_without_the_gram_angle():
    """`search.md` Phase D gives two outcomes for roots this close -- the same eigenfunction
    found twice, which must collapse and RECOUNT, or two independent ones, which must merge and
    SUM -- and warns "do NOT sum" on the first, because summation is the double count. A single
    read on the merged group recounts by construction, and is right either way: one well found
    twice reads its own multiplicity, and two genuine eigenvalues within `prec` both dip and are
    both counted.

    What is still missing is the case the angle is really for: two roots at the SAME separation
    where one pair is a duplicate and the other is not. Distance cannot tell those apart and
    neither can this -- but recounting is the safe branch.
    """
    lam = 5 * np.pi**2
    res = solve(_cone([2 * np.pi**2, lam, lam * (1 + 1e-10)]), prec=1e-8)
    near = [x for x in res.levels if abs(x.lam - lam) / lam < 1e-6]
    assert len(near) == 1
    assert near[0].mult == 2, f"recounted, not summed away: got {near[0].mult}"


def test_a_declared_width_is_HELD_TO_by_the_scorer():
    """A level may span what it declares and no more. Plain tuples declare nothing, which is
    why the adversaries are unaffected by this allowance."""
    lam = 5 * np.pi**2
    truth = np.array([lam, lam * (1 + 4e-8)])

    class Wide:
        def __init__(self, lam, mult, width):
            self.lam, self.mult, self.width = lam, mult, width

    honest = ls.score_levels([Wide(lam, 2, 4e-7)], truth, 1e-8)
    assert honest.n_impure == 0 and honest.n_declared_wide == 1
    assert honest.total_right

    silent = ls.score_levels([(lam, 2)], truth, 1e-8)
    assert not silent.ok, "declaring nothing means getting no allowance"
    assert silent.verdict == "missed", (
        "and the second eigenvalue is not even assigned to it -- 4e-08 away against a match "
        "radius of prec, so the level is not merely impure, it does not reach")


# --- real pencils ------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("name", ["disk", "eq_tri", "L_shape"])
def test_the_pipeline_on_the_hard_roster(name):
    import hard_roster as hr

    case = hr.ROSTER[name]()
    prec = 1e-8
    obj = bs.objective_for(case, prec)
    res = solve(obj, prec=prec)
    assert not res.declined, res.diagnostics.decline_reason
    s = ls.score_levels(list(res), case.truth, prec, label=name)
    assert s.ok, (f"{s.verdict}: coverage {s.coverage:.2f}, separation {s.separation:.2f}, "
                  f"{s.n_mult_right}/{s.n_matched} multiplicities")
    assert res.diagnostics.multiplicity_evals == 0


# was ulp_sensitive; 0 of 32 jitter seeds on the rung-sized suite pencil (2026-10-11)
@pytest.mark.slow
@pytest.mark.parametrize("mult", [2, 3, 4])
def test_the_pipeline_on_an_exact_degeneracy(mult):
    """The case composition failed on: all three scored separation 0.00 before the seam defects
    were fixed, while every stage passed its own tests."""
    (case,) = [c for c in rs.family_D() if f"mult={mult}" in c.label]
    obj = bs.objective_for(case, 1e-8)
    res = solve(obj, prec=1e-8)
    assert not res.declined
    s = ls.score_levels(list(res), case.truth, 1e-8, label=case.label)
    assert s.ok, f"{s.verdict}"
    assert max(x.mult for x in res.levels) == mult


@pytest.mark.slow
@pytest.mark.parametrize("band", [0.5, 0.8, 1.3, 3.0])
@pytest.mark.parametrize("window", ["pair", "cluster"])
def test_the_pipeline_across_the_crossover(window, band):
    """Both sides of the merge licence. A method that always merges and one that always splits
    both look correct on a sweep that stays on one side.

    The contract is right or flagged. Bands 0.8 and 1.3 sit where the licence ratio is about 1,
    where whether two eigenvalues are one level or two is not decided by the evidence, and a
    count the solve flags there is the honest answer -- as `prec_score`'s `flagged` verdict
    accepts. Every verdict `score_levels` returns is a count verdict, so a flag excuses any of
    them; an unflagged one fails."""
    cs = rs.crossover_cases(1e-8, window=window, bands=(band,))
    if not cs:
        pytest.skip("not representable at this band")
    case = cs[0]
    obj = bs.objective_for(case, 1e-8)
    res = solve(obj, prec=1e-8)
    assert not res.declined, res.diagnostics.decline_reason
    s = ls.score_levels(list(res), case.truth, 1e-8, label=case.label)
    d = res.diagnostics
    flagged = (not d.certified_counts or d.n_ambiguous > 0
               or any(L.count_uncertain or L.grouping_ambiguous for L in res.levels))
    assert s.ok or flagged, (
        f"{s.verdict}, unflagged: {s.n_mult_right}/{s.n_matched} multiplicities")


@pytest.mark.slow
def test_the_cost_stays_near_thirty_per_eigenvalue_end_to_end():
    """The whole pipeline, charged as one budget: bracket, refine and multiplicity through one
    shared cache, so the cost is the union of what they sampled and not the sum."""
    import hard_roster as hr

    for case in (hr.ROSTER["disk"](), hr.ROSTER["eq_tri"](), *rs.family_A()):
        obj = bs.objective_for(case, 1e-8)
        res = solve(obj, prec=1e-8)
        per = res.diagnostics.evaluations / len(case.truth)
        assert per < bs.COST_CAP_PER_EIG, f"{case.label}: {per:.1f} per eigenvalue"


def test_a_level_with_NO_attained_bound_has_NOT_met_the_request():
    """Clause (2) is an OR: within `prec`, or REPORT the bound reached. A level whose `delta_res`
    could not be computed -- no closing parabola, so NaN -- has done neither, and counting it
    as met put it outside `diagnostics.unmet` where no caller would see it."""
    from chladni.mps.search import Level

    assert not Level(lam=5.0, mult=1).met(1e-8), "NaN is not a bound"
    assert not Level(lam=5.0, mult=1, delta_res=5.0e-7).met(1e-8)
    assert Level(lam=5.0, mult=1, delta_res=5.0e-9).met(1e-8)


# --- a refused bracket that held an eigenvalue ------------------------------------------------
#
# Prompted by the one wrong cell CI ever had: `test_the_cluster_total_GRID[N65-1e-06-b2.6]` on
# Linux/OpenBLAS with `n_brackets=4, n_rejected=1`, three levels reported for four eigenvalues,
# `n_uncertain=0` -- a total short by a bracket the refiner refused, and CERTIFIED.
#
# **The rule these test does NOT catch that cell**, measured on a Linux runner 2026-09-21: the
# refusal there was at a rank-flipped flank point, `sigma = 5.18e-07`, SHALLOWER than the worst
# located level, `4.08e-07`, so `rejected_deep` is `()`. It catches refusals that are deep -- the
# starved-disk drop in `test_solve_interval.py` is the real one on record -- and these tests pin
# that behaviour. `notes/session_2026-09-19.md`, the 09-21 section, has the trace.

def test_a_rejection_that_dipped_as_DEEP_as_a_located_root_is_flagged():
    """The discriminator, and it costs nothing: an empty bracket is refused on the barrier
    between wells, orders above any root, while this one reached a located level's own depth."""
    from chladni.mps.refine import RefineResult
    from chladni.mps.search import Level, _deep_rejections

    levels = [Level(lam=10.0, mult=1, sigma=4.1e-07), Level(lam=20.0, mult=1, sigma=2.6e-11)]
    deep = RefineResult(lam=15.0, sigma=9.0e-11, accepted=False, reason="no_parabola",
                        evaluations=6)
    (got,) = _deep_rejections([deep], levels)
    assert got == (15.0, 9.0e-11, "no_parabola")


def test_a_rejection_on_the_BARRIER_stays_quiet():
    """The false-alarm side. 1.2e-05 is where this cell's barrier between wells sits, four
    orders above the deepest located root, and refusing there is the refiner working correctly."""
    from chladni.mps.refine import RefineResult
    from chladni.mps.search import Level, _deep_rejections

    levels = [Level(lam=10.0, mult=1, sigma=4.1e-07), Level(lam=20.0, mult=1, sigma=2.6e-11)]
    shallow = RefineResult(lam=15.0, sigma=1.2e-05, accepted=False, reason="not_a_minimum",
                           evaluations=3)
    assert _deep_rejections([shallow], levels) == ()


def test_with_NOTHING_located_every_refusal_is_flagged():
    """A solve that located no eigenvalue and refused a bracket has not earned a certified count
    either, and `max(..., default=inf)` is what makes that fall out rather than being special
    cased."""
    from chladni.mps.refine import RefineResult
    from chladni.mps.search import _deep_rejections

    r = RefineResult(lam=15.0, sigma=1.2e-05, accepted=False, reason="flat", evaluations=3)
    assert len(_deep_rejections([r], [])) == 1


def test_a_deep_rejection_reaches_n_uncertain_and_UNCERTIFIES_the_solve(monkeypatch):
    """End to end. The refiner is made to refuse the first bracket AT THE DEPTH IT REACHED, and
    everything else is the shipped pipeline.

    NOT a reproduction of the Linux cell, which this was once described as. There the refusal
    came at a rank-flipped bracket END, shallower than every located level, and the rule stays
    silent on it. This pins the deep case, which is the case the rule is for."""
    import dataclasses
    import chladni.mps.search as S

    lams = [2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]
    real, seen = S.refine, []

    def refuse_the_first(obj, lo, hi, **kw):
        r = real(obj, lo, hi, **kw)
        seen.append(r)
        if len(seen) == 1:
            return dataclasses.replace(r, accepted=False, reason="no_parabola")
        return r

    monkeypatch.setattr(S, "refine", refuse_the_first)
    res = S.solve(_cone(lams), prec=1e-8)
    d = res.diagnostics

    assert d.n_brackets == 3 and d.n_rejected == 1, (d.n_brackets, d.n_rejected)
    assert len(res.levels) == 2, "the total IS short -- that is the premise, not the defect"
    assert len(d.rejected_deep) == 1, d.rejected_deep
    assert d.n_uncertain == 1, "and it says so"
    assert not d.certified_counts
    (_, _, _, why) = d.uncertain[-1]
    assert "REFUSED" in why and "no_parabola" in why, why


def test_the_shipped_pipeline_flags_NOTHING_on_a_clean_cone():
    """The guard against the rule being loud everywhere: on a landscape with three separated
    roots and nothing refused, it must be inert."""
    from chladni.mps.search import solve as _solve

    lams = [2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]
    d = _solve(_cone(lams), prec=1e-8).diagnostics
    assert d.n_rejected == 0 and d.rejected_deep == ()
    assert d.certified_counts


# --- the wall slope stage 3 reads from the cache ------------------------------------------
#
# `_slope_from_cache` sizes the side search's exclusion radius. It assumed a cone that reaches
# zero; a well with a floor broke it, and one eigenvalue came back as several levels.


def _hyperbolic_well(floor, c, lam0):
    """sigma(lam) = sqrt(floor^2 + c^2 (lam - lam0)^2): a cone rounded off at `floor`."""
    return lambda lam: np.sqrt(floor**2 + (c * (np.asarray(lam, float) - lam0))**2)


def test_the_cache_slope_is_unchanged_where_there_is_no_floor():
    """`floor = 0` must be the estimator it replaced, bit for bit, or the reach rule's bounds --
    calibrated on it -- no longer mean what they were measured to mean."""
    from chladni.mps.search import _slope_from_cache

    rng = np.random.default_rng(7)
    lams = np.sort(rng.uniform(0.0, 10.0, 40))
    vs = rng.uniform(1e-3, 1.0, 40)
    for lam in (2.5, 5.0, 7.5):
        left, right = lams[lams < lam], lams[lams > lam]
        lo, hi = left.max(), right.min()
        want = (vs[lams == lo][0] + vs[lams == hi][0]) / (hi - lo)
        assert _slope_from_cache(lams, vs, lam) == want
        assert _slope_from_cache(lams, vs, lam, floor=0.0) == want


def test_a_FLOOR_no_longer_turns_the_cache_slope_into_noise():
    """The defect, in isolation. A well with a 9.6e-7 floor and a true wall slope of 0.05, cached
    the way a refinement leaves it -- two samples a few ulps either side of the root, the rest out
    on the flanks. Blind to the floor, the nearest pair reads `2*floor/(hi - lo)`: the 8.45e6
    measured at 50*pi^2, prec 1e-6, against a true 0.02-0.08. Aware of it, the pair is skipped as
    flat bottom and the flanks give the wall."""
    from chladni.mps.search import _slope_from_cache

    floor, c, lam0 = 9.6e-7, 0.05, 493.4795966366
    f = _hyperbolic_well(floor, c, lam0)
    ulp = np.spacing(lam0)
    lams = np.array([lam0 - 3e-4, lam0 - 1e-4, lam0 - 2 * ulp, lam0 + 2 * ulp,
                     lam0 + 1e-4, lam0 + 3e-4])
    vs = f(lams)

    blind = _slope_from_cache(lams, vs, lam0)
    assert blind > 1e5 * c, "the blind estimate should reproduce the defect it replaces"
    aware = _slope_from_cache(lams, vs, lam0, floor=floor)
    assert aware == pytest.approx(c, rel=0.05)


def test_the_exclusion_radius_survives_a_floor():
    """What the slope is FOR. `solve` excludes `r +- half` from the side search with
    `half = 0.5 * sigma_j / c`, and relies on it: a find inside the sub-brackets is taken to be
    NOT `r`. With the blind slope that radius was 1e-13 and the side search found `r` again 1e-12
    away, as a new level -- one eigenvalue reported twice. The radius must stay on the scale of
    the well, not of the ulps between the refiner's last samples."""
    from chladni.mps.search import _slope_from_cache

    floor, c, lam0 = 9.6e-7, 0.05, 493.4795966366
    f = _hyperbolic_well(floor, c, lam0)
    ulp = np.spacing(lam0)
    lams = np.array([lam0 - 3e-4, lam0 - 2 * ulp, lam0 + 2 * ulp, lam0 + 3e-4])
    sigma_j = 1e-5                                  # a second branch a well-width away
    half = 0.5 * sigma_j / _slope_from_cache(lams, f(lams), lam0, floor=floor)
    assert half > 1e3 * ulp


def test_a_MILD_disagreement_keeps_the_calibrated_blind_slope():
    """The guard's other half. At a hump top accepted as a root -- 65*pi^2, prec 1e-8, band 0.3,
    `sigma` 1.35e-8 there is the hump, not a floor -- the floor-aware estimate differs by 2.3x,
    and applying it flipped four calibrated tests. Below the guard the blind estimate must come
    back exactly, not merely close."""
    from chladni.mps.search import _SLOPE_FLOOR_GUARD, _slope_from_cache

    lams = np.array([0.0, 1.0, 3.0, 4.0])
    vs = np.array([4.0, 3.0, 3.0, 4.0])             # samples well up a gentle bowl
    blind = _slope_from_cache(lams, vs, 2.0)
    aware_floor = 1.2                                # a "floor" the samples sit a few times above
    got = _slope_from_cache(lams, vs, 2.0, floor=aware_floor)
    assert blind < _SLOPE_FLOOR_GUARD * got or got == blind
    assert got == blind


def test_phase_d_defaults_to_DISTANCE_and_refuses_an_unknown_rule():
    """Moved from the retired Phase D file (now `notes/evidence_tests/test_search_collapse.py`):
    the shipped default is named for what it does, and a typo is an error, not a silent fallback."""
    import inspect
    assert inspect.signature(solve).parameters["phase_d"].default == "distance"
    obj = Objective(lambda x: np.array([abs(x - 50.0) + 1e-9, 1.0]), 15.0, 110.0, variable="lam")
    with pytest.raises(ValueError, match="phase_d"):
        solve(obj, prec=1e-8, phase_d="gram")
